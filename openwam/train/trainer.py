"""OpenWAM joint video-action trainer: construction + the training loop.

Call order:

  __init__():  seed -> build the architecture for the run mode -> place -> freeze -> param report
  train():     optimizer / dataloader / LR schedule (plan_steps) -> open the run dir
               -> accelerate prepare -> wandb / vis -> (resume) load full state -> text cache
               -> loop{ per-step seed -> loss -> backward / clip / step -> reduce metrics
                        -> StepLogger -> RunDir.save_step -> vis }
               -> finish (RunDir.finish: final weights, drop resume state; close wandb)

The pieces live next to this module in ``openwam.train``: the run mode and its architecture build (``run_mode``),
parameter groups / step budget / LR schedule (``optim``), phase timing / metric reduction / per-step sinks
(``step_log``), the run dir and checkpoint files (``checkpointing``), seeding and the training-time
rollouts (``visualization``).

Usage (``scripts/train.py``):
    run_mode, ckpt_path = decide_run_mode(cfg.training)
    trainer = OpenWAMTrainer(cfg, accelerator, dataset, run_mode, ckpt_path, rank=rank)
    trainer.train()
"""

import itertools
import logging
import os

import torch
from omegaconf import DictConfig

from openwam.dataloader.collate import collate
from openwam.dataloader.utils.normalization import normalization_on
from openwam.model.common.dtypes import DTYPES
from openwam.train.checkpointing import (
    RunDir,
    compute_resume_position,
    create_run_dir,
    latest_resume_state,
    load_full_state,
    save_config,
    save_normalization_stats,
    verify_resume_normalization_stats,
)
from openwam.train.optim import build_cosine_scheduler, build_trainable_parameters, plan_steps
from openwam.train.run_mode import RunMode, build_training_architecture
from openwam.train.seeding import per_step_seed, seed_process
from openwam.train.step_log import StepLogger, StepTimer, init_wandb, log_parameter_counts, reduce_step_metrics
from openwam.train.visualization import TrainVisualizer

logger = logging.getLogger(__name__)


class OpenWAMTrainer:
    """Joint video-action trainer for OpenWAM. See module docstring for call order.

    Args:
        cfg: Hydra DictConfig with model, training, data, project sections, resolved by the entry point
            (``openwam.config.resolve_config``: ``cfg.common.layout`` is set).
        accelerator: HuggingFace Accelerator (DeepSpeed); construction reads its ``device`` and ``is_main_process``.
        dataset: training dataset (dataloader, rollouts, text cache, normalization stats); None only for
            construction-only tests.
        run_mode: RunMode the entry point decided (``run_mode.decide_run_mode``).
        ckpt_path: str, that mode's checkpoint source (finetune source / resumed run dir); None for FRESH.
        rank: int, this process's global rank (read once by the entry point); offsets every seed.
    """

    # (1) Constructor — seed -> build architecture -> place -> freeze -> param report.
    def __init__(
        self, cfg: DictConfig, accelerator, dataset, run_mode: RunMode, ckpt_path: str | None, *, rank: int
    ):
        self.cfg = cfg
        self.dataset = dataset
        self.accelerator = accelerator
        self.run_mode = run_mode
        self.ckpt_path = ckpt_path

        # ---- Reproducible seed (FastWAM-style, yaml-driven) ----
        # Seed before build_architecture so DiT/ActionDiT weight init is
        # deterministic. seed_process uses the same RANK_OFFSET rank stride as
        # per_step_seed and the launcher's seed_everything, so a process's init
        # and per-step seeds share one rank window (cudnn left to the launcher).
        # resolve_config already checked project.seed is an int.
        self._rank = rank
        self._run_seed = int(cfg.project.seed)
        seed_process(self._run_seed, rank=self._rank)
        if self._rank == 0:
            logger.info("cfg.project.seed=%d (rank=%d)", self._run_seed, self._rank)

        t = cfg.training

        from openwam.model.architectures.base import TrainOptions

        # Training knobs the architecture keeps from construction on (compute_loss reads them).
        train_options = TrainOptions(
            use_gradient_checkpointing=bool(t.use_gradient_checkpointing),
            use_gradient_checkpointing_offload=bool(t.use_gradient_checkpointing_offload),
            lambda_video=float(t.lambda_video),
            lambda_action=float(t.lambda_action),
            mot_checkpoint_mixed_attn=bool(t.mot_checkpoint_mixed_attn),
        )
        # Checked here, not in TrainOptions / compute_loss: INFERENCE_ONLY legitimately has both weights 0, and only
        # a training run needs a loss with at least one term.
        if train_options.lambda_video == 0 and train_options.lambda_action == 0:
            raise ValueError("training.lambda_video and training.lambda_action are both 0: the loss has no term.")
        # The run mode (decided by the entry point) picks the build; finetune weights land here, BEFORE the freeze.
        self.architecture = build_training_architecture(run_mode, ckpt_path, cfg, train_options)
        logger.info("Architecture: %s", type(self.architecture).__name__)

        # The weight / batch dtype, decided once from cfg (resolve_config checked the key); both
        # set_dtype_device calls (here and prepare_accelerate) use it.
        self._dtype = DTYPES[str(t.mixed_precision)]
        # Device placement: skip .to(device) when initialize_model_on_cpu — DeepSpeed's prepare() then handles the move.
        if not bool(t.initialize_model_on_cpu):
            # The accelerator's device, the same (dtype, device) prepare_accelerate passes again after prepare.
            self.architecture.set_dtype_device(self._dtype, self.accelerator.device)

        # --- Freeze: the backbone yaml's training.freeze; every path must exist on the model.
        for name in self.architecture.freeze_modules(list(t.freeze)):
            logger.info("Frozen: %s", name)

        log_parameter_counts(self.architecture, is_main=self.accelerator.is_main_process)

    # (2) Driver — setup, then the epoch / step loop.
    def train(self) -> None:
        """Run the training loop (HuggingFace Accelerate + DeepSpeed), every knob from ``cfg.training``.

        The run mode was decided at construction: fresh / finetune start at step 0 in a new run dir (finetune
        weights already loaded); resume reuses the run dir and loads its full state after prepare.

        Returns:
            None; checkpoints / logs land in the run dir.
        """
        t = self.cfg.training
        # num_epochs null = step-only training; max_steps null = no step limit.
        num_epochs = int(t.num_epochs) if t.num_epochs is not None else None
        max_steps = int(t.max_steps) if t.max_steps is not None else None
        batch_size = int(t.batch_size)
        grad_accum = int(t.gradient_accumulation_steps)
        is_main = self.accelerator.is_main_process

        optimizer = self.build_optimizer()
        dataloader = self.build_dataloader(batch_size)
        # null = no clipping (resolve_config checked a set value is > 0).
        max_grad_norm = None if t.max_grad_norm is None else float(t.max_grad_norm)

        # Per-process step budget from the pre-prepare loader: micro steps (progress bar), optimizer steps (LR horizon).
        total_steps, total_opt_steps = plan_steps(
            len(dataloader), grad_accum, self.accelerator.num_processes, num_epochs, max_steps
        )
        scheduler = self.build_lr_scheduler(optimizer, total_opt_steps)

        # Fresh / finetune: a new run dir; resume: the run's own dir and its latest state.
        run_dir, resume_state_dir = self.open_run_dir()
        optimizer, dataloader, scheduler = self.prepare_accelerate(optimizer, dataloader, scheduler)

        all_params = [p for group in optimizer.param_groups for p in group["params"]]

        # wandb on the main rank only; project.wandb.project null = off (init_wandb returns None).
        wandb_run = init_wandb(self.cfg) if is_main else None

        # Periodic world-model rollouts (GT | pred clips to wandb). Every rank holds one
        # fixed sample, picked before epoch 0 so the clips are comparable across vis steps.
        vis_steps = None if t.vis_steps is None else int(t.vis_steps)  # null = off
        visualizer = None
        if vis_steps is not None:
            # Eval semantics (no caption dropout, fixed history, no jitter), sharing the preloaded readers.
            visualizer = TrainVisualizer(
                self.dataset.eval_view(),
                rank=self._rank,
                num_samples=int(t.vis_num_samples),
                denoise_steps=int(t.vis_denoise_steps),
                fps=int(t.vis_fps),
                seed=self._run_seed,
            )

        opt_step = 0
        global_step = 0
        start_epoch = 0
        skip_first = 0

        # Resume: restore full state AFTER prepare, then map global_step -> (epoch, skip).
        if self.run_mode is RunMode.RESUME:
            global_step, opt_step, start_epoch, skip_first = self.load_resume_state(
                resume_state_dir, len(dataloader), grad_accum
            )
            already_done = (num_epochs is not None and start_epoch >= num_epochs) or (
                max_steps is not None and global_step >= max_steps
            )
            if already_done:
                logger.info("[resume] global_step=%d already complete; finishing.", global_step)
                self.finish(run_dir, global_step, wandb_run)
                return

        # Text cache only now that the weights are final (prepare -> resume load_full_state -> cache, the same
        # order for every mode): on resume the text encoder holds real values only after load_full_state.
        if bool(t.text_cache):
            self.build_text_cache()

        # Per-phase step timing (wandb timing/*); DeepSpeed's engine.step is hooked so
        # backward and optimizer time are billed separately.
        log_timing = bool(t.log_step_timing)
        timer = StepTimer(log_timing)
        timer.wrap_engine_step(self.accelerator.deepspeed_engine_wrapped.engine)
        step_logger = StepLogger(
            total_steps=total_steps,
            initial_step=global_step,
            wandb_run=wandb_run,
            run_path=run_dir.path,
            is_main=is_main,
            samples_per_step=batch_size * self.accelerator.num_processes,
            log_timing=log_timing,
        )
        # Batches go to the inner module's (device, dtype); the loop drives the prepared wrapper.
        arch = self.accelerator.unwrap_model(self.architecture)

        # Per-step manual_seed makes timestep + noise sampling in compute_loss
        # reproducible across runs and ZeRO stages: same (rank, step) -> same RNG,
        # different ranks at the same step keep in-batch timestep diversity.
        epochs = itertools.count(start_epoch) if num_epochs is None else range(start_epoch, num_epochs)
        for epoch in epochs:
            # The dataset keys its draws on the epoch; the loader's set_epoch keeps accelerate's shard state in step.
            dataloader.set_epoch(epoch)
            self.dataset.set_epoch(epoch)
            # On the resumed epoch, skip the batches already consumed before the checkpoint.
            if epoch == start_epoch and skip_first > 0:
                from accelerate import skip_first_batches

                epoch_iter = skip_first_batches(dataloader, skip_first)
            else:
                epoch_iter = dataloader
            for batch in epoch_iter:
                timer.mark("data")
                step_seed = per_step_seed(self._run_seed, rank=self._rank, step=global_step)
                torch.manual_seed(step_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(step_seed)
                with self.accelerator.accumulate(self.architecture):
                    # The architecture's loss dict: loss (carries the graph), loss_video, loss_action.
                    losses = self.architecture.compute_loss(batch.to(arch.device, arch.dtype))
                    timer.mark("forward")
                    loss = losses["loss"]
                    self.accelerator.backward(loss)
                    timer.mark("backward")  # ~0 under DeepSpeed: the engine.step hook already closed backward/optim

                    grad_norm = torch.tensor(0.0, device=loss.device)
                    if self.accelerator.sync_gradients:
                        if max_grad_norm is not None:
                            grad_norm_val = self.accelerator.clip_grad_norm_(all_params, max_grad_norm)
                            grad_norm = torch.tensor(float(grad_norm_val), device=loss.device)
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        optimizer.zero_grad()
                        opt_step += 1
                    timer.mark("optim")

                global_step += 1

                # Both reductions are collectives: every rank runs them, in this order.
                metrics = reduce_step_metrics(self.accelerator, losses, grad_norm)
                timing = timer.snapshot(self.accelerator)
                step_logger.log(
                    metrics,
                    global_step=global_step,
                    opt_step=opt_step,
                    epoch=epoch,
                    lr=optimizer.param_groups[0]["lr"],
                    timing=timing,
                )

                # Every save_steps: weights (+ resume state), pruned in lockstep.
                run_dir.save_step(self.accelerator, self.architecture, global_step, opt_step, epoch)

                # Step 1 too: an early clip shows the hook works long before the first vis_steps.
                if visualizer is not None and (global_step % vis_steps == 0 or global_step == 1):
                    visualizer.run(self.accelerator, self.architecture, global_step, wandb_run, run_dir.path)

                timer.end_step()  # bills metric reduce / logging / save / vis to "other"

                if max_steps is not None and global_step >= max_steps:
                    step_logger.close()
                    self.finish(run_dir, global_step, wandb_run)
                    return

        step_logger.close()
        self.finish(run_dir, global_step, wandb_run)

    # (3) Called by train() first — AdamW over the per-module (action/video) LR param groups.
    def build_optimizer(self) -> torch.optim.Optimizer:
        """AdamW over the per-module (action/video) LR parameter groups."""
        t = self.cfg.training
        params = build_trainable_parameters(
            self.architecture,
            # null = that module uses the base learning_rate; a number (0 included) is used as written.
            action_lr=None if t.action_lr is None else float(t.action_lr),
            video_lr=None if t.video_lr is None else float(t.video_lr),
        )
        betas = tuple(t.adam_betas)
        return torch.optim.AdamW(params, lr=float(t.learning_rate), weight_decay=float(t.weight_decay), betas=betas)

    # (4) Called by train() — build the training DataLoader (indices in order; the dataset does the random draw).
    def build_dataloader(self, batch_size: int) -> torch.utils.data.DataLoader:
        """Build the training DataLoader: indices read in order, no shuffle and no loader RNG.

        The dataset already draws each sample at random from an RNG keyed on ``(seed, epoch, index)``, so a
        loader shuffle would only permute keys. Without one, the sample at a step is a pure function of
        ``(epoch, position)``: a resume needs only ``set_epoch`` + ``skip_first_batches``, no saved RNG state.

        Args:
            batch_size: int, per-GPU batch size.

        Returns:
            torch DataLoader over ``self.dataset``.
        """
        t = self.cfg.training
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(t.dataset_num_workers),
            collate_fn=collate,
            pin_memory=True,
        )

    # (5) Called by train() — linear-warmup + cosine schedule, or None (constant LR).
    def build_lr_scheduler(self, optimizer, total_opt_steps: int):
        """Linear-warmup + cosine schedule, or None (constant LR).

        Args:
            optimizer: the AdamW from :meth:`build_optimizer`.
            total_opt_steps: int, per-process optimizer steps of the run.

        Returns:
            SequentialLR for ``training.lr_scheduler: cosine``; None for null (constant LR).
        """
        t = self.cfg.training
        # null = constant LR; resolve_config allows only null / cosine.
        if t.lr_scheduler is None:
            return None
        return build_cosine_scheduler(
            optimizer, total_opt_steps=total_opt_steps, training_cfg=t, num_processes=self.accelerator.num_processes
        )

    # (6) Called by train() — the run dir (new, or the resumed run's) and the state a resume continues from.
    def open_run_dir(self) -> tuple[RunDir, str | None]:
        """Create (fresh / finetune) or reuse (resume) the run dir, with its save policy from ``cfg.training``.

        Every rank resolves the resume state independently (shared FS), so a missing state raises on every rank
        without deadlocking a broadcast.

        Returns:
            (run_dir, resume_state_dir): the RunDir; the ``accel_state_step_*`` dir to load (None unless RESUME).
        """
        t = self.cfg.training
        if self.run_mode is RunMode.RESUME:
            resume_state_dir = latest_resume_state(self.ckpt_path)
            path = os.path.dirname(resume_state_dir)
            # Keep the run dir's deploy stats; refuse the resume when they differ from the dataset's in any way.
            if normalization_on(self.cfg.dataloader.normalize_mode):
                verify_resume_normalization_stats(path, self.dataset.normalization_stats_path)
        else:
            path = create_run_dir(self.accelerator, t.output_path, t.run_dir_name, self.write_run_assets)
            resume_state_dir = None
        run_dir = RunDir(
            path,
            # null = never save (the yaml's "disable saving").
            save_steps=None if t.save_steps is None else int(t.save_steps),
            keep_last_k=int(t.keep_last_k_ckpts),
            save_full_states=bool(t.save_full_states_for_resume),
        )
        return run_dir, resume_state_dir

    # (6b) Called by create_run_dir on rank 0 — the self-contained deploy artifacts of a new run dir.
    def write_run_assets(self, path: str) -> None:
        """Backbone non-weight assets (from model_path, or the source ckpt dir on finetune), config, action stats.

        Args:
            path: str, the new run dir.
        """
        self.architecture.save_assets_for_deployment(path)
        save_config(path, self.cfg)
        if normalization_on(self.cfg.dataloader.normalize_mode):
            save_normalization_stats(path, self.dataset.normalization_stats_path)

    # (7) Called by train() — wrap architecture/optimizer/dataloader with the DeepSpeed Accelerator.
    def prepare_accelerate(self, optimizer, dataloader, scheduler):
        """Wrap architecture / optimizer / dataloader (and the scheduler, when there is one) with the Accelerator.

        Args:
            optimizer: the AdamW from :meth:`build_optimizer`.
            dataloader: the DataLoader from :meth:`build_dataloader`.
            scheduler: the LR scheduler, or None (constant LR: nothing to prepare).

        Returns:
            (optimizer, dataloader, scheduler), prepared; ``self.architecture`` becomes the prepared handle.
        """
        prepare_args = [self.architecture, optimizer, dataloader]
        if scheduler is not None:
            prepare_args.append(scheduler)
            self.architecture, optimizer, dataloader, scheduler = self.accelerator.prepare(*prepare_args)
        else:
            self.architecture, optimizer, dataloader = self.accelerator.prepare(*prepare_args)
        # The loop drives the prepared handle; architecture-level setup runs on the inner module
        # (unwrap_model: the module itself when Accelerate adds no wrapper).
        arch = self.accelerator.unwrap_model(self.architecture)
        # Propagate device down through architecture; frozen modules (T5/VAE) idempotent move.
        arch.set_dtype_device(self._dtype, self.accelerator.device)
        arch.move_frozen_to_device(self.accelerator.device)
        logger.info(
            "architecture wrapped (%s), device=%s",
            type(self.architecture).__name__,
            self.accelerator.device,
        )
        return optimizer, dataloader, scheduler

    # (7b) Called by train() after prepare (and the resume load) when training.text_cache is on.
    def build_text_cache(self) -> None:
        """Pre-encode the dataset's full prompt set; the frozen text encoder then sits on CPU.

        Input: none (uses self.dataset.all_prompts() and the unwrapped architecture's video backbone).
        Returns: None. The backbone raises if it cannot cache (cosmos3: its yaml says training.text_cache: false).
        """
        arch = self.accelerator.unwrap_model(self.architecture)
        prompts = self.dataset.all_prompts()
        arch.video_backbone.build_text_cache(prompts)
        logger.info("text cache: cached %d prompts; text encoder moved to CPU", len(prompts))

    # (8) Called by train() on the resume path — restore full state, map step -> (start_epoch, skip).
    def load_resume_state(self, resume_state_dir: str, batches_per_epoch: int, grad_accum: int) -> tuple:
        """Load the full state (after prepare) and map its global_step to where the loop restarts.

        Args:
            resume_state_dir: str, the ``accel_state_step_*`` dir from :meth:`open_run_dir`.
            batches_per_epoch: int, ``len(dataloader)`` of the PREPARED loader (per process).
            grad_accum: int, ``training.gradient_accumulation_steps``.

        Returns:
            (global_step, opt_step, start_epoch, skip): ints; global_step pulled back to the grad_accum boundary
            ``skip`` (batches to skip in ``start_epoch``) was floored to. ValueError for a saved step <= 0.
        """
        is_main = self.accelerator.is_main_process
        if is_main:
            logger.info("[resume] loading Accelerate state from %s", resume_state_dir)
        meta = load_full_state(self.accelerator, resume_state_dir)
        global_step = int(meta["global_step"])
        # A state is only saved after a step, so a resumed step 0 means a broken trainer_state.json.
        if global_step <= 0:
            raise ValueError(f"[resume] {resume_state_dir}/trainer_state.json has global_step={global_step}; expected > 0")
        # Align global_step to the grad_accum boundary skip was floored to, then derive
        # opt_step from it — otherwise floored-off batches re-train and per-step seeds
        # (keyed on global_step) drift. No-op at grad_accum=1.
        start_epoch, skip, global_step = compute_resume_position(global_step, batches_per_epoch, grad_accum)
        opt_step = global_step // grad_accum
        # load_state restored accelerate's micro-step counter as saved (possibly mid-accumulation); the loop
        # restarts on a boundary, so the counter must too, or every optimizer step after it fires one micro early.
        self.accelerator.step = 0
        if is_main:
            logger.info(
                "[resume] resumed at global_step=%d opt_step=%d epoch=%d skip_first=%d",
                global_step,
                opt_step,
                start_epoch,
                skip,
            )
        return global_step, opt_step, start_epoch, skip

    # (9) Called on every train() exit — final weights, drop resume state, close wandb.
    def finish(self, run_dir: RunDir, global_step: int, wandb_run) -> None:
        """Teardown for every exit path: the run dir's final save / prune, then wandb.

        Args:
            run_dir: RunDir of this run.
            global_step: int, the last micro step.
            wandb_run: the wandb Run, or None (off / non-main rank).
        """
        run_dir.finish(self.accelerator, self.architecture, global_step)
        if wandb_run is not None:
            wandb_run.finish()
