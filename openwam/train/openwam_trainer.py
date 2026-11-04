"""OpenWAM joint video-action trainer (self-contained).

Call order — core skeleton only:

  __init__():  seed -> build_architecture -> freeze
               -> [latent setup] -> log param counts
  train():     build optimizer/dataloader/scheduler -> setup output dir
               -> accelerate prepare -> (resume) load full state -> text cache
               -> loop{ compute_loss -> backward/clip/step
               -> reduce metrics -> log step -> maybe save ckpt }
               -> finish_training (final ckpt, drop resume state, close wandb)

Methods below are ordered by call sequence: __init__, train, then the
helpers in the order train() reaches them (sub-helpers follow their caller).

Stateless helpers live in ``openwam.train.utils`` (config / param report / LR /
wandb / metric reduction / debug-CSV / checkpoint mgmt / optimizer groups /
seeding).

Usage:
    trainer = OpenWAMTrainer(cfg, accelerator, dataset)
    trainer.train()
"""

import itertools
import logging
import math
import os

import torch
from omegaconf import DictConfig, OmegaConf

from openwam.dataloader.collate import Batch, collate
from openwam.train.utils.checkpointing import (
    compute_resume_position,
    finalize_keep_weights_only,
    find_latest_accel_state,
    load_full_state,
    manage_checkpoints,
    save_config,
    save_full_state,
    save_normalization_stats,
    save_weights,
    verify_resume_normalization_stats,
)
from openwam.train.utils.optimizer_groups import build_trainable_parameters
from openwam.train.utils.seeding import per_step_seed, seed_process, wire_sampler_seed
from openwam.train.utils.step_timer import StepTimer
from openwam.train.utils.training_utils import (
    build_cosine_scheduler,
    init_wandb,
    log_parameter_counts,
    reduce_step_metrics,
    write_debug_loss_row,
)
from openwam.train.utils.visualization import TrainVisualizer

logger = logging.getLogger(__name__)


class OpenWAMTrainer:
    """Joint video-action trainer for OpenWAM. See module docstring for call order.

    Args:
        cfg: Hydra DictConfig with model, training, data, project sections, resolved by the entry point
            (``openwam.config.resolve_config``: ``cfg.common.layout`` is set).
        accelerator: HuggingFace Accelerator instance.
        dataset: Training dataset (used for action stats loading).
    """

    # (1) Constructor — seed -> build architecture -> freeze -> init schedulers -> param report.
    def __init__(self, cfg: DictConfig, accelerator=None, dataset=None):
        self.cfg = cfg
        self.dataset = dataset
        self.accelerator = accelerator

        # ---- Reproducible seed (FastWAM-style, yaml-driven) ----
        # Seed before build_architecture so DiT/ActionDiT weight init is
        # deterministic. seed_process uses the same RANK_OFFSET rank stride as
        # per_step_seed and the launcher's seed_everything, so a process's init
        # and per-step seeds share one rank window (cudnn left to the launcher).
        # resolve_config already checked project.seed is an int.
        self._rank = int(os.environ.get("RANK", 0))
        self._run_seed = int(cfg.project.seed)
        seed_process(self._run_seed, rank=self._rank)
        if self._rank == 0:
            logger.info("cfg.project.seed=%d (rank=%d)", self._run_seed, self._rank)

        t = cfg.training

        # Build architecture (creates video_backbone internally from config). The load mode is decided
        # here, once: no ckpt -> upstream pretrained weights; finetune / resume -> the checkpoint's shell
        # (same shell for both; they differ only in how its weights are filled, see ckpt_model_loader).
        from openwam.model import build_architecture
        from openwam.model.architectures.base import TrainOptions
        from openwam.model.video_backbone import LoadMode

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
        # null (or "") = that mode is off.
        finetune_path = t.finetune_ckpt_path or None
        resume_path = t.resume_ckpt_path or None
        if finetune_path and resume_path:
            raise ValueError("finetune_ckpt_path and resume_ckpt_path are mutually exclusive; set at most one.")
        if finetune_path is not None:
            from openwam.train.utils.ckpt_model_loader import build_for_finetune

            # Finetune starts a NEW run from cfg.model, which must equal the ckpt's model (checked inside), so
            # the architecture built here and the config save_config() writes are the same.
            # Weights land here, BEFORE the freeze below.
            self.architecture = build_for_finetune(finetune_path, cfg, train_options)
        elif resume_path is not None:
            from openwam.train.utils.ckpt_model_loader import build_for_resume

            # Resume continues ONE run: cfg is its saved config.yaml (checked inside); its weights are
            # restored by load_full_state after prepare.
            self.architecture = build_for_resume(resume_path, cfg, train_options)
        else:
            # The architecture's slice: the shared values (action / state dims, layout) under the model block;
            # the upstream file paths come separately (assets, not part of the model).
            self.architecture = build_architecture(
                OmegaConf.merge(cfg.common, cfg.model),
                mode=LoadMode.PRETRAINED,
                assets=cfg.assets.video_backbone,
                ckpt_dir=None,
                train_options=train_options,
            )
        logger.info("Architecture: %s", type(self.architecture).__name__)

        # Device placement: skip .to(device) when initialize_model_on_cpu and a real
        # training Accelerator is present — DeepSpeed's prepare() then handles the move.
        _init_on_cpu = bool(t.initialize_model_on_cpu)
        if not (_init_on_cpu and self.accelerator is not None):
            self.architecture.set_dtype_device(self.architecture.dtype, self.architecture.device)

        # --- Freeze: the backbone yaml's training.freeze; every path must exist on the model.
        for name in self.architecture.freeze_modules(list(t.freeze)):
            logger.info("Frozen: %s", name)

        self.model = self  # self-reference some external callers expect

        is_main = self.accelerator is None or self.accelerator.is_main_process
        log_parameter_counts(self.architecture, is_main=is_main)

    # (2) Driver — build optimizer/dataloader/scheduler -> setup dir -> accelerate prepare
    #     -> (resume) -> epoch/step loop{compute_loss -> log_step -> save} -> finish_training.
    def train(self) -> None:
        """Run the training loop (HuggingFace Accelerate distributed), every knob from ``cfg.training``.

        Three entry modes (train.yaml finetune/resume fields): fresh, finetune warm-start
        (weights loaded at construction), or resume (load full state after prepare).

        Returns:
            None; checkpoints / logs land in the run dir.
        """
        t = self.cfg.training
        # num_epochs null = step-only training; max_steps null = no step limit.
        num_epochs = int(t.num_epochs) if t.num_epochs is not None else None
        max_steps = int(t.max_steps) if t.max_steps is not None else None
        batch_size = int(t.batch_size)
        grad_accum = int(t.gradient_accumulation_steps)

        debug = bool(t.debug)
        if debug:
            max_steps = 20
            save_steps_override = 10
            logger.info("DEBUG mode: max_steps=20, save@10, constant LR")

        # num_epochs=null means step-only training: max_steps is the sole stop condition.
        if num_epochs is None and not max_steps:
            raise ValueError(
                "training.num_epochs and training.max_steps are both unset; "
                "set num_epochs, or set max_steps for step-only training."
            )

        # finetune/resume validation + architecture construction happen in
        # __init__ (self-contained ckpt-dir path); only resume needs a path here.
        resume_path = t.resume_ckpt_path or None

        optimizer = self.build_optimizer()
        dataloader = self.build_dataloader(batch_size)
        # 0 / null = no clipping.
        max_grad_norm = float(t.max_grad_norm) if t.max_grad_norm else None

        n_proc = self.accelerator.num_processes
        steps_per_epoch = math.ceil(len(dataloader) / grad_accum)
        # max_steps counts micro-steps (global_step); convert to optimizer steps for the LR horizon.
        max_opt_steps = math.ceil(max_steps / grad_accum) if max_steps else None
        if num_epochs is not None:
            # prepare() shards the dataloader ~1/n_proc; fold that in so total_opt_steps is the
            # per-process optimizer steps the loop actually runs (same unit as max_opt_steps).
            steps_per_epoch = math.ceil(steps_per_epoch / n_proc)
            total_opt_steps = steps_per_epoch * num_epochs
            if max_opt_steps:
                total_opt_steps = min(total_opt_steps, max_opt_steps)
        else:
            total_opt_steps = max_opt_steps
        scheduler = self.build_lr_scheduler(optimizer, total_opt_steps, debug=debug)

        if debug:
            save_steps = save_steps_override
        else:
            # 0 / null = never save (the yaml's "disable saving").
            save_steps = int(t.save_steps or 0)
        keep_last_k = int(t.keep_last_k_ckpts)
        save_full_states_for_resume = bool(t.save_full_states_for_resume)

        # Finetune warm-start weights were already loaded at architecture
        # construction (__init__, self-contained ckpt-dir path). Step stays 0.

        output_path, resume_state_dir = self.setup_output_dir(debug, resume_path)
        optimizer, dataloader, scheduler = self.prepare_accelerate(optimizer, dataloader, scheduler)

        wire_sampler_seed(dataloader, self._run_seed, rank=self._rank)

        all_params = [p for group in optimizer.param_groups for p in group["params"]]

        is_main = self.accelerator is None or self.accelerator.is_main_process
        wandb_run = None if (debug or not is_main) else init_wandb(self.cfg)

        # Periodic world-model rollouts (GT | pred clips to wandb). Every rank holds one
        # fixed sample, picked before epoch 0 so the clips are comparable across vis steps.
        vis_steps = int(t.vis_steps or 0)  # 0 / null = off
        visualizer = None
        if vis_steps > 0 and self.dataset is not None:
            # Eval semantics (no caption dropout, fixed history, no jitter), sharing the preloaded readers.
            visualizer = TrainVisualizer(
                self.dataset.eval_view(),
                rank=self._rank,
                num_samples=int(t.vis_num_samples),
                denoise_steps=int(t.vis_denoise_steps),
                fps=int(t.vis_fps),
                seed=self._run_seed,
            )

        from tqdm import tqdm

        if num_epochs is not None:
            total_steps = len(dataloader) * num_epochs
            if max_steps:
                total_steps = min(total_steps, max_steps)
        else:
            total_steps = max_steps

        import time as _time

        opt_step = 0
        global_step = 0
        start_epoch = 0
        skip_first = 0

        # Resume: restore full state AFTER prepare, then map global_step -> (epoch, skip).
        if resume_state_dir is not None:
            global_step, opt_step, start_epoch, skip_first = self.resume_if_configured(
                resume_state_dir, dataloader, grad_accum
            )
            already_done = (num_epochs is not None and start_epoch >= num_epochs) or (
                max_steps and global_step >= max_steps
            )
            if already_done:
                logger.info("[resume] global_step=%d already complete; finishing.", global_step)
                self.finish_training(output_path, global_step, save_steps, keep_last_k, is_main, wandb_run)
                return

        # Text cache only now that the weights are final (prepare -> resume load_full_state -> cache, the same
        # order for every mode): on resume the text encoder holds real values only after load_full_state.
        if bool(t.text_cache):
            self.build_text_cache()

        _step_t0 = _time.monotonic()
        # Per-phase step timing (wandb timing/*); DeepSpeed's engine.step is hooked so
        # backward and optimizer time are billed separately.
        self._timer = StepTimer(bool(t.log_step_timing))
        self._timer.wrap_engine_step(getattr(self.accelerator.deepspeed_engine_wrapped, "engine", None))
        pbar = tqdm(total=total_steps, desc="Training", unit="step", initial=min(global_step, total_steps))

        assert self.accelerator is not None, "OpenWAMTrainer requires an Accelerator"

        # Per-step manual_seed makes timestep + noise sampling in compute_loss
        # reproducible across runs and ZeRO stages: same (rank, step) -> same RNG,
        # different ranks at the same step keep in-batch timestep diversity.
        epochs = itertools.count(start_epoch) if num_epochs is None else range(start_epoch, num_epochs)
        for epoch in epochs:
            if hasattr(dataloader, "set_epoch"):
                dataloader.set_epoch(epoch)
            if hasattr(self.dataset, "set_epoch"):
                self.dataset.set_epoch(epoch)
            # On the resumed epoch, skip the batches already consumed before the checkpoint.
            if epoch == start_epoch and skip_first > 0:
                from accelerate import skip_first_batches

                epoch_iter = skip_first_batches(dataloader, skip_first)
            else:
                epoch_iter = dataloader
            for batch in epoch_iter:
                self._timer.mark("data")
                step_seed = per_step_seed(self._run_seed, rank=self._rank, step=global_step)
                torch.manual_seed(step_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(step_seed)
                with self.accelerator.accumulate(self.architecture):
                    losses = self.compute_loss(batch)
                    self._timer.mark("forward")
                    loss = losses["total"]
                    self.accelerator.backward(loss)
                    self._timer.mark("backward")  # ~0 under DeepSpeed: the engine.step hook already closed backward/optim

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
                    self._timer.mark("optim")

                global_step += 1

                metrics = reduce_step_metrics(self.accelerator, losses, grad_norm)

                current_lr = optimizer.param_groups[0]["lr"]
                _now = _time.monotonic()
                steps_per_sec = 1.0 / max(_now - _step_t0, 1e-9)
                _step_t0 = _now

                self.log_step(
                    metrics=metrics,
                    global_step=global_step,
                    opt_step=opt_step,
                    epoch=epoch,
                    lr=current_lr,
                    steps_per_sec=steps_per_sec,
                    batch_size=batch_size,
                    pbar=pbar,
                    wandb_run=wandb_run,
                    debug=debug,
                    output_path=output_path,
                    timing=self._timer.snapshot(self.accelerator),
                )

                # save_steps: write the weights line (+ the resumable full state when
                # save_full_states_for_resume=true), then prune in lockstep.
                if save_steps and global_step > 0 and global_step % save_steps == 0:
                    save_weights(self.accelerator, self.architecture, output_path, global_step, final=False)
                    if save_full_states_for_resume:
                        save_full_state(self.accelerator, output_path, global_step, opt_step, epoch)
                    if is_main:
                        manage_checkpoints(output_path, keep_last_k)

                # Step 1 too: an early clip shows the hook works long before the first vis_steps.
                if visualizer is not None and (global_step % vis_steps == 0 or global_step == 1):
                    visualizer.run(self.accelerator, self.architecture, global_step, wandb_run, output_path)

                self._timer.end_step()  # bills metric reduce / logging / save / vis to "other"

                if max_steps and global_step >= max_steps:
                    pbar.close()
                    self.finish_training(output_path, global_step, save_steps, keep_last_k, is_main, wandb_run)
                    return

        pbar.close()
        self.finish_training(output_path, global_step, save_steps, keep_last_k, is_main, wandb_run)

    # (3) Called by train() first — AdamW over the per-module (action/video) LR param groups.
    def build_optimizer(self) -> torch.optim.Optimizer:
        """AdamW over the per-module (action/video) LR parameter groups."""
        t = self.cfg.training
        params = build_trainable_parameters(
            self,
            # null = that module uses the base learning_rate.
            action_lr=float(t.action_lr) if t.action_lr else None,
            video_lr=float(t.video_lr) if t.video_lr else None,
        )
        betas = tuple(t.adam_betas)
        return torch.optim.AdamW(params, lr=float(t.learning_rate), weight_decay=float(t.weight_decay), betas=betas)

    # (4) Called by train() — build the training DataLoader (seeded generator when reproducible).
    def build_dataloader(self, batch_size: int) -> torch.utils.data.DataLoader:
        """Build the training DataLoader (standard shuffle).

        Wires a per-rank ``generator`` (from ``cfg.project.seed``) and a ``worker_init_fn`` so dataset-side
        randomness is reproducible across runs while keeping per-epoch / per-worker variation.

        Args:
            batch_size: int, per-GPU batch size.

        Returns:
            torch DataLoader over ``self.dataset``.
        """
        t = self.cfg.training
        kwargs: dict = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(t.dataset_num_workers),
            collate_fn=collate,
            pin_memory=True,
        )
        from openwam.train.utils.seeding import dataloader_worker_init_fn, make_dataloader_generator

        kwargs["generator"] = make_dataloader_generator(self._run_seed, rank=self._rank)
        kwargs["worker_init_fn"] = dataloader_worker_init_fn
        return torch.utils.data.DataLoader(self.dataset, **kwargs)

    # (5) Called by train() — linear-warmup + cosine schedule, or None (constant LR / debug).
    def build_lr_scheduler(self, optimizer, total_opt_steps: int, debug: bool = False):
        """Linear-warmup + cosine schedule, or None (constant LR / debug).

        Args:
            optimizer: the AdamW from :meth:`build_optimizer`.
            total_opt_steps: int, per-process optimizer steps of the run.
            debug: bool, ``training.debug`` (forces constant LR).

        Returns:
            SequentialLR for ``training.lr_scheduler: cosine``; None for null (constant LR) or debug.
        """
        t = self.cfg.training
        if debug:
            return None
        if t.lr_scheduler == "cosine":
            return build_cosine_scheduler(
                optimizer, total_opt_steps=total_opt_steps, training_cfg=t, num_processes=self.accelerator.num_processes
            )
        return None

    # (6) Called by train() — locate/create the run dir and resolve the resume state dir.
    def setup_output_dir(self, debug: bool, resume_path: str | None) -> tuple[str, str | None]:
        """Locate/create the run dir and resolve the resume state dir.

        With a usable resume state the run dir is REUSED (assets/config/norm already
        present); otherwise rank-0 creates a fresh timestamped dir and broadcasts it.
        All ranks resolve ``resume_state_dir`` independently (shared FS, deterministic),
        so a missing-state error raises on every rank without deadlocking the broadcast.
        Returns ``(output_path, resume_state_dir)``.
        """
        base_output_path = self.cfg.training.output_path
        is_main = self.accelerator.is_main_process

        resume_state_dir = find_latest_accel_state(resume_path) if resume_path else None
        if resume_path and resume_state_dir is None:
            raise FileNotFoundError(
                f"resume_ckpt_path={resume_path} has no usable accel_state_step_*; full states "
                f"are only written when training.save_full_states_for_resume=true (and a finished run drops "
                f"them) — use finetune_ckpt_path to warm-start from weights instead."
            )
        if resume_state_dir is not None:
            output_path = os.path.dirname(resume_state_dir)
            logger.info("[resume] reusing run dir %s (state=%s)", output_path, os.path.basename(resume_state_dir))
            # Keep the checkpoint-dir deploy artifact; refuse resume when it
            # diverges from the dataset transform (legacy or regenerated stats).
            if self.dataset is not None:
                verify_resume_normalization_stats(output_path, self.dataset)
            return output_path, resume_state_dir

        # Fixed run dir name (e.g. wandb run name) if configured; else the launch timestamp.
        fixed_name = self.cfg.training.run_dir_name
        if fixed_name:
            # Checked on every rank (shared FS) so a clash raises everywhere instead of deadlocking the broadcast.
            existing = os.path.join(base_output_path, str(fixed_name), "config.yaml")
            if os.path.exists(existing):
                raise FileExistsError(
                    f"training.run_dir_name={fixed_name}: {existing} already exists; pick a new run name "
                    f"(or use training.resume_ckpt_path to continue that run)."
                )
        import torch.distributed as dist

        # Every rank finishes the clash check before rank 0 writes config.yaml below; otherwise a slower
        # rank sees rank 0's fresh config.yaml and raises (Modal runs 88/90, 2026-10-16).
        dist.barrier()

        if is_main:
            from datetime import datetime

            run_dir_name = str(fixed_name) if fixed_name else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if debug:
                run_dir_name += "_debug"
            output_path = os.path.join(base_output_path, run_dir_name)
            os.makedirs(output_path, exist_ok=True)
            # Self-contained deploy: backbones copy their non-weight assets (from model_path, or from the
            # source ckpt dir on finetune), then config + action stats.
            self.architecture.save_assets_for_deployment(output_path)
            save_config(output_path, self.cfg)
            if self.dataset is not None:
                save_normalization_stats(output_path, self.dataset)
        else:
            output_path = None
        path_list = [output_path] if is_main else [None]
        dist.broadcast_object_list(path_list, src=0)
        output_path = path_list[0]
        logger.info("Checkpoints will be saved to %s", output_path)
        return output_path, None

    # (7) Called by train() — wrap architecture/optimizer/dataloader with the DeepSpeed Accelerator.
    def prepare_accelerate(self, optimizer, dataloader, scheduler):
        """Wrap architecture/optimizer/dataloader with the DeepSpeed Accelerator."""
        prepare_args = [self.architecture, optimizer, dataloader]
        if scheduler is not None:
            prepare_args.append(scheduler)
            self.architecture, optimizer, dataloader, scheduler = self.accelerator.prepare(*prepare_args)
        else:
            self.architecture, optimizer, dataloader = self.accelerator.prepare(*prepare_args)
        # After prepare, self.architecture is the wrapped handle used for the loop's
        # forward/backward. Architecture-level helpers must run on the UNDERLYING
        # module, not the wrapper: DeepSpeedEngine.__getattr__ forwards unknown attrs
        # to the inner module, but DistributedDataParallel does NOT — so calling
        # set_dtype_device/move_frozen_to_device on the wrapper would raise
        # AttributeError under a hypothetical plain-DDP accelerator (world_size>1).
        # unwrap_model returns the inner module for both backends (no-op single-GPU,
        # where Accelerate adds no wrapper). NB: this only covers these setup-time
        # helpers — the training loop's self.architecture.compute_loss
        # call would hit the same non-forwarding wrapper if a non-DeepSpeed multi-GPU
        # path is ever reintroduced.
        arch = self.accelerator.unwrap_model(self.architecture)
        # Propagate device down through architecture; frozen modules (T5/VAE) idempotent move.
        arch.set_dtype_device(arch.dtype, self.accelerator.device)
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
        arch = self.architecture if self.accelerator is None else self.accelerator.unwrap_model(self.architecture)
        prompts = self.dataset.all_prompts()
        arch.video_backbone.build_text_cache(prompts)
        logger.info("text cache: cached %d prompts; text encoder moved to CPU", len(prompts))

    # (8) Called by train() on the resume path — restore full state, map step -> (start_epoch, skip).
    def resume_if_configured(self, resume_state_dir: str, dataloader, grad_accum: int) -> tuple[int, int, int, int]:
        """Load full state (after prepare) and map global_step -> (start_epoch, skip_first_batches).

        Returns (global_step, opt_step, start_epoch, skip).
        """
        is_main = self.accelerator.is_main_process
        if is_main:
            logger.info("[resume] loading Accelerate state from %s", resume_state_dir)
        meta = load_full_state(self.accelerator, resume_state_dir)
        global_step = int(meta.get("global_step", 0))
        # Align global_step to the grad_accum boundary skip was floored to, then derive
        # opt_step from it — otherwise floored-off batches re-train and per-step seeds
        # (keyed on global_step) drift. No-op at grad_accum=1.
        start_epoch, skip, global_step = compute_resume_position(global_step, len(dataloader), grad_accum)
        opt_step = global_step // grad_accum
        if is_main:
            logger.info(
                "[resume] resumed at global_step=%d opt_step=%d epoch=%d skip_first=%d",
                global_step,
                opt_step,
                start_epoch,
                skip,
            )
        return global_step, opt_step, start_epoch, skip

    # (9) Called each step in train()'s loop — joint video-action loss dict (total/video/action).
    def compute_loss(self, batch: Batch) -> dict:
        """Joint video-action loss of one collated batch.

        Args:
            batch: Batch from the DataLoader (``collate``).

        Returns:
            dict of 0-dim tensors: total (carries the graph), video, action.
        """
        arch = self.accelerator.unwrap_model(self.architecture)
        result = self.architecture.compute_loss(batch.to(arch.device, arch.dtype))
        return {
            "total": result["loss"],
            "video": result["loss_video"],
            "action": result["loss_action"],
        }

    # (10) Called each step in train()'s loop — progress bar, wandb log, debug loss-history CSV.
    def log_step(
        self,
        *,
        metrics,
        global_step,
        opt_step,
        epoch,
        lr,
        steps_per_sec,
        batch_size,
        pbar,
        wandb_run,
        debug,
        output_path,
        timing=None,
    ) -> None:
        """Update progress bar, log to wandb, and (debug) write the loss-history CSV row.

        timing: dict from StepTimer.snapshot (timing/<phase>[_max] seconds) or None/{}.
        """
        timing = timing or {}
        labels = [("action", "loss_action")]
        loss_total = metrics["loss_total"]
        loss_video = metrics["loss_video"]
        grad_norm = metrics["grad_norm"]

        if pbar is not None:
            postfix = {"loss": f"{loss_total:.4f}", "video": f"{loss_video:.4f}"}
            for name, key in labels:
                postfix[name] = f"{metrics[key]:.4f}"
            postfix["lr"] = f"{lr:.2e}"
            postfix["epoch"] = epoch
            # Rank-mean phase seconds, so plain stdout logs (local / Modal) show the split too.
            for short, key in (("data", "data"), ("enc", "encode"), ("fwd", "forward"), ("bwd", "backward"), ("opt", "optim"), ("oth", "other")):
                if f"timing/{key}" in timing:
                    postfix[short] = f"{timing[f'timing/{key}']:.2f}"
            pbar.set_postfix(postfix)
            pbar.update(1)

        if wandb_run is not None:
            num_procs = self.accelerator.num_processes if self.accelerator is not None else 1
            log_dict = {
                "train/loss": loss_total,
                "train/loss_video": loss_video,
                "train/grad_norm": grad_norm,
                "train/lr": lr,
                "performance/steps_per_sec": steps_per_sec,
                "performance/samples_per_sec": steps_per_sec * batch_size * num_procs,
            }
            for name, key in labels:
                log_dict[f"train/loss_{name}"] = metrics[key]
            log_dict.update(timing)
            wandb_run.log(log_dict, step=global_step)

        # Same phase times as a file (<run>/timing.jsonl) so runs on different wandb servers
        # (local vs Modal) can be compared offline; main rank only, values are rank-reduced.
        is_main = self.accelerator is None or self.accelerator.is_main_process
        if timing and is_main and output_path:
            import json

            with open(os.path.join(output_path, "timing.jsonl"), "a") as f:
                f.write(json.dumps({"step": global_step, **timing}) + "\n")

        if not debug:
            return
        is_main = self.accelerator is None or self.accelerator.is_main_process
        if not is_main:
            return
        loss_parts = " ".join(f"{name}={metrics[key]:.6f}" for name, key in labels)
        msg = (
            f"[debug][step {global_step:04d} opt {opt_step:04d}] "
            f"loss={loss_total:.6f} video={loss_video:.6f} {loss_parts} "
            f"grad_norm={grad_norm:.6f} lr={lr:.3e} epoch={epoch} "
            f"steps_per_sec={steps_per_sec:.3f}"
        )
        logger.info(msg)
        if pbar is not None:
            pbar.write(msg)
        else:
            print(msg, flush=True)

        if output_path:
            write_debug_loss_row(
                output_path,
                labels=labels,
                metrics=metrics,
                global_step=global_step,
                opt_step=opt_step,
                epoch=epoch,
                lr=lr,
                steps_per_sec=steps_per_sec,
            )

    # (11) Called on every train() exit — final weights, drop resume state, close wandb.
    def finish_training(
        self,
        output_path: str,
        global_step: int,
        save_steps,
        keep_last_k: int,
        is_main: bool,
        wandb_run,
    ) -> None:
        """Unified teardown for every exit path: final weights, drop resume state, close wandb.

        Falsy ``save_steps`` = a profiling/no-write run, so no final artifact (matches the
        periodic-save gating). After the final weights land, ``finalize_keep_weights_only``
        removes every ``accel_state_step_*`` and retains the configured number of recent
        weight checkpoints (rank-0, post-barrier).
        """
        if save_steps:
            save_weights(self.accelerator, self.architecture, output_path, global_step, final=True)
        self.accelerator.wait_for_everyone()
        if is_main:
            finalize_keep_weights_only(output_path, keep_last_k)
        if wandb_run is not None:
            wandb_run.finish()
