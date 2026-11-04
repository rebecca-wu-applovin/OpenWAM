"""Per-step reporting: parameter report, wandb, phase timing, cross-rank metric reduction, per-step logging.

:class:`StepTimer` splits each step's wall time into phases so a slow s/step can be attributed:

    data     waiting on the DataLoader iterator (incl. accelerate's H2D copy)
    forward  architecture.compute_loss (frozen VAE / text encode + trainable forward)
    backward loss.backward (+ ZeRO grad reduce-scatter; waits for slower ranks)
    optim    clip + optimizer step + zero_grad (DeepSpeed engine.step)
    other    metric all-reduce, logging, checkpoint save, vis (previous step)

Each ``mark`` calls ``torch.cuda.synchronize`` so GPU work is billed to the phase that
launched it, not to whichever later phase first blocks on it. The syncs remove a little
CPU/GPU overlap; disable with ``training.log_step_timing=false``.

:class:`StepLogger` owns the per-step sinks (progress bar, wandb, timing.jsonl, loss CSV); the rest is stateless.
"""

import json
import logging
import os
import time

import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

PHASES = ("data", "forward", "backward", "optim", "other")


class StepTimer:
    """Accumulates seconds per phase between consecutive ``mark`` calls."""

    def __init__(self, enabled: bool):
        """enabled: False makes every method a no-op (and ``snapshot`` returns {})."""
        self.enabled = enabled
        self._last = time.monotonic()
        self._cur = dict.fromkeys(PHASES, 0.0)
        self._prev_other = 0.0  # "other" is closed after log time, so report last step's

    def mark(self, phase: str) -> None:
        """Close the running interval and bill it to ``phase``.

        phase: one of PHASES.
        Returns: None.
        """
        if not self.enabled:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.monotonic()
        self._cur[phase] += now - self._last
        self._last = now

    def end_step(self) -> None:
        """Bill the step's tail to ``other`` and reset for the next step. Returns: None."""
        if not self.enabled:
            return
        self.mark("other")
        self._prev_other = self._cur["other"]
        self._cur = dict.fromkeys(PHASES, 0.0)

    def snapshot(self, accelerator) -> dict:
        """This step's phase times, reduced across ranks.

        accelerator: Accelerate handle; with >1 process the (world, len(PHASES)) times are gathered.
        Returns: dict ``timing/<phase>`` = rank-mean seconds and ``timing/<phase>_max`` =
            slowest-rank seconds (one per phase), plus ``timing/step`` = sum of the means.
            ``other`` is the previous step's value. Empty dict if disabled.
        """
        if not self.enabled:
            return {}
        vals = [self._cur[p] for p in PHASES[:-1]] + [self._prev_other]
        local = torch.tensor(vals, dtype=torch.float32)
        if accelerator.num_processes > 1:
            g = accelerator.gather(local.to(accelerator.device).reshape(1, -1)).float().cpu()  # (world, len(PHASES))
            mean, mx = g.mean(dim=0), g.max(dim=0).values
        else:
            mean = mx = local
        out = {}
        for i, p in enumerate(PHASES):
            out[f"timing/{p}"] = float(mean[i])
            out[f"timing/{p}_max"] = float(mx[i])
        out["timing/step"] = float(mean.sum())
        return out

    def wrap_engine_step(self, engine) -> None:
        """Bill DeepSpeed's in-backward ``engine.step`` to ``optim`` instead of ``backward``.

        Under DeepSpeed, ``accelerator.backward`` runs backward *and* engine.step (clip,
        optimizer, zero_grad); without this hook both would be billed to ``backward``.
        engine: DeepSpeedEngine (training always runs DeepSpeed, see scripts/train.py::_build_accelerator).
        Returns: None.
        """
        if not self.enabled:
            return
        orig_step = engine.step

        def step(*args, **kwargs):
            self.mark("backward")
            out = orig_step(*args, **kwargs)
            self.mark("optim")  # close here, else the loop's post-backward mark absorbs the step
            return out

        engine.step = step


def log_parameter_counts(architecture, *, is_main: bool) -> None:
    """Print per-backbone total/trainable param counts (rank-0 only).

    Uses print() not logger so it survives Hydra's default logging filter;
    the is_main gate keeps it correct regardless of the launcher's stdout
    suppression on non-main ranks.
    """
    if not is_main:
        return

    def _count(module):
        if module is None:
            return 0, 0
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    bb_counts = {name: _count(module) for name, module in architecture.backbones.items()}
    extra_counts = {}
    for name, module in architecture.named_children():
        if name in bb_counts:
            continue
        total, trainable = _count(module)
        if total:
            extra_counts[name] = (total, trainable)
    arch_total = sum(total for total, _ in bb_counts.values()) + sum(total for total, _ in extra_counts.values())
    arch_train = sum(train for _, train in bb_counts.values()) + sum(train for _, train in extra_counts.values())
    print("=" * 60)
    print("Parameter counts")
    for name, (total, trainable) in {**bb_counts, **extra_counts}.items():
        print(f"  {name:<15}: total={total / 1e6:7.1f}M  trainable={trainable / 1e6:7.1f}M")
    print(f"  Architecture  : total={arch_total / 1e6:7.1f}M  trainable={arch_train / 1e6:7.1f}M")
    print("=" * 60, flush=True)


def init_wandb(cfg: DictConfig):
    """Init a wandb run from ``cfg.project.wandb``.

    Args:
        cfg: the run config (``project.wandb.{project, run_name, entity, group}``; the whole config is logged).

    Returns:
        the wandb Run, or None when ``project.wandb.project`` is null (the yaml's "off") or wandb is not installed.
    """
    wandb_cfg = cfg.project.wandb
    project = wandb_cfg.project
    # null project = wandb off (train_debug.yaml).
    if not project:
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed, skipping wandb logging")
        return None

    run = wandb.init(
        project=project,
        name=wandb_cfg.run_name,
        entity=wandb_cfg.entity,
        group=wandb_cfg.group,  # groups runs of one ablation in the wandb UI
        config=OmegaConf.to_container(cfg, resolve=True),
        resume="allow",
    )
    logger.info("wandb initialized: %s/%s", project, run.name)
    return run


# The architecture's loss keys (compute_loss), kept end to end: reduction, progress bar, wandb, CSV.
LOSS_KEYS = ("loss", "loss_video", "loss_action")


def reduce_step_metrics(accelerator, losses: dict, grad_norm: torch.Tensor) -> dict:
    """One step's losses and grad norm as floats, averaged over ranks (a collective when there are several).

    Args:
        accelerator: the Accelerator (``num_processes``, ``gather``).
        losses: dict of 0-dim tensors from ``compute_loss`` with the :data:`LOSS_KEYS`.
        grad_norm: 0-dim tensor, the clipped-step grad norm (0 on non-sync / unclipped steps).

    Returns:
        dict ``{loss, loss_video, loss_action, grad_norm}`` of Python floats (rank means).
    """
    values = [losses[key].detach().float().item() for key in LOSS_KEYS] + [grad_norm.item()]
    if accelerator.num_processes > 1:
        local = torch.tensor(values, device=losses["loss"].device, dtype=torch.float32).reshape(1, -1)
        # (world, 4) -> rank mean per column.
        values = accelerator.gather(local).mean(dim=0).tolist()
    return dict(zip((*LOSS_KEYS, "grad_norm"), values))


def _drop_rows_after(path: str, step: int, header_lines: int, row_step) -> None:
    """Rewrite a per-step log file keeping its header and only the rows of steps <= ``step``.

    Args:
        path: str, ``loss_history.csv`` or ``timing.jsonl``; a missing file is left missing.
        step: int, the last step whose row stays.
        header_lines: int, leading lines that are not step rows (1 for the CSV, 0 for jsonl).
        row_step: callable(line: str) -> int, the step a row belongs to.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    kept = lines[:header_lines] + [line for line in lines[header_lines:] if row_step(line) <= step]
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(line + "\n" for line in kept))


class StepLogger:
    """Reports each training step to every sink: progress bar (all ranks), wandb, ``<run>/timing.jsonl`` and
    ``<run>/loss_history.csv`` (main rank).

    Args:
        total_steps: int, per-process micro steps of the run (progress-bar total).
        initial_step: int, global_step the loop starts at (0, or the resumed step).
        wandb_run: the wandb Run, or None (wandb off / non-main rank).
        run_path: str, the run dir.
        is_main: bool, this process writes the files.
        samples_per_step: int, batch_size * num_processes (for performance/samples_per_sec).
        log_timing: bool, ``training.log_step_timing``: phase times go to the bar and timing.jsonl.
    """

    # Progress-bar label -> StepTimer phase.
    _BAR_PHASES = (("data", "data"), ("fwd", "forward"), ("bwd", "backward"), ("opt", "optim"), ("oth", "other"))

    def __init__(
        self,
        *,
        total_steps: int,
        initial_step: int,
        wandb_run,
        run_path: str,
        is_main: bool,
        samples_per_step: int,
        log_timing: bool,
    ):
        from tqdm import tqdm

        self._pbar = tqdm(total=total_steps, desc="Training", unit="step", initial=min(initial_step, total_steps))
        self._wandb_run = wandb_run
        self._run_path = run_path
        self._is_main = is_main
        self._samples_per_step = samples_per_step
        self._log_timing = log_timing
        # steps/sec is measured between consecutive log() calls.
        self._last = time.monotonic()
        if is_main:
            # A resumed run re-trains the steps after initial_step: their earlier rows go, so each step has one row.
            # (A fresh run dir has neither file yet.)
            _drop_rows_after(
                os.path.join(run_path, "loss_history.csv"), initial_step, 1, lambda line: int(line.split(",")[0])
            )
            _drop_rows_after(
                os.path.join(run_path, "timing.jsonl"), initial_step, 0, lambda line: json.loads(line)["step"]
            )

    def log(self, metrics: dict, *, global_step: int, opt_step: int, epoch: int, lr: float, timing: dict) -> None:
        """Report one step.

        Args:
            metrics: dict from :func:`reduce_step_metrics` (``loss``, ``loss_video``, ``loss_action``, ``grad_norm``).
            global_step: int, micro steps done (wandb step).
            opt_step: int, optimizer steps done.
            epoch: int, current epoch.
            lr: float, the first param group's learning rate.
            timing: dict from ``StepTimer.snapshot`` (``timing/<phase>[_max]`` seconds; empty when timing is off).
        """
        now = time.monotonic()
        steps_per_sec = 1.0 / max(now - self._last, 1e-9)
        self._last = now

        postfix = {
            "loss": f"{metrics['loss']:.4f}",
            "video": f"{metrics['loss_video']:.4f}",
            "action": f"{metrics['loss_action']:.4f}",
            "lr": f"{lr:.2e}",
            "epoch": epoch,
        }
        if self._log_timing:
            # Rank-mean phase seconds, so plain stdout logs (local / Modal) show the split too.
            postfix.update({short: f"{timing[f'timing/{phase}']:.2f}" for short, phase in self._BAR_PHASES})
        self._pbar.set_postfix(postfix)
        self._pbar.update(1)

        if self._wandb_run is not None:
            log_dict = {f"train/{key}": metrics[key] for key in (*LOSS_KEYS, "grad_norm")}
            log_dict["train/lr"] = lr
            log_dict["performance/steps_per_sec"] = steps_per_sec
            log_dict["performance/samples_per_sec"] = steps_per_sec * self._samples_per_step
            self._wandb_run.log({**log_dict, **timing}, step=global_step)

        if not self._is_main:
            return
        # Same phase times as a file so runs on different wandb servers (local vs Modal) compare offline.
        if self._log_timing:
            with open(os.path.join(self._run_path, "timing.jsonl"), "a") as f:
                f.write(json.dumps({"step": global_step, **timing}) + "\n")
        self._write_loss_row(metrics, global_step, opt_step, epoch, lr, steps_per_sec)

    def _write_loss_row(
        self, metrics: dict, global_step: int, opt_step: int, epoch: int, lr: float, steps_per_sec: float
    ) -> None:
        """Append one row to ``<run>/loss_history.csv`` (header on first write; a resumed run appends).

        Args:
            metrics: dict from :func:`reduce_step_metrics`.
            global_step / opt_step / epoch: ints, the step's position.
            lr: float, learning rate; steps_per_sec: float, measured rate.
        """
        path = os.path.join(self._run_path, "loss_history.csv")
        write_header = not os.path.exists(path)
        with open(path, "a", encoding="utf-8") as f:
            if write_header:
                f.write("step,opt_step,epoch,loss,loss_video,loss_action,grad_norm,lr,steps_per_sec\n")
            values = ",".join(f"{metrics[key]:.10g}" for key in (*LOSS_KEYS, "grad_norm"))
            f.write(f"{global_step},{opt_step},{epoch},{values},{lr:.10g},{steps_per_sec:.10g}\n")

    def close(self) -> None:
        """Close the progress bar (wandb is closed by the trainer's finish)."""
        self._pbar.close()
