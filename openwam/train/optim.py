"""Optimizer setup: trainable-parameter groups, the run's step budget, and the warmup + cosine LR schedule."""

from __future__ import annotations

import logging
import math

from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

logger = logging.getLogger(__name__)


def build_trainable_parameters(architecture, *, action_lr: float | None, video_lr: float | None):
    """Bucket every trainable top-level param by owner, then build optimizer groups.

    - ``video_backbone`` → ``video_lr`` (None = the optimizer base ``learning_rate``)
    - ``action_backbone`` → ``action_lr`` (None = the base LR)
    - every other trainable top-level module → the base LR, never a per-group override.

    Source of truth: ``BaseWAMArchitecture.get_trainable_modules()`` (top-level
    children that hold at least one ``requires_grad`` param). Only ``requires_grad``
    params are collected, so the ``training.freeze`` list — which flips
    ``requires_grad`` — is the single gate for trainability; frozen params never
    reach the optimizer.

    Args:
        architecture: BaseWAMArchitecture whose trainable params are grouped.
        action_lr: float LR of the ActionDiT params, or None for the base LR.
        video_lr: float LR of the video backbone params, or None for the base LR.

    Returns:
        With both LRs None, a flat list of params (action, video, other order); otherwise a list of AdamW param
        group dicts (``params`` plus ``lr`` where set), empty groups left out.
    """
    action_params, video_params, other_params = [], [], []
    for mod_name, mod in architecture.get_trainable_modules().items():
        if mod_name == "video_backbone":
            bucket = video_params
        elif mod_name == "action_backbone":
            bucket = action_params
        else:
            bucket = other_params
        bucket.extend(param for param in mod.parameters() if param.requires_grad)

    if action_lr is None and video_lr is None:
        return action_params + video_params + other_params

    # action / video may carry their own LR; "other" always rides the base LR (lr=None).
    groups = []
    for params, lr in [(action_params, action_lr), (video_params, video_lr), (other_params, None)]:
        if not params:
            continue
        group = {"params": params}
        if lr is not None:
            group["lr"] = lr
        groups.append(group)
    return groups


def plan_steps(
    batches_per_epoch: int, grad_accum: int, num_processes: int, num_epochs: int | None, max_steps: int | None
) -> tuple[int, int]:
    """The run's step budget per process: micro steps (``global_step``, progress bar) and optimizer steps (LR horizon).

    ``prepare()`` shards the loader to ``ceil(batches_per_epoch / num_processes)`` batches per process (accelerate's
    even_batches), and accumulation syncs at every epoch end, so an epoch is ``ceil(that / grad_accum)`` optimizer
    steps. ``max_steps`` counts micro steps and caps both.

    Args:
        batches_per_epoch: int, ``len(dataloader)`` BEFORE ``prepare()`` (all processes together).
        grad_accum: int, ``training.gradient_accumulation_steps``.
        num_processes: int, the Accelerator's process count.
        num_epochs: int, ``training.num_epochs``; None = step-only (``max_steps`` alone stops the run).
        max_steps: int, ``training.max_steps`` in micro steps; None = no step limit.

    Returns:
        (total_micro_steps, total_opt_steps): ints, per process. ValueError when num_epochs and max_steps are both None.
    """
    if num_epochs is None and max_steps is None:
        raise ValueError(
            "training.num_epochs and training.max_steps are both unset; "
            "set num_epochs, or set max_steps for step-only training."
        )
    # Step-only: max_steps micro steps, ceil'd to optimizer steps.
    if num_epochs is None:
        return max_steps, math.ceil(max_steps / grad_accum)
    micro_per_epoch = math.ceil(batches_per_epoch / num_processes)
    opt_per_epoch = math.ceil(micro_per_epoch / grad_accum)
    total_micro, total_opt = micro_per_epoch * num_epochs, opt_per_epoch * num_epochs
    if max_steps is not None:
        total_micro, total_opt = min(total_micro, max_steps), min(total_opt, math.ceil(max_steps / grad_accum))
    return total_micro, total_opt


def build_cosine_scheduler(
    optimizer, *, total_opt_steps: int, training_cfg: DictConfig, num_processes: int
) -> SequentialLR:
    """Linear-warmup + cosine-anneal LR schedule.

    total_opt_steps is in per-process optimizer steps; prepare() wraps the scheduler in
    AcceleratedScheduler which advances it num_processes times per opt step, so scale the
    horizons by num_processes to cancel that out.

    Args:
        optimizer: torch Optimizer whose LR the schedule drives.
        total_opt_steps: int, per-process optimizer steps of the whole run.
        training_cfg: ``cfg.training`` (``learning_rate``, ``warmup_ratio``, ``lr_min_ratio``).
        num_processes: int, the Accelerator's process count.

    Returns:
        SequentialLR: LinearLR warmup then CosineAnnealingLR down to ``learning_rate * lr_min_ratio``.
    """
    lr = float(training_cfg.learning_rate)
    warmup_ratio = float(training_cfg.warmup_ratio)
    lr_min_ratio = float(training_cfg.lr_min_ratio)
    n = max(int(num_processes), 1)
    warmup_steps = int(total_opt_steps * warmup_ratio)
    cosine_steps = max(total_opt_steps - warmup_steps, 1)
    # Horizons in scheduler.step() calls: n per optimizer step under AcceleratedScheduler.
    warmup_iters = warmup_steps * n
    warmup_sched = LinearLR(optimizer, start_factor=1.0 / max(warmup_iters, 1), total_iters=warmup_iters)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=cosine_steps * n, eta_min=lr * lr_min_ratio)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_iters])
    logger.info(
        "LR scheduler: cosine | total_opt_steps=%d warmup=%d eta_min=%.2e",
        total_opt_steps,
        warmup_steps,
        lr * lr_min_ratio,
    )
    return scheduler
