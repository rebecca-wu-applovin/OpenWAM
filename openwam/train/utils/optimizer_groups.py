"""Optimizer group helpers for scale-oriented OpenWAM training."""

from __future__ import annotations


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


__all__ = ["build_trainable_parameters"]
