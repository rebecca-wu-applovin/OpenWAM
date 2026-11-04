"""``sample_grid``: the training noise-level draw shared by the video and action flow-matching schedulers.

A neutral module (neither ``video_backbone`` nor ``action_backbone``): the Wan / cosmos3 video schedulers and the
action ``ActionScheduler`` all draw their training noise levels uniformly from their own discrete grid with it.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def sample_grid(
    timesteps: Optional[Tensor], sigmas: Optional[Tensor], weights: Optional[Tensor], batch_size: int, *, device, dtype
) -> Tuple[Tensor, Tensor, Tensor]:
    """Draw one training noise level per sample uniformly from a discrete schedule grid.

    One ``torch.randint`` on the global CPU RNG picks the grid indices (the draw ``compute_loss`` always made,
    so the RNG stream is unchanged); the grid tensors are then indexed with them.

    Args:
        timesteps: (N,) grid timesteps. sigmas: (N,) grid sigmas. weights: (N,) per-step loss weights; None until
            the scheduler's ``set_timesteps(training=True)`` ran.
        batch_size: int, B.
        device / dtype: target device and latent dtype of the timesteps / sigmas.

    Returns:
        (timesteps (B,) dtype, sigmas (B,) dtype, loss_weights (B,) float32), all on ``device``. Raises RuntimeError
        when the training grid was never built.
    """
    if weights is None:
        raise RuntimeError("sample_train needs the training grid: call scheduler.set_timesteps(..., training=True)")
    ids = torch.randint(0, len(timesteps), (batch_size,))
    # Weights stay float32: the loss multiplies them in fp32.
    return (
        timesteps[ids].to(dtype=dtype, device=device),
        sigmas[ids].to(dtype=dtype, device=device),
        weights[ids].to(dtype=torch.float32, device=device),
    )


__all__ = ["sample_grid"]
