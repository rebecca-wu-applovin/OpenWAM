"""GE's flow-matching scheduler: the shared shifted grid for deploy, GE's native law for training noise.

The grid (``set_timesteps`` / ``step``) is :class:`FlowScheduler`'s; with the yaml's ``shift_video`` 1.0 the 1000-step
grid ``sigma_i = 1 - i/1000`` is diffusers' default ``FlowMatchEulerDiscreteScheduler`` (shift 1) GE trains with.
Only :meth:`GEFlowScheduler.sample_train` differs: GE's ``flow_weighting_scheme: none`` draw (ge_trainer.py:574-596,
646-648), moved verbatim from the backbone's former ``sample_training_noise``.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from openwam.model.video_backbone.common.flow_scheduler import FlowScheduler

GE_NUM_TRAIN_TIMESTEPS = 1000


class GEFlowScheduler(FlowScheduler):
    """:class:`FlowScheduler` with GE's native training-noise draw.

    Args:
        shift_video: float, the deploy grid's α-shift (GE yaml: 1.0).
    """

    def __init__(self, *, shift_video: float):
        super().__init__(shift_video=shift_video, num_train_timesteps=GE_NUM_TRAIN_TIMESTEPS)

    def sample_train(self, batch_size: int, *, device, dtype) -> Tuple[Tensor, Tensor, Tensor]:
        """GE's ``flow_weighting_scheme: none`` sampling.

        u ~ U[0, 1) per sample; index = floor(u * 1000) into the shift-1 FlowMatchEuler grid
        sigma_i = (1000 - i) / 1000; transformer timestep = floor(sigma * 1000); loss weight 1.

        Args:
            batch_size: int, B. device / dtype: target device and latent dtype.

        Returns:
            (timesteps (B,), sigmas (B,), loss_weights (B,) float32 ones).
        """
        u = torch.rand(batch_size)
        idx = (u * GE_NUM_TRAIN_TIMESTEPS).long()
        sigmas = (GE_NUM_TRAIN_TIMESTEPS - idx).float() / GE_NUM_TRAIN_TIMESTEPS
        timesteps = (sigmas * GE_NUM_TRAIN_TIMESTEPS).long().float()
        return (
            timesteps.to(device=device, dtype=dtype),
            sigmas.to(device=device, dtype=dtype),
            torch.ones(batch_size, device=device, dtype=torch.float32),
        )


__all__ = ["GE_NUM_TRAIN_TIMESTEPS", "GEFlowScheduler"]
