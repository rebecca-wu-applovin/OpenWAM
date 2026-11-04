"""flux's flow-matching scheduler: the shared shifted grid for deploy, flux's native law for training noise.

The grid (``set_timesteps`` / ``step``) is :class:`FlowScheduler`'s (flux DROID inference shift 5). Only
:meth:`FluxFlowScheduler.sample_train` differs: flux's video timestep law (policy.py:662,
``packing.sample_timesteps``), moved verbatim from the backbone's former ``sample_training_noise``.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from openwam.model.video_backbone.common.flow_scheduler import FlowScheduler


def rational_time_shift(t: Tensor, shift: float) -> Tensor:
    """flux processing/packing.py ``rational_time_shift``.

    Args:
        t: (B,) times in (0, 1). shift: float, the rational shift.

    Returns:
        (B,) shifted times ``shift t / (1 + (shift - 1) t)``.
    """
    return shift * t / (1.0 + (shift - 1.0) * t)


class FluxFlowScheduler(FlowScheduler):
    """:class:`FlowScheduler` with flux's native training-noise draw.

    Args:
        shift_video: float, the deploy grid's α-shift (flux DROID inference: 5).
        train_timestep_width: float, logistic width of ``logit(t)`` (native 0.75).
        train_timestep_shift: float, rational shift applied after the logistic draw (native 42).
    """

    def __init__(self, *, shift_video: float, train_timestep_width: float, train_timestep_shift: float):
        super().__init__(shift_video=shift_video)
        self.train_timestep_width = float(train_timestep_width)
        self.train_timestep_shift = float(train_timestep_shift)

    def sample_train(self, batch_size: int, *, device, dtype) -> Tuple[Tensor, Tensor, Tensor]:
        """flux training timestep law for the video stream (policy.py:662, packing.sample_timesteps).

        ``logit(t) ~ Logistic(0, 0.75)`` then the rational shift 42; ``t`` is rounded to the latent dtype
        as the native noising does. OpenWAM's ``x_t = (1 - s) x0 + s eps`` / target ``eps - x0`` equal flux's
        ``add_noise``, so sigma = t.

        Args:
            batch_size: int, B.
            device / dtype: target device and latent dtype.

        Returns:
            ``(timesteps (B,) fp32 = 1000 t, sigmas (B,) dtype, loss_weights (B,) ones)``.
        """
        eps = torch.finfo(torch.float32).eps
        u = torch.rand(batch_size).clamp(eps, 1 - eps)
        t = torch.sigmoid(self.train_timestep_width * (torch.log(u) - torch.log1p(-u)))
        t = rational_time_shift(t, self.train_timestep_shift).to(device=device, dtype=dtype)
        # Native flow_loss weighs every video token equally (video_loss_weight = 1).
        return t.float() * 1000.0, t, torch.ones(batch_size, device=device)


__all__ = ["FluxFlowScheduler", "rational_time_shift"]
