"""``FlowScheduler``: the one flow-matching scheduler of the video and the action stream.

Shared math: the α-shifted sigma grid, the noising ``x_t = (1 - σ) x + σ n``, the velocity target ``n - x`` and the
Euler step. The only per-family difference is how training draws its noise level, written explicitly in the yaml
(``model.video_backbone.train_timestep`` / ``model.action_backbone.train_timestep``) and picked once, in
:class:`FlowScheduler`'s constructor, from :data:`TRAIN_LAWS`:

* ``grid`` (Wan / longlive / tiny, cosmos3, action): uniform index into the shifted 1000-step training grid; loss
  weights ``bell`` (Wan BSMNTW) or ``uniform`` (Cosmos-Predict2.5 upstream ``TrainTimeWeight("uniform")``).
* ``logit_logistic`` (flux): ``logit(t) ~ Logistic(0, width)`` then the rational shift (flux packing.sample_timesteps).
* ``ge_native`` (GE): ``flow_weighting_scheme: none`` on diffusers' shift-1 FlowMatchEuler grid (ge_trainer.py).

Grids are never rewritten in place: the training grid is built once in the constructor, and inference grids come
from the pure :meth:`FlowScheduler.inference_timesteps`. Neutral package (``openwam.model.common``) so the action
side never imports the video package.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Tuple

import torch
from omegaconf import DictConfig
from torch import Tensor

# Timestep unit of every stream: t = σ · 1000 (training grid size and the denoise loop's divisor).
NUM_TRAIN_TIMESTEPS = 1000
# Each law's exact yaml keys; anything missing or extra is a config error.
_LAW_KEYS = {
    "grid": {"law", "weighting"},
    "logit_logistic": {"law", "width", "shift"},
    "ge_native": {"law"},
}
TRAIN_LAWS = tuple(_LAW_KEYS)
GRID_WEIGHTINGS = ("bell", "uniform")


def alpha_shift(x: Tensor, shift: float) -> Tensor:
    """Rational time shift ``shift x / (1 + (shift - 1) x)`` (SD3; flux packing.py rational time shift).

    Args:
        x: (N,) float tensor of unshifted times / sigmas in [0, 1].
        shift: float, the α-shift (python float, so the op order and rounding are fixed).

    Returns:
        (N,) shifted values, same dtype as ``x``.
    """
    return shift * x / (1.0 + (shift - 1.0) * x)


def shifted_sigmas(num_steps: int, shift: float) -> Tensor:
    """The shifted sigma grid ``alpha_shift(linspace(1, 0, N + 1)[:-1], shift)``.

    Args:
        num_steps: int, N grid points (1000 for training, the denoise steps at inference).
        shift: float, α-shift.

    Returns:
        (N,) float32 sigmas, decreasing from 1.
    """
    s = torch.linspace(1.0, 0.0, num_steps + 1)[:-1]
    return alpha_shift(s, shift)


def bell_weights(timesteps: Tensor) -> Tensor:
    """Wan BSMNTW loss weights: a bell over the timestep axis, shifted to min 0, scaled to mean 1.

    Args:
        timesteps: (N,) float32 timesteps of the 1000-step training grid.

    Returns:
        (N,) float32 per-grid-point loss weights.
    """
    steps = NUM_TRAIN_TIMESTEPS
    y = torch.exp(-2 * ((timesteps - steps / 2) / steps) ** 2)
    y_shifted = y - y.min()
    # int / tensor (reciprocal then scale) exactly as the former Wan / action schedulers computed it.
    return y_shifted * (steps / y_shifted.sum())


def sample_grid(
    timesteps: Tensor, sigmas: Tensor, weights: Tensor, batch_size: int, *, device, dtype
) -> Tuple[Tensor, Tensor, Tensor]:
    """``grid`` law: one noise level per sample, uniform over a discrete grid.

    One ``torch.randint`` on the global CPU RNG picks the indices, then the grid tensors are indexed with them.

    Args:
        timesteps: (N,) grid timesteps. sigmas: (N,) grid sigmas. weights: (N,) float32 per-point loss weights.
        batch_size: int, B.
        device / dtype: target device and latent dtype of the timesteps / sigmas.

    Returns:
        (timesteps (B,) dtype, sigmas (B,) dtype, loss_weights (B,) float32), all on ``device``.
    """
    ids = torch.randint(0, len(timesteps), (batch_size,))
    # Weights stay float32: the loss multiplies them in fp32.
    return (
        timesteps[ids].to(dtype=dtype, device=device),
        sigmas[ids].to(dtype=dtype, device=device),
        weights[ids].to(dtype=torch.float32, device=device),
    )


def sample_logit_logistic(
    batch_size: int, *, width: float, shift: float, device, dtype
) -> Tuple[Tensor, Tensor, Tensor]:
    """``logit_logistic`` law (flux policy.py:662, packing.sample_timesteps): ``logit(t) ~ Logistic(0, width)``.

    The draw is rationally shifted by ``shift`` and rounded to the latent dtype like the native noising; sigma = t.

    Args:
        batch_size: int, B.
        width: float, logistic width of ``logit(t)`` (flux 0.75). shift: float, rational shift (flux 42).
        device / dtype: target device and latent dtype.

    Returns:
        (timesteps (B,) float32 = 1000 t, sigmas (B,) dtype = t, loss_weights (B,) float32 ones), on ``device``.
    """
    eps = torch.finfo(torch.float32).eps
    u = torch.rand(batch_size).clamp(eps, 1 - eps)
    t = torch.sigmoid(width * (torch.log(u) - torch.log1p(-u)))
    t = alpha_shift(t, shift).to(device=device, dtype=dtype)
    # Native flow_loss weighs every video token equally (video_loss_weight = 1).
    return t.float() * 1000.0, t, torch.ones(batch_size, device=device)


def sample_ge_native(batch_size: int, *, device, dtype) -> Tuple[Tensor, Tensor, Tensor]:
    """``ge_native`` law: GE's ``flow_weighting_scheme: none`` draw (ge_trainer.py:574-596, 646-648).

    u ~ U[0, 1) per sample; index = floor(1000 u) into the shift-1 FlowMatchEuler grid sigma_i = (1000 - i) / 1000;
    transformer timestep = floor(1000 sigma); loss weight 1.

    Args:
        batch_size: int, B. device / dtype: target device and latent dtype.

    Returns:
        (timesteps (B,) dtype, sigmas (B,) dtype, loss_weights (B,) float32 ones), on ``device``.
    """
    n = NUM_TRAIN_TIMESTEPS
    u = torch.rand(batch_size)
    idx = (u * n).long()
    sigmas = (n - idx).float() / n
    timesteps = (sigmas * n).long().float()
    return (
        timesteps.to(device=device, dtype=dtype),
        sigmas.to(device=device, dtype=dtype),
        torch.ones(batch_size, device=device, dtype=torch.float32),
    )


class FlowScheduler:
    """Flow-matching scheduler of one stream (video or action); immutable after construction.

    Args:
        shift: float, α-shift of the stream's grids (yaml ``shift_video`` / ``shift_action``).
        train_timestep: DictConfig (or plain mapping), the yaml ``train_timestep`` block:
            ``{law: grid, weighting: bell | uniform}``, ``{law: logit_logistic, width, shift}`` or ``{law: ge_native}``.
    """

    num_train_timesteps: int = NUM_TRAIN_TIMESTEPS

    def __init__(self, *, shift: float, train_timestep: DictConfig | Mapping):
        self.shift = float(shift)
        law = str(train_timestep["law"])
        if law not in _LAW_KEYS:
            raise ValueError(f"train_timestep.law must be one of {TRAIN_LAWS}, got {law!r}")
        keys = set(train_timestep.keys())
        if keys != _LAW_KEYS[law]:
            raise ValueError(
                f"train_timestep for law {law!r} takes exactly {sorted(_LAW_KEYS[law])}, got {sorted(keys)}"
            )
        # The one place the law is read: bind the training draw once.
        if law == "grid":
            weighting = str(train_timestep["weighting"])
            if weighting not in GRID_WEIGHTINGS:
                raise ValueError(f"train_timestep.weighting must be one of {GRID_WEIGHTINGS}, got {weighting!r}")
            # Training grid built here, once; never rewritten.
            sigmas = shifted_sigmas(NUM_TRAIN_TIMESTEPS, self.shift)
            timesteps = sigmas * NUM_TRAIN_TIMESTEPS
            if weighting == "bell":
                weights = bell_weights(timesteps)
            else:
                weights = torch.ones_like(timesteps, dtype=torch.float32)
            self._draw = functools.partial(sample_grid, timesteps, sigmas, weights)
        elif law == "logit_logistic":
            self._draw = functools.partial(
                sample_logit_logistic, width=float(train_timestep["width"]), shift=float(train_timestep["shift"])
            )
        else:
            self._draw = sample_ge_native

    def sample_train(self, batch_size: int, *, device, dtype) -> Tuple[Tensor, Tensor, Tensor]:
        """Training noise levels drawn with the yaml's law.

        Args:
            batch_size: int, B. device / dtype: target device and latent dtype.

        Returns:
            (timesteps (B,), sigmas (B,) dtype, loss_weights (B,) float32), on ``device``; timesteps are in
            ``dtype`` except ``logit_logistic`` (float32 = 1000 t, as flux feeds them).
        """
        return self._draw(batch_size, device=device, dtype=dtype)

    def inference_timesteps(self, num_steps: int) -> Tensor:
        """The deploy / rollout grid; pure (the training draw is untouched).

        Args:
            num_steps: int, N denoise steps.

        Returns:
            (N,) float32 timesteps ``shifted_sigmas(N, shift) * 1000``, decreasing.
        """
        return shifted_sigmas(num_steps, self.shift) * NUM_TRAIN_TIMESTEPS

    @staticmethod
    def add_noise(sample: Tensor, noise: Tensor, sigma: Tensor) -> Tensor:
        """Flow-matching noising ``(1 - σ) x + σ n``.

        Args:
            sample: (B, ...) clean tensor. noise: same shape. sigma: (B, 1, ...) broadcastable sigmas.

        Returns:
            (B, ...) noisy sample.
        """
        return (1 - sigma) * sample + sigma * noise

    @staticmethod
    def training_target(sample: Tensor, noise: Tensor) -> Tensor:
        """Velocity target ``n - x``.

        Args:
            sample: (B, ...) clean tensor. noise: same shape.

        Returns:
            (B, ...) target.
        """
        return noise - sample

    @staticmethod
    def step(pred: Tensor, sigma, sigma_next, sample: Tensor) -> Tensor:
        """One Euler step ``x + pred (σ_next - σ)``.

        Args:
            pred: (B, ...) predicted velocity. sigma / sigma_next: float (or broadcastable tensor) current and next
                noise level. sample: (B, ...) current sample.

        Returns:
            (B, ...) sample at ``sigma_next``.
        """
        return sample + pred * (sigma_next - sigma)


__all__ = [
    "NUM_TRAIN_TIMESTEPS",
    "TRAIN_LAWS",
    "GRID_WEIGHTINGS",
    "alpha_shift",
    "shifted_sigmas",
    "bell_weights",
    "sample_grid",
    "sample_logit_logistic",
    "sample_ge_native",
    "FlowScheduler",
]
