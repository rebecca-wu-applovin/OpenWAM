"""Denoising schedules composed from the two streams' :class:`FlowScheduler` s (video and action).

Each stream's grid comes from its scheduler (``inference_timesteps`` / ``shift``); nothing here mutates a scheduler.
``schedule_sync`` returns a list of ``(t_video, t_action)`` pairs
describing the per-iteration noise levels for the joint denoising loop,
terminated with a ``(0.0, 0.0)`` sentinel.

Two denoising modes are supported:

- ``sync``           — both streams advance in lockstep on their own
  deterministic timestep series (default; unchanged behavior).
- ``async``          — Latent-Forcing-style trajectory: one
  stream denoises earlier than the other along an alpha-shift curve
  (``alpha``, arXiv:2602.11401) and/or a linear ``offset`` delaying the
  lag stream, with ``lead`` choosing which stream leads. Each stream
  rides its own ``alpha_shift`` grid (matching training), and
  ``alpha=1, offset=0`` reproduces ``sync`` bit-for-bit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import List, Tuple

import torch
from omegaconf import DictConfig

from openwam.config import cfg_get
from openwam.model.common.flow_scheduler import NUM_TRAIN_TIMESTEPS, FlowScheduler, alpha_shift

Schedule = List[Tuple[float, float]]
VALID_DENOISE_MODES = ("sync", "async")


@dataclass(frozen=True)
class DenoiseConfig:
    """Validated denoising settings (see :func:`normalize_denoise_config`); every field is required."""

    denoise_mode: str
    lead_modality: str
    variance_shift_alpha: float
    linear_offset: float


def _finite_float(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return value


def normalize_denoise_config(sampler: DictConfig | Mapping) -> DenoiseConfig:
    """Read and validate the four denoising settings (``deploy.sampler``, or the dict :func:`make_schedule` builds).

    Args:
        sampler: DictConfig / dict holding ``denoise_mode``, ``lead_modality``, ``variance_shift_alpha`` and
            ``linear_offset`` (all required; other keys such as ``denoise_steps`` are ignored).

    Returns:
        DenoiseConfig with the normalized values.

    Raises:
        KeyError: one of the four keys is missing. ValueError: an invalid value, or a non-identity async control
            (``lead_modality != "video"``, ``alpha != 1``, ``offset != 0``) under ``denoise_mode="sync"``.
    """
    mode = str(cfg_get(sampler, "denoise_mode")).strip().lower()
    if mode not in VALID_DENOISE_MODES:
        raise ValueError(f"Unsupported denoise mode {mode!r}; expected one of {VALID_DENOISE_MODES}")

    lead = str(cfg_get(sampler, "lead_modality")).strip().lower()
    if lead not in ("action", "video"):
        raise ValueError(f"lead_modality must be 'action' or 'video', got {lead!r}")

    alpha = _finite_float(cfg_get(sampler, "variance_shift_alpha"), "variance_shift_alpha")
    if alpha < 1.0:
        raise ValueError(f"variance_shift_alpha must be >= 1, got {alpha!r}")

    offset = _finite_float(cfg_get(sampler, "linear_offset"), "linear_offset")
    if not 0.0 <= offset < 1.0:
        raise ValueError(f"linear_offset must satisfy 0 <= value < 1, got {offset!r}")

    if mode == "sync":
        inactive = []
        if lead != "video":
            inactive.append("lead_modality")
        if alpha != 1.0:
            inactive.append("variance_shift_alpha")
        if offset != 0.0:
            inactive.append("linear_offset")
        if inactive:
            verb = "requires" if len(inactive) == 1 else "require"
            raise ValueError(f"{', '.join(inactive)} {verb} denoise_mode='async'")

    return DenoiseConfig(
        denoise_mode=mode,
        lead_modality=lead,
        variance_shift_alpha=alpha,
        linear_offset=offset,
    )


def denoise_async_is_noop(config: DenoiseConfig) -> bool:
    """True when ``async`` reproduces the ``sync`` trajectory bit-for-bit.

    ``alpha=1, offset=0`` is the diagonal, so a run labelled ``async`` at those
    values produces exactly the ``sync`` schedule — worth a warning rather than
    silently attributing the numbers to a shifted trajectory.

    Args:
        config: DenoiseConfig from :func:`normalize_denoise_config`.

    Returns:
        bool, True for ``async`` on the identity trajectory.
    """
    # alpha = 1.0 and offset = 0.0 define the identity (diagonal) trajectory.
    return config.denoise_mode == "async" and config.variance_shift_alpha == 1.0 and config.linear_offset == 0.0


def schedule_sync(video_scheduler: FlowScheduler, action_scheduler: FlowScheduler, num_steps: int) -> Schedule:
    """Both streams advance in lockstep, each on its own scheduler's inference grid.

    The streams were trained on independent ``(sigma_v, sigma_a)`` draws, so any per-stream shift pair is
    in-distribution; each shift is the scheduler's own (yaml ``shift_video`` / ``shift_action``).

    Args:
        video_scheduler / action_scheduler: FlowScheduler of each stream (read only; grids are not mutated).
        num_steps: int, N denoise steps.

    Returns:
        Schedule: N (t_video, t_action) pairs plus the ``(0.0, 0.0)`` sentinel.
    """
    v_ts = video_scheduler.inference_timesteps(num_steps).tolist()
    a_ts = action_scheduler.inference_timesteps(num_steps).tolist()
    return [(v, a) for v, a in zip(v_ts, a_ts)] + [(0.0, 0.0)]


def schedule_variance_shift(
    video_scheduler: FlowScheduler,
    action_scheduler: FlowScheduler,
    num_steps: int,
    *,
    lead: str,
    alpha: float,
    offset: float,
) -> Schedule:
    """Latent-Forcing-style ordered schedule: one stream denoises earlier.

    Both streams share a global progress ``u = k / num_steps``. The **lead**
    stream takes cleanness ``f_alpha(u) >= u`` (Latent Forcing arXiv:2602.11401
    Eq. 4) so it reaches "clean" earlier; the **lag** stream takes ``u``,
    optionally delayed by ``offset``: cleanness stays 0 (sigma 1) until global
    progress passes ``offset``, then advances linearly (the piecewise variant).
    Each stream's sigma stays on the backbone's training grid via
    ``alpha_shift(1 - cleanness, scheduler.shift)``.

    Computed in float32 on the schedulers' own base grid
    (``linspace(1, 0, n+1)[:-1]``), with the lead curve applied as the
    algebraically identical ``1 - f_alpha(1 - s) == f_{1/alpha}(s)`` -- exact
    at ``alpha=1`` in floating point -- and the ``offset == 0`` path leaving
    the lag grid untouched, so ``alpha=1, offset=0`` reproduces
    ``schedule_sync`` bit-for-bit.

    Args:
        video_scheduler / action_scheduler: FlowScheduler of each stream (only ``shift`` is read).
        num_steps: int, N denoising steps per stream.
        lead: ``"action"`` or ``"video"``, the stream that denoises earlier. alpha: float >= 1, lead-curve
            strength (1 = sync diagonal). offset: float in [0, 1), delay of the lag stream's start. All three
            already validated by :func:`normalize_denoise_config`.

    Returns:
        Schedule: N (t_video, t_action) pairs plus the ``(0.0, 0.0)`` sentinel; sigma decreases monotonically.
    """
    num_train = float(NUM_TRAIN_TIMESTEPS)

    # s[k] = 1 - k/num_steps: the schedulers' float32 base sigma grid.
    s = torch.linspace(1.0, 0.0, num_steps + 1)[:-1]
    # Lead pre-shift sigma 1 - f_alpha(1-s) rewritten as f_{1/alpha}(s), which
    # leaves s bitwise untouched at alpha=1; the lag stream stays on s unless
    # delayed by offset below.
    lead_sigma = alpha_shift(s, 1.0 / alpha)
    if offset > 0.0:
        # lag cleanness = clamp((u - off)/(1 - off), 0, 1) in sigma form;
        # the off == 0 passthrough keeps alpha=1 bitwise == sync.
        lag_sigma = torch.clamp(s / (1.0 - offset), max=1.0)
    else:
        lag_sigma = s
    if lead == "video":
        v_sigma, a_sigma = lead_sigma, lag_sigma
    else:
        v_sigma, a_sigma = lag_sigma, lead_sigma
    # Each stream's sigma rides its own alpha-shift grid (matches training).
    v_ts = (alpha_shift(v_sigma, video_scheduler.shift) * num_train).tolist()
    a_ts = (alpha_shift(a_sigma, action_scheduler.shift) * num_train).tolist()

    return [(v, a) for v, a in zip(v_ts, a_ts)] + [(0.0, 0.0)]


def make_schedule(
    config: DenoiseConfig, video_scheduler: FlowScheduler, action_scheduler: FlowScheduler, num_steps: int
) -> Schedule:
    """Build the sync or async denoising trajectory named by ``config.denoise_mode``.

    Args:
        config: DenoiseConfig from :func:`normalize_denoise_config` (validated once there).
        video_scheduler / action_scheduler: FlowScheduler of each stream (e.g. ``arch.video_scheduler``).
        num_steps: int, N denoise steps.

    Returns:
        Schedule: N (t_video, t_action) pairs plus the ``(0.0, 0.0)`` sentinel.
    """
    if config.denoise_mode == "sync":
        return schedule_sync(video_scheduler, action_scheduler, num_steps)
    return schedule_variance_shift(
        video_scheduler,
        action_scheduler,
        num_steps,
        lead=config.lead_modality,
        alpha=config.variance_shift_alpha,
        offset=config.linear_offset,
    )


__all__ = [
    "Schedule",
    "DenoiseConfig",
    "VALID_DENOISE_MODES",
    "normalize_denoise_config",
    "denoise_async_is_noop",
    "schedule_sync",
    "schedule_variance_shift",
    "make_schedule",
]
