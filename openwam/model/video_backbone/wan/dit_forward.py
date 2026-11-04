"""Wan DiT forward-path helper: time modulation.

Free function, dependency-injected, no backbone import: ``build_time_modulation`` takes the DiT and the
plain layout numbers explicitly and has one output layout (per token); callers adapt to it.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from openwam.model.video_backbone.wan.dit import sinusoidal_embedding_1d


def build_time_modulation(
    dit, timestep: Tensor, num_hist_latents: int, num_latents: int, tokens_per_frame: int
) -> Tuple[Tensor, Tensor]:
    """Per-token time embedding and AdaLN modulation: clean history frames at t = 0, the rest at ``timestep``.

    The time MLPs run once per latent frame (every token of a frame shares its timestep), then each frame's
    rows are repeated over its tokens.

    Args:
        dit: WanModel (``freq_dim``, ``dim``, ``time_embedding``, ``time_projection``).
        timestep: (B,) video timestep in the model dtype.
        num_hist_latents: int, leading clean latent frames (``layout.num_hist_latents``).
        num_latents: int, latent frames F of the clip (``layout.num_latents``).
        tokens_per_frame: int, DiT tokens per latent frame (patchified h * w).

    Returns:
        time_embed: (B, L, dim) head time embedding, L = num_latents * tokens_per_frame (frame-major).
        time_mod: (B, L, 6, dim) block modulation.
    """
    batch_size = timestep.shape[0]
    # Per-frame timesteps: the sampled t everywhere, then the clean history pinned to 0.
    frame_t = torch.ones(batch_size, num_latents, dtype=timestep.dtype, device=timestep.device) * timestep.view(
        batch_size, 1
    )
    frame_t[:, :num_hist_latents] = 0
    # Time MLPs on the (B * F) frame timesteps, in the timestep's (= model) dtype.
    frame_embed = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, frame_t.reshape(-1)))
    frame_embed = frame_embed.reshape(batch_size, num_latents, dit.dim)
    frame_mod = dit.time_projection(frame_embed).unflatten(2, (6, dit.dim))
    # Frame rows -> token rows (tokens of a frame are contiguous in the (f h w) sequence).
    time_embed = frame_embed.repeat_interleave(tokens_per_frame, dim=1)
    time_mod = frame_mod.repeat_interleave(tokens_per_frame, dim=1)
    return time_embed, time_mod
