"""Per-layer joint attention masks for the GE multi-view backbone under the OpenWAM MoT.

Token order of the video stream is frame-major with the views stacked inside a
frame: index = ((t * V + v) * h + y) * w + x, i.e. the latent ``(T, V*h, w)``
grid of the vertically stacked canvas flattened row by row. So a "frame" (what
the shared mask helpers call ``video_tokens_per_frame``) holds every view.

GE attends within a view in two of every three blocks and across views in the
third (``block_idx % 3 == 0``). On top of that pattern the frame mask's rules apply
(``architectures/utils/mask_modes.build_frame_mask``): the clean latents never read
noisy latents or the action; noisy latents and the action read everything.
"""

from __future__ import annotations

from typing import List

import torch

from openwam.model.video_backbone.base import expand_frame_mask


def view_index(n_frames: int, n_view: int, view_h: int, view_w: int, device=None) -> torch.Tensor:
    """View id of every video token.

    Args:
        n_frames / n_view / view_h / view_w: latent grid sizes (T, V, h, w).
        device: torch device.

    Returns:
        (T*V*h*w,) long, token -> view id under the frame-major / view-inside order.
    """
    per_view = view_h * view_w
    idx = torch.arange(n_frames * n_view * per_view, device=device)
    return (idx // per_view) % n_view


def build_ge_joint_masks(
    frame_mask: torch.Tensor,
    *,
    n_view: int,
    view_h: int,
    view_w: int,
    s_action: int,
    num_layers: int,
    cross_view_every: int = 3,
    device=None,
) -> List[torch.Tensor]:
    """One bool mask (True = attend) per layer over ``[video tokens, action tokens]``.

    Args:
        frame_mask: (T + 1, T + 1) bool frame mask (``build_frame_mask``); T latent frames, S_v = T*V*h*w.
        n_view / view_h / view_w: latent grid of one view (V, h, w).
        s_action: int, action tokens appended after the video (0 = video only).
        num_layers: int, number of blocks.
        cross_view_every: int, block_idx % cross_view_every == 0 attends across views (GE: 3).
        device: torch device.

    Returns:
        list of ``num_layers`` (S_v + s_action, S_v + s_action) bool tensors; layers share
        the two distinct tensors (cross-view / per-view).
    """
    n_frames = int(frame_mask.shape[0]) - 1
    tpf = n_view * view_h * view_w  # tokens per latent frame (all views)
    s_video = n_frames * tpf
    # Cross-view layers: only the shared frame rules.
    cross = expand_frame_mask(frame_mask, [tpf] * n_frames, s_action, device if device is not None else "cpu")
    # Per-view layers: additionally restrict video->video to the same view.
    views = view_index(n_frames, n_view, view_h, view_w, device=cross.device)
    per_view = cross.clone()
    per_view[:s_video, :s_video] &= views[:, None] == views[None, :]
    return [cross if layer % cross_view_every == 0 else per_view for layer in range(num_layers)]


__all__ = ["build_ge_joint_masks", "view_index"]
