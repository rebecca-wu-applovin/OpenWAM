"""Per-layer joint attention masks for the GE multi-view backbone under the OpenWAM MoT.

Token order of the video stream is frame-major with the views stacked inside a
frame: index = ((t * V + v) * h + y) * w + x, i.e. the latent ``(T, V*h, w)``
grid of the vertically stacked canvas flattened row by row. So a "frame" (what
the shared mask helpers call ``video_tokens_per_frame``) holds every view.

GE attends within a view in two of every three blocks and across views in the
third (``block_idx % 3 == 0``). On top of that pattern the OpenWAM rules apply
(see ``architectures/utils/mask_modes.py``): the clean history frames never read
noisy frames or action; noisy frames and action read everything they may.
"""

from __future__ import annotations

from typing import List

import torch

from openwam.model.architectures.utils.mask_modes import build_cross_modal_attention_mask, build_history_video_mask


class _ModeShim:
    """Minimal stand-in exposing the v<->v mode to ``build_cross_modal_attention_mask``."""

    def __init__(self, mode: str):
        self.video_attention_mask_mode = mode

    def build_video_to_video_mask(self, *, video_seq_len, video_tokens_per_frame, device):
        # Single clean frame (legacy path): same rule as the H-frame helper with H = 1.
        return build_history_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            num_clean_frames=1,
            mode=self.video_attention_mask_mode,
            device=device,
        )


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
    *,
    n_frames: int,
    n_view: int,
    view_h: int,
    view_w: int,
    s_action: int,
    num_clean_frames: int,
    attention_mask_mode: str,
    video_mask_mode: str,
    num_layers: int,
    cross_view_every: int = 3,
    device=None,
) -> List[torch.Tensor]:
    """One bool mask (True = attend) per layer over ``[video tokens, action tokens]``.

    Args:
        n_frames / n_view / view_h / view_w: latent grid (T, V, h, w); S_v = T*V*h*w.
        s_action: int, action tokens appended after the video (0 = video only).
        num_clean_frames: int, leading clean history frames H (1 = legacy first frame).
        attention_mask_mode: v<->a mode (``mutual`` in 07).
        video_mask_mode: v<->v history mode (``first_frame_causal`` = history bidirectional,
            history rows never read noisy frames; ``bidirectional`` = GE native, no history rule).
        num_layers: int, number of blocks.
        cross_view_every: int, block_idx % cross_view_every == 0 attends across views (GE: 3).
        device: torch device.

    Returns:
        list of ``num_layers`` (S_v + s_action, S_v + s_action) bool tensors; layers share
        the two distinct tensors (cross-view / per-view).
    """
    tpf = n_view * view_h * view_w  # tokens per latent frame (all views)
    s_video = n_frames * tpf
    # Cross-view layers: only the shared OpenWAM rules.
    cross = build_cross_modal_attention_mask(
        _ModeShim(video_mask_mode),
        s_video=s_video,
        s_action=s_action,
        video_tokens_per_frame=tpf,
        mode=attention_mask_mode,
        device=torch.device(device) if device is not None else torch.device("cpu"),
        num_clean_frames=num_clean_frames,
    )
    # Per-view layers: additionally restrict video->video to the same view.
    views = view_index(n_frames, n_view, view_h, view_w, device=cross.device)
    per_view = cross.clone()
    per_view[:s_video, :s_video] &= views[:, None] == views[None, :]
    return [cross if layer % cross_view_every == 0 else per_view for layer in range(num_layers)]


__all__ = ["build_ge_joint_masks", "view_index"]
