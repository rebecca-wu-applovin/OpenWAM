"""Stream position ids and per-layer joint attention masks of the flux video-side sequence.

Sequence layout ``[ T | F | H | P | A ]`` (see ``backbone.py``); everything here is
batch-independent and CPU-only.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

_TEMPORAL_COMPRESSION = 4
_VIDEO_POSITION_FPS = 24.0  # PolicyConfig.video_position_fps: latent i sits at 4 i / 24 s
_STATE_FPS = 1.0  # our 5 proprio frames are 1 s apart (07 layout)
_P_STREAM = "action_cond"
# Segment order of the video-side sequence.
_SEGMENTS = ("txt", "video", "video_cond", _P_STREAM)

__all__ = ["build_flux_layer_masks", "build_stream_ids", "times_to_ids", "video_time_ids"]


def times_to_ids(time: Tensor) -> Tensor:
    return (time * 1000 // 10).to(dtype=torch.int64)


def video_time_ids(latent_index: Tensor) -> Tensor:
    """Scheme (a) time ids: latent i at ``times_to_ids(i * 4 / 24)`` (10 ms ticks).

    Args:
        latent_index: (N,) int latent indices.

    Returns:
        (N,) int64 ids (0, 16, 33, 50, 66, 83, 100 for i = 0..6).
    """
    seconds = latent_index.float() * _TEMPORAL_COMPRESSION / _VIDEO_POSITION_FPS
    return times_to_ids(seconds)


def build_stream_ids(n_hist: int, n_fut: int, h: int, w: int, text_len: int, n_prop: int) -> dict[str, Tensor]:
    """4-axis RoPE ids ``(t, h, w, l)`` of every stream (batch-independent).

    Args:
        n_hist / n_fut: H history / future latent frames.
        h / w: latent grid.
        text_len: int, padded text length L.
        n_prop: int, proprio tokens (one per history frame).

    Returns:
        dict ``txt`` (L, 4), ``video`` (n_fut*h*w, 4), ``video_cond`` (n_hist*h*w, 4), ``action_cond`` (n_prop, 4).
    """

    def grid(first: int, n: int) -> Tensor:
        # (t h w) token order of prc_vid; l = 0 for video.
        t = video_time_ids(torch.arange(first, first + n))
        return torch.cartesian_prod(t, torch.arange(h), torch.arange(w), torch.zeros(1, dtype=torch.long))

    txt = torch.stack(
        [torch.zeros(text_len, dtype=torch.long)] * 3 + [torch.arange(text_len)], dim=-1
    )  # prc_txt: (0, 0, 0, l)
    # history_profile pack_actions: times (j - (n - 1)) / fps, l = -1.
    t_prop = times_to_ids((torch.arange(n_prop).float() - (n_prop - 1)) / _STATE_FPS)
    zeros = torch.zeros(n_prop, dtype=torch.long)
    prop = torch.stack([t_prop, zeros, zeros, torch.full((n_prop,), -1, dtype=torch.long)], dim=-1)
    return {"txt": txt, "video": grid(n_hist, n_fut), "video_cond": grid(0, n_hist), _P_STREAM: prop}


def build_flux_layer_masks(
    seg_lens: dict[str, int],
    s_action: int,
    text_valid: Optional[Tensor],
    *,
    num_mode_layers: int,
    num_layers: int,
    device,
    strict_clean_rows: bool = False,
) -> list[Tensor]:
    """Per-layer joint masks over ``[T | F | H | P | A]`` (True = attend).

    Args:
        seg_lens: token count per segment ``txt / video / video_cond / action_cond``.
        s_action: int, OpenWAM action tokens appended after the video-side sequence (0 = none).
        text_valid: optional (B,) int, each sample's own text bucket length (<= seg_lens['txt']).
        num_mode_layers: int, leading mode-block layers (5).
        num_layers: int, total layers (33).
        device: torch device.
        strict_clean_rows: bool, test-only: also hide F / A from the T and P rows (native flux lets them
            read everything, so H reads noisy content one layer later through T / P).

    Returns:
        list of ``num_layers`` masks, each (S, S) or (B, 1, S, S) when text lengths differ per sample;
        the mode-layer entries share one tensor, as do the single-layer ones.
    """
    names = list(_SEGMENTS) + ["action"]
    lens = [seg_lens[n] for n in _SEGMENTS] + [s_action]
    starts = torch.tensor([0] + lens[:-1]).cumsum(0).tolist()
    s = sum(lens)
    sl = {n: slice(a, a + n_) for n, a, n_ in zip(names, starts, lens)}

    # Mode blocks: block-diagonal, every stream attends only to itself.
    mode = torch.zeros((s, s), dtype=torch.bool, device=device)
    for n in names:
        mode[sl[n], sl[n]] = True
    # Single blocks: native full attention; clean history never reads the noisy future / action.
    single = torch.ones((s, s), dtype=torch.bool, device=device)
    single[sl["video_cond"], sl["video"]] = False
    single[sl["video_cond"], sl["action"]] = False
    if strict_clean_rows:
        for clean in ("txt", _P_STREAM):
            single[sl[clean], sl["video"]] = False
            single[sl[clean], sl["action"]] = False

    def with_text_padding(m: Tensor) -> Tensor:
        # Per sample: text keys past the sample's own bucket are invisible (rows keep their diagonal).
        if text_valid is None or bool((text_valid == seg_lens["txt"]).all()):
            return m
        b = text_valid.shape[0]
        key_ok = torch.ones((b, s), dtype=torch.bool, device=device)
        pos = torch.arange(seg_lens["txt"], device=device)
        key_ok[:, sl["txt"]] = pos[None] < text_valid.to(device)[:, None]
        out = m[None] & key_ok[:, None, :]
        out |= torch.eye(s, dtype=torch.bool, device=device)[None]
        return out[:, None]

    mode, single = with_text_padding(mode), with_text_padding(single)
    return [mode] * num_mode_layers + [single] * (num_layers - num_mode_layers)
