"""Unit tests of the frame-level joint attention mask (``build_frame_mask``).

Rows / columns are the T latents of the layout plus one action block. Clean latents never see noisy latents or
the action; noisy latents and the action see everything (``mutual``, the only implemented v<->a mode).
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from openwam.config import latent_layout
from openwam.model.architectures.utils.mask_modes import build_frame_mask

# 3 history frames per frame (3 clean latents) + 8 future frames at 4x (2 noisy latents): T = 5, action = row 5.
LAYOUT = OmegaConf.create(latent_layout([-2, -1, 0], list(range(1, 9)), 33, "per_frame", 4))
N_CLEAN, T = 3, 5


def test_mutual_bidirectional():
    mask = build_frame_mask(LAYOUT, "bidirectional", "mutual")
    assert mask.shape == (T + 1, T + 1) and mask.dtype == torch.bool
    assert mask[:N_CLEAN, :N_CLEAN].all()  # clean <-> clean
    assert not mask[:N_CLEAN, N_CLEAN:].any()  # clean -> noisy / action blocked
    assert mask[N_CLEAN:].all()  # noisy latents and the action see everything


def test_causal_clean_rows():
    mask = build_frame_mask(LAYOUT, "causal", "mutual")
    # Clean latent i sees clean latents 0..i only; the rest of the mask is the bidirectional one.
    assert torch.equal(mask[:N_CLEAN, :N_CLEAN], torch.tril(torch.ones(N_CLEAN, N_CLEAN, dtype=torch.bool)))
    assert torch.equal(mask[N_CLEAN:], build_frame_mask(LAYOUT, "bidirectional", "mutual")[N_CLEAN:])


def test_unknown_clean_attention_raises():
    with pytest.raises(ValueError):
        build_frame_mask(LAYOUT, "first_frame_causal", "mutual")


@pytest.mark.parametrize("mode", ["action_sees_video", "video_sees_action", "isolated"])
def test_unverified_modes_raise(mode):
    with pytest.raises(NotImplementedError):
        build_frame_mask(LAYOUT, "bidirectional", mode)
