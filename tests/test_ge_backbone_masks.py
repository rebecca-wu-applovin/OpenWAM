"""GE-Base-fast per-layer MoT masks (ge/masks.py, 07_sparse_history_backbones).

Grid: T frames x V views x h x w tokens, frame-major with views inside a frame;
H clean history frames; A action tokens after the video.
"""

from __future__ import annotations

import pytest
import torch

from openwam.model.video_backbone.ge.masks import build_ge_joint_masks, view_index

T, V, HH, WW, H, A, L = 4, 2, 2, 3, 2, 5, 6
TPF = V * HH * WW
SV = T * TPF
NH = H * TPF


@pytest.fixture(scope="module")
def masks():
    return build_ge_joint_masks(
        n_frames=T, n_view=V, view_h=HH, view_w=WW, s_action=A, num_clean_frames=H,
        attention_mask_mode="mutual", video_mask_mode="first_frame_causal", num_layers=L,
    )


def test_layer_pattern(masks):
    assert len(masks) == L
    # GE: block_idx % 3 == 0 is cross-view, the rest per-view.
    for i, m in enumerate(masks):
        assert m.shape == (SV + A, SV + A)
        assert torch.equal(m, masks[0]) == (i % 3 == 0)


def test_view_index_order():
    vid = view_index(T, V, HH, WW)
    # Inside each frame: first HH*WW tokens are view 0, next view 1.
    assert vid[: HH * WW].eq(0).all() and vid[HH * WW : TPF].eq(1).all()
    assert torch.equal(vid[TPF : 2 * TPF], vid[:TPF])


def test_history_never_reads_noisy_or_action(masks):
    for m in masks:
        assert not m[:NH, NH:].any()


def test_action_reads_all_views_every_layer(masks):
    for m in masks:
        assert m[SV:, :].all()


def test_future_rows(masks):
    vid = view_index(T, V, HH, WW)
    same = vid[:, None] == vid[None, :]
    cross, per_view = masks[0], masks[1]
    # Future rows see all video + all action in cross-view layers.
    assert cross[NH:SV, :].all()
    # Per-view layers: future rows see only their own view's video, but all action.
    assert torch.equal(per_view[NH:SV, :SV], same[NH:SV])
    assert per_view[NH:SV, SV:].all()


def test_history_view_pattern(masks):
    vid = view_index(T, V, HH, WW)
    same = vid[:NH, None] == vid[None, :NH]
    assert masks[0][:NH, :NH].all()  # cross-view: history bidirectional across views
    assert torch.equal(masks[1][:NH, :NH], same)  # per-view: own view only


def test_ge_native_bidirectional_video_only():
    m = build_ge_joint_masks(
        n_frames=T, n_view=V, view_h=HH, view_w=WW, s_action=0, num_clean_frames=H,
        attention_mask_mode="mutual", video_mask_mode="bidirectional", num_layers=3,
    )
    vid = view_index(T, V, HH, WW)
    assert m[0].all()  # cross-view block = plain full attention
    assert torch.equal(m[1], vid[:, None] == vid[None, :])  # per-view block = GE's (b v) batching


def test_native_noise_sampler_matches_ge_trainer():
    # ge_trainer.py: idx = floor(u*1000), sigma = FlowMatchEuler(shift 1).sigmas[idx], t = floor(sigma*1000), w = 1.
    from openwam.model.video_backbone.ge_backbone import GEBaseFastVideoBackbone

    torch.manual_seed(0)
    ts, sig, w = GEBaseFastVideoBackbone.sample_training_noise(None, 4096, device="cpu", dtype=torch.float32)
    assert ((sig > 0) & (sig <= 1)).all() and w.eq(1).all()
    torch.testing.assert_close(ts, torch.floor(sig * 1000))
    # Grid sigma_i = (1000 - i) / 1000: every sample sits on it, and it is covered ~uniformly.
    idx = torch.round(1000 - sig * 1000)
    torch.testing.assert_close(sig, (1000 - idx) / 1000)
    assert 0.45 < sig.mean().item() < 0.55
