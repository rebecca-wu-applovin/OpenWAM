"""flux3_action_base backbone (07_sparse_history_backbones): RoPE ids, per-layer masks, native noise law and the
per-layer MoT geometry on a tiny random trunk (CPU)."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from openwam.model.architectures.utils.mask_modes import build_frame_mask
from openwam.model.video_backbone.flux.backbone import _N_MODE, FluxActionVideoBackbone
from openwam.model.video_backbone.flux.dit import JointSingleSeq, JointSingleSeqParams
from openwam.model.video_backbone.flux.masks import build_flux_layer_masks, build_stream_ids, video_time_ids

SEG = {"txt": 4, "video": 6, "video_cond": 9, "action_cond": 3}  # T, F, H, P token counts
SA = 2  # OpenWAM action tokens
# 3 clean (H) + 2 noisy (F) latents of 3 tokens each, as in SEG.
FM = build_frame_mask(OmegaConf.create({"num_latents": 5, "num_hist_latents": 3}), "bidirectional", "mutual")
ORDER = ["txt", "video", "video_cond", "action_cond", "action"]


def _slices(seg=SEG, s_action=SA):
    out, a = {}, 0
    for n, ln in zip(ORDER, [seg[k] for k in ORDER[:-1]] + [s_action]):
        out[n] = slice(a, a + ln)
        a += ln
    return out


def test_video_time_ids_scheme_a():
    # latent i at 4 i / 24 s on the 10 ms clock
    assert video_time_ids(torch.arange(7)).tolist() == [0, 16, 33, 50, 66, 83, 100]


def test_stream_ids():
    ids = build_stream_ids(n_hist=5, n_fut=2, h=2, w=3, text_len=4, n_prop=5)
    assert ids["txt"].tolist() == [[0, 0, 0, l] for l in range(4)]
    # video tokens in (t h w) order, l = 0; future frames continue the latent index after the history
    assert ids["video"].shape == (2 * 6, 4) and ids["video"][0].tolist() == [83, 0, 0, 0]
    assert ids["video"][-1].tolist() == [100, 1, 2, 0]
    assert ids["video_cond"][:, 0].unique().tolist() == [0, 16, 33, 50, 66]
    # proprio: negative real time (1 s apart), l = -1
    assert ids["action_cond"].tolist() == [[-400, 0, 0, -1], [-300, 0, 0, -1], [-200, 0, 0, -1],
                                           [-100, 0, 0, -1], [0, 0, 0, -1]]


def test_layer_masks():
    masks = build_flux_layer_masks(FM, SEG, SA, None, n_hist=3, num_mode_layers=5, num_layers=33, device="cpu")
    assert len(masks) == 33
    sl = _slices()
    mode, single = masks[0], masks[5]
    # mode blocks: every stream (incl. the OpenWAM action tokens) only itself
    for a in ORDER:
        for b in ORDER:
            assert bool(mode[sl[a], sl[b]].all()) == (a == b) and bool(mode[sl[a], sl[b]].any()) == (a == b)
    # single blocks: H rows never read F / A; everything else reads everything
    assert not single[sl["video_cond"], sl["video"]].any() and not single[sl["video_cond"], sl["action"]].any()
    assert single[sl["video_cond"], sl["txt"]].all() and single[sl["video_cond"], sl["action_cond"]].all()
    for rows in ("txt", "video", "action_cond", "action"):
        assert single[sl[rows]].all()


def test_text_padding_masks():
    masks = build_flux_layer_masks(
        FM, SEG, SA, torch.tensor([4, 2]), n_hist=3, num_mode_layers=5, num_layers=33, device="cpu"
    )
    m = masks[5]
    assert m.shape == (2, 1, sum(SEG.values()) + SA, sum(SEG.values()) + SA)
    # sample 1: text keys 2, 3 hidden from every row except their own diagonal
    assert not m[1, 0, 6, 2:4].any() and m[1, 0, 6, :2].all() and m[1, 0, 3, 3]
    assert m[0, 0, 6, :4].all()


def test_native_noise_law():
    vb = FluxActionVideoBackbone(dit=torch.nn.Identity(), vae=torch.nn.Identity(), text_encoder=torch.nn.Identity(), text_encoder_dir="", action_cond_dim=4)
    torch.manual_seed(0)
    ts, sig, w = vb.scheduler.sample_train(4096, device="cpu", dtype=torch.float32)
    assert ((sig > 0) & (sig <= 1)).all() and torch.allclose(ts, sig * 1000) and (w == 1).all()
    # shift 42 puts most mass near pure noise
    assert sig.median() > 0.95


def _tiny_trunk():
    torch.manual_seed(0)
    params = JointSingleSeqParams(
        in_channels={"video": 8, "video_cond": 8, "action_cond": 4},
        sequence={"x_video": "video", "x_video_cond": "video_cond", "x_action_cond": "action_cond"},
        vec_in_dim=768, context_in_dim=16, hidden_size=64, num_heads=2, depth=_N_MODE, depth_single_blocks=2,
        axes_dim=[8, 8, 8, 8],
    )
    dit = JointSingleSeq(params).float()
    for p in dit.parameters():  # LastLayer / modulations are zero-sized at init in practice: randomize all
        torch.nn.init.normal_(p, std=0.05)
    return dit


def test_mot_layer_count_and_geometry():
    vb = FluxActionVideoBackbone(dit=_tiny_trunk(), vae=torch.nn.Identity(), text_encoder=torch.nn.Identity(), text_encoder_dir="", action_cond_dim=4)
    assert vb.num_layers == _N_MODE + 2 and vb.head_dim * vb.num_heads == vb.dim
    # The frame mask must describe exactly the stream's latents.
    st = type("S", (), {"extras": {"seg_lens": SEG, "text_valid": None, "n_hist": 3}, "grid_frames": 4,
                        "hidden_states": torch.zeros(1)})()
    with pytest.raises(ValueError):
        vb.build_joint_attention_masks(FM, st, 2)
