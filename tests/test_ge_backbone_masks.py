"""GE-Base-fast per-layer MoT masks (ge/masks.py, 07_sparse_history_backbones).

Grid: T frames x V views x h x w tokens, frame-major with views inside a frame;
H clean history frames; A action tokens after the video.
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from openwam.model.architectures.utils.mask_modes import build_frame_mask
from openwam.model.video_backbone.ge.masks import build_ge_joint_masks, view_index

T, V, HH, WW, H, A, L = 4, 2, 2, 3, 2, 5, 6
TPF = V * HH * WW
SV = T * TPF
NH = H * TPF
# T latents, the first H clean (bidirectional among themselves), mutual v<->a.
FM = build_frame_mask(OmegaConf.create({"num_latents": T, "num_hist_latents": H}), "bidirectional", "mutual")


@pytest.fixture(scope="module")
def masks():
    return build_ge_joint_masks(FM, n_view=V, view_h=HH, view_w=WW, s_action=A, num_layers=L)


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


def test_video_only_masks(masks):
    m = build_ge_joint_masks(FM, n_view=V, view_h=HH, view_w=WW, s_action=0, num_layers=3)
    vid = view_index(T, V, HH, WW)
    # Without action tokens the masks are the video blocks of the joint ones.
    assert torch.equal(m[0], masks[0][:SV, :SV])
    assert torch.equal(m[1], m[0] & (vid[:, None] == vid[None, :]))  # per-view block = GE's (b v) batching


def test_native_noise_sampler_matches_ge_trainer():
    # ge_trainer.py: idx = floor(u*1000), sigma = FlowMatchEuler(shift 1).sigmas[idx], t = floor(sigma*1000), w = 1.
    from openwam.model.video_backbone.ge.scheduler import GEFlowScheduler

    torch.manual_seed(0)
    ts, sig, w = GEFlowScheduler(shift_video=1.0).sample_train(4096, device="cpu", dtype=torch.float32)
    assert ((sig > 0) & (sig <= 1)).all() and w.eq(1).all()
    torch.testing.assert_close(ts, torch.floor(sig * 1000))
    # Grid sigma_i = (1000 - i) / 1000: every sample sits on it, and it is covered ~uniformly.
    idx = torch.round(1000 - sig * 1000)
    torch.testing.assert_close(sig, (1000 - idx) / 1000)
    assert 0.45 < sig.mean().item() < 0.55


def test_dropped_caption_is_the_empty_prompt_with_its_own_mask():
    """Task 12: a caption-dropped sample reaches GE as "" and gets the empty prompt's embedding *and* mask.

    (GE's native trainer swapped in the empty embedding but kept the original prompt's mask; that model-side
    dropout is gone, the dataset now replaces the prompt itself.)
    """
    from types import SimpleNamespace

    from torch import nn

    from openwam.model.video_backbone.ge.backbone import GEBaseFastVideoBackbone

    def _tokenizer(prompts, **kw):
        """Token count = 1 + len(prompt) (EOS), max 4 tokens."""
        mask = torch.zeros(len(prompts), 4, dtype=torch.long)
        for i, p in enumerate(prompts):
            mask[i, : min(len(p) + 1, 4)] = 1
        return SimpleNamespace(input_ids=mask * 7, attention_mask=mask)

    class _T5(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(1))

        def forward(self, ids):
            return (ids.float().unsqueeze(-1).expand(-1, -1, 3) * self.w,)

    shell = GEBaseFastVideoBackbone.__new__(GEBaseFastVideoBackbone)
    nn.Module.__init__(shell)
    shell._tokenizer, shell.text_encoder, shell._text_max_length = _tokenizer, _T5(), 4
    shell._device, shell._dtype = torch.device("cpu"), torch.float32
    shell.text_lookup = shell._t5_entries
    batch = shell.encode_text(["pick", ""], grid=None, num_frames=1)
    alone = shell.encode_text([""], grid=None, num_frames=1)
    assert torch.equal(batch.context[1:], alone.context)
    assert batch.context_mask[1].tolist() == [True, False, False, False]  # "" -> EOS only, not "pick"'s mask
