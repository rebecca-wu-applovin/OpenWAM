"""Framework pieces of the sparse-history layout (07_sparse_history_backbones).

- joint attention mask with H clean history frames ("clean never reads noisy")
- H = 1 stays byte-identical to the legacy first-frame path
- N proprio tokens with a learned slot embedding, and its optimizer visibility
"""

from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.utils.mask_modes import (
    MUTUAL,
    build_cross_modal_attention_mask,
    build_history_video_mask,
)

H, F, TPF, SA = 3, 2, 2, 4  # 3 history + 2 future frames x 2 tokens, 4 action tokens
SV = (H + F) * TPF
NH = H * TPF  # history token count


class _VB:
    """Minimal video backbone; ``build_video_to_video_mask`` is the legacy first-frame rule."""

    def __init__(self, mode):
        self.video_attention_mask_mode = mode

    def build_video_to_video_mask(self, *, video_seq_len, video_tokens_per_frame, device):
        mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
        mask[:video_tokens_per_frame, video_tokens_per_frame:] = False
        return mask


def _joint(mode, n_clean=H):
    """(SV+SA, SV+SA) joint mask for the mutual mode."""
    return build_cross_modal_attention_mask(
        _VB(mode),
        s_video=SV,
        s_action=SA,
        video_tokens_per_frame=TPF,
        mode=MUTUAL,
        device=torch.device("cpu"),
        num_clean_frames=n_clean,
    )


@pytest.mark.parametrize("mode", ["first_frame_causal", "history_per_frame_causal"])
def test_history_rows_never_read_noisy(mode):
    m = _joint(mode)
    # History rows: no future video, no action.
    assert not m[:NH, NH:].any()
    # Future rows: all video + all action.
    assert m[NH:SV, :].all()
    # Action rows: everything.
    assert m[SV:, :].all()


def test_history_inner_pattern():
    bi = _joint("first_frame_causal")
    assert bi[:NH, :NH].all()  # bidirectional inside the history
    causal = _joint("history_per_frame_causal")
    for i in range(H):
        rows = slice(i * TPF, (i + 1) * TPF)
        # history frame i sees frames 0..i only (Long-WAM)
        assert causal[rows, : (i + 1) * TPF].all()
        assert not causal[rows, (i + 1) * TPF : NH].any()


def test_bidirectional_mode_is_all_true():
    assert build_history_video_mask(
        video_seq_len=SV, video_tokens_per_frame=TPF, num_clean_frames=H, mode="bidirectional", device="cpu"
    ).all()


def test_single_clean_frame_matches_legacy():
    # H = 1 must go through the backbone's own v<->v builder, exactly as before.
    legacy = build_cross_modal_attention_mask(
        _VB("first_frame_causal"),
        s_video=SV,
        s_action=SA,
        video_tokens_per_frame=TPF,
        mode=MUTUAL,
        device=torch.device("cpu"),
    )
    assert torch.equal(legacy, _joint("first_frame_causal", n_clean=1))
    # and the generic H-frame rule with H = 1 agrees with it
    generic = _joint("first_frame_causal", n_clean=1)
    assert not generic[:TPF, TPF:SV].any() and not generic[:TPF, SV:].any()


def _proprio_holder(n_tokens, state_dim=3, ctx=8):
    """Stand-in object carrying exactly what ``_append_proprio_context_token`` reads."""
    holder = types.SimpleNamespace(
        _use_proprioception_context=True,
        proprio_encoder=nn.Linear(state_dim, ctx),
        proprio_slot_emb=nn.Embedding(n_tokens, ctx) if n_tokens > 1 else None,
        num_proprio_tokens=n_tokens,
        proprio_dim=state_dim,
    )
    return holder


def _append(holder, proprio, b=2, ctx=8, text_len=4):
    from openwam.model.video_backbone.base import TextCondition

    mask = torch.arange(text_len)[None] < torch.tensor([2, 4])[:, None]
    text = TextCondition(context=torch.zeros(b, text_len, ctx), context_mask=mask)
    return BaseWAMArchitecture._append_proprio_context_token(holder, text, proprio, None)


def test_n_proprio_tokens_with_slot_embedding():
    torch.manual_seed(0)
    holder = _proprio_holder(5)
    nn.init.normal_(holder.proprio_slot_emb.weight)
    proprio = torch.randn(1, 3).expand(2, 5, 3).clone()  # same state in all 5 slots
    out = _append(holder, proprio)
    assert out.context.shape == (2, 4 + 5, 8)
    assert out.context_mask.shape == (2, 9)
    assert out.context_mask[:, 4:].all()
    tok = out.context[:, 4:]
    # Identical states differ only by the slot embedding → order is recoverable.
    expected = holder.proprio_encoder(proprio) + holder.proprio_slot_emb.weight
    torch.testing.assert_close(tok, expected)
    assert not torch.allclose(tok[:, 0], tok[:, 1])


def test_single_proprio_token_unchanged():
    torch.manual_seed(0)
    holder = _proprio_holder(1)
    proprio = torch.randn(2, 3)
    out = _append(holder, proprio)
    assert out.context.shape == (2, 5, 8)
    torch.testing.assert_close(out.context[:, 4], holder.proprio_encoder(proprio))


def test_wrong_token_count_raises():
    with pytest.raises(ValueError):
        _append(_proprio_holder(5), torch.randn(2, 4, 3))


def test_slot_embedding_reaches_optimizer():
    # build_trainable_parameters only walks top-level child modules.
    class _Arch(nn.Module):
        get_trainable_modules = BaseWAMArchitecture.get_trainable_modules

    arch = _Arch()
    arch.proprio_encoder = nn.Linear(3, 8)
    arch.proprio_slot_emb = nn.Embedding(5, 8)
    assert "proprio_slot_emb" in arch.get_trainable_modules()
