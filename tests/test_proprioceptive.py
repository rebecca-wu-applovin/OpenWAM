"""Tests for FastWAM-style proprio context conditioning."""

import pytest
import torch

from openwam.model.video_backbone.base import TextCondition
from tests._arch import proprio_context_arch


def _text(context, seq_lens=None):
    """context: (B, L, D); seq_lens: (B,) valid text tokens (default all). Returns: TextCondition."""
    b, l = context.shape[:2]
    lens = torch.full((b,), l) if seq_lens is None else seq_lens
    return TextCondition(context=context, context_mask=torch.arange(l)[None] < lens[:, None])


def test_proprio_context_token_extends_context_and_mask():
    arch = proprio_context_arch(state_dim=7, text_dim=16)
    text = _text(torch.randn(2, 4, 16), torch.tensor([2, 4]))

    _, out = arch.build_contexts(text, torch.randn(2, 1, 7), torch.ones(2, 1, 7, dtype=torch.bool))

    assert out.context.shape == (2, 5, 16)
    assert out.context_mask.shape == (2, 5)
    assert out.context_mask[:, -1].all()
    # The token follows the padded text, so the valid tokens are not a prefix any more.
    assert out.context_mask[0].tolist() == [True, True, False, False, True]


def test_proprio_context_needs_one_row_per_history_frame():
    arch = proprio_context_arch(state_dim=7, text_dim=16, h=3)
    with pytest.raises(ValueError, match="proprio must be"):
        arch.build_contexts(_text(torch.randn(1, 4, 16)), torch.randn(1, 1, 7), torch.ones(1, 1, 7, dtype=torch.bool))


def test_proprio_context_validates_dim():
    arch = proprio_context_arch(state_dim=7, text_dim=16)
    with pytest.raises(ValueError, match="proprio must be"):
        arch.build_contexts(_text(torch.randn(1, 4, 16)), torch.randn(1, 1, 6), torch.ones(1, 1, 6, dtype=torch.bool))


def test_action_dit_rejects_per_token_timestep():
    from openwam.model.action_backbone.separate_action_dit import ActionDiT

    dit = ActionDiT(
        action_dim=7,
        dim=32,
        ffn_dim=64,
        num_heads=4,
        num_layers=1,
        video_dim=32,
        bridge_layers=(0,),
        variant="joint_self_attn",
    )
    with pytest.raises(ValueError, match="action timestep must be 1D"):
        context = torch.randn(2, 4, dit.text_dim)
        context_mask = torch.ones(2, 4, dtype=torch.bool)
        dit.prepare_state(torch.randn(2, 5, 7), torch.randn(2, 5), context=context, context_mask=context_mask)


def test_action_dit_rejects_proprio_constructor_flags():
    from openwam.model.action_backbone.separate_action_dit import ActionDiT

    with pytest.raises(TypeError, match="use_proprioception"):
        ActionDiT(
            action_dim=7,
            dim=32,
            ffn_dim=64,
            num_heads=4,
            num_layers=1,
            video_dim=32,
            bridge_layers=(0,),
            variant="joint_self_attn",
            use_proprioception=True,
            state_dim=7,
        )
