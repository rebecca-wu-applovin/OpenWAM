"""Per-row proprio_mask gating in ``build_contexts`` (the proprio tokens of ActionDiT's context).

Covers (no GPU required):
  - mask all False             zeros the proprio tokens and sets their context_mask entries False
  - mixed batch                routes each sample independently
  - per-dim mask               a row is real when any of its dims is
  - gradient isolation         masked rows contribute zero gradient to proprio_encoder
  - global switch off          the text comes back unchanged when context proprio is off
  - shape check                a mask that does not match the proprio raises
"""

from __future__ import annotations

import pytest
import torch

from openwam.model.video_backbone.base import TextCondition
from tests._arch import proprio_context_arch


def _make_text(B: int = 2, L: int = 4, D: int = 16) -> TextCondition:
    """Returns: TextCondition with a random (B, L, D) context, every text token valid."""
    torch.manual_seed(0)
    return TextCondition(context=torch.randn(B, L, D), context_mask=torch.ones(B, L, dtype=torch.bool))


def _action_context(arch, text, proprio, mask):
    """Returns: ActionDiT's TextCondition from ``build_contexts``."""
    return arch.build_contexts(text, proprio, mask)[1]


# ---------------------------------------------------------------------------
# Per-sample mask behaviour
# ---------------------------------------------------------------------------


def test_mask_all_true_is_the_encoded_proprio():
    """All-True mask: the appended token is proprio_encoder(proprio) and attendable."""
    arch = proprio_context_arch(state_dim=7, text_dim=16)
    text = _make_text(B=2, L=4, D=16)
    proprio = torch.randn(2, 1, 7)

    out = _action_context(arch, text, proprio, torch.ones(2, 1, 7, dtype=torch.bool))

    torch.testing.assert_close(out.context[:, -1:], arch.proprio_encoder(proprio))
    assert out.context_mask[:, -1].all()


def test_mask_all_false_zeros_token():
    """All-False mask zeros the appended proprio token and masks it out."""
    arch = proprio_context_arch(state_dim=7, text_dim=16)
    text = _make_text(B=2, L=4, D=16)

    out = _action_context(arch, text, torch.randn(2, 1, 7), torch.zeros(2, 1, 7, dtype=torch.bool))

    assert out.context[:, -1, :].abs().sum().item() == 0.0
    assert out.context_mask[:, -1].any().item() is False


def test_mixed_batch_routing():
    """Per-row routing: True rows get real token+True mask, False rows zero+False."""
    arch = proprio_context_arch(state_dim=7, text_dim=16)
    text = _make_text(B=3, L=4, D=16)

    mask = torch.tensor([True, False, True]).view(3, 1, 1).expand(3, 1, 7)
    out = _action_context(arch, text, torch.randn(3, 1, 7), mask)

    appended = out.context[:, -1, :]
    assert appended[0].abs().sum().item() > 0  # mask=True
    assert appended[1].abs().sum().item() == 0.0  # mask=False
    assert appended[2].abs().sum().item() > 0  # mask=True
    assert out.context_mask[:, -1].tolist() == [True, False, True]


def test_per_dim_mask_keeps_a_row_with_any_real_dim():
    """A row with some padded dims (single-arm proprio) is still a real token; each history row is gated alone."""
    arch = proprio_context_arch(state_dim=7, text_dim=16, h=3)
    text = _make_text(B=1, L=4, D=16)
    mask = torch.zeros(1, 3, 7, dtype=torch.bool)
    mask[0, 0, :3] = True  # row 0: partial dims
    mask[0, 2] = True  # row 2: every dim; row 1: none

    out = _action_context(arch, text, torch.randn(1, 3, 7), mask)

    assert out.context.shape == (1, 4 + 3, 16)
    assert out.context_mask[0, 4:].tolist() == [True, False, True]
    assert out.context[0, 5].abs().sum().item() == 0.0


# ---------------------------------------------------------------------------
# Gradient isolation
# ---------------------------------------------------------------------------


def test_gradient_isolation_weight():
    """mask=False everywhere: proprio_encoder.weight.grad is None or all zero."""
    arch = proprio_context_arch(state_dim=7, text_dim=16)
    arch.proprio_encoder.zero_grad(set_to_none=True)

    out = _action_context(arch, _make_text(B=2), torch.randn(2, 1, 7), torch.zeros(2, 1, 7, dtype=torch.bool))
    out.context.sum().backward()

    g = arch.proprio_encoder.weight.grad
    assert g is None or g.abs().sum().item() == 0.0


def test_gradient_isolation_mixed_batch():
    """Mixed batch [True, False]: weight.grad equals the grad from running only row 0."""
    state_dim, text_dim = 7, 16
    torch.manual_seed(123)
    text_full = _make_text(B=2, L=4, D=text_dim)
    proprio_full = torch.randn(2, 1, state_dim)

    # --- Mixed batch ---
    arch_mixed = proprio_context_arch(state_dim=state_dim, text_dim=text_dim)
    arch_mixed.proprio_encoder.zero_grad(set_to_none=True)
    mask = torch.tensor([True, False]).view(2, 1, 1).expand(2, 1, state_dim)
    _action_context(arch_mixed, text_full, proprio_full, mask).context.sum().backward()
    g_mixed = arch_mixed.proprio_encoder.weight.grad.clone()

    # --- Single-row (mask-True row only) reference, same encoder init ---
    arch_solo = proprio_context_arch(state_dim=state_dim, text_dim=text_dim)
    arch_solo.proprio_encoder.load_state_dict(arch_mixed.proprio_encoder.state_dict())
    arch_solo.proprio_encoder.zero_grad(set_to_none=True)
    text_solo = TextCondition(
        context=text_full.context[:1].detach().clone(), context_mask=text_full.context_mask[:1].clone()
    )
    solo = _action_context(
        arch_solo, text_solo, proprio_full[:1].detach().clone(), torch.ones(1, 1, state_dim, dtype=torch.bool)
    )
    solo.context.sum().backward()
    g_solo = arch_solo.proprio_encoder.weight.grad.clone()

    assert torch.allclose(g_mixed, g_solo, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# Switch / shape
# ---------------------------------------------------------------------------


def test_global_switch_off_returns_text_unchanged():
    """When proprio context is globally disabled, both streams get the text as is (the mask is ignored)."""
    arch = proprio_context_arch(state_dim=7, text_dim=16, enabled=False)
    text = _make_text(B=2, L=4, D=16)

    video_text, action_text = arch.build_contexts(text, torch.randn(2, 1, 7), torch.ones(2, 1, 7, dtype=torch.bool))
    assert video_text is text and action_text is text


def test_mask_shape_mismatch_raises():
    """A mask that does not match the proprio raises ValueError with a clear message."""
    arch = proprio_context_arch(state_dim=7, text_dim=16)
    with pytest.raises(ValueError, match="proprio must be"):
        arch.build_contexts(_make_text(B=2), torch.randn(2, 1, 7), torch.ones(3, 1, 7, dtype=torch.bool))


def test_proprio_token_follows_the_padded_text():
    """The proprio token is appended after the padded text, so valid context tokens are not a prefix any more."""
    arch = proprio_context_arch(state_dim=7, text_dim=16)
    text = TextCondition(
        context=torch.randn(2, 4, 16), context_mask=torch.arange(4)[None] < torch.tensor([2, 4])[:, None]
    )

    _, out = arch.build_contexts(text, torch.randn(2, 1, 7), torch.ones(2, 1, 7, dtype=torch.bool))

    assert out.context.shape == (2, 5, 16) and out.context_mask.shape == (2, 5)
    assert out.context_mask[0].tolist() == [True, True, False, False, True]
    assert out.context_mask[:, -1].all()


def test_action_dit_rejects_per_token_timestep():
    from openwam.model.action_backbone.separate_action_dit import MoTActionDiT

    dit = MoTActionDiT(
        shift_action=5.0,
        train_timestep={"law": "grid", "weighting": "bell"},
        action_dim=7,
        dim=32,
        ffn_dim=64,
        num_heads=4,
        num_layers=1,
        attn_head_dim=8,
        text_dim=4096,
        freq_dim=256,
        max_action_len=1024,
        eps=1e-6,
    )
    with pytest.raises(ValueError, match="action timestep must be 1D"):
        context = torch.randn(2, 4, dit.text_dim)
        context_mask = torch.ones(2, 4, dtype=torch.bool)
        dit.prepare_state(torch.randn(2, 5, 7), torch.randn(2, 5), context=context, context_mask=context_mask)
