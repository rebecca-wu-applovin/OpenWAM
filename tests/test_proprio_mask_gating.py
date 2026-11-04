"""Per-sample proprio_mask gating in ``_append_proprio_context_token``.

Covers (no GPU required):
  - mask all True              matches the pre-change baseline bit-exactly
  - mask all False             zeros the proprio token and sets context_mask[:, -1] = False
  - mixed batch                routes each row independently
  - gradient isolation         masked rows contribute zero gradient to proprio_encoder
  - back-compat                no sample mask (None) falls back to all-True
  - global switch off          the text comes back unchanged when context proprio is off
  - shape normalization        (B,) and (B, 1) inputs both succeed; bad shapes raise

Mirrors the ``_ContextProprioArch`` fixture pattern from ``test_proprioceptive.py``.
"""

from __future__ import annotations

import pytest
import torch

from openwam.model.architectures.dual_system.joint_self_attn import (
    DualSystemSelfAttnArchitecture,
)
from openwam.model.video_backbone.base import TextCondition


class _ContextProprioArch(DualSystemSelfAttnArchitecture):
    """Minimal architecture shell that only exercises BaseWAMArchitecture helpers."""

    def __init__(self, *, state_dim: int = 7, text_dim: int = 16, enabled: bool = True):
        super().__init__(None)
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self._init_proprio_context(
            {
                "use_proprioception": enabled,
                "state_dim": state_dim,
                "text_dim": text_dim,
            },
            text_dim=text_dim,
        )


def _make_text(B: int = 2, L: int = 4, D: int = 16) -> TextCondition:
    """Returns: TextCondition with a random (B, L, D) context, every text token valid."""
    torch.manual_seed(0)
    return TextCondition(context=torch.randn(B, L, D), context_mask=torch.ones(B, L, dtype=torch.bool))


# ---------------------------------------------------------------------------
# Per-sample mask behaviour
# ---------------------------------------------------------------------------


def test_mask_all_true_matches_baseline():
    """All-True mask must produce numerically identical output to the no-mask path."""
    arch = _ContextProprioArch(state_dim=7, text_dim=16)
    torch.manual_seed(42)
    text = _make_text(B=2, L=4, D=16)
    proprio = torch.randn(2, 7)

    out_no_mask = arch._append_proprio_context_token(text, proprio, None)
    out_with_mask = arch._append_proprio_context_token(text, proprio, torch.ones(2, 1, dtype=torch.bool))

    assert torch.equal(out_no_mask.context, out_with_mask.context)
    assert torch.equal(out_no_mask.context_mask, out_with_mask.context_mask)


def test_mask_all_false_zeros_token():
    """All-False mask zeros the appended proprio token."""
    arch = _ContextProprioArch(state_dim=7, text_dim=16)
    text = _make_text(B=2, L=4, D=16)
    proprio = torch.randn(2, 7)

    out = arch._append_proprio_context_token(text, proprio, torch.zeros(2, 1, dtype=torch.bool))

    # last token (the appended proprio token) must be all-zero for every batch row
    appended = out.context[:, -1, :]
    assert appended.abs().sum().item() == 0.0


def test_mask_all_false_attn_mask_false():
    """All-False mask sets context_mask[:, -1] = False."""
    arch = _ContextProprioArch(state_dim=7, text_dim=16)
    text = _make_text(B=2, L=4, D=16)
    proprio = torch.randn(2, 7)

    out = arch._append_proprio_context_token(text, proprio, torch.zeros(2, 1, dtype=torch.bool))

    assert out.context_mask[:, -1].any().item() is False


def test_mixed_batch_routing():
    """Per-row routing: True rows get real token+True mask, False rows zero+False."""
    arch = _ContextProprioArch(state_dim=7, text_dim=16)
    text = _make_text(B=3, L=4, D=16)
    proprio = torch.randn(3, 7)

    mask = torch.tensor([[True], [False], [True]], dtype=torch.bool)
    out = arch._append_proprio_context_token(text, proprio, mask)

    appended = out.context[:, -1, :]
    assert appended[0].abs().sum().item() > 0  # mask=True
    assert appended[1].abs().sum().item() == 0.0  # mask=False
    assert appended[2].abs().sum().item() > 0  # mask=True

    assert out.context_mask[:, -1].tolist() == [True, False, True]


# ---------------------------------------------------------------------------
# Gradient isolation
# ---------------------------------------------------------------------------


def test_gradient_isolation_weight():
    """Single-batch mask=False: proprio_encoder.weight.grad is None or all zero."""
    arch = _ContextProprioArch(state_dim=7, text_dim=16)
    arch.proprio_encoder.zero_grad(set_to_none=True)

    text = _make_text(B=2, L=4, D=16)
    proprio = torch.randn(2, 7)

    out = arch._append_proprio_context_token(text, proprio, torch.zeros(2, 1, dtype=torch.bool))
    out.context.sum().backward()

    g = arch.proprio_encoder.weight.grad
    assert g is None or g.abs().sum().item() == 0.0


def test_gradient_isolation_mixed_batch():
    """Mixed batch [True, False]: weight.grad equals the grad from running only row 0."""
    state_dim, text_dim = 7, 16
    torch.manual_seed(123)
    text_full = _make_text(B=2, L=4, D=text_dim)
    proprio_full = torch.randn(2, state_dim)

    # --- Mixed batch ---
    arch_mixed = _ContextProprioArch(state_dim=state_dim, text_dim=text_dim)
    arch_mixed.proprio_encoder.zero_grad(set_to_none=True)
    mask = torch.tensor([[True], [False]], dtype=torch.bool)
    out_mixed = arch_mixed._append_proprio_context_token(text_full, proprio_full, mask)
    out_mixed.context.sum().backward()
    g_mixed = arch_mixed.proprio_encoder.weight.grad.clone()

    # --- Single-row (mask-True row only) reference, same encoder init ---
    arch_solo = _ContextProprioArch(state_dim=state_dim, text_dim=text_dim)
    arch_solo.proprio_encoder.load_state_dict(arch_mixed.proprio_encoder.state_dict())
    arch_solo.proprio_encoder.zero_grad(set_to_none=True)
    text_solo = TextCondition(
        context=text_full.context[:1].detach().clone(), context_mask=text_full.context_mask[:1].clone()
    )
    out_solo = arch_solo._append_proprio_context_token(
        text_solo, proprio_full[:1].detach().clone(), torch.ones(1, 1, dtype=torch.bool)
    )
    out_solo.context.sum().backward()
    g_solo = arch_solo.proprio_encoder.weight.grad.clone()

    assert torch.allclose(g_mixed, g_solo, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_missing_field_backward_compat():
    """No sample mask -> behaviour matches the all-True path."""
    arch = _ContextProprioArch(state_dim=7, text_dim=16)
    text = _make_text(B=2, L=4, D=16)
    proprio = torch.randn(2, 7)

    out = arch._append_proprio_context_token(text, proprio, None)
    assert out.context.shape == (2, 5, 16)
    assert out.context_mask[:, -1].tolist() == [True, True]


def test_global_switch_off_returns_text_unchanged():
    """When proprio context is globally disabled, the text comes back as is (the mask is ignored)."""
    arch = _ContextProprioArch(state_dim=7, text_dim=16, enabled=False)
    text = _make_text(B=2, L=4, D=16)

    out = arch._append_proprio_context_token(text, None, torch.tensor([[True], [False]], dtype=torch.bool))
    assert out is text


# ---------------------------------------------------------------------------
# Shape normalization
# ---------------------------------------------------------------------------


def test_mask_1d_vs_2d_normalization():
    """Both (B,) and (B, 1) shapes produce identical outputs."""
    arch = _ContextProprioArch(state_dim=7, text_dim=16)
    text = _make_text(B=3, L=4, D=16)
    proprio = torch.randn(3, 7)

    mask_2d = torch.tensor([[True], [False], [True]], dtype=torch.bool)
    mask_1d = torch.tensor([True, False, True], dtype=torch.bool)

    out_2d = arch._append_proprio_context_token(text, proprio, mask_2d)
    out_1d = arch._append_proprio_context_token(text, proprio, mask_1d)

    assert torch.equal(out_2d.context, out_1d.context)
    assert torch.equal(out_2d.context_mask, out_1d.context_mask)


def test_mask_shape_mismatch_raises():
    """Wrong batch dim in mask raises ValueError with a clear message."""
    arch = _ContextProprioArch(state_dim=7, text_dim=16)
    text = _make_text(B=2, L=4, D=16)
    proprio = torch.randn(2, 7)

    with pytest.raises(ValueError, match="_proprio_sample_mask"):
        arch._append_proprio_context_token(text, proprio, torch.ones(3, 1, dtype=torch.bool))
