"""Regression tests for fused-attention backend dispatch.

FA2/FA3/sage are CUDA half-precision kernels: they raise rather than degrade.
The dispatcher must therefore check device *and* dtype before selecting
one, otherwise merely having ``flash-attn`` importable breaks fp32 and CPU
inputs -- including the CPU tensors the rest of this suite runs on.

The backends are faked here, so these tests exercise the "flash-attn is
installed" configuration without it being installed. That matters: CI runs on
CPU-only torch and would otherwise never cover this dispatch at all.
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange


def _sdpa_reference(q, k, v, num_heads):
    qq, kk, vv = (rearrange(t, "b s (n d) -> b n s d", n=num_heads) for t in (q, k, v))
    return rearrange(F.scaled_dot_product_attention(qq, kk, vv), "b n s d -> b s (n d)", n=num_heads)


def _install_fake_flash_attn_2(monkeypatch, attn, kernel):
    """Make ``flash_attention`` believe FA2 -- and only FA2 -- is installed."""
    monkeypatch.setattr(attn, "flash_attn", types.SimpleNamespace(flash_attn_func=kernel), raising=False)
    monkeypatch.setattr(attn, "FLASH_ATTN_2_AVAILABLE", True)
    monkeypatch.setattr(attn, "FLASH_ATTN_3_AVAILABLE", False)
    monkeypatch.setattr(attn, "SAGE_ATTN_AVAILABLE", False)


# --------------------------------------------------------------- shared dispatcher


def test_flash_attention_falls_back_to_sdpa_on_cpu(monkeypatch):
    """CPU tensors must never reach the fused kernel."""
    import openwam.model.common.attention as attn

    def _reject(*args, **kwargs):
        raise AssertionError("fused kernel must not receive CPU tensors")

    _install_fake_flash_attn_2(monkeypatch, attn, _reject)

    num_heads = 4
    q, k, v = (torch.randn(2, 8, num_heads * 16) for _ in range(3))

    out = attn.flash_attention(q, k, v, num_heads=num_heads)

    assert torch.allclose(out, _sdpa_reference(q, k, v, num_heads), atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_flash_attention_falls_back_to_sdpa_for_cuda_fp32(monkeypatch):
    """fp32 is rejected by the fused kernels even on CUDA."""
    import openwam.model.common.attention as attn

    def _reject(*args, **kwargs):
        raise AssertionError("fused kernel must not receive fp32 tensors")

    _install_fake_flash_attn_2(monkeypatch, attn, _reject)

    num_heads = 4
    q, k, v = (torch.randn(2, 8, num_heads * 16, device="cuda", dtype=torch.float32) for _ in range(3))

    out = attn.flash_attention(q, k, v, num_heads=num_heads)

    assert torch.allclose(out, _sdpa_reference(q, k, v, num_heads), atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_flash_attention_keeps_fused_path_for_cuda_half(monkeypatch):
    """The guard must not cost us the fast path it is protecting."""
    import openwam.model.common.attention as attn

    calls = []

    def _record(q, k, v, *args, **kwargs):
        calls.append(1)
        return q

    _install_fake_flash_attn_2(monkeypatch, attn, _record)

    num_heads = 4
    q, k, v = (torch.randn(2, 8, num_heads * 16, device="cuda", dtype=torch.bfloat16) for _ in range(3))

    out = attn.flash_attention(q, k, v, num_heads=num_heads)

    assert calls, "cuda/bf16 input should still reach the fused kernel"
    assert torch.equal(out, q)


def test_fused_backend_name_follows_availability(monkeypatch):
    import openwam.model.common.attention as attn

    for flag in ("FLASH_ATTN_3_AVAILABLE", "FLASH_ATTN_2_AVAILABLE", "SAGE_ATTN_AVAILABLE"):
        monkeypatch.setattr(attn, flag, False)
    assert attn.fused_backend_name() == "torch_sdpa"

    monkeypatch.setattr(attn, "SAGE_ATTN_AVAILABLE", True)
    assert attn.fused_backend_name() == "sage_attention"

    monkeypatch.setattr(attn, "FLASH_ATTN_2_AVAILABLE", True)
    assert attn.fused_backend_name() == "flash_attention_2"

    monkeypatch.setattr(attn, "FLASH_ATTN_3_AVAILABLE", True)
    assert attn.fused_backend_name() == "flash_attention_3"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused path needs CUDA half-precision tensors")
def test_flash_attention_dispatches_on_fused_backend_name(monkeypatch):
    """flash_attention runs the kernel fused_backend_name names (one priority list, not a second copy of it)."""
    import openwam.model.common.attention as attn

    calls = []
    monkeypatch.setattr(attn, "fused_backend_name", lambda: "sage_attention")
    monkeypatch.setattr(attn, "sageattn", lambda q, k, v: calls.append(q.shape) or q, raising=False)
    q = torch.randn(1, 3, 8, device="cuda", dtype=torch.bfloat16)
    attn.flash_attention(q, q, q, num_heads=2)
    assert calls == [(1, 2, 3, 4)]


def test_action_stream_uses_the_shared_dispatcher():
    """The action attention modules call common.attention.flash_attention (no second dispatcher)."""
    import openwam.model.action_backbone.components as comp
    import openwam.model.action_backbone.separate_action_dit as sad
    import openwam.model.common.attention as common

    assert sad.flash_attention is common.flash_attention
    assert not hasattr(comp, "get_attention_fn")
