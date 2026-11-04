"""Fused-attention dispatcher shared by the video (Wan) and action streams."""

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange

try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn

    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False


def fused_backend_name() -> str:
    """Pick the fused backend by install priority FA3 > FA2 > sage > SDPA.

    Inputs: none (reads the module-level availability flags).

    Returns: one of "flash_attention_3", "flash_attention_2", "sage_attention", "torch_sdpa". It applies to
    CUDA fp16/bf16 input only; flash_attention falls back to SDPA for anything else.
    """
    if FLASH_ATTN_3_AVAILABLE:
        return "flash_attention_3"
    if FLASH_ATTN_2_AVAILABLE:
        return "flash_attention_2"
    if SAGE_ATTN_AVAILABLE:
        return "sage_attention"
    return "torch_sdpa"


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    attn_mask: Optional[torch.Tensor] = None,
):
    """Multi-head attention on packed heads, on the fused kernel :func:`fused_backend_name` picks when it applies.

    Args:
        q: (B, Sq, num_heads * head_dim) query.
        k, v: (B, Skv, num_heads * head_dim) key / value (Skv == Sq for self-attention).
        num_heads: int, head count.
        attn_mask: (B or 1, 1 or num_heads, Sq, Skv) bool / additive SDPA mask, or None; a mask forces SDPA.

    Returns:
        (B, Sq, num_heads * head_dim) attention output.
    """
    # FA2/FA3/sage are CUDA half-precision kernels without masks; anything else runs SDPA. fused_backend_name is
    # the one place the fused priority lives.
    fused_ok = q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)
    backend = fused_backend_name() if fused_ok and attn_mask is None else "torch_sdpa"
    # FA2/FA3 take (B, S, N, D); sage and SDPA take (B, N, S, D)
    if backend in ("flash_attention_3", "flash_attention_2"):
        q, k, v = (rearrange(t, "b s (n d) -> b s n d", n=num_heads) for t in (q, k, v))
        if backend == "flash_attention_3":
            x = flash_attn_interface.flash_attn_func(q, k, v)
            if isinstance(x, tuple):
                x = x[0]
        else:
            x = flash_attn.flash_attn_func(q, k, v)
        return rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    q, k, v = (rearrange(t, "b s (n d) -> b n s d", n=num_heads) for t in (q, k, v))
    if backend == "sage_attention":
        x = sageattn(q, k, v)
    else:
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return rearrange(x, "b n s d -> b s (n d)", n=num_heads)
