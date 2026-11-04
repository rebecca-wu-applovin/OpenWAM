"""Joint attention masks of the Cosmos3 gen stream: the shared frame-mask expansion plus the und prefix keys.

Cosmos3's per-layer keys / values carry a leading block of cached und (text) tokens with no matching query rows
(``BlockLoopState.prefix_kv_len`` / ``prefix_kv_mask``). Every query row may read that prefix (the video stream
reads its own text natively; action rows reading it mirrors the cross-attention context of the other families),
gated per sample by ``prefix_kv_mask`` when the und text is padded.
"""

from __future__ import annotations

import torch
from torch import Tensor


def widen_mask_for_prefix_kv(mask: Tensor, prefix_len: int, prefix_mask) -> Tensor:
    """Prepend the prefix-K/V key columns to a query x key mask.

    Args:
        mask: (S_q, S_k) bool, or (B, 1, S_q, S_k).
        prefix_len: int P, prefix keys (0 = nothing to add).
        prefix_mask: (B, P) bool per-sample validity of the prefix keys, or None when every prefix key is real
            (keeps the cheap batch-shared 2-D mask; testing a tensor here would sync / break the compiled MoT loop).

    Returns:
        (S_q, P + S_k) bool when ``prefix_mask`` is None (rank kept), else (B, 1, S_q, P + S_k).
    """
    if prefix_len <= 0:
        return mask
    rows = mask.shape[-2]
    if prefix_mask is None:
        pad = torch.ones((*mask.shape[:-1], prefix_len), dtype=torch.bool, device=mask.device)
        return torch.cat([pad, mask], dim=-1)
    bsz = prefix_mask.shape[0]
    # Per-sample prefix columns, broadcast over every query row.
    pad = prefix_mask.view(bsz, 1, 1, prefix_len).expand(bsz, 1, rows, prefix_len)
    base = mask if mask.dim() == 4 else mask.view(1, 1, rows, mask.shape[-1]).expand(bsz, 1, rows, mask.shape[-1])
    return torch.cat([pad, base], dim=-1)


__all__ = ["widen_mask_for_prefix_kv"]
