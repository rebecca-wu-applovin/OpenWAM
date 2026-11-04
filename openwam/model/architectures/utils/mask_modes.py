"""Frame-level attention mask of the joint video + action sequence, built once from the latent layout.

Rows / columns 0 .. T-1 are the T = ``layout.num_latents`` latent frames (the first ``layout.num_hist_latents``
are clean conditioning, the rest noisy), row / column T is the whole action block. True = the row may attend
the column. The visibility rules live only here:

* clean latents never see noisy latents or the action; among themselves they see each other
  (``clean_attention=bidirectional``) or the ones up to themselves (``causal``);
* noisy latents and the action see everything (``attention_mask_mode=mutual``, the only verified v<->a mode;
  ``resolve_config`` rejects the other three).

A backbone expands this to token level with ``build_joint_attention_masks``.
"""

from __future__ import annotations

import torch


def build_frame_mask(layout, clean_attention: str, attention_mask_mode: str) -> torch.Tensor:
    """Frame-level mask of ``[T latents, action block]``.

    Args:
        layout: ``cfg.common.layout`` (``num_latents`` T, ``num_hist_latents`` clean prefix).
        clean_attention: ``bidirectional`` | ``causal``, visibility among the clean latents.
        attention_mask_mode: v<->a mode; only ``mutual`` is implemented.

    Returns:
        (T + 1, T + 1) bool on CPU, True = attend. Raises ValueError for an unknown clean_attention and
        NotImplementedError for a v<->a mode other than ``mutual``.
    """
    if attention_mask_mode != "mutual":
        raise NotImplementedError(
            f"build_frame_mask implements only the mutual v<->a mode, got {attention_mask_mode!r}"
        )
    t, n_clean = int(layout.num_latents), int(layout.num_hist_latents)
    # Noisy latents and the action (rows n_clean .. T) see everything.
    mask = torch.ones((t + 1, t + 1), dtype=torch.bool)
    # Clean rows: nothing outside the clean prefix (no noisy latent, no action).
    mask[:n_clean, n_clean:] = False
    if clean_attention == "causal":
        mask[:n_clean, :n_clean] = torch.tril(torch.ones((n_clean, n_clean), dtype=torch.bool))
    elif clean_attention != "bidirectional":
        raise ValueError(f"clean_attention must be bidirectional | causal, got {clean_attention!r}")
    return mask


__all__ = ["build_frame_mask"]
