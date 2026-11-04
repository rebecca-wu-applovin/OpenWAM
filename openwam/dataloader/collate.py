"""``collate``: dataset samples -> one ``Batch`` (the DataLoader's ``collate_fn``).

A sample is data only (``dataset.__getitem__``): ``video`` H + F PIL frames, ``prompt``, ``action`` (K, D),
``action_mask`` (K, D), ``proprio`` (H, D), ``proprio_mask`` (H, D) and ``source`` (the sub-dataset's name).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Batch:
    """One training batch, the single input of ``BaseWAMArchitecture.compute_loss``.

    Fields:
        video: list of B clips, each H + F PIL frames (history, the current frame t last, then the future).
        prompt: list of B str.
        action: (B, K, D) float, the normalized action chunk.
        action_is_pad: (B, K, D) bool, True = this (step, dim) has no supervision.
        proprio: (B, H, D) float, one normalized proprio row per history frame (oldest first).
        proprio_mask: (B, H, D) bool, True = this proprio entry is real.
        source: list of B str, the sub-dataset each sample came from (visualization titles).
    """

    video: list
    prompt: list
    action: Tensor
    action_is_pad: Tensor
    proprio: Tensor
    proprio_mask: Tensor
    source: list

    def to(self, device, dtype: torch.dtype = torch.float32, non_blocking: bool = False) -> "Batch":
        """Move every tensor to ``device``; the float fields also take ``dtype``.

        Args:
            device: torch.device (or str) to move to.
            dtype: torch.dtype of ``action`` / ``proprio`` (default float32 = what ``collate`` builds, so
                accelerate's dataloader, which calls ``.to(device, non_blocking=...)``, leaves values unchanged).
            non_blocking: bool, forwarded to ``Tensor.to``.

        Returns:
            A new Batch; the masks stay bool, the clip / prompt / source lists are shared.
        """
        floats = {"device": device, "dtype": dtype, "non_blocking": non_blocking}
        return dataclasses.replace(
            self,
            action=self.action.to(**floats),
            action_is_pad=self.action_is_pad.to(device=device, non_blocking=non_blocking),
            proprio=self.proprio.to(**floats),
            proprio_mask=self.proprio_mask.to(device=device, non_blocking=non_blocking),
        )


def collate(samples: list) -> Batch:
    """Stack B dataset samples into one Batch (numpy or tensor fields alike; clips stay PIL lists).

    Args:
        samples: list of B sample dicts (keys in the module docstring); every sample has the same H, K, D.

    Returns:
        Batch with action (B, K, D), action_is_pad (B, K, D), proprio (B, H, D), proprio_mask (B, H, D) on CPU.
    """

    def stack(key: str) -> Tensor:
        # as_tensor: readers may hand numpy arrays or tensors.
        return torch.stack([torch.as_tensor(s[key]) for s in samples])

    return Batch(
        video=[s["video"] for s in samples],
        prompt=[s["prompt"] for s in samples],
        action=stack("action"),
        action_is_pad=~stack("action_mask").bool(),
        proprio=stack("proprio"),
        proprio_mask=stack("proprio_mask").bool(),
        source=[s["source"] for s in samples],
    )


__all__ = ["Batch", "collate"]
