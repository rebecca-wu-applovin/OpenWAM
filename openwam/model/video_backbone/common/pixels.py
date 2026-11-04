"""PIL <-> tensor video helpers shared by the cosmos3 / ge / flux backbones.

Moved from the deleted ``cosmos_predict25/_vae_utils.py`` (renamed, logic unchanged).
"""

from __future__ import annotations

from typing import Any

import torch


def pil_clips_to_tensor(frames: Any, *, device) -> "torch.Tensor":
    """Convert ``list[list[PIL.Image]]`` → ``(B, 3, T, H, W)`` float32 in ``[-1, 1]`` on ``device``.

    frames: B clips of T same-size PIL frames.
    device: where the tensor is built; pass the encoder's GPU so only uint8 crosses PCIe and the
        float conversion runs there (the CPU float32 path cost up to ~1 s/step on Modal, 13_modal_profile).
    Returns: (B, 3, T, H, W) float32, uint8 RGB → ``float / 127.5 - 1``, bit-identical on any device
        (14_pil_to_tensor_on_gpu).
    Self-contained (no Wan imports) so any backbone can use it without importing Wan.
    """
    import numpy as np

    if not isinstance(frames, (list, tuple)) or not frames:
        raise ValueError(
            f"Expected `frames` as a non-empty list of clips (each a list of PIL frames); got {type(frames).__name__}."
        )
    arrs = []
    for clip in frames:
        if not isinstance(clip, (list, tuple)) or not clip:
            raise ValueError("Each per-sample entry in `frames` must be a non-empty list of PIL.Image frames.")
        clip_arr = np.stack([np.asarray(f.convert("RGB"), dtype=np.uint8) for f in clip], axis=0)
        arrs.append(clip_arr)
    stack = np.stack(arrs, axis=0)  # (B, T, H, W, 3) uint8
    # The 256 possible values, computed on the CPU with the original formula. CUDA's scalar division
    # can round differently (1 ulp), so the device only looks values up and never divides.
    lut = (torch.arange(256, dtype=torch.float32) / 127.5 - 1.0).to(device)
    # uint8 to the device first (4x less to copy than float32), then map it there.
    idx = torch.from_numpy(stack).to(device, non_blocking=True).int()
    t = lut[idx].permute(0, 4, 1, 2, 3).contiguous()  # (B, 3, T, H, W)
    return t


def tensor_to_pil_frames(video: "torch.Tensor") -> list:
    """Convert ``(B, 3, T, H, W)`` in ``[-1, 1]`` → ``list[PIL.Image]`` (B=1 only)."""
    from PIL import Image  # local import; not always present on minimal CI

    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError(f"tensor_to_pil_frames expected (B, 3, T, H, W); got shape {tuple(video.shape)}.")
    if video.shape[0] != 1:
        raise NotImplementedError(
            f"video decode currently supports B=1 only; got B={video.shape[0]}. Deploy paths call decode per-sample."
        )
    frame_uint8 = (
        ((video[0].float().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(1, 2, 3, 0)
        .contiguous()
        .cpu()
        .numpy()
    )  # (T, H, W, 3) uint8
    return [Image.fromarray(frame) for frame in frame_uint8]
