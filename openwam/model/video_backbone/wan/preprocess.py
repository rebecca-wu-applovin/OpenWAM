"""Stateless pixel/latent preprocessing for the Wan backbone.

Pure functions lifted verbatim from the former ``BasePipeline`` helpers: Wan's own
PIL -> pixel and VAE output -> PIL conversions (kept until Task 12 unifies them with
``common/pixels.py``). ``dtype`` / ``device`` are passed explicitly.
"""

from __future__ import annotations

import numpy as np
import torch
from einops import reduce, repeat
from PIL import Image


def preprocess_image(image, *, dtype, device, pattern="B C H W", min_value=-1, max_value=1):
    """PIL.Image -> tensor in ``[min_value, max_value]`` on ``(dtype, device)``."""
    image = torch.Tensor(np.array(image, dtype=np.float32))
    image = image.to(dtype=dtype, device=device)
    image = image * ((max_value - min_value) / 255) + min_value
    image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))
    return image


def preprocess_video(video, *, dtype, device, pattern="B C T H W", min_value=-1, max_value=1):
    """List[PIL.Image] -> stacked tensor along the ``T`` axis of ``pattern``.

    video: T PIL images (or HxWxC arrays), all the same size.
    Returns: (1, C, T, H, W) for the default pattern, values in [min_value, max_value], on (dtype, device).
    """
    frames = [np.asarray(image) for image in video]
    if frames and all(f.dtype == np.uint8 for f in frames) and torch.device(device).type == "cuda":
        # Fast path: one pinned uint8 host->device copy per clip, converted on the GPU. The per-frame
        # float32 pageable copies below cost ~2.7 s/step on Modal (vs 0.06 s locally). uint8 -> dtype
        # is exact and the scaling ops match preprocess_image, so the output is bit-identical.
        clip = torch.from_numpy(np.stack(frames)).to(device, non_blocking=True)  # (T, H, W, C) uint8
        clip = clip.to(dtype=dtype) * ((max_value - min_value) / 255) + min_value
        return repeat(clip, f"T H W C -> {pattern}", **({"B": 1} if "B" in pattern else {})).contiguous()
    video = [
        preprocess_image(image, dtype=dtype, device=device, min_value=min_value, max_value=max_value) for image in video
    ]
    video = torch.stack(video, dim=pattern.index("T") // 2)
    return video


def _vae_output_to_image(vae_output, pattern="B C H W", min_value=-1, max_value=1):
    if pattern != "H W C":
        vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
    image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(0, 255)
    image = image.to(device="cpu", dtype=torch.uint8)
    image = Image.fromarray(image.numpy())
    return image


def vae_output_to_video(vae_output, pattern="B C T H W", min_value=-1, max_value=1):
    """VAE pixel tensor -> list[PIL.Image]."""
    if pattern != "T H W C":
        vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
    return [
        _vae_output_to_image(image, pattern="H W C", min_value=min_value, max_value=max_value) for image in vae_output
    ]
