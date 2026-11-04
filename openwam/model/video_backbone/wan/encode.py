"""Wan VAE / text-encoder IO.

Free functions — ``vae`` / ``tokenizer`` / ``text_encoder`` passed explicitly so
the backbone class holds no IO logic.
"""

from __future__ import annotations

from typing import Tuple

from torch import Tensor

from openwam.model.video_backbone.wan.preprocess import preprocess_video as _preprocess_video_native
from openwam.model.video_backbone.wan.preprocess import vae_output_to_video


def encode_text(prompts: list, *, tokenizer, text_encoder, device, max_length: int = 512) -> Tuple[Tensor, Tensor]:
    ids, mask = tokenizer(
        prompts,
        return_mask=True,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    ids = ids.to(device)
    mask = mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    context = text_encoder(ids, mask)
    for i, v in enumerate(seq_lens):
        context[i, v:] = 0
    return context, seq_lens


def encode_text_for_inference(
    prompt, *, prompt_embed_cache, tokenizer, text_encoder, device, max_length: int = 512
) -> Tuple[Tensor, Tensor]:
    """Deploy ``(context, seq_lens)``, reusing a prompt-keyed cached embed."""
    if prompt_embed_cache is not None and prompt in prompt_embed_cache:
        return prompt_embed_cache[prompt]
    context, seq_lens = encode_text(
        [prompt], tokenizer=tokenizer, text_encoder=text_encoder, device=device, max_length=max_length
    )
    if prompt_embed_cache is not None:
        prompt_embed_cache[prompt] = (context, seq_lens)
    return context, seq_lens


def preprocess_video(frames, *, dtype, device) -> Tensor:
    """frames: list of PIL frames. Returns: (1, 3, T, H, W) pixels in [-1, 1] on ``device``."""
    return _preprocess_video_native(frames, dtype=dtype, device=device)


def encode_video(video_tensor: Tensor, *, vae) -> Tensor:
    """video_tensor: (B, 3, T, H, W) pixels. Returns: (B, z, T_lat, h, w) latents."""
    return vae.batch_encode(video_tensor, device=video_tensor.device)


def decode_latents(latents: Tensor, *, vae, device, tiled: bool = True) -> Tensor:
    """latents: (B, z, T_lat, h, w). Returns: decoded pixels from the VAE."""
    return vae.decode(latents.to(device), device=device, tiled=tiled)


def latents_to_frames(video_tensor: Tensor) -> list:
    """video_tensor: VAE output pixels. Returns: list of PIL frames."""
    return vae_output_to_video(video_tensor)
