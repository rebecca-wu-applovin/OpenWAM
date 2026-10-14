# Vendored into OpenWAM from black-forest-labs/flux-action @ 830f1ce (src/flux_action/models/text_encoder.py),
# Apache-2.0. Changes: only the Qwen3-VL embedder is kept (no Hub spec parsing, mock encoder or compile hook).
"""Qwen3-VL-4B text context for the flux-3-action trunk.

Every caption goes through the chat template, is right-padded to its own bucket (the next multiple
of 80 tokens) and encoded with the padding hidden by the attention mask; the hidden states of 8
layers are stacked along channels into ``(1, L, 8 * 2560 = 20480)``. The DiT sees the whole bucket
(padding tokens included) with no mask, exactly as the native policy does.
"""

from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import Tensor, nn

TEXT_PAD_MULTIPLE = 80
TEXT_PAD_MAX_LENGTH = 8192
OUTPUT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
CTX_DIM = 20480  # 8 Qwen3-VL-4B layers x 2560 hidden


def padded_text_token_count(real_len: int, pad_multiple: int = TEXT_PAD_MULTIPLE, pad_max: int = TEXT_PAD_MAX_LENGTH) -> int:
    """Bucket length of a caption: ``real_len`` rounded up to ``pad_multiple``, capped at ``pad_max``."""
    return min(math.ceil(real_len / pad_multiple) * pad_multiple, pad_max)


class Qwen3VLEmbedder(nn.Module):
    """Frozen Qwen3-VL-4B; ``self.model`` is the HF ``Qwen3VLForConditionalGeneration``."""

    def __init__(self, model_dir: str, dtype: torch.dtype = torch.bfloat16):
        """
        Args:
            model_dir: str, local Qwen3-VL-4B-Instruct directory (flux-3-action-base/text_encoder).
            dtype: torch.dtype of the weights and of the returned context.
        """
        super().__init__()
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, torch_dtype=dtype)
        self.processor = AutoProcessor.from_pretrained(model_dir)
        if self.processor.tokenizer.padding_side != "right":
            raise ValueError("the text encoder requires a right-padding tokenizer")
        self.output_layer = list(OUTPUT_LAYERS)

    @torch.no_grad()
    def encode(self, texts: list[str], *, fixed_length: int | None = None) -> list[Tensor]:
        """Captions -> contexts ``[(1, L_i, 20480)]`` in input order, one forward per length bucket.

        Args:
            texts: list[str] of B captions ("" = the CFG null caption).
            fixed_length: optional int, pad every caption to this length instead of its own bucket.

        Returns:
            list of B tensors ``(1, L_i, 20480)`` in the weight dtype, ``L_i`` a multiple of 80.
        """
        tokenizer = self.processor.tokenizer
        formatted = [
            self.processor.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
            )
            for text in texts
        ]
        # Group captions by their padded length so each bucket is one forward.
        buckets: dict[int, list[int]] = {}
        for i, prompt in enumerate(formatted):
            if fixed_length is not None:
                buckets.setdefault(fixed_length, []).append(i)
                continue
            real_len = tokenizer(
                prompt, return_tensors="pt", padding=False, truncation=True, max_length=TEXT_PAD_MAX_LENGTH
            )["input_ids"].shape[1]
            buckets.setdefault(padded_text_token_count(real_len), []).append(i)
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        results: list[Tensor | None] = [None] * len(texts)
        for target_length, members in sorted(buckets.items()):
            toks = tokenizer(
                [formatted[i] for i in members],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=target_length,
                padding_side="right",
            )
            out = self.model.model(
                input_ids=toks["input_ids"].to(device),
                attention_mask=toks["attention_mask"].to(device),
                output_hidden_states=True,
                use_cache=False,
            )
            # (b, 8, L, 2560) -> (b, L, 20480): the 8 layers stacked along channels.
            stacked = torch.stack([out.hidden_states[k] for k in self.output_layer], dim=1)
            result = rearrange(stacked, "b c l d -> b l (c d)").to(dtype)
            for j, i in enumerate(members):
                results[i] = result[j : j + 1].clone()
        return results


__all__ = ["CTX_DIM", "Qwen3VLEmbedder", "padded_text_token_count"]
