# Vendored into OpenWAM from black-forest-labs/flux-action @ 830f1ce (src/flux_action/models/text_encoder.py),
# Apache-2.0. Changes: only the Qwen3-VL embedder is kept (no Hub spec parsing, mock encoder or compile hook);
# it wraps an already-built model, made by ``from_pretrained`` (weights) or ``from_config_dir`` (meta shell).
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

    def __init__(self, model: nn.Module, processor):
        """Wrap a built Qwen3-VL; use :meth:`from_pretrained` / :meth:`from_config_dir`.

        Args:
            model: ``Qwen3VLForConditionalGeneration`` (real or meta).
            processor: its ``AutoProcessor`` (chat template + right-padding tokenizer).
        """
        super().__init__()
        self.model = model
        self.processor = processor
        if self.processor.tokenizer.padding_side != "right":
            raise ValueError("the text encoder requires a right-padding tokenizer")
        self.output_layer = list(OUTPUT_LAYERS)

    @classmethod
    def from_pretrained(cls, model_dir: str, dtype: torch.dtype = torch.bfloat16) -> "Qwen3VLEmbedder":
        """Load the weights and processor of a local Qwen3-VL-4B-Instruct dir.

        Args:
            model_dir: str, e.g. flux-3-action-base/text_encoder.
            dtype: torch.dtype of the weights and of the returned context.

        Returns:
            Qwen3VLEmbedder on CPU.
        """
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, torch_dtype=dtype)
        return cls(model, AutoProcessor.from_pretrained(model_dir))

    @classmethod
    def from_config_dir(cls, model_dir: str, dtype: torch.dtype = torch.bfloat16) -> "Qwen3VLEmbedder":
        """Build the model from ``model_dir/config.json`` with parameters on ``meta``; the processor is real.

        Args:
            model_dir: str, a dir with Qwen3-VL's config.json + processor / tokenizer files (no weights needed).
            dtype: torch.dtype of the (meta) parameters.

        Returns:
            Qwen3VLEmbedder whose parameters are on ``meta``; non-persistent rotary buffers are real.
        """
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

        config = AutoConfig.from_pretrained(model_dir)
        with init_empty_weights(include_buffers=False):
            model = Qwen3VLForConditionalGeneration._from_config(config, dtype=dtype)
        # Tie lm_head to embed_tokens as from_pretrained does (one tensor; the strict load keeps the tie).
        model.tie_weights()
        return cls(model, AutoProcessor.from_pretrained(model_dir))

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
