"""Prompt -> text-encoder output cache, built once from a fixed prompt set (training with a frozen text encoder).

A family's ``encode_text`` reads its text through one slot, ``self.text_lookup(prompts)``: by default the online
text encoder, after ``build_text_cache`` a :class:`PromptCache` built from that same encoder. Both return the same
thing (one entry per prompt), so ``encode_text`` never checks whether a cache exists.
"""

from __future__ import annotations

from typing import Any, Callable


class PromptCache:
    """Per-prompt encoder outputs, looked up by prompt.

    Args:
        encode: callable list[str] -> list of per-prompt entries (whatever the family's text encoder yields for
            one prompt, e.g. a tuple ``(context (1, L, D), seq_lens (1,))``), in input order.
        prompts: list[str], every prompt the cache must answer, encoded in this order (the family decides
            order and dedup; a repeated prompt keeps its last encoding).
        batch_size: int, prompts per ``encode`` call while building.
    """

    def __init__(self, encode: Callable[[list], list], prompts: list, batch_size: int):
        self.entries: dict[str, Any] = {}
        prompts = list(prompts)
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]
            # Each entry stays on the device the encoder produced it on.
            for prompt, entry in zip(chunk, encode(chunk)):
                self.entries[prompt] = entry

    def __call__(self, prompts: list) -> list:
        """Look the prompts up.

        Args:
            prompts: list[str] of B prompts.

        Returns:
            list of the B cached entries, in input order. Raises KeyError on a prompt the cache was not built
            with (the text encoder is parked on CPU, so a miss means the prompt set was incomplete).
        """
        missing = [p for p in prompts if p not in self.entries]
        if missing:
            raise KeyError(f"text cache has no entry for prompts {missing[:3]} (cache size {len(self.entries)})")
        return [self.entries[p] for p in prompts]


__all__ = ["PromptCache"]
