"""Stand-ins for the Cosmos3EdgeVideoBackbone constructor arguments a tiny CPU test shell never exercises.

The real constructor takes the VAE, tokenizer, tokenizer dir and latent normalization as required arguments
(production always builds them from the bundle / checkpoint). Block-loop tests run on random tiny nets with no
pixel IO and no prompt encoding, so they pass these stand-ins instead.
"""

from __future__ import annotations

import torch
from torch import nn


class _NoTokenizer:
    """Tokenizer stand-in for shells that never encode a prompt: any attribute read fails loudly."""

    def __getattr__(self, name):
        """name: str attribute. Returns: never; raises AttributeError naming the stand-in."""
        raise AttributeError(f"test shell has no tokenizer (read {name!r})")


def unused_io_kwargs() -> dict:
    """Constructor kwargs for the IO a tiny shell does not exercise.

    Returns:
        dict with ``vae`` (parameterless nn.Module), ``tokenizer`` (_NoTokenizer), ``tokenizer_dir`` (str),
        ``latents_mean`` (48,) zeros and ``latents_std`` (48,) ones (identity normalization).
    """
    return {
        "vae": nn.Module(),
        "tokenizer": _NoTokenizer(),
        "tokenizer_dir": "<no tokenizer dir>",
        "latents_mean": torch.zeros(48),
        "latents_std": torch.ones(48),
    }
