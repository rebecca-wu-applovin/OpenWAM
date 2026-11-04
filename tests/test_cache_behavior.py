"""Wan's text slot: ``encode_text`` reads ``text_lookup``, the online umT5 or (after ``build_text_cache``) a PromptCache.

A mock tokenizer / text encoder (no weights) records each encoder call, so the tests can check:

1. Online: every ``encode_text`` runs the encoder.
2. After ``build_text_cache``: no encoder call, same context / mask as online, encoder parked on CPU.
3. A prompt outside the cached set raises KeyError.
"""

from __future__ import annotations

import torch
from torch import nn

from openwam.model.video_backbone.wan.backbone import WanBackbone


class _MockTokenizer:
    """Token count = len(prompt) (capped at 4), so seq_lens differ per prompt."""

    def __call__(self, prompts, **kw):
        n = len(prompts)
        mask = torch.zeros(n, 4, dtype=torch.long)
        for i, p in enumerate(prompts):
            mask[i, : min(len(p), 4)] = 1
        return torch.arange(4).repeat(n, 1), mask


class _MockTextEncoder(nn.Module):
    """Records each call; the context depends on the token ids and mask."""

    def __init__(self):
        super().__init__()
        self.calls: list[int] = []
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, ids, mask):
        self.calls.append(ids.shape[0])
        return (ids.float() + mask.float()).unsqueeze(-1).expand(-1, -1, 8).clone()


def _wan(text_encoder):
    """A WanBackbone around the mock text side only (no DiT / VAE), on CPU."""
    vb = WanBackbone(
        dit=None,
        vae=None,
        text_encoder=text_encoder,
        tokenizer=_MockTokenizer(),
        tokenizer_dir="",
        shift_video=5.0,
        text_max_length=4,
    )
    vb._device = torch.device("cpu")
    return vb


def _encode(vb, prompts):
    return vb.encode_text(prompts, grid=(1, 1, 1), num_frames=1)


def test_online_encodes_every_call():
    enc = _MockTextEncoder()
    vb = _wan(enc)
    text = _encode(vb, ["ab", "abcdef"])
    _encode(vb, ["ab"])
    assert enc.calls == [2, 1]
    # Mask = position < token count; context zeroed past it.
    assert text.context_mask.tolist() == [[True, True, False, False], [True, True, True, True]]
    assert text.context[0, 2:].abs().sum() == 0 and text.extra == {}


def test_cache_matches_online_and_skips_the_encoder():
    prompts = ["ab", "abcdef", "abc"]
    online = _encode(_wan(_MockTextEncoder()), prompts)
    enc = _MockTextEncoder()
    vb = _wan(enc)
    vb.build_text_cache(prompts)
    n_build = len(enc.calls)
    cached = _encode(vb, prompts[::-1])
    assert len(enc.calls) == n_build, "cached encode_text must not run the encoder"
    assert torch.equal(cached.context, online.context.flip(0))
    assert torch.equal(cached.context_mask, online.context_mask.flip(0))
    assert enc.weight.device.type == "cpu" and "text_encoder.weight" in vb.state_dict()


def test_cache_answers_the_empty_prompt():
    # A caption-dropped sample asks for "": the cache always holds it.
    online = _encode(_wan(_MockTextEncoder()), [""])
    vb = _wan(_MockTextEncoder())
    vb.build_text_cache(["ab"])
    cached = _encode(vb, [""])
    assert torch.equal(cached.context, online.context) and torch.equal(cached.context_mask, online.context_mask)


def test_cache_miss_raises():
    vb = _wan(_MockTextEncoder())
    vb.build_text_cache(["ab"])
    try:
        _encode(vb, ["never seen"])
    except KeyError:
        return
    raise AssertionError("an uncached prompt must raise KeyError")
