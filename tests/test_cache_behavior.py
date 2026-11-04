"""Unit tests for prompt_embed_cache in WanBase.preprocess_input_for_inference.

The deploy method builds every conditioning signal with explicit helpers (no
WanVideoPipeline unit-runner). The only cached quantity is the text embedding
``(context, seq_lens)``, reused via ``prompt_embed_cache``. These tests spy on
the text encoder to verify:

1. Cold start: no cache → text is encoded.
2. Seen prompt: text encode skipped.
3. Unseen prompt: cache miss → text encoded.
4. prompt_A -> prompt_B -> prompt_A: prompt_A embed still cached.
5. No cache: runs cleanly every call.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from openwam.model.video_backbone.wan.backbone import WanBase

# ---------------------------------------------------------------------------
# Minimal mock infrastructure
# ---------------------------------------------------------------------------


class _MockScheduler:
    def set_timesteps(self, num_inference_steps, shift):
        pass


class _MockPipe:
    def __init__(self):
        self.scheduler = _MockScheduler()


class _MockTokenizer:
    def __call__(self, prompts, **kw):
        n = len(prompts)
        return torch.zeros(n, 4, dtype=torch.long), torch.ones(n, 4, dtype=torch.long)


class _MockTextEncoder:
    """Records each call so a test can count how often text was encoded."""

    def __init__(self, calls):
        self._calls = calls

    def __call__(self, ids, mask):
        self._calls.append(list(range(ids.shape[0])))
        return torch.zeros(ids.shape[0], 4, 8)


class _MockWanVB:
    """WanVideoBackbone-like object exercising the deploy cache seam.

    The encode seams (text / noise) are stubbed so the test needs
    no real weights or GPU; ``_encode_text`` records its calls so a test can
    assert whether the text path was hit or served from a cache.
    """

    def __init__(self, pipe=None):
        self._pipe = pipe if pipe is not None else _MockPipe()
        self._device = "cpu"
        self._dtype = torch.float32
        self._encode_text_calls: list[list] = []
        # Encode + conditioning seams now live in wan.encode / wan.conditioning
        # free functions; provide the state they read explicitly.
        self._tokenizer = _MockTokenizer()
        self.text_encoder = _MockTextEncoder(self._encode_text_calls)
        self._text_cache = None  # training text cache off (WanVideoBackbone.__init__ default): encode online
        self.text_max_length = 512  # WanVideoBackbone class default (umT5 token budget)
        self._height_division_factor = 16
        self._width_division_factor = 16
        self._time_division_factor = 4
        self._time_division_remainder = 1
        # build_deploy_noise reads z_dim / upsampling_factor off the native VAE.
        self.vae = SimpleNamespace(model=SimpleNamespace(z_dim=4), upsampling_factor=8)

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def scheduler(self):
        return self._pipe.scheduler

    # --- real method under test (conditioning seams now in wan.conditioning) ---
    def preprocess_input_for_inference(self, **kw):
        return WanBase.preprocess_input_for_inference(self, **kw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prep(vb, prompt, prompt_embed_cache=None, seed=0):
    return vb.preprocess_input_for_inference(
        prompt=prompt,
        first_frame_image=None,
        num_frames=17,
        height=32,
        width=32,
        seed=seed,
        tiled=False,
        num_inference_steps=2,
        shift=5.0,
        prompt_embed_cache=prompt_embed_cache,
    )


def _text_calls(vb) -> int:
    return len(vb._encode_text_calls)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cold_start_encodes_text():
    """No cache provided — text is encoded."""
    vb = _MockWanVB()
    _prep(vb, "prompt_A")
    assert _text_calls(vb) == 1


def test_prompt_embed_cache_hit_skips_text_encode():
    """After first call with prompt_A, a second call skips text encode via prompt_embed_cache."""
    vb = _MockWanVB()
    prompt_embed_cache: dict = {}

    _prep(vb, "prompt_A", prompt_embed_cache=prompt_embed_cache)
    assert "prompt_A" in prompt_embed_cache
    assert _text_calls(vb) == 1

    _prep(vb, "prompt_A", prompt_embed_cache=prompt_embed_cache)
    assert _text_calls(vb) == 1, "text should be served from prompt_embed_cache"


def test_prompt_embed_cache_miss_on_new_prompt():
    """First time seeing prompt_B: cache miss → text encoded."""
    vb = _MockWanVB()
    prompt_embed_cache: dict = {}

    _prep(vb, "prompt_A", prompt_embed_cache=prompt_embed_cache)
    assert "prompt_A" in prompt_embed_cache

    _prep(vb, "prompt_B", prompt_embed_cache=prompt_embed_cache)
    assert _text_calls(vb) == 2, "expected text encode for an unseen prompt"
    assert "prompt_B" in prompt_embed_cache


def test_prompt_embed_cache_survives_prompt_change():
    """After prompt_A -> prompt_B -> prompt_A: prompt_A embed is still cached."""
    vb = _MockWanVB()
    prompt_embed_cache: dict = {}

    _prep(vb, "prompt_A", prompt_embed_cache=prompt_embed_cache)
    _prep(vb, "prompt_B", prompt_embed_cache=prompt_embed_cache)
    calls_before = _text_calls(vb)

    _prep(vb, "prompt_A", prompt_embed_cache=prompt_embed_cache)
    assert _text_calls(vb) == calls_before, "prompt_A embed should still be cached"


def test_both_caches_none_does_not_crash():
    """With no caches, the function encodes text every call."""
    vb = _MockWanVB()
    for i in range(3):
        _prep(vb, f"prompt_{i}")
    assert _text_calls(vb) == 3
