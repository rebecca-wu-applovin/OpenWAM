"""The shared preprocessing flow of :class:`VideoBackbone` (Task 7), on a CPU fake subclass.

The base class owns pixels -> latents (``encode_frames`` / ``encode_clip``), the condition / train preprocessing
(``preprocess_condition`` / ``preprocess_train``) and ``decode_video``; a family only implements
``encode_pixels`` / ``decode_latents`` / ``encode_text``. ``PromptCache`` is the text cache ``build_text_cache``
swaps into a family's ``text_lookup`` slot.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from openwam.model.video_backbone.base import TextCondition, VideoBackbone
from openwam.model.video_backbone.common.text_cache import PromptCache

TC = 4  # temporal compression of the fake VAE
SC = 8  # spatial compression of the fake VAE
Z = 3  # latent channels


class _FakeBackbone(VideoBackbone):
    """Minimal family: a fake causal VAE (keeps frames 0, TC, 2TC, ... and every SC-th pixel, plus the call's T so
    a frame encoded alone differs from the same frame inside a clip) and a fake text encoder."""

    def __init__(self):
        super().__init__()
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self._temporal_compression = TC
        self.calls_T: list[int] = []  # T of every encode_pixels call
        self.calls_N: list[int] = []  # batch N of every encode_pixels call
        self.text_calls: list[dict] = []  # kwargs of every encode_text call

    # --- the three family hooks ---
    def encode_pixels(self, pixels):
        """pixels: (N, 3, T, H, W). Returns: (N, 3, 1 + (T-1)//TC, H/SC, W/SC)."""
        self.calls_T.append(int(pixels.shape[2]))
        self.calls_N.append(int(pixels.shape[0]))
        return pixels[:, :, ::TC, ::SC, ::SC] + float(pixels.shape[2])

    def decode_latents(self, latents):
        """latents: (N, z, T_lat, h, w). Returns: (N, 3, 1 + TC*(T_lat-1), h*SC, w*SC) in [-1, 1]."""
        n, _, t, h, w = latents.shape
        return torch.zeros(n, 3, 1 + TC * (t - 1), h * SC, w * SC)

    def encode_text(self, prompts, *, grid, num_frames):
        """prompts: B str. grid: (T_lat, h, w). num_frames: int. Returns: TextCondition (B, 2, 4)."""
        self.text_calls.append({"prompts": list(prompts), "grid": tuple(grid), "num_frames": int(num_frames)})
        b = len(prompts)
        return TextCondition(context=torch.zeros(b, 2, 4), context_mask=torch.ones(b, 2, dtype=torch.bool), extra={})

    # --- the rest of the ABC (unused here) ---
    @classmethod
    def from_pretrained(cls, cfg):
        return cls()

    @classmethod
    def from_checkpoint_dir(cls, cfg, ckpt_dir):
        return cls()

    dim = num_layers = num_heads = head_dim = 1
    scheduler = None

    def prepare(self, **kw):
        raise NotImplementedError

    def run_block(self, block_id, state):
        raise NotImplementedError

    def finalize(self, state):
        raise NotImplementedError


@pytest.fixture
def fake():
    return _FakeBackbone()


def _clips(b: int, t: int, size=(32, 32)):
    """b clips of t PIL frames with distinct colors. Returns: list[list[PIL.Image]]."""
    return [
        [Image.new("RGB", size, ((40 * i + 7 * j) % 256, 10 * j % 256, 3 * i % 256)) for j in range(t)]
        for i in range(b)
    ]


def test_encode_frames_encodes_each_frame_alone(fake):
    lat = fake.encode_frames(torch.zeros(2, 3, 5, 32, 32))
    # One call with all B*H frames, each a 1-frame clip.
    assert fake.calls_T == [1] and fake.calls_N == [10]
    assert lat.shape == (2, Z, 5, 4, 4)


def test_encode_frames_keeps_sample_and_frame_order(fake):
    pixels = torch.randn(2, 3, 3, 16, 16)
    lat = fake.encode_frames(pixels)
    # Latent t of sample b is frame t of sample b encoded alone.
    for b in range(2):
        for t in range(3):
            assert torch.equal(lat[b, :, t], pixels[b, :, t, ::SC, ::SC] + 1.0)


def test_preprocess_train_sparse_concat(fake):
    clean, latents, text = fake.preprocess_train(_clips(2, 10), ["a", "b"], num_hist_frames=5)
    # History: every frame alone; future: one causal clip of F = 5 frames -> 2 latents.
    assert fake.calls_T == [1, 5]
    assert clean.shape == (2, Z, 5, 4, 4) and latents.shape == (2, Z, 7, 4, 4)
    assert torch.equal(latents[:, :, :5], clean)
    # Text sees the whole latent grid and the equivalent contiguous clip length.
    assert fake.text_calls == [{"prompts": ["a", "b"], "grid": (7, 4, 4), "num_frames": 1 + TC * 6}]
    assert isinstance(text, TextCondition)


def test_preprocess_train_first_frame(fake):
    clips = _clips(2, 9)
    clean, latents, _ = fake.preprocess_train(clips, ["a", "b"], num_hist_frames=0)
    # One whole-clip encode; the clean prefix is its latent 0 (not a separate 1-frame encode).
    assert fake.calls_T == [9]
    assert torch.equal(latents, fake.encode_clip(fake.to_pixels(clips)))
    assert torch.equal(clean, latents[:, :, :1])
    assert fake.text_calls[0]["grid"] == (3, 4, 4) and fake.text_calls[0]["num_frames"] == 9


def test_preprocess_condition(fake):
    hist, text = fake.preprocess_condition(_clips(1, 5), "a", num_frames=25, num_latents=7)
    assert fake.calls_T == [1] and hist.shape == (1, Z, 5, 4, 4)
    # One str prompt is expanded to the batch at the entry.
    assert fake.text_calls == [{"prompts": ["a"], "grid": (7, 4, 4), "num_frames": 25}]
    assert text.context.shape[0] == 1


def test_prompt_count_must_match_batch(fake):
    with pytest.raises(ValueError):
        fake.preprocess_train(_clips(2, 9), ["only one"], num_hist_frames=0)


def test_decode_video_rejects_batch(fake):
    with pytest.raises(ValueError):
        fake.decode_video(torch.zeros(2, Z, 3, 4, 4))
    frames = fake.decode_video(torch.zeros(1, Z, 3, 4, 4))
    assert len(frames) == 9 and frames[0].size == (32, 32)


def test_num_future_latents_rejects(fake):
    with pytest.raises(ValueError):
        fake.num_future_latents(6)
    assert fake.num_future_latents(9) == 3


def _fake_text_encoder(prompts):
    """prompts: list[str]. Returns: per-prompt (context (1, 3, 2) from the prompt's chars, length (1,)), batched."""
    ctx = torch.stack([torch.tensor([[float(ord(c)) for c in (p + "___")[:3]]] * 2).T for p in prompts])
    lens = torch.tensor([len(p) for p in prompts])
    return [(ctx[j : j + 1], lens[j : j + 1]) for j in range(len(prompts))]


def test_prompt_cache_matches_uncached():
    prompts = ["pick", "place the cup", ""]
    cache = PromptCache(_fake_text_encoder, prompts, batch_size=1)
    for got, want in zip(cache(prompts[::-1]), _fake_text_encoder(prompts[::-1])):
        assert all(torch.equal(g, w) for g, w in zip(got, want))
    assert set(cache.entries) == set(prompts)
    with pytest.raises(KeyError):
        cache(["never seen"])
