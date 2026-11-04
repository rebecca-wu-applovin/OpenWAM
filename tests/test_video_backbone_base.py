"""The shared preprocessing flow of :class:`VideoBackbone`, the latent layout and the frame-level mask, on CPU.

The base class owns pixels -> latents (``encode_frames`` / ``encode_clip``), the one latent layout of
``preprocess_condition`` / ``preprocess_train`` (past frames per_frame | clip, the current frame alone, the
future = the clip [t, future] minus its latent 0) and ``decode_future``; a family only implements
``encode_pixels`` / ``decode_latents`` / ``encode_text``. ``resolve_config`` derives the layout into
``cfg.common.layout``; ``build_frame_mask`` + ``build_joint_attention_masks`` replace the old mask builders
(copied below as references for the equivalence test). ``PromptCache`` is the text cache ``build_text_cache``
swaps into a family's ``text_lookup`` slot.
"""

from __future__ import annotations

from typing import List, Optional

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import Tensor, nn

from openwam.config import latent_layout, resolve_config
from openwam.model.architectures.utils.mask_modes import build_frame_mask
from openwam.model.video_backbone.base import BlockLoopState, TextCondition, VideoBackbone
from openwam.model.video_backbone.common.text_cache import PromptCache

TC = 4  # temporal compression of the fake VAE
SC = 8  # spatial compression of the fake VAE
Z = 3  # latent channels
FUT8 = [4, 8, 12, 16, 20, 24, 28, 32]  # 8 future frames, every 4th row


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
        self.decoded: list[Tensor] = []  # latents of every decode_latents call
        self.text_calls: list[dict] = []  # kwargs of every encode_text call

    # --- the three family hooks ---
    def encode_pixels(self, pixels):
        """pixels: (N, 3, T, H, W). Returns: (N, 3, 1 + (T-1)//TC, H/SC, W/SC)."""
        self.calls_T.append(int(pixels.shape[2]))
        self.calls_N.append(int(pixels.shape[0]))
        return pixels[:, :, ::TC, ::SC, ::SC] + float(pixels.shape[2])

    def decode_latents(self, latents):
        """latents: (N, z, T_lat, h, w). Returns: (N, 3, 1 + TC*(T_lat-1), h*SC, w*SC) in [-1, 1]."""
        self.decoded.append(latents)
        n, _, t, h, w = latents.shape
        return torch.zeros(n, 3, 1 + TC * (t - 1), h * SC, w * SC)

    def encode_text(self, prompts, *, grid, num_frames):
        """prompts: B str. grid: (T_lat, h, w). num_frames: int. Returns: TextCondition (B, 2, 4)."""
        self.text_calls.append({"prompts": list(prompts), "grid": tuple(grid), "num_frames": int(num_frames)})
        b = len(prompts)
        return TextCondition(context=torch.zeros(b, 2, 4), context_mask=torch.ones(b, 2, dtype=torch.bool), extra={})

    # --- the rest of the ABC (unused here) ---
    @classmethod
    def from_pretrained(cls, cfg, assets):
        return cls()

    @classmethod
    def from_checkpoint_dir(cls, cfg, ckpt_dir):
        return cls()

    dim = num_heads = head_dim = 1
    num_layers = 3
    text_dim = 4
    scheduler = None

    def prepare(self, **kw):
        raise NotImplementedError

    def run_block(self, block_id, state, attn_mask):
        raise NotImplementedError

    def finalize(self, state):
        raise NotImplementedError


@pytest.fixture
def fake():
    return _FakeBackbone()


def _layout(h: int, f: int = 8, encoding: str = "per_frame", tc: int = TC):
    """Layout of H history frames (ending at 0) and F future frames. Returns: DictConfig like cfg.common.layout."""
    return OmegaConf.create(latent_layout(list(range(-h + 1, 1)), list(range(1, f + 1)), 33, encoding, tc))


def _clips(b: int, t: int, size=(32, 32)):
    """b clips of t PIL frames with distinct colors. Returns: list[list[PIL.Image]]."""
    return [
        [Image.new("RGB", size, ((40 * i + 7 * j) % 256, 10 * j % 256, 3 * i % 256)) for j in range(t)]
        for i in range(b)
    ]


# ---------------------------------------------------------------- encode primitives


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


def test_encode_clip_rejects(fake):
    with pytest.raises(ValueError):
        fake.encode_clip(torch.zeros(1, 3, 4, 16, 16))  # tc = 4: 4 frames is not 1 + 4k
    assert fake.calls_T == []  # rejected before the VAE runs


# ---------------------------------------------------------------- the one latent layout


def test_layout_per_frame(fake):
    latents, text = fake.preprocess_train(_clips(2, 13), ["a", "b"], _layout(5))
    # Past frames alone (B*4), the current frame alone (B), the clip [t, 8 future] (9 frames).
    assert fake.calls_T == [1, 1, 9] and fake.calls_N == [8, 2, 2]
    assert latents.shape == (2, Z, 7, 4, 4)  # 4 past + 1 current + 2 future
    assert fake.text_calls == [{"prompts": ["a", "b"], "grid": (7, 4, 4), "num_frames": 1 + TC * 6}]
    assert isinstance(text, TextCondition)


def test_layout_clip_history(fake):
    layout = _layout(6, encoding="clip")  # P = 5 past frames = 1 + 4
    assert (layout.num_past_latents, layout.num_hist_latents, layout.num_latents) == (2, 3, 5)
    latents, _ = fake.preprocess_train(_clips(1, 14), "a", layout)
    # The past frames as one clip (T = 5), then the current frame alone, then the future clip.
    assert fake.calls_T == [5, 1, 9]
    assert latents.shape == (1, Z, 5, 4, 4)


def test_h1_is_first_frame(fake):
    clips = _clips(2, 9)
    latents, _ = fake.preprocess_train(clips, ["a", "b"], _layout(1))
    assert latents.shape[2] == 3  # 9 frames -> 3 latents, the original first-frame layout
    # No past frames: the condition is the current frame encoded alone.
    assert fake.calls_T == [1, 9]
    pixels = fake.to_pixels(clips)
    assert torch.equal(latents[:, :, :1], fake.encode_frames(pixels[:, :, :1]))


def test_target_excludes_current_frame(fake):
    clips = _clips(1, 13)
    latents, _ = fake.preprocess_train(clips, "a", _layout(5))
    pixels = fake.to_pixels(clips)
    # The future latents are the clip [t, future] encoded as one, minus its latent 0 (= t).
    assert torch.equal(latents[:, :, 5:], fake.encode_clip(pixels[:, :, 4:].contiguous())[:, :, 1:])


def test_any_nhist(fake):
    for h in range(1, 6):
        layout = _layout(h)
        latents, _ = fake.preprocess_train(_clips(1, h + 8), "a", layout)
        hist, _ = fake.preprocess_condition(_clips(1, h), "a", layout)
        assert layout.num_hist_latents == h and hist.shape[2] == h
        assert latents.shape[2] == h + 8 // TC
        # The deploy condition is the training prefix (same code, same frames).
        assert torch.equal(hist, latents[:, :, :h])


def test_preprocess_condition_text_grid(fake):
    hist, text = fake.preprocess_condition(_clips(1, 5), "a", _layout(5))
    assert hist.shape == (1, Z, 5, 4, 4)
    # One str prompt is expanded to the batch; the text sees the whole clip's grid.
    assert fake.text_calls == [{"prompts": ["a"], "grid": (7, 4, 4), "num_frames": 1 + TC * 6}]
    assert text.context.shape[0] == 1


def test_prompt_count_must_match_batch(fake):
    with pytest.raises(ValueError):
        fake.preprocess_train(_clips(2, 9), ["only one"], _layout(1))


def test_clip_frame_count_must_match_layout(fake):
    with pytest.raises(ValueError):
        fake.preprocess_train(_clips(1, 10), "a", _layout(1))


def test_decode_future_drops_current(fake):
    layout = _layout(5)
    latents = torch.randn(1, Z, 7, 4, 4)
    frames = fake.decode_future(latents, layout)
    # Decoded: [current, future] (latents 4..6) -> 9 frames; frame 0 (t) dropped.
    assert torch.equal(fake.decoded[0], latents[:, :, 4:])
    assert len(frames) == 8 and frames[0].size == (32, 32)
    with pytest.raises(ValueError):
        fake.decode_future(torch.zeros(2, Z, 7, 4, 4), layout)  # one clip only
    with pytest.raises(ValueError):
        fake.decode_future(torch.zeros(1, Z, 6, 4, 4), layout)  # latent count != layout


# ---------------------------------------------------------------- resolve_config


def _cfg(h=5, f=8, encoding="per_frame", tc=4, clean="bidirectional", va="mutual", **arch_extra):
    """A run config with the keys resolve_config reads. Returns: DictConfig."""
    return OmegaConf.create(
        {
            "project": {"seed": 0},
            "common": {"action_dim": 48, "state_dim": 48},
            "dataloader": {
                "history_offsets": list(range(-h + 1, 1)),
                "future_offsets": list(range(1, f + 1)),
                "num_frames": 33,
                "history_sampling": "fixed",
                "history_window": 100,
                "caption_dropout_p": 0.0,
            },
            "model": {
                "architecture": {
                    "history_encoding": encoding,
                    "clean_attention": clean,
                    "attention_mask_mode": va,
                    "proprio_input": "context_token",
                    **arch_extra,
                },
                "video_backbone": {
                    "name": "fake",
                    "temporal_compression": tc,
                    "required_clean_attention": None,
                    "required_proprio_input": "context_token",
                },
            },
        }
    )


def test_resolve_config_rejects_null_seed():
    # Every run RNG (dataset per-sample, per-step, vis sample) derives from project.seed: null is not a mode.
    cfg = _cfg()
    cfg.project.seed = None
    with pytest.raises(ValueError, match="project.seed"):
        resolve_config(cfg)


def test_resolve_config_rejects_bool_seed():
    # bool is an int subclass in Python; a yaml `true` is a typo, not a seed.
    cfg = _cfg()
    cfg.project.seed = True
    with pytest.raises(ValueError, match="project.seed"):
        resolve_config(cfg)


def test_resolve_config_layout():
    layout = resolve_config(_cfg()).common.layout
    assert (layout.num_hist_latents, layout.num_future_latents, layout.num_latents) == (5, 2, 7)
    assert (layout.action_horizon, layout.num_proprio_tokens) == (32, 5)
    assert resolve_config(_cfg(h=6, encoding="clip")).common.layout.num_hist_latents == 3
    assert resolve_config(_cfg(tc=8)).common.layout.num_future_latents == 1  # GE
    # The layout block is read-only.
    with pytest.raises(Exception):
        layout.num_latents = 9
    with pytest.raises(ValueError):
        resolve_config(_cfg(f=5))  # F not a multiple of tc
    with pytest.raises(ValueError):
        resolve_config(_cfg(h=5, encoding="clip"))  # P = 4 past frames is not 1 + 4k
    # A saved config re-resolves to the same layout; a different stored layout raises.
    saved = resolve_config(_cfg())
    assert resolve_config(OmegaConf.create(OmegaConf.to_container(saved))).common.layout == saved.common.layout
    stale = OmegaConf.to_container(saved)
    stale["common"]["layout"]["num_latents"] = 9
    with pytest.raises(ValueError):
        resolve_config(OmegaConf.create(stale))
    # The hand-written keys the layout replaced are rejected.
    for key in ("num_history_frames", "num_proprio_tokens"):
        with pytest.raises(ValueError):
            resolve_config(_cfg(**{key: 5}))
    # A backbone's required clean attention (LongLive: causal) must match the architecture's.
    cfg = _cfg(clean="bidirectional")
    cfg.model.video_backbone.required_clean_attention = "causal"
    with pytest.raises(ValueError):
        resolve_config(cfg)
    cfg.model.architecture.clean_attention = "causal"
    resolve_config(cfg)


def test_resolve_config_freezes_whole_cfg():
    # The run config is read-only after resolve_config: a write anywhere (not only in common.layout) raises.
    from omegaconf.errors import ReadonlyConfigError

    cfg = resolve_config(_cfg())
    with pytest.raises(ReadonlyConfigError):
        cfg.dataloader.num_frames = 17


@pytest.mark.parametrize(
    "name, key",
    [
        ("fake", "caption_dropout_p"),
        ("ge_base_fast", "seed"),
        ("fake", "text_dropout_p"),
        ("fake", "text_encoder_dropout"),
        ("fake", "text_encoder_dropout_seed"),
    ],
)
def test_backbone_caption_dropout_keys_rejected(name, key):
    """The old per-backbone caption-dropout keys (GE / flux / cosmos3) are stale: dataloader.caption_dropout_p."""
    cfg = _cfg()
    cfg.model.video_backbone.name = name
    cfg.model.video_backbone[key] = 0.1
    with pytest.raises(ValueError, match="caption-dropout"):
        resolve_config(cfg)


def test_seed_key_is_stale_for_ge_only():
    """``seed`` was GE's caption-dropout seed; another backbone may own a ``seed`` key."""
    cfg = _cfg()
    cfg.model.video_backbone.seed = 0
    resolve_config(cfg)


def test_history_window_must_cover_offsets():
    # Random history must be able to draw the fixed (val / deploy) layout: window >= the oldest offset.
    cfg = _cfg()
    cfg.dataloader.history_offsets = [-80, -60, -40, -20, 0]
    cfg.dataloader.history_sampling, cfg.dataloader.history_window = "random", 50
    with pytest.raises(ValueError, match="history_window"):
        resolve_config(cfg)
    cfg.dataloader.history_window = 80
    resolve_config(cfg)
    # H - 1 distinct past rows must fit in the window too (offsets -4..0 need 4 rows).
    cfg = _cfg(h=5)
    cfg.dataloader.history_sampling, cfg.dataloader.history_window = "random", 3
    with pytest.raises(ValueError, match="history_window"):
        resolve_config(cfg)
    # Fixed sampling never reads the window: the fixed offsets are the layout.
    cfg.dataloader.history_sampling = "fixed"
    resolve_config(cfg)
    # Unknown mode, or a missing key (OmegaConf names its full path), is rejected.
    cfg = _cfg()
    cfg.dataloader.history_sampling = "uniform"
    with pytest.raises(ValueError, match="history_sampling"):
        resolve_config(cfg)
    for key in ("history_sampling", "history_window"):
        cfg = _cfg()
        cfg.dataloader.pop(key)
        with pytest.raises(Exception, match=f"dataloader.{key}"):
            resolve_config(cfg)


def test_caption_dropout_p_checked():
    # Required and a probability; resolve_config is the one place it is checked (the dataset reads it as is).
    cfg = _cfg()
    cfg.dataloader.pop("caption_dropout_p")
    with pytest.raises(Exception, match="dataloader.caption_dropout_p"):
        resolve_config(cfg)
    for bad in (-0.1, 1.5):
        cfg = _cfg()
        cfg.dataloader.caption_dropout_p = bad
        with pytest.raises(ValueError, match="caption_dropout_p"):
            resolve_config(cfg)
    cfg = _cfg()
    cfg.dataloader.caption_dropout_p = 1.0
    resolve_config(cfg)


def test_unverified_va_mode_raises():
    for mode in ("action_sees_video", "video_sees_action", "isolated"):
        with pytest.raises(NotImplementedError):
            resolve_config(_cfg(va=mode))
        with pytest.raises(NotImplementedError):
            build_frame_mask(_layout(5), "bidirectional", mode)
    with pytest.raises(ValueError):
        resolve_config(_cfg(va="sideways"))
    with pytest.raises(ValueError):
        resolve_config(_cfg(clean="first_frame_causal"))


# ---------------------------------------------------------------- frame mask


def test_frame_mask_clip_prefix():
    mask = build_frame_mask(_layout(6, encoding="clip"), "bidirectional", "mutual")
    assert mask.shape == (6, 6)  # 3 clean + 2 noisy latents + the action block
    # The clean prefix is the 3 history latents, not the 6 history frames.
    assert mask[:3, :3].all() and not mask[:3, 3:].any()
    assert mask[3:].all()


def _state(t: int, tpf: int, *, b: int = 1, prefix: int = 0, prefix_mask=None, **extras) -> BlockLoopState:
    """Video-stream state of T latent frames x tpf tokens. Returns: BlockLoopState (only shapes are real)."""
    return BlockLoopState(
        hidden_states=torch.zeros(b, t * tpf, 2),
        time_mod=torch.zeros(()),
        rope_freqs=torch.zeros(()),
        context=torch.zeros(()),
        grid_frames=t,
        grid_height=1,
        grid_width=tpf,
        prefix_kv_len=prefix,
        prefix_kv_mask=prefix_mask,
        extras=dict(extras),
    )


def test_latent_count_mismatch_raises(fake):
    frame_mask = build_frame_mask(_layout(5), "bidirectional", "mutual")  # 7 latents
    with pytest.raises(ValueError):
        fake.build_joint_attention_masks(frame_mask, _state(6, 4), 3)
    masks = fake.build_joint_attention_masks(frame_mask, _state(7, 4), 3)
    assert len(masks) == fake.num_layers and masks[0].shape == (31, 31)


# --- reference: the mask builders this task deleted (verbatim copies of the pre-Task-8 code) ---------------


def _old_build_history_video_mask(*, video_seq_len, video_tokens_per_frame, num_clean_frames, mode, device):
    mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
    if mode == "bidirectional":
        return mask
    n_hist = min(num_clean_frames * video_tokens_per_frame, video_seq_len)
    mask[:n_hist, n_hist:] = False
    if mode == "history_per_frame_causal":
        frame_causal = torch.tril(torch.ones((num_clean_frames, num_clean_frames), dtype=torch.bool, device=device))
        mask[:n_hist, :n_hist] = frame_causal.repeat_interleave(video_tokens_per_frame, 0).repeat_interleave(
            video_tokens_per_frame, 1
        )
    return mask


def _old_fill_cross_modal_va_blocks(
    mask, *, v_start, v_end, a_start, a_end, mode, video_tokens_per_frame, num_clean_frames=1
):
    s_video = v_end - v_start
    ff = min(video_tokens_per_frame * max(int(num_clean_frames), 1), s_video)
    if mode in ("action_sees_video", "mutual"):
        mask[a_start:a_end, v_start:v_end] = True
    else:
        mask[a_start:a_end, v_start:v_end] = False
        mask[a_start:a_end, v_start : v_start + ff] = True
    if mode in ("mutual", "video_sees_action"):
        mask[v_start:v_end, a_start:a_end] = True
        mask[v_start : v_start + ff, a_start:a_end] = False
    else:
        mask[v_start:v_end, a_start:a_end] = False


def _old_wan_build_video_to_video_mask(mode, video_seq_len, video_tokens_per_frame, device):
    """Wan / cosmos3 ``build_video_to_video_mask`` (bidirectional | first_frame_causal branches)."""
    if mode == "bidirectional":
        return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
    video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
    first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
    video_mask[:first_frame_tokens, first_frame_tokens:] = False
    return video_mask


class _OldModeShim:
    """The old v<->v mode carrier: a backbone's ``video_attention_mask_mode`` + ``build_video_to_video_mask``."""

    def __init__(self, mode):
        self.video_attention_mask_mode = mode

    def build_video_to_video_mask(self, *, video_seq_len, video_tokens_per_frame, device):
        return _old_wan_build_video_to_video_mask(
            self.video_attention_mask_mode, video_seq_len, video_tokens_per_frame, device
        )


def _old_build_cross_modal_attention_mask(
    video_backbone, *, s_video, s_action, video_tokens_per_frame, mode, device, num_clean_frames=1
):
    total = s_video + s_action
    mask = torch.zeros((total, total), dtype=torch.bool, device=device)
    if num_clean_frames > 1:
        mask[:s_video, :s_video] = _old_build_history_video_mask(
            video_seq_len=s_video,
            video_tokens_per_frame=video_tokens_per_frame,
            num_clean_frames=num_clean_frames,
            mode=video_backbone.video_attention_mask_mode,
            device=device,
        )
    else:
        mask[:s_video, :s_video] = video_backbone.build_video_to_video_mask(
            video_seq_len=s_video,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
    a_start, a_end = s_video, s_video + s_action
    mask[a_start:a_end, a_start:a_end] = True
    _old_fill_cross_modal_va_blocks(
        mask,
        v_start=0,
        v_end=s_video,
        a_start=a_start,
        a_end=a_end,
        mode=mode,
        video_tokens_per_frame=video_tokens_per_frame,
        num_clean_frames=num_clean_frames,
    )
    return mask


def _old_widen_mask_for_prefix_kv(mask, state):
    prefix = int(getattr(state, "prefix_kv_len", 0) or 0)
    if prefix <= 0:
        return mask
    rows = mask.shape[-2]
    prefix_mask = getattr(state, "prefix_kv_mask", None)
    if prefix_mask is None:
        pad = torch.ones((*mask.shape[:-1], prefix), dtype=torch.bool, device=mask.device)
        return torch.cat([pad, mask], dim=-1)
    bsz = prefix_mask.shape[0]
    pad = prefix_mask.view(bsz, 1, 1, prefix).expand(bsz, 1, rows, prefix)
    base = mask if mask.dim() == 4 else mask.view(1, 1, rows, mask.shape[-1]).expand(bsz, 1, rows, mask.shape[-1])
    return torch.cat([pad, base], dim=-1)


def _old_view_index(n_frames, n_view, view_h, view_w, device=None):
    per_view = view_h * view_w
    idx = torch.arange(n_frames * n_view * per_view, device=device)
    return (idx // per_view) % n_view


def _old_build_ge_joint_masks(
    *,
    n_frames,
    n_view,
    view_h,
    view_w,
    s_action,
    num_clean_frames,
    attention_mask_mode,
    video_mask_mode,
    num_layers,
    cross_view_every=3,
    device=None,
):
    class _GEShim:  # ge/masks._ModeShim: the history rule with H = 1
        def __init__(self, mode):
            self.video_attention_mask_mode = mode

        def build_video_to_video_mask(self, *, video_seq_len, video_tokens_per_frame, device):
            return _old_build_history_video_mask(
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
                num_clean_frames=1,
                mode=self.video_attention_mask_mode,
                device=device,
            )

    tpf = n_view * view_h * view_w
    s_video = n_frames * tpf
    cross = _old_build_cross_modal_attention_mask(
        _GEShim(video_mask_mode),
        s_video=s_video,
        s_action=s_action,
        video_tokens_per_frame=tpf,
        mode=attention_mask_mode,
        device=torch.device(device) if device is not None else torch.device("cpu"),
        num_clean_frames=num_clean_frames,
    )
    views = _old_view_index(n_frames, n_view, view_h, view_w, device=cross.device)
    per_view = cross.clone()
    per_view[:s_video, :s_video] &= views[:, None] == views[None, :]
    return [cross if layer % cross_view_every == 0 else per_view for layer in range(num_layers)]


_OLD_FLUX_SEGMENTS = ("txt", "video", "video_cond", "action_cond")


def _old_build_flux_layer_masks(
    seg_lens, s_action, text_valid: Optional[Tensor], *, num_mode_layers, num_layers, device
) -> List[Tensor]:
    names = list(_OLD_FLUX_SEGMENTS) + ["action"]
    lens = [seg_lens[n] for n in _OLD_FLUX_SEGMENTS] + [s_action]
    starts = torch.tensor([0] + lens[:-1]).cumsum(0).tolist()
    s = sum(lens)
    sl = {n: slice(a, a + n_) for n, a, n_ in zip(names, starts, lens)}
    mode = torch.zeros((s, s), dtype=torch.bool, device=device)
    for n in names:
        mode[sl[n], sl[n]] = True
    single = torch.ones((s, s), dtype=torch.bool, device=device)
    single[sl["video_cond"], sl["video"]] = False
    single[sl["video_cond"], sl["action"]] = False

    def with_text_padding(m):
        if text_valid is None or bool((text_valid == seg_lens["txt"]).all()):
            return m
        b = text_valid.shape[0]
        key_ok = torch.ones((b, s), dtype=torch.bool, device=device)
        pos = torch.arange(seg_lens["txt"], device=device)
        key_ok[:, sl["txt"]] = pos[None] < text_valid.to(device)[:, None]
        out = m[None] & key_ok[:, None, :]
        out |= torch.eye(s, dtype=torch.bool, device=device)[None]
        return out[:, None]

    mode, single = with_text_padding(mode), with_text_padding(single)
    return [mode] * num_mode_layers + [single] * (num_layers - num_mode_layers)


# (H, old v<->v mode, new clean_attention) of every launched layout: the Wan first-frame (old: the backbone's
# first_frame_causal v<->v block), the sparse arms (first_frame_causal) and LongLive (history_per_frame_causal).
_LAYOUTS = [
    (1, "first_frame_causal", "bidirectional"),
    (5, "first_frame_causal", "bidirectional"),
    (5, "history_per_frame_causal", "causal"),
]


@pytest.mark.parametrize("h, old_mode, clean", _LAYOUTS)
def test_frame_mask_matches_old_mask(fake, h, old_mode, clean):
    from openwam.model.video_backbone.cosmos3.backbone import Cosmos3EdgeVideoBackbone
    from openwam.model.video_backbone.flux.masks import build_flux_layer_masks
    from openwam.model.video_backbone.ge.masks import build_ge_joint_masks

    layout = _layout(h)
    t, tpf, s_action = int(layout.num_latents), 6, 5
    frame_mask = build_frame_mask(layout, clean, "mutual")
    cpu = torch.device("cpu")

    # joint_self_attn (Wan / LongLive): the whole [video, action] mask.
    old = _old_build_cross_modal_attention_mask(
        _OldModeShim(old_mode),
        s_video=t * tpf,
        s_action=s_action,
        video_tokens_per_frame=tpf,
        mode="mutual",
        device=cpu,
        num_clean_frames=h,
    )
    new = fake.build_joint_attention_masks(frame_mask, _state(t, tpf), s_action)
    assert all(torch.equal(m, old) for m in new)

    # joint_cross_attn: the video self-attention block (the old v<->v builder).
    if h == 1:
        old_v = _old_wan_build_video_to_video_mask(old_mode, t * tpf, tpf, cpu)
    else:
        old_v = _old_build_history_video_mask(
            video_seq_len=t * tpf, video_tokens_per_frame=tpf, num_clean_frames=h, mode=old_mode, device=cpu
        )
    assert torch.equal(fake.build_joint_attention_masks(frame_mask, _state(t, tpf), 0)[0], old_v)

    # GE: per-view / cross-view layers (2 views of 2x3 tokens).
    for s_a in (s_action, 0):
        old_ge = _old_build_ge_joint_masks(
            n_frames=t,
            n_view=2,
            view_h=2,
            view_w=3,
            s_action=s_a,
            num_clean_frames=h,
            attention_mask_mode="mutual",
            video_mask_mode=old_mode,
            num_layers=7,
        )
        new_ge = build_ge_joint_masks(frame_mask, n_view=2, view_h=2, view_w=3, s_action=s_a, num_layers=7)
        assert all(torch.equal(a, b) for a, b in zip(old_ge, new_ge)) and len(new_ge) == 7

    # flux: [T | F | H | P | A]; flux never had a per-frame causal history, so only the bidirectional layouts.
    if clean == "bidirectional":
        hw = 4
        seg = {"txt": 5, "video": (t - h) * hw, "video_cond": h * hw, "action_cond": h}
        for text_valid in (None, torch.tensor([5, 3])):
            old_fx = _old_build_flux_layer_masks(seg, s_action, text_valid, num_mode_layers=2, num_layers=4, device=cpu)
            new_fx = build_flux_layer_masks(
                frame_mask, seg, s_action, text_valid, n_hist=h, num_mode_layers=2, num_layers=4, device=cpu
            )
            assert all(torch.equal(a, b) for a, b in zip(old_fx, new_fx))

    # cosmos3: the frame-major mask with the und prefix keys prepended (shared, or gated per sample).
    shell = Cosmos3EdgeVideoBackbone.__new__(Cosmos3EdgeVideoBackbone)
    nn.Module.__init__(shell)
    shell._num_layers = 3
    for prefix_mask in (None, torch.tensor([[True, True, False], [True, True, True]])):
        state = _state(t, tpf, b=2, prefix=3, prefix_mask=prefix_mask)
        old_c = _old_widen_mask_for_prefix_kv(old, state)
        new_c = shell.build_joint_attention_masks(frame_mask, state, s_action)
        assert len(new_c) == 3 and all(torch.equal(m, old_c) for m in new_c)


# ---------------------------------------------------------------- text cache


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


# ---------------------------------------------------------------- no family exceptions in the base flow

# The shared flow the base class owns: a family may only implement encode_pixels / decode_latents / encode_text.
_BASE_FLOW = (
    "to_pixels",
    "encode_frames",
    "encode_clip",
    "_encode_history",
    "_encode_layout_text",
    "preprocess_condition",
    "preprocess_train",
    "decode_future",
)


@pytest.mark.parametrize("method", _BASE_FLOW)
def test_families_do_not_override_base_flow(method):
    """Every registered family inherits ``method`` from VideoBackbone unchanged."""
    import openwam.model.video_backbone  # noqa: F401  (registers the built-in families)
    from openwam.model.video_backbone.registry import _VIDEO_BACKBONE_REGISTRY

    families = set(_VIDEO_BACKBONE_REGISTRY.values())
    assert families, "no family registered"
    overriding = sorted(cls.__name__ for cls in families if getattr(cls, method) is not getattr(VideoBackbone, method))
    assert not overriding, f"{overriding} override VideoBackbone.{method}"


def test_decode_future_converts_with_tensor_to_pil_frames(fake):
    """decode_future turns the decoded pixels into PIL frames with the shared ``tensor_to_pil_frames`` alone."""
    from openwam.model.video_backbone.common.pixels import tensor_to_pil_frames

    layout = _layout(1)
    pixels = torch.linspace(-1.2, 1.2, 9 * 3 * 32 * 32).reshape(1, 3, 9, 32, 32)
    fake.decode_latents = lambda latents: pixels  # (1, 3, 1 + TC * 2, 32, 32), values past [-1, 1] too
    frames = fake.decode_future(torch.zeros(1, Z, 3, 4, 4), layout)
    want = tensor_to_pil_frames(pixels)[1:]
    assert [f.tobytes() for f in frames] == [w.tobytes() for w in want]
    assert "_pixels_to_frames" not in dir(VideoBackbone)  # no per-family conversion hook


def test_prompt_cache_dedups_in_order_with_the_empty_prompt():
    """One build order for every family: ``dict.fromkeys(prompts + [""])`` ("" = a caption-dropped sample)."""
    calls = []

    def _encode(chunk):
        calls.append(list(chunk))
        return _fake_text_encoder(chunk)

    cache = PromptCache(_encode, ["place", "pick", "place"], batch_size=2)
    assert list(cache.entries) == ["place", "pick", ""]
    assert calls == [["place", "pick"], [""]]  # each prompt encoded once, in first-seen order
