"""flux-3-action-base video trunk as an OpenWAM video backbone (07_sparse_history_backbones).

Only the video side of FLUX 3 Action is used; its action stream is replaced by OpenWAM's ActionDiT,
aligned 1:1 with the trunk's 5 mode blocks + 28 single-stream blocks (33 MoT layers).

Token layout of the video-side sequence (native single-block order: context, then the active
content streams in ``sequence`` order):

    [ T (text, L_max) | F (x_video, future) | H (x_video_cond, history) | P (x_action_cond, proprio) ]

* T: Qwen3-VL-4B context (8 stacked layers, 20480 ch), ids ``(0, 0, 0, l)``, timestep 0.
* F: future latents i = H..T-1, ids ``(times_to_ids(4 i / 24), h, w, 0)``, noisy timestep.
* H: single-frame history latents i = 0..H-1, same id rule, timestep 0 (clean).
* P: the H proprio frames through flux's own ``action_cond`` stream (fresh input Linear, mode blocks /
  modulations from the trunk's shared ``action_prediction_cond``), ids ``(times_to_ids(j - (H-1)), 0, 0, -1)``
  (native: negative real time on the state clock), timestep 0.

Attention (``build_joint_attention_masks``; A = OpenWAM action tokens appended by the MoT driver):
* layers 0-4 (mode blocks): every stream only itself (T, F, H, P, A block-diagonal);
* layers 5-32 (single blocks): native full joint attention over T/F/H/P, plus the framework rule that
  H rows never read F or A; F, T, P and A rows read everything.
Per sample, text keys beyond that caption's own bucket length are masked (native packs captions of
different lengths without padding them to each other).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import Tensor, nn

from openwam.model.video_backbone.base import BlockLoopState, VideoBackbone
from openwam.model.video_backbone.flux_action.positional import times_to_ids
from openwam.model.video_backbone.flux_action.transformer import (
    JointSingleSeq,
    JointSingleSeqParams,
    apply_rope,
    timestep_embedding,
)

logger = logging.getLogger(__name__)

# Native geometry (models/video_vae.py ViTNormInference, models/transformer.py JointSingleSeqParams).
_LATENT_CHANNELS = 96
_TEMPORAL_COMPRESSION = 4
_SPATIAL_COMPRESSION = 32
_DECODER_TEMPORAL_KERNEL = 5  # decoder NATTEN kernel along latent time; shorter latents cannot decode
_VIDEO_POSITION_FPS = 24.0  # PolicyConfig.video_position_fps: latent i sits at 4 i / 24 s
_STATE_FPS = 1.0  # our 5 proprio frames are 1 s apart (07 layout)
_VEC_DIM = 768  # pooled-vector width; flux feeds zeros
_SHARED_ACTION_COND = "action_prediction_cond"  # co-trained action branch routed to our P stream
_P_STREAM = "action_cond"
_N_MODE = 5  # JointSingleSeqParams.depth
# Segment order of the video-side sequence.
_SEGMENTS = ("txt", "video", "video_cond", _P_STREAM)


def _rational_time_shift(t: Tensor, shift: float) -> Tensor:
    """flux processing/packing.py ``rational_time_shift``."""
    return shift * t / (1.0 + (shift - 1.0) * t)


def video_time_ids(latent_index: Tensor) -> Tensor:
    """Scheme (a) time ids: latent i at ``times_to_ids(i * 4 / 24)`` (10 ms ticks).

    Args:
        latent_index: (N,) int latent indices.

    Returns:
        (N,) int64 ids (0, 16, 33, 50, 66, 83, 100 for i = 0..6).
    """
    seconds = latent_index.float() * _TEMPORAL_COMPRESSION / _VIDEO_POSITION_FPS
    return times_to_ids(seconds)


def build_stream_ids(n_hist: int, n_fut: int, h: int, w: int, text_len: int, n_prop: int) -> dict[str, Tensor]:
    """4-axis RoPE ids ``(t, h, w, l)`` of every stream (batch-independent).

    Args:
        n_hist / n_fut: H history / future latent frames.
        h / w: latent grid.
        text_len: int, padded text length L.
        n_prop: int, proprio tokens (one per history frame).

    Returns:
        dict ``txt`` (L, 4), ``video`` (n_fut*h*w, 4), ``video_cond`` (n_hist*h*w, 4), ``action_cond`` (n_prop, 4).
    """

    def grid(first: int, n: int) -> Tensor:
        # (t h w) token order of prc_vid; l = 0 for video.
        t = video_time_ids(torch.arange(first, first + n))
        return torch.cartesian_prod(t, torch.arange(h), torch.arange(w), torch.zeros(1, dtype=torch.long))

    txt = torch.stack(
        [torch.zeros(text_len, dtype=torch.long)] * 3 + [torch.arange(text_len)], dim=-1
    )  # prc_txt: (0, 0, 0, l)
    # history_profile pack_actions: times (j - (n - 1)) / fps, l = -1.
    t_prop = times_to_ids((torch.arange(n_prop).float() - (n_prop - 1)) / _STATE_FPS)
    zeros = torch.zeros(n_prop, dtype=torch.long)
    prop = torch.stack([t_prop, zeros, zeros, torch.full((n_prop,), -1, dtype=torch.long)], dim=-1)
    return {"txt": txt, "video": grid(n_hist, n_fut), "video_cond": grid(0, n_hist), _P_STREAM: prop}


def build_flux_layer_masks(
    seg_lens: dict[str, int],
    s_action: int,
    text_valid: Optional[Tensor],
    *,
    num_mode_layers: int,
    num_layers: int,
    device,
    strict_clean_rows: bool = False,
) -> list[Tensor]:
    """Per-layer joint masks over ``[T | F | H | P | A]`` (True = attend).

    Args:
        seg_lens: token count per segment ``txt / video / video_cond / action_cond``.
        s_action: int, OpenWAM action tokens appended after the video-side sequence (0 = none).
        text_valid: optional (B,) int, each sample's own text bucket length (<= seg_lens['txt']).
        num_mode_layers: int, leading mode-block layers (5).
        num_layers: int, total layers (33).
        device: torch device.
        strict_clean_rows: bool, test-only: also hide F / A from the T and P rows (native flux lets them
            read everything, so H reads noisy content one layer later through T / P).

    Returns:
        list of ``num_layers`` masks, each (S, S) or (B, 1, S, S) when text lengths differ per sample;
        the mode-layer entries share one tensor, as do the single-layer ones.
    """
    names = list(_SEGMENTS) + ["action"]
    lens = [seg_lens[n] for n in _SEGMENTS] + [s_action]
    starts = torch.tensor([0] + lens[:-1]).cumsum(0).tolist()
    s = sum(lens)
    sl = {n: slice(a, a + n_) for n, a, n_ in zip(names, starts, lens)}

    # Mode blocks: block-diagonal, every stream attends only to itself.
    mode = torch.zeros((s, s), dtype=torch.bool, device=device)
    for n in names:
        mode[sl[n], sl[n]] = True
    # Single blocks: native full attention; clean history never reads the noisy future / action.
    single = torch.ones((s, s), dtype=torch.bool, device=device)
    single[sl["video_cond"], sl["video"]] = False
    single[sl["video_cond"], sl["action"]] = False
    if strict_clean_rows:
        for clean in ("txt", _P_STREAM):
            single[sl[clean], sl["video"]] = False
            single[sl[clean], sl["action"]] = False

    def with_text_padding(m: Tensor) -> Tensor:
        # Per sample: text keys past the sample's own bucket are invisible (rows keep their diagonal).
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


class FluxActionVideoBackbone(VideoBackbone):
    """flux-3-action-base trunk + ViTNorm video VAE + Qwen3-VL-4B behind the VideoBackbone ABC."""

    # The P stream needs the raw proprio frames (joint_self_attn passes them as ``raw_proprio``).
    consumes_raw_proprio = True

    def __init__(
        self,
        *,
        dit: JointSingleSeq,
        vae: nn.Module,
        text_encoder: Optional[nn.Module],
        state_dim: int,
        shift_video: float = 5.0,
        text_dropout_p: float = 0.1,
        train_timestep_width: float = 0.75,
        train_timestep_shift: float = 42.0,
    ):
        """
        Args:
            dit: JointSingleSeq with streams video / video_cond / action_cond.
            vae: ViTNormInference (frozen).
            text_encoder: Qwen3VLEmbedder (frozen) or None (deploy rebuild).
            state_dim: int, proprio width fed to the P stream.
            shift_video: float, inference shift of the OpenWAM scheduler (flux DROID inference: 5).
            text_dropout_p: float, training caption dropout to "" (flux PolicyConfig.caption_dropout).
            train_timestep_width / train_timestep_shift: native training timestep law (logit-logistic + rational shift).
        """
        super().__init__()
        from openwam.model.video_backbone.cosmos_predict25.scheduler import CosmosFlowSchedulerAdapter

        self.dit = dit
        self.vae = vae
        if text_encoder is not None:
            self.text_encoder = text_encoder
        self.state_dim = int(state_dim)
        self._shift_video = float(shift_video)
        self._scheduler = CosmosFlowSchedulerAdapter(shift_video=self._shift_video)
        self._text_dropout_p = float(text_dropout_p)
        self._ts_width = float(train_timestep_width)
        self._ts_shift = float(train_timestep_shift)
        self._dit_patch_size = (1, 1, 1)
        self._temporal_compression = _TEMPORAL_COMPRESSION
        self._causal_temporal = True
        self._video_attention_mask_mode = "first_frame_causal"
        self._text_cache: Optional[dict] = None
        # Test-only switch of build_flux_layer_masks (see verify_layout.py); never set in training.
        self.strict_clean_rows = False

    # ---------------------------------------------------------------- metadata
    @property
    def dim(self) -> int:
        return self.dit.hidden_size

    @property
    def num_layers(self) -> int:
        return self.dit.depth + self.dit.depth_single_blocks

    @property
    def num_heads(self) -> int:
        return self.dit.num_heads

    @property
    def head_dim(self) -> int:
        return self.dit.hidden_size // self.dit.num_heads

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def text_dim(self) -> Optional[int]:
        return self.dit.params.context_in_dim

    @property
    def video_attention_mask_mode(self) -> str:
        return self._video_attention_mask_mode

    @video_attention_mask_mode.setter
    def video_attention_mask_mode(self, mode: str) -> None:
        self._video_attention_mask_mode = str(mode)

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_pretrained(cls, source: Any, **kw) -> "FluxActionVideoBackbone":
        """Build from the architecture params (training: ``video_backbone`` + ``state_dim``).

        Args:
            source: dict / DictConfig holding ``video_backbone`` (model_path, ...) and ``state_dim``.
            kw: ignored (``text_dim`` etc.).

        Returns:
            FluxActionVideoBackbone with trunk, VAE and text encoder loaded.
        """
        vb_cfg = source.get("video_backbone") if hasattr(source, "get") and "video_backbone" in source else None
        if vb_cfg is None or vb_cfg.get("_source") is not None:
            raise NotImplementedError("flux3_action_base: deploy rebuild from a checkpoint is not implemented yet")
        root = str(vb_cfg.get("model_path"))
        state_dim = int(source.get("state_dim", 48))
        dit = build_flux_dit(f"{root}/flux-3-action-base.safetensors", state_dim=state_dim, head_seed=0)
        vae = build_flux_vae(f"{root}/video_vae.safetensors")
        from openwam.model.video_backbone.flux_action.text_encoder import Qwen3VLEmbedder

        text_encoder = Qwen3VLEmbedder(f"{root}/text_encoder")
        return cls(
            dit=dit,
            vae=vae,
            text_encoder=text_encoder,
            state_dim=state_dim,
            shift_video=float(vb_cfg.get("shift_video", 5.0)),
            text_dropout_p=float(vb_cfg.get("text_dropout_p", 0.1)),
            train_timestep_width=float(vb_cfg.get("train_timestep_width", 0.75)),
            train_timestep_shift=float(vb_cfg.get("train_timestep_shift", 42.0)),
        )

    # ---------------------------------------------------------------- training noise (native)
    def sample_training_noise(self, batch: int, *, device, dtype) -> Tuple[Tensor, Tensor, Tensor]:
        """flux training timestep law for the video stream (policy.py:662, packing.sample_timesteps).

        ``logit(t) ~ Logistic(0, 0.75)`` then the rational shift 42; ``t`` is rounded to the latent dtype
        as the native noising does. OpenWAM's ``x_t = (1 - s) x0 + s eps`` / target ``eps - x0`` equal flux's
        ``add_noise``, so sigma = t.

        Args:
            batch: int, B.
            device / dtype: target device and latent dtype.

        Returns:
            ``(timesteps (B,) fp32 = 1000 t, sigmas (B,) dtype, loss_weights (B,) ones)``.
        """
        eps = torch.finfo(torch.float32).eps
        u = torch.rand(batch).clamp(eps, 1 - eps)
        t = torch.sigmoid(self._ts_width * (torch.log(u) - torch.log1p(-u)))
        t = _rational_time_shift(t, self._ts_shift).to(device=device, dtype=dtype)
        # Native flow_loss weighs every video token equally (video_loss_weight = 1).
        return t.float() * 1000.0, t, torch.ones(batch, device=device)

    # ---------------------------------------------------------------- encoders
    @torch.no_grad()
    def _vae_encode(self, video: Tensor) -> Tensor:
        """``(N, 3, T, H, W)`` in [-1, 1] -> normalized latents ``(N, 96, 1 + (T-1)//4, H/32, W/32)``.

        A single frame is encoded alone (native ``encode_task`` / ``encode_frame``: the 2-D kernel path).
        A clip is black-padded at the end to the 45-frame VAE chunk, encoded and cropped back
        (native ``packing.encode_videos``, the DROID training path): its temporal NATTEN kernel is 5, so
        short clips such as our 1+4 future cannot run unpadded.
        """
        from openwam.model.video_backbone.flux_action.video_vae import padded_chunk_length

        dtype = next(self.vae.parameters()).dtype
        n, c, t, hp, wp = video.shape
        if t > 1:
            chunk = self.vae.chunk_size_frames
            total = chunk if t < chunk else padded_chunk_length(t, chunk)
            pad = torch.full((n, c, total - t, hp, wp), -1.0, dtype=video.dtype, device=video.device)
            video = torch.cat([video, pad], dim=2)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=video.device.type == "cuda"):
            lat = self.vae.encode(video.to(dtype), chunked_encode=video.shape[2] > self.vae.chunk_size_frames)
        return lat[:, :, : 1 + (t - 1) // _TEMPORAL_COMPRESSION].clone()

    @torch.no_grad()
    def decode_video(self, latents: Tensor, *, tiled: bool = True) -> list:
        """Normalized latents ``(1, 96, T_lat, h, w)`` -> ``4 * T_lat - 3`` PIL frames (training vis).

        The decoder's temporal NATTEN kernel spans 5 latents, so a shorter clip (our 1+4 future = 2 latents)
        is padded by repeating its last latent and the extra frames are cropped. The kernel is not causal,
        so the last decoded frames are approximate; fine for visualization, not a reconstruction metric.
        ``tiled`` is unused: the VAE's own looped decode (``decoder_max_t``) bounds the memory.
        """
        from openwam.model.video_backbone.cosmos_predict25._vae_utils import _video_tensor_to_pil

        _ = tiled
        t = latents.shape[2]
        if t < _DECODER_TEMPORAL_KERNEL:
            pad = latents[:, :, -1:].expand(-1, -1, _DECODER_TEMPORAL_KERNEL - t, -1, -1)
            latents = torch.cat([latents, pad], dim=2)
        dtype = next(self.vae.parameters()).dtype
        # Same autocast as _vae_encode; ViTNormInference.decode denormalizes and clamps to [-1, 1].
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=latents.device.type == "cuda"):
            video = self.vae.decode(latents.to(dtype))
        video = video[:, :, : _TEMPORAL_COMPRESSION * (t - 1) + 1]
        return _video_tensor_to_pil(video.float().clamp(-1, 1))

    @staticmethod
    def _pixels(frames) -> Tensor:
        """list of B clips of PIL frames -> (B, 3, T, H, W) float in [-1, 1]."""
        from openwam.model.video_backbone.cosmos_predict25._vae_utils import _pil_video_to_tensor

        return _pil_video_to_tensor(frames)

    def encode_sparse(self, frames, num_history_frames: int) -> Tensor:
        """Each history frame alone, the future as one causal clip (flux history.encode_windows).

        Args:
            frames: list of B clips (H + F PIL frames), or (B, 3, H + F, Hp, Wp) in [-1, 1].
            num_history_frames: int, H.

        Returns:
            (B, 96, H + 1 + (F-1)//4, h, w) latents.
        """
        x = frames if torch.is_tensor(frames) else self._pixels(frames)
        x = x.to(self.device)
        b, h = x.shape[0], int(num_history_frames)
        hist = x[:, :, :h].transpose(1, 2).flatten(0, 1).unsqueeze(2)  # (B*H, 3, 1, Hp, Wp)
        hist_lat = self._vae_encode(hist).unflatten(0, (b, h)).squeeze(3).transpose(1, 2)
        fut_lat = self._vae_encode(x[:, :, h:])
        return torch.cat([hist_lat, fut_lat], dim=2)

    @torch.no_grad()
    def build_text_cache(self, prompts: list, *, offload_device: str = "cpu") -> None:
        """Encode every training prompt (and the "" dropout caption) once, then park Qwen on the CPU.

        Args:
            prompts: list[str] of every training prompt.
            offload_device: where the frozen encoder lives afterwards.
        """
        texts = sorted(set(prompts) | {""})
        cache = {}
        for i in range(0, len(texts), 16):
            chunk = texts[i : i + 16]
            for p, ctx in zip(chunk, self.text_encoder.encode(chunk)):
                cache[p] = ctx.to(self.device)
        self._text_cache = cache
        self.text_encoder.to(offload_device)
        torch.cuda.empty_cache()

    def encode_text(self, prompts: list) -> Tuple[Tensor, Tensor]:
        """Captions -> ``(context (B, L_max, 20480), seq_lens (B,))``; each caption keeps its own bucket.

        Args:
            prompts: list[str] of B captions.

        Returns:
            context zero-padded to the longest bucket, seq_lens = each caption's bucket length.
        """
        if self._text_cache is not None:
            missing = [p for p in prompts if p not in self._text_cache]
            if missing:
                raise KeyError(f"flux text cache has no entry for {missing[:3]}")
            ctxs = [self._text_cache[p] for p in prompts]
        else:
            ctxs = self.text_encoder.encode(list(prompts))
        lens = torch.tensor([c.shape[1] for c in ctxs], device=self.device)
        ctx = torch.zeros(len(ctxs), int(lens.max()), ctxs[0].shape[-1], device=self.device, dtype=self.dtype)
        for i, c in enumerate(ctxs):
            ctx[i, : c.shape[1]] = c[0].to(device=self.device, dtype=self.dtype)
        return ctx, lens

    # ---------------------------------------------------------------- train preprocessing
    def preprocess_input_for_train(self, *, frames=None, text=None, **kw) -> dict:
        """Frames + prompts -> latents (sparse history when ``num_history_frames`` > 0) + text context.

        Returns:
            dict ``input_latents`` (B, 96, T, h, w), ``first_frame_latents`` (clean prefix), ``num_clean_prefix_frames``,
            ``context`` (B, L, 20480), ``seq_lens`` (B,), ``text_len`` (int L), ``height`` / ``width`` / ``num_frames``.
        """
        n_hist = int(kw.get("num_history_frames", 0) or 0)
        prompts = [text] * len(frames) if isinstance(text, str) else list(text)
        if self.training and self._text_dropout_p > 0:
            # Native caption dropout: the empty caption through the same chat template.
            keep = torch.rand(len(prompts)) >= self._text_dropout_p
            prompts = [p if k else "" for p, k in zip(prompts, keep.tolist())]
        with torch.no_grad():
            if n_hist > 0:
                latents = self.encode_sparse(frames, n_hist)
            else:
                latents = self._vae_encode(self._pixels(frames).to(self.device))
            context, seq_lens = self.encode_text(prompts)
        latents = latents.to(device=self.device, dtype=self.dtype)
        n_clean = max(n_hist, 1)
        w0, h0 = frames[0][0].size
        return {
            "input_latents": latents,
            "first_frame_latents": latents[:, :, :n_clean].clone(),
            "num_clean_prefix_frames": n_clean,
            "context": context,
            "seq_lens": seq_lens,
            "text_len": int(context.shape[1]),
            "height": h0,
            "width": w0,
            "num_frames": len(frames[0]),
        }

    # ---------------------------------------------------------------- block loop
    def prepare(self, **kw) -> BlockLoopState:
        """Embed T / F / H / P, build per-token modulations and RoPE, stash per-segment metadata.

        Expects ``latents`` (B, 96, T, h, w), ``timestep`` (B,) = 1000 t, ``context`` (B, L(+N_p), 20480)
        whose first ``text_len`` tokens are text, ``seq_lens`` (B,), ``num_clean_prefix_frames`` H and
        ``raw_proprio`` (B, N_p, state_dim) (zeros when absent).
        """
        dit = self.dit
        latents = kw["latents"]
        b, _, t_lat, h, w = latents.shape
        n_hist = int(kw.get("num_clean_prefix_frames", 1) or 1)
        text_len = int(kw.get("text_len", kw["context"].shape[1]))
        ctx = kw["context"][:, :text_len]
        seq_lens = kw.get("seq_lens")
        prop = kw.get("raw_proprio")
        if prop is None:
            prop = torch.zeros(b, n_hist, self.state_dim, device=latents.device, dtype=latents.dtype)
        prop = prop.reshape(b, -1, self.state_dim)
        n_prop = prop.shape[1]
        dtype = latents.dtype
        dev = latents.device

        # Content tokens in prc_vid (t h w) order.
        hist_tok = rearrange(latents[:, :, :n_hist], "b c t h w -> b (t h w) c")
        fut_tok = rearrange(latents[:, :, n_hist:], "b c t h w -> b (t h w) c")
        x = {
            "txt": dit.txt_in(ctx.to(dtype)),
            "video": dit.emb_in["video"](fut_tok),
            "video_cond": dit.emb_in["video_cond"](hist_tok),
            _P_STREAM: dit.emb_in[_P_STREAM](prop.to(dtype)),
        }
        seg_lens = {k: int(v.shape[1]) for k, v in x.items()}

        # Per-token timesteps: future noisy (t in [0, 1]), every conditioning stream clean.
        t = (kw["timestep"].float() / 1000.0).to(dev)
        ts = {k: torch.zeros(b, n, device=dev) for k, n in seg_lens.items()}
        ts["video"] = t[:, None].expand(b, seg_lens["video"])
        vector_emb = dit.vector_in(torch.zeros(b, _VEC_DIM, device=dev, dtype=dtype))
        vec = {k: dit.embed_timesteps(v.reshape(-1), dtype).reshape(b, -1, dit.hidden_size) + vector_emb[:, None]
               for k, v in ts.items()}

        # Modulations: early (mode blocks) and single, per segment, concatenated in sequence order.
        def mods(table):
            parts = [table[k if k != "txt" else "txt"](vec[k])[0] for k in _SEGMENTS]
            return tuple(torch.cat([p[i] for p in parts], dim=1) for i in range(3))

        early = mods(dit.early_stream_modulations)
        single = mods(dit.single_stream_modulations)

        ids = build_stream_ids(n_hist, t_lat - n_hist, h, w, text_len, n_prop)
        all_ids = torch.cat([ids[k] for k in _SEGMENTS], dim=0).to(dev)
        pe = dit.pe_embedder(all_ids[None].expand(b, -1, -1))

        hidden = torch.cat([x[k] for k in _SEGMENTS], dim=1)
        extras = {
            "seg_lens": seg_lens,
            "early_mod": early,
            "single_mod": single,
            "pe": pe,
            "vec": vec,
            "n_hist": n_hist,
            "text_valid": seq_lens.to(dev) if seq_lens is not None else None,
        }
        return BlockLoopState(
            hidden_states=hidden,
            time_mod=torch.zeros((), device=dev, dtype=dtype),
            rope_freqs=torch.zeros((), device=dev),
            context=ctx,
            grid_frames=t_lat,
            grid_height=h,
            grid_width=w,
            use_gradient_checkpointing=bool(kw.get("use_gradient_checkpointing", False)),
            use_gradient_checkpointing_offload=bool(kw.get("use_gradient_checkpointing_offload", False)),
            extras=extras,
        )

    def _segments(self, state: BlockLoopState):
        """Segment slices of the video-side sequence, in ``_SEGMENTS`` order."""
        out, start = {}, 0
        for k in _SEGMENTS:
            n = state.extras["seg_lens"][k]
            out[k] = slice(start, start + n)
            start += n
        return out

    def _mode_block(self, layer_id: int, seg: str):
        """Mode block of ``seg`` at depth ``layer_id``."""
        if seg == "txt":
            return self.dit.txt_mode_blocks[layer_id]
        return self.dit.content_mode_blocks[seg][layer_id]

    @staticmethod
    def _qkv_mlp(block, x: Tensor, mod, pe: Tensor):
        """Pre-attention half shared by ModeBlock / SingleStreamBlock: modulate, project, QK-norm, RoPE.

        Returns:
            q, k, v as (B, L, H*D) and the MLP pre-activation (B, L, 2 * mlp_hidden).
        """
        shift, scale, _ = mod
        x_mod = (1 + scale) * block.pre_norm(x) + shift
        heads = block.num_heads
        q = rearrange(block.q_proj(x_mod), "B L (H D) -> B H L D", H=heads)
        k = rearrange(block.k_proj(x_mod), "B L (H D) -> B H L D", H=heads)
        v = rearrange(block.v_proj(x_mod), "B L (H D) -> B H L D", H=heads)
        mlp = block.mlp_in(x_mod)
        q, k = block.norm(q, k, v)
        q, k = apply_rope(q, k, pe)
        flat = lambda t: rearrange(t, "B H L D -> B L (H D)")  # noqa: E731
        return flat(q), flat(k), flat(v), mlp

    @staticmethod
    def _slice_mod(mod, sl: slice):
        return tuple(m[:, sl] for m in mod)

    def pre_attn_at_layer(self, layer_id: int, state: BlockLoopState) -> Tuple[Tensor, Tensor, Tensor, dict]:
        """Q/K/V of the whole video-side sequence at ``layer_id`` (mode block per segment, or single block)."""
        x, ex = state.hidden_states, state.extras
        pe = ex["pe"]
        if layer_id < _N_MODE:
            qs, ks, vs, mlps = [], [], [], []
            for seg, sl in self._segments(state).items():
                q, k, v, mlp = self._qkv_mlp(
                    self._mode_block(layer_id, seg), x[:, sl], self._slice_mod(ex["early_mod"], sl), pe[:, :, sl]
                )
                qs.append(q), ks.append(k), vs.append(v), mlps.append(mlp)
            return torch.cat(qs, 1), torch.cat(ks, 1), torch.cat(vs, 1), {"mlp": mlps}
        block = self.dit.single_blocks[layer_id - _N_MODE]
        q, k, v, mlp = self._qkv_mlp(block, x, ex["single_mod"], pe)
        return q, k, v, {"mlp": [mlp]}

    def post_attn_at_layer(self, layer_id: int, state: BlockLoopState, attn_out: Tensor, post_state: dict):
        """Output projection + gated MLP + gated residual, per segment (mode) or over the sequence (single)."""
        x, ex = state.hidden_states, state.extras
        if layer_id < _N_MODE:
            outs = []
            for (seg, sl), mlp in zip(self._segments(state).items(), post_state["mlp"]):
                block = self._mode_block(layer_id, seg)
                out = block.attn_out(attn_out[:, sl]) + block.mlp_out(block.mlp_act(mlp))
                outs.append(x[:, sl] + ex["early_mod"][2][:, sl] * out)
            state.hidden_states = torch.cat(outs, dim=1)
            return state
        block = self.dit.single_blocks[layer_id - _N_MODE]
        out = block.attn_out(attn_out) + block.mlp_out(block.mlp_act(post_state["mlp"][0]))
        state.hidden_states = x + ex["single_mod"][2] * out
        return state

    def build_joint_attention_masks(self, vstate: BlockLoopState, *, s_action: int, attention_mask_mode: str,
                                    num_clean_frames: int) -> list[Tensor]:
        """Per-layer masks for the MoT driver (see module docstring); only ``mutual`` is defined here."""
        if s_action and attention_mask_mode != "mutual":
            raise NotImplementedError(f"flux3_action_base defines only the mutual v<->a mask, got {attention_mask_mode}")
        return build_flux_layer_masks(
            vstate.extras["seg_lens"],
            s_action,
            vstate.extras["text_valid"],
            num_mode_layers=_N_MODE,
            num_layers=self.num_layers,
            device=vstate.hidden_states.device,
            strict_clean_rows=self.strict_clean_rows,
        )

    def run_block(self, block_id: int, state: BlockLoopState) -> BlockLoopState:
        """Video-only layer (no action tokens): same masks with ``s_action = 0``."""
        masks = state.extras.get("_masks")
        if masks is None:
            masks = self.build_joint_attention_masks(state, s_action=0, attention_mask_mode="mutual", num_clean_frames=1)
            state.extras = {**state.extras, "_masks": masks}
        q, k, v, post = self.pre_attn_at_layer(block_id, state)
        heads = self.num_heads
        split = lambda t: rearrange(t, "b s (n d) -> b n s d", n=heads)  # noqa: E731
        out = F.scaled_dot_product_attention(split(q), split(k), split(v), attn_mask=masks[block_id])
        return self.post_attn_at_layer(block_id, state, rearrange(out, "b n s d -> b s (n d)"), post)

    def finalize(self, state: BlockLoopState) -> Tensor:
        """Per-stream heads -> ``(B, 96, T, h, w)``: history from ``video_cond``'s head, future from ``video``'s."""
        x, ex = state.hidden_states, state.extras
        seg = self._segments(state)
        h, w, n_hist = state.grid_height, state.grid_width, ex["n_hist"]
        fut = self.dit.final_layer["video"](x[:, seg["video"]], ex["vec"]["video"])
        hist = self.dit.final_layer["video_cond"](x[:, seg["video_cond"]], ex["vec"]["video_cond"])
        hist = rearrange(hist, "b (t h w) c -> b c t h w", t=n_hist, h=h, w=w)
        fut = rearrange(fut, "b (t h w) c -> b c t h w", h=h, w=w)
        return torch.cat([hist, fut], dim=2)


# -------------------------------------------------------------------- builders
def build_flux_dit(path: str, *, state_dim: int, head_seed: int = 0, dtype: torch.dtype = torch.bfloat16) -> JointSingleSeq:
    """JointSingleSeq with streams video / video_cond / action_cond, loaded from the action-base trunk.

    ``action_prediction_cond`` mode blocks + modulations are routed to ``action_cond`` (flux wiring); the
    ``action_cond`` input Linear is fresh (xavier, seeded; the trunk's ee50 / gaming heads are 50 / 64 wide)
    and its unused final layer zero. The trunk's action-prediction stream and heads are dropped.

    Args:
        path: flux-3-action-base.safetensors.
        state_dim: int, P input width.
        head_seed: int, seed of the fresh input Linear.
        dtype: parameter dtype.

    Returns:
        JointSingleSeq in eval mode on CPU.
    """
    from safetensors import safe_open

    params = JointSingleSeqParams(
        in_channels={"video": _LATENT_CHANNELS, "video_cond": _LATENT_CHANNELS, _P_STREAM: int(state_dim)},
        sequence={"x_video": "video", "x_video_cond": "video_cond", f"x_{_P_STREAM}": _P_STREAM},
    )
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = JointSingleSeq(params)
    finally:
        torch.set_default_dtype(prev)
    sd = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            parts = key.split(".")
            if parts[0] in ("emb_in", "final_layer") and parts[1].startswith("action_prediction"):
                continue  # embodiment heads of other robots
            if len(parts) >= 3 and parts[1] == _SHARED_ACTION_COND:
                parts[1] = _P_STREAM
            elif len(parts) >= 3 and parts[1] == "action_prediction":
                continue  # the trunk's action-prediction stream is replaced by OpenWAM's ActionDiT
            sd[".".join(parts)] = f.get_tensor(key)
    res = model.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=False)
    fresh = {f"emb_in.{_P_STREAM}.weight", f"final_layer.{_P_STREAM}.linear.weight",
             f"final_layer.{_P_STREAM}.adaLN_modulation.1.weight"}
    if set(res.missing_keys) - fresh or res.unexpected_keys:
        raise ValueError(f"flux trunk load mismatch: missing {sorted(set(res.missing_keys) - fresh)[:6]}, "
                         f"unexpected {res.unexpected_keys[:6]}")
    # Reference init of a fresh embodiment head (wiring.fresh_head_state_dict): xavier input, zero output.
    gen = torch.Generator().manual_seed(head_seed)
    weight = torch.empty(model.hidden_size, int(state_dim))
    nn.init.xavier_uniform_(weight, generator=gen)
    with torch.no_grad():
        model.emb_in[_P_STREAM].weight.copy_(weight.to(dtype))
        for p in model.final_layer[_P_STREAM].parameters():
            p.zero_()
    model.eval()
    return model


def build_flux_vae(path: str) -> nn.Module:
    """Frozen ViTNorm video VAE (``ViTNormInference``, bf16) from ``video_vae.safetensors``."""
    from safetensors.torch import load_file

    from openwam.model.video_backbone.flux_action.video_vae import ViTNormInference, ViTNormInferenceParams

    vae = ViTNormInference(ViTNormInferenceParams(use_compile=False, chunked_encode=False, decoder_max_t=8))
    vae.load_state_dict(load_file(path, device="cpu"))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


__all__ = [
    "FluxActionVideoBackbone",
    "build_flux_dit",
    "build_flux_layer_masks",
    "build_flux_vae",
    "build_stream_ids",
    "video_time_ids",
]
