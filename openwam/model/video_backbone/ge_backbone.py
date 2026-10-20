"""GE-Base-fast (Genie Envisioner, AgiBot) video backbone — 07_sparse_history_backbones.

Components (all GE-native):
  * DiT: GE's multi-view LTX transformer (28 x 2048, 32 heads x 64, patch 1),
    weights ``GE_base_fast_v0.1.safetensors``; vendored in ``ge/transformer.py``.
  * VAE: LTX-Video ``AutoencoderKLLTXVideo`` (8x temporal causal, 32x spatial,
    128 ch), latents normalized with the VAE's ``latents_mean/std``.
  * Text: LTX-Video T5-XXL encoder, 128 tokens, 6% caption dropout (GE swaps the
    dropped prompt's embedding for the empty prompt's and keeps its mask).

Views: the dataloader stacks the cameras vertically (ego over chest). The
backbone splits the canvas into ``n_view`` tiles, VAE-encodes each tile on its
own, and stacks the tile latents back along height, so latents are
``(B, 128, T, V*h, w)`` and the loss / clean-prefix logic of the architecture
is unchanged. Tokens are frame-major with the views inside a frame (see
``ge/masks.py``); every token carries its own view's RoPE grid (shared by all
views, as in GE) plus GE's view embedding.

Under the MoT (``dual_system / joint_self_attn``) the per-layer masks come from
:meth:`build_joint_attention_masks`: GE's per-view / cross-view block pattern
plus OpenWAM's "clean history never reads noisy frames or action" rule.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from openwam.model.video_backbone.base import BlockLoopState, VideoBackbone
from openwam.model.video_backbone.cosmos_predict25.scheduler import CosmosFlowSchedulerAdapter

logger = logging.getLogger(__name__)

_GE_TEMPORAL_COMPRESSION = 8
_GE_SPATIAL_COMPRESSION = 32
_GE_LATENT_CHANNELS = 128
_GE_FRAME_RATE = 30.0  # GE's forward_pass hard-codes frame_rate=30 for the RoPE time scale
_GE_NUM_TRAIN_TIMESTEPS = 1000


def _cfg(vb_cfg, key, default=None):
    """Read ``key`` from a dict / DictConfig backbone cfg."""
    if vb_cfg is None:
        return default
    val = vb_cfg.get(key, default) if hasattr(vb_cfg, "get") else getattr(vb_cfg, key, default)
    return default if val is None else val


class GEBaseFastVideoBackbone(VideoBackbone):
    """GE multi-view LTX transformer + LTX VAE + T5 behind the VideoBackbone ABC."""

    def __init__(
        self,
        *,
        dit,
        vae=None,
        tokenizer=None,
        text_encoder=None,
        n_view: int = 2,
        text_max_length: int = 128,
        caption_dropout_p: float = 0.06,
        vae_sample: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        # Flat named children so the model yaml freeze list hits vae / text_encoder.
        self.dit = dit
        if vae is not None:
            self.vae = vae
        if text_encoder is not None:
            self.text_encoder = text_encoder
        self._tokenizer = tokenizer
        cfg = dit.config
        self._dim = int(cfg.num_attention_heads * cfg.attention_head_dim)
        self._num_layers = int(cfg.num_layers)
        self._num_heads = int(cfg.num_attention_heads)
        self._head_dim = int(cfg.attention_head_dim)
        self._n_view = int(n_view)
        self._text_max_length = int(text_max_length)
        self._caption_dropout_p = float(caption_dropout_p)
        self._vae_sample = bool(vae_sample)
        self._rng = random.Random(seed)
        # GE trains with diffusers' default FlowMatchEulerDiscreteScheduler (shift 1): the
        # adapter's 1000-step grid sigma_i = 1 - i/1000 is the same grid (used by deploy).
        self._scheduler = CosmosFlowSchedulerAdapter(shift_video=1.0, num_train_timesteps=_GE_NUM_TRAIN_TIMESTEPS)
        self._shift_video = 1.0
        self._dit_patch_size = (1, 1, 1)
        self._temporal_compression = _GE_TEMPORAL_COMPRESSION
        self._causal_temporal = True
        self._video_attention_mask_mode = "bidirectional"
        self._text_cache: Optional[dict] = None
        self._mask_cache: dict = {}

    # ================================================================
    # Structural metadata
    # ================================================================

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def text_dim(self) -> Optional[int]:
        """T5-XXL hidden width (the context the framework appends proprio to)."""
        return int(self.dit.config.caption_channels)

    @property
    def video_attention_mask_mode(self) -> str:
        return self._video_attention_mask_mode

    @video_attention_mask_mode.setter
    def video_attention_mask_mode(self, mode: str) -> None:
        self._video_attention_mask_mode = str(mode)
        self._mask_cache = {}

    # ================================================================
    # Construction
    # ================================================================

    @classmethod
    def from_pretrained(
        cls, source: Any, *, external_encoder=None, text_dim: Optional[int] = None, ckpt_dir: Optional[str] = None, **kw
    ):
        """Build from the pretrained GE / LTX weights.

        Args:
            source: training: dict / DictConfig of architecture params holding ``video_backbone``;
                run-dir rebuild: that ``video_backbone`` cfg itself. It has ``model_path`` (GE ckpt dir),
                ``ltx_path`` (LTX-Video dir: vae/, text_encoder/, tokenizer/) and optional
                ``n_view``, ``text_max_length``, ``caption_dropout_p``, ``vae_sample``.
            ckpt_dir: run dir being rebuilt from (resume / finetune); None for a fresh training build.
            external_encoder / text_dim: must be None / 4096 (native components only).

        Returns:
            GEBaseFastVideoBackbone on CPU in bf16.
        """
        from diffusers import AutoencoderKLLTXVideo
        from safetensors.torch import load_file
        from transformers import T5EncoderModel, T5Tokenizer

        from openwam.model.video_backbone.ge.transformer import LTXVideoTransformer3DModel

        if external_encoder is not None:
            raise ValueError("ge_base_fast keeps the native LTX VAE; external encoders are not supported")
        if ckpt_dir is not None:
            # Rebuild from a run dir (resume / finetune, ckpt_model_loader): `source` is that run's own
            # video_backbone cfg. Build the same pretrained model from it; the caller then overwrites
            # every weight (accelerate load_state / the safetensors load).
            vb_cfg = source
        else:
            vb_cfg = source.get("video_backbone")
        model_path = str(_cfg(vb_cfg, "model_path"))
        ltx_path = str(_cfg(vb_cfg, "ltx_path"))

        dit = LTXVideoTransformer3DModel()
        state = load_file(os.path.join(model_path, str(_cfg(vb_cfg, "weights_file", "GE_base_fast_v0.1.safetensors"))))
        dit.load_state_dict(state, strict=True)  # 718 tensors, no action expert
        dit.to(torch.bfloat16)
        vae = AutoencoderKLLTXVideo.from_pretrained(ltx_path, subfolder="vae", torch_dtype=torch.bfloat16)
        tokenizer = T5Tokenizer.from_pretrained(ltx_path, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(ltx_path, subfolder="text_encoder", torch_dtype=torch.bfloat16)
        if text_dim is not None and int(text_dim) != int(dit.config.caption_channels):
            raise ValueError(f"ge_base_fast text_dim is {dit.config.caption_channels}, architecture asked {text_dim}")
        return cls(
            dit=dit,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            n_view=int(_cfg(vb_cfg, "n_view", 2)),
            text_max_length=int(_cfg(vb_cfg, "text_max_length", 128)),
            caption_dropout_p=float(_cfg(vb_cfg, "caption_dropout_p", 0.06)),
            vae_sample=bool(_cfg(vb_cfg, "vae_sample", True)),
            seed=_cfg(vb_cfg, "seed", None),
        )

    # ================================================================
    # Text
    # ================================================================

    @torch.no_grad()
    def _t5(self, prompts: List[str]) -> Tuple[Tensor, Tensor]:
        """GE ``_encode_prompt``: (B, 128, 4096) embeddings + (B, 128) bool mask."""
        tok = self._tokenizer(
            prompts, padding="max_length", max_length=self._text_max_length, truncation=True,
            add_special_tokens=True, return_tensors="pt",
        )
        dev = next(self.text_encoder.parameters()).device
        emb = self.text_encoder(tok.input_ids.to(dev))[0]
        return emb.to(device=self.device, dtype=self.dtype), tok.attention_mask.bool().to(self.device)

    @torch.no_grad()
    def build_text_cache(self, prompts: list, *, batch_size: int = 8, offload_device: str = "cpu") -> None:
        """Encode every training prompt (+ the empty dropout prompt) once, then park T5 off the GPU.

        Args:
            prompts: list[str], the dataset's prompt set.
            batch_size: int, prompts per T5 forward.
            offload_device: where T5 lives afterwards (stays a registered child).
        """
        cache = {}
        allp = list(dict.fromkeys(list(prompts) + [""]))
        for i in range(0, len(allp), batch_size):
            chunk = allp[i : i + batch_size]
            emb, mask = self._t5(chunk)
            for j, p in enumerate(chunk):
                cache[p] = (emb[j : j + 1].clone(), mask[j : j + 1].clone())
        self._text_cache = cache
        self.text_encoder.to(offload_device)
        torch.cuda.empty_cache()

    def _encode_text(self, prompts: List[str]) -> Tuple[Tensor, Tensor]:
        """(B, 128, 4096) context + (B, 128) mask, with GE's training-time caption dropout.

        Dropped samples (p = caption_dropout_p, training only) take the empty prompt's
        embedding but keep their own mask, exactly as ge_trainer.py does.
        """
        need = list(prompts) + [""]
        if self._text_cache is not None:
            missing = [p for p in need if p not in self._text_cache]
            if missing:
                raise KeyError(f"text cache has no entry for prompts {missing[:3]}")
            emb = torch.cat([self._text_cache[p][0] for p in need], dim=0)
            mask = torch.cat([self._text_cache[p][1] for p in need], dim=0)
        else:
            emb, mask = self._t5(need)
        emb, mask, uncond = emb[:-1].clone(), mask[:-1].clone(), emb[-1:]
        if self.training and self._caption_dropout_p > 0:
            drop = torch.tensor([self._rng.random() < self._caption_dropout_p for _ in prompts], device=emb.device)
            emb = torch.where(drop[:, None, None], uncond.expand_as(emb), emb)
        return emb, mask

    # ================================================================
    # Video IO
    # ================================================================

    def _split_views(self, frames: Tensor) -> Tensor:
        """(B, 3, T, V*h_px, W) canvas → (B*V, 3, T, h_px, W) per-view clips (top tile = view 0)."""
        return rearrange(frames, "b c t (v h) w -> (b v) c t h w", v=self._n_view)

    @torch.no_grad()
    def _vae_encode(self, pixels: Tensor) -> Tensor:
        """(N, 3, T, h_px, w_px) in [-1, 1] → normalized (N, 128, T_lat, h, w) latents (GE get_latents).

        The posterior is sampled only in training mode with ``vae_sample`` on (GE's training recipe); eval
        (training vis, deploy) takes its mode, so inference is deterministic.
        """
        vae = self.vae
        dist = vae.encode(pixels.to(device=self.device, dtype=next(vae.parameters()).dtype)).latent_dist
        lat = dist.sample() if (self._vae_sample and self.training) else dist.mode()
        mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(lat.device, lat.dtype)
        std = vae.latents_std.view(1, -1, 1, 1, 1).to(lat.device, lat.dtype)
        return (lat - mean) * float(vae.config.scaling_factor) / std

    def _encode_history(self, frames: Tensor) -> Tensor:
        """Every history frame encoded alone (GE memory frames), per view.

        Args:
            frames: (B, 3, H, V*h_px, W) canvas pixels in [-1, 1].

        Returns: (B, 128, H, V*h, w) latents, views stacked along height.
        """
        b, h = frames.shape[0], frames.shape[2]
        views = self._split_views(frames)  # (B*V, 3, H, h_px, W)
        hist = views.transpose(1, 2).flatten(0, 1).unsqueeze(2)  # (B*V*H, 3, 1, h_px, W)
        lat = self._vae_encode(hist).unflatten(0, (views.shape[0], h)).squeeze(3).transpose(1, 2)
        return rearrange(lat, "(b v) c t h w -> b c t (v h) w", b=b)

    def _encode_clip(self, frames: Tensor) -> Tensor:
        """A clip encoded as one causal clip, per view.

        Args:
            frames: (B, 3, T, V*h_px, W) canvas pixels in [-1, 1], T = 1 + 8k.

        Returns: (B, 128, 1 + (T-1)//8, V*h, w) latents, views stacked along height.
        """
        lat = self._vae_encode(self._split_views(frames))
        return rearrange(lat, "(b v) c t h w -> b c t (v h) w", b=frames.shape[0])

    def _to_pixels(self, frames) -> Tensor:
        """frames: list of B clips of T PIL canvases, or (B, 3, T, H_px, W) in [-1, 1]. Returns: the tensor form."""
        from openwam.model.video_backbone.cosmos_predict25._vae_utils import _pil_video_to_tensor

        return frames if torch.is_tensor(frames) else _pil_video_to_tensor(frames, device=self.device)

    def preprocess_sparse_condition(self, history_frames, text, *, num_future_frames: int) -> dict:
        """Sparse-history conditioning: H history canvases encoded one by one + the T5 context.

        Args:
            history_frames: list of B clips (H PIL canvases, ego over chest), or (B, 3, H, H_px, W) in [-1, 1].
            text: list of B prompts (or one str for all).
            num_future_frames: int F (1 + 8k).

        Returns:
            dict with ``first_frame_latents`` (B, 128, H, V*h, w), ``num_clean_prefix_frames`` = H, ``context``
            (B, 128, 4096), ``context_mask``, ``seq_lens``, ``num_frames`` = H + F, ``height``, ``width``.
        """
        frames_t = self._to_pixels(history_frames)
        self.num_future_latents(num_future_frames)  # validates F = 1 + 8k
        latents = self._encode_history(frames_t).to(dtype=self.dtype)
        prompts = [text] * len(frames_t) if isinstance(text, str) else list(text)
        context, mask = self._encode_text(prompts)
        return {
            "first_frame_latents": latents,
            "num_clean_prefix_frames": int(frames_t.shape[2]),
            "context": context,
            "context_mask": mask,
            "seq_lens": mask.sum(dim=1).long(),
            "num_frames": int(frames_t.shape[2]) + int(num_future_frames),
            "height": int(frames_t.shape[3]),
            "width": int(frames_t.shape[4]),
        }

    def encode_sparse_future(self, future_frames) -> Tensor:
        """future_frames: list of B clips of F PIL canvases (or the tensor form). Returns: (B, 128, 1 + (F-1)//8, V*h, w)."""
        return self._encode_clip(self._to_pixels(future_frames)).to(dtype=self.dtype)

    def preprocess_input_for_train(self, *, frames=None, text=None, **kw) -> dict:
        """Raw clips + prompts → denoising-loop inputs.

        Args:
            frames: list of B clips (H + F PIL canvases, ego over chest) or (B, 3, T, H_px, W) in [-1, 1].
            text: list of B prompts.
            num_history_frames (kw): H clean single-frame history latents (07: 5); 0 = first frame only.

        Returns:
            dict with ``input_latents`` (B, 128, T_lat, V*h, w), ``first_frame_latents`` (the clean
            prefix), ``num_clean_prefix_frames``, ``context`` (B, 128, 4096), ``context_mask``,
            ``seq_lens``.
        """
        frames_t = self._to_pixels(frames)
        n_hist = int(kw.get("num_history_frames", 0) or 0)
        n_fut = frames_t.shape[2] - n_hist
        if (n_fut - 1) % self._temporal_compression:
            raise ValueError(f"ge_base_fast: future frames must be 1 + 8k, got {n_fut}")
        if n_hist > 0:
            # Sparse history = the shared condition (also built by deploy) + the future clip as target.
            cond = self.preprocess_sparse_condition(frames_t[:, :, :n_hist], text, num_future_frames=n_fut)
            fut = self.encode_sparse_future(frames_t[:, :, n_hist:])
            return {**cond, "input_latents": torch.cat([cond["first_frame_latents"], fut], dim=2)}
        latents = self._encode_clip(frames_t).to(dtype=self.dtype)
        prompts = [text] * len(frames_t) if isinstance(text, str) else list(text)
        context, mask = self._encode_text(prompts)
        return {
            "input_latents": latents,
            "first_frame_latents": latents[:, :, :1].clone(),
            "num_clean_prefix_frames": 1,
            "context": context,
            "context_mask": mask,
            "seq_lens": mask.sum(dim=1).long(),
            "num_frames": int(frames_t.shape[2]),
            "height": int(frames_t.shape[3]),
            "width": int(frames_t.shape[4]),
        }

    # ================================================================
    # Native training noise (ge_trainer.py:574-596, 646-648)
    # ================================================================

    def sample_training_noise(self, batch_size: int, *, device, dtype) -> Tuple[Tensor, Tensor, Tensor]:
        """GE's ``flow_weighting_scheme: none`` sampling.

        u ~ U[0, 1) per sample; index = floor(u * 1000) into the shift-1 FlowMatchEuler grid
        sigma_i = (1000 - i) / 1000; transformer timestep = floor(sigma * 1000); loss weight 1.

        Returns:
            (timesteps (B,), sigmas (B,), loss_weights (B,)).
        """
        u = torch.rand(batch_size)
        idx = (u * _GE_NUM_TRAIN_TIMESTEPS).long()
        sigmas = (_GE_NUM_TRAIN_TIMESTEPS - idx).float() / _GE_NUM_TRAIN_TIMESTEPS
        timesteps = (sigmas * _GE_NUM_TRAIN_TIMESTEPS).long().float()
        return (
            timesteps.to(device=device, dtype=dtype),
            sigmas.to(device=device, dtype=dtype),
            torch.ones(batch_size, device=device, dtype=torch.float32),
        )

    # ================================================================
    # Three-step execution
    # ================================================================

    def _grid(self, latents: Tensor) -> Tuple[int, int, int]:
        """(T, h_view, w) of a (B, C, T, V*h, w) latent."""
        t, vh, w = latents.shape[2:]
        if vh % self._n_view:
            raise ValueError(f"latent height {vh} does not split into {self._n_view} views")
        return t, vh // self._n_view, w

    def prepare(self, **kw) -> BlockLoopState:
        """Patchify (patch 1: one token per latent pixel), per-token time / view modulation, RoPE, text.

        Inputs (kw): ``latents`` (B, 128, T, V*h, w); ``timestep`` (B,) in [0, 1000];
        ``num_clean_prefix_frames`` H (their tokens get t = 0, GE pixel_wise_timestep);
        ``context`` (B, L, 4096) (text, + proprio appended by the framework); ``context_mask`` (B, L).
        """
        dit = self.dit
        latents = kw["latents"]
        timestep = kw["timestep"]
        b = latents.shape[0]
        t_lat, vh, vw = self._grid(latents)
        n_view = self._n_view
        tpf = n_view * vh * vw
        n_clean = int(kw.get("num_clean_prefix_frames", 0) or 0)

        # Tokens: frame-major, views inside a frame (= flatten the stacked latent grid).
        hidden = dit.proj_in(rearrange(latents, "b c t h w -> b (t h w) c").to(dit.proj_in.weight.dtype))

        # Per-token timestep: clean history frames get 0 (GE pixel_wise_timestep).
        ts = timestep.reshape(-1).float()
        ts = ts.expand(b) if ts.numel() == 1 else ts
        frame_ts = ts[:, None].expand(b, t_lat).clone()
        frame_ts[:, :n_clean] = 0
        tok_ts = frame_ts.repeat_interleave(tpf, dim=1)  # (B, S)
        temb, emb_t = dit.time_embed(tok_ts.flatten(), batch_size=b, hidden_dtype=hidden.dtype)
        temb = temb.view(b, -1, temb.shape[-1])  # (B, S, 6*dim)
        emb_t = emb_t.view(b, -1, emb_t.shape[-1])  # (B, S, dim)

        # GE view embedding: per-view AdaLN offset + final-layer embedding offset.
        from openwam.model.video_backbone.ge.masks import view_index

        vid = view_index(t_lat, n_view, vh, vw, device=hidden.device)
        view_emb = dit.view_embed[:n_view].to(hidden.dtype)
        temb = temb + dit.view_ada(view_emb)[vid][None]
        emb_t = emb_t + view_emb[vid][None]

        # RoPE on each view's own (t, h, w) grid, shared by the views (GE native), rescaled
        # by (1 / latent_frame_rate, 32, 32) with frame_rate fixed to 30.
        scale = (1.0 / (_GE_FRAME_RATE / _GE_TEMPORAL_COMPRESSION), _GE_SPATIAL_COMPRESSION, _GE_SPATIAL_COMPRESSION)
        cos, sin = dit.rope(hidden[:1], scale, t_lat, vh, vw)  # (1, T*h*w, dim), order (t, y, x)
        cos = rearrange(cos, "o (t y x) d -> o t y x d", t=t_lat, y=vh)
        sin = rearrange(sin, "o (t y x) d -> o t y x d", t=t_lat, y=vh)
        cos = rearrange(cos[:, :, None].expand(-1, -1, n_view, -1, -1, -1), "o t v y x d -> o (t v y x) d")
        sin = rearrange(sin[:, :, None].expand(-1, -1, n_view, -1, -1, -1), "o t v y x d -> o (t v y x) d")

        # Text (+ proprio) → caption projection; mask True = attend.
        context = dit.caption_projection(kw["context"].to(hidden.dtype))
        cmask = kw.get("context_mask")
        if cmask is None:
            seq_lens = kw.get("seq_lens")
            pos = torch.arange(context.shape[1], device=context.device)[None]
            cmask = pos < seq_lens.to(context.device)[:, None] if seq_lens is not None else torch.ones_like(pos, dtype=torch.bool).expand(b, -1)
        cmask = cmask.to(device=context.device, dtype=torch.bool)

        return BlockLoopState(
            hidden_states=hidden,
            time_mod=temb,
            rope_freqs=torch.zeros((), device=hidden.device),
            context=context,
            context_mask=cmask,
            grid_frames=t_lat,
            grid_height=n_view * vh,
            grid_width=vw,
            use_gradient_checkpointing=bool(kw.get("use_gradient_checkpointing", False)),
            use_gradient_checkpointing_offload=bool(kw.get("use_gradient_checkpointing_offload", False)),
            extras={"cos": cos, "sin": sin, "emb_t": emb_t, "view_h": vh, "n_clean": max(n_clean, 1)},
        )

    def _ada(self, layer_id: int, state: BlockLoopState):
        """Six AdaLN tensors (each (B, S, dim)) of block ``layer_id``."""
        block = self.dit.transformer_blocks[layer_id]
        b, s = state.hidden_states.shape[:2]
        ada = block.scale_shift_table[None, None] + state.time_mod.reshape(b, s, 6, -1)
        return ada.unbind(dim=2)

    def pre_attn_at_layer(self, layer_id: int, state: BlockLoopState) -> Tuple[Tensor, Tensor, Tensor, dict]:
        """norm1 + AdaLN → attn1 q/k/v (RMS-normed across heads, RoPE'd), each (B, S, 2048)."""
        from openwam.model.video_backbone.ge.transformer import apply_rotary_emb

        block = self.dit.transformer_blocks[layer_id]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._ada(layer_id, state)
        x = block.norm1(state.hidden_states) * (1 + scale_msa) + shift_msa
        attn = block.attn1
        rope = (state.extras["cos"], state.extras["sin"])
        q = apply_rotary_emb(attn.norm_q(attn.to_q(x)), rope)
        k = apply_rotary_emb(attn.norm_k(attn.to_k(x)), rope)
        v = attn.to_v(x)
        post = {"gate_msa": gate_msa, "shift_mlp": shift_mlp, "scale_mlp": scale_mlp, "gate_mlp": gate_mlp}
        return q, k, v, post

    def post_attn_at_layer(self, layer_id: int, state: BlockLoopState, attn_out: Tensor, post_state: dict) -> BlockLoopState:
        """attn1 out-proj + gate, text cross-attn (attn2, no pre-norm, as GE), gated FFN."""
        block = self.dit.transformer_blocks[layer_id]
        x = state.hidden_states + block.attn1.to_out[1](block.attn1.to_out[0](attn_out)) * post_state["gate_msa"]
        x = x + self._cross_attn(block.attn2, x, state.context, state.context_mask)
        h = block.norm2(x) * (1 + post_state["scale_mlp"]) + post_state["shift_mlp"]
        state.hidden_states = x + block.ff(h) * post_state["gate_mlp"]
        return state

    def _cross_attn(self, attn, x: Tensor, context: Tensor, mask: Tensor) -> Tensor:
        """All views' queries of a sample attend that sample's text (+ proprio) tokens."""
        heads = attn.heads
        q = attn.norm_q(attn.to_q(x)).unflatten(2, (heads, -1)).transpose(1, 2)
        k = attn.norm_k(attn.to_k(context)).unflatten(2, (heads, -1)).transpose(1, 2)
        v = attn.to_v(context).unflatten(2, (heads, -1)).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
        out = out.transpose(1, 2).flatten(2, 3).to(q.dtype)
        return attn.to_out[1](attn.to_out[0](out))

    def _layer_masks(self, state: BlockLoopState, s_action: int, attention_mask_mode: str, num_clean_frames: int):
        """Cached per-layer masks for this grid (see ``ge/masks.py``)."""
        key = (state.grid_frames, state.extras["view_h"], state.grid_width, s_action, attention_mask_mode,
               num_clean_frames, self._video_attention_mask_mode, str(state.hidden_states.device))
        if key not in self._mask_cache:
            # Lazy: masks.py imports the architecture mask helpers.
            from openwam.model.video_backbone.ge.masks import build_ge_joint_masks

            self._mask_cache[key] = build_ge_joint_masks(
                n_frames=state.grid_frames,
                n_view=self._n_view,
                view_h=state.extras["view_h"],
                view_w=state.grid_width,
                s_action=s_action,
                num_clean_frames=num_clean_frames,
                attention_mask_mode=attention_mask_mode,
                video_mask_mode=self._video_attention_mask_mode,
                num_layers=self._num_layers,
                device=state.hidden_states.device,
            )
        return self._mask_cache[key]

    def build_joint_attention_masks(self, vstate: BlockLoopState, *, s_action: int, attention_mask_mode: str, num_clean_frames: int):
        """MoT hook: one (S_v + S_a)^2 bool mask per layer (GE per-view / cross-view pattern + history rule)."""
        return self._layer_masks(vstate, s_action, attention_mask_mode, num_clean_frames)

    def run_block(self, block_id: int, state: BlockLoopState) -> BlockLoopState:
        """Video-only block (no action stream): same masks with zero action tokens."""

        def _body(x):
            local = BlockLoopState(**{**state.__dict__, "hidden_states": x})
            q, k, v, post = self.pre_attn_at_layer(block_id, local)
            mask = self._layer_masks(state, 0, "mutual", state.extras["n_clean"])[block_id]
            n = self._num_heads
            q, k, v = (t.unflatten(2, (n, -1)).transpose(1, 2) for t in (q, k, v))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).flatten(2, 3)
            return self.post_attn_at_layer(block_id, local, out, post).hidden_states

        if state.use_gradient_checkpointing and torch.is_grad_enabled():
            state.hidden_states = torch.utils.checkpoint.checkpoint(_body, state.hidden_states, use_reentrant=False)
        else:
            state.hidden_states = _body(state.hidden_states)
        return state

    def finalize(self, state: BlockLoopState) -> Tensor:
        """Final AdaLN (time + view embedding) + proj_out → (B, 128, T, V*h, w) velocity."""
        dit = self.dit
        ss = dit.scale_shift_table[None, None] + state.extras["emb_t"][:, :, None]
        shift, scale = ss[:, :, 0], ss[:, :, 1]
        out = dit.proj_out(dit.norm_out(state.hidden_states) * (1 + scale) + shift)
        return rearrange(
            out, "b (t h w) c -> b c t h w", t=state.grid_frames, h=state.grid_height, w=state.grid_width
        )

    # ================================================================
    # Deploy (decode only; inference preprocessing is not wired yet)
    # ================================================================

    def build_video_to_video_mask(self, video_seq_len: int, video_tokens_per_frame: int, device) -> Tensor:
        raise NotImplementedError("ge_base_fast builds its masks per layer (build_joint_attention_masks)")

    def preprocess_input_for_inference(self, **kw) -> dict:
        raise NotImplementedError("ge_base_fast deploy preprocessing is not wired yet (07 trains only)")

    def decode_video(self, latents: Tensor, *, tiled: bool = True) -> list:
        """(1, 128, T, V*h, w) normalized latents → list of vstacked PIL canvases."""
        from openwam.model.video_backbone.cosmos_predict25._vae_utils import _video_tensor_to_pil

        vae = self.vae
        lat = rearrange(latents, "b c t (v h) w -> (b v) c t h w", v=self._n_view)
        mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(lat.device, lat.dtype)
        std = vae.latents_std.view(1, -1, 1, 1, 1).to(lat.device, lat.dtype)
        lat = lat * std / float(vae.config.scaling_factor) + mean
        video = vae.decode(lat.to(next(vae.parameters()).dtype)).sample  # (V, 3, T, h, w)
        video = rearrange(video, "(b v) c t h w -> b c t (v h) w", v=self._n_view)
        return _video_tensor_to_pil(video.float().clamp(-1, 1))

    def save_deploy_assets(self, output_dir: str, cfg) -> None:
        """No self-contained deploy bundle yet; checkpoints reload via the training cfg paths."""
        logger.info("ge_base_fast: deploy-asset embedding not implemented; skipping.")


__all__ = ["GEBaseFastVideoBackbone"]
