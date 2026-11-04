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

Construction (everything from ``cfg = cfg.model.video_backbone``):
* :meth:`FluxActionVideoBackbone.from_pretrained`: the trunk built from the yaml ``dit`` geometry + the
  ``action_cond_dim`` P input, loaded from ``flux-3-action-base.safetensors``; VAE and Qwen3-VL from the bundle;
* :meth:`FluxActionVideoBackbone.from_checkpoint_dir`: the same modules as meta shells, Qwen3-VL from the
  ``config.json`` / processor files :meth:`save_deploy_assets` copied into ``<ckpt>/text_encoder/``.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Tuple

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from openwam.model.video_backbone.base import BlockLoopState, TextCondition, VideoBackbone
from openwam.model.video_backbone.common.text_cache import PromptCache
from openwam.model.video_backbone.flux.dit import (
    JointSingleSeq,
    JointSingleSeqParams,
    apply_rope,
    timestep_embedding,
)
from openwam.model.video_backbone.flux.masks import (
    _P_STREAM,
    _SEGMENTS,
    _TEMPORAL_COMPRESSION,
    build_flux_layer_masks,
    build_stream_ids,
)

logger = logging.getLogger(__name__)

# Native geometry (models/video_vae.py ViTNormInference, models/transformer.py JointSingleSeqParams).
_LATENT_CHANNELS = 96
_SPATIAL_COMPRESSION = 32
_DECODER_TEMPORAL_KERNEL = 5  # decoder NATTEN kernel along latent time; shorter latents cannot decode
# Shortest clip the encoder runs on: its last Swin blocks see ceil(T/2) frames (after the first
# TemporalMerging), which NATTEN needs >= the temporal kernel 5, so T >= 9 (and T = 1 mod 4).
_ENCODER_MIN_FRAMES = 9
_VEC_DIM = 768  # pooled-vector width; flux feeds zeros
_SHARED_ACTION_COND = "action_prediction_cond"  # co-trained action branch routed to our P stream
_N_MODE = 5  # JointSingleSeqParams.depth
# Qwen3-VL dir: ``text_encoder/`` in the bundle and, copied by save_deploy_assets (weights excluded), in a ckpt dir.
_TEXT_ENCODER_SUBDIR = "text_encoder"
_WEIGHT_PATTERNS = ("*.safetensors", "*.safetensors.index.json")


def _rational_time_shift(t: Tensor, shift: float) -> Tensor:
    """flux processing/packing.py ``rational_time_shift``."""
    return shift * t / (1.0 + (shift - 1.0) * t)


class FluxActionVideoBackbone(VideoBackbone):
    """flux-3-action-base trunk + ViTNorm video VAE + Qwen3-VL-4B behind the VideoBackbone ABC."""

    def __init__(
        self,
        *,
        dit: JointSingleSeq,
        vae: nn.Module,
        text_encoder: nn.Module,
        text_encoder_dir: str,
        action_cond_dim: int,
        shift_video: float = 5.0,
        text_dropout_p: float = 0.1,
        train_timestep_width: float = 0.75,
        train_timestep_shift: float = 42.0,
    ):
        """
        Args:
            dit: JointSingleSeq with streams video / video_cond / action_cond.
            vae: ViTNormInference (frozen).
            text_encoder: Qwen3VLEmbedder (frozen).
            text_encoder_dir: str, the dir its config / processor files were read from (save_deploy_assets copies it).
            action_cond_dim: int, proprio width fed to the P stream (``resolve_config`` checks it equals
                ``model.architecture.state_dim``).
            shift_video: float, inference shift of the OpenWAM scheduler (flux DROID inference: 5).
            text_dropout_p: float, training caption dropout to "" (flux PolicyConfig.caption_dropout).
            train_timestep_width / train_timestep_shift: native training timestep law (logit-logistic + rational shift).
        """
        super().__init__()
        from openwam.model.video_backbone.common.flow_scheduler import FlowScheduler

        self.dit = dit
        self.vae = vae
        self.text_encoder = text_encoder
        self._text_encoder_dir = text_encoder_dir
        self.state_dim = int(action_cond_dim)
        self._shift_video = float(shift_video)
        self._scheduler = FlowScheduler(shift_video=self._shift_video)
        self._text_dropout_p = float(text_dropout_p)
        self._ts_width = float(train_timestep_width)
        self._ts_shift = float(train_timestep_shift)
        self._dit_patch_size = (1, 1, 1)
        self._temporal_compression = _TEMPORAL_COMPRESSION
        self._causal_temporal = True
        # Text slot read by encode_text: the online Qwen3-VL here, a PromptCache after build_text_cache.
        self.text_lookup = self._qwen_entries
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
    def text_dim(self) -> int:
        return self.dit.params.context_in_dim

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_pretrained(cls, cfg: DictConfig) -> "FluxActionVideoBackbone":
        """Build from the pretrained flux bundle at ``cfg.model_path``.

        Args:
            cfg: ``cfg.model.video_backbone``: ``dit`` geometry, ``action_cond_dim``, ``model_path`` and the
                scalar knobs.

        Returns:
            FluxActionVideoBackbone with trunk, VAE and text encoder loaded (CPU). Raises ValueError when the
            trunk weights do not match the yaml geometry.
        """
        from openwam.model.video_backbone.flux.text_encoder import Qwen3VLEmbedder

        root = str(cfg.model_path)
        # Built in this order (trunk, VAE): their random inits are part of the training seed stream.
        dit = build_flux_dit(f"{root}/flux-3-action-base.safetensors", cfg.dit, action_cond_dim=int(cfg.action_cond_dim), head_seed=0)
        vae = build_flux_vae(f"{root}/video_vae.safetensors")
        text_dir = os.path.join(root, _TEXT_ENCODER_SUBDIR)
        return cls._assemble(cfg, dit, vae, Qwen3VLEmbedder.from_pretrained(text_dir), text_dir)

    @classmethod
    def from_checkpoint_dir(cls, cfg: DictConfig, ckpt_dir: str) -> "FluxActionVideoBackbone":
        """Build the empty skeleton of a saved checkpoint; the caller loads its weights.

        Args:
            cfg: the checkpoint's ``cfg.model.video_backbone``.
            ckpt_dir: str, checkpoint dir holding ``text_encoder/`` (Qwen3-VL config + processor files).

        Returns:
            FluxActionVideoBackbone whose parameters are all on ``meta`` (bf16); non-persistent buffers are real.
        """
        from accelerate import init_empty_weights

        from openwam.model.video_backbone.flux.text_encoder import Qwen3VLEmbedder
        from openwam.model.video_backbone.flux.vae import ViTNormInference

        text_dir = os.path.join(ckpt_dir, _TEXT_ENCODER_SUBDIR)
        if not os.path.isdir(text_dir):
            raise FileNotFoundError(f"flux text encoder assets not found at {text_dir} (written by save_deploy_assets)")
        with init_empty_weights(include_buffers=False):
            dit = _flux_dit(cfg.dit, int(cfg.action_cond_dim), torch.bfloat16)
            vae = ViTNormInference(_flux_vae_params())
        # Same dtypes as the pretrained path: the trunk is built in bf16, the VAE casts itself (params.dtype).
        dit.eval()
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        return cls._assemble(cfg, dit, vae, Qwen3VLEmbedder.from_config_dir(text_dir), text_dir)

    @classmethod
    def _assemble(cls, cfg: DictConfig, dit, vae, text_encoder, text_encoder_dir: str) -> "FluxActionVideoBackbone":
        """Shared tail of both constructors: wrap the modules with the yaml's scalar knobs.

        Args:
            cfg: ``cfg.model.video_backbone``. dit / vae / text_encoder: built modules (real or meta).
            text_encoder_dir: str, where the text encoder's config / processor files were read from.

        Returns:
            FluxActionVideoBackbone.
        """
        return cls(
            dit=dit,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_dir=text_encoder_dir,
            action_cond_dim=int(cfg.action_cond_dim),
            shift_video=float(cfg.shift_video),
            text_dropout_p=float(cfg.text_dropout_p),
            train_timestep_width=float(cfg.train_timestep_width),
            train_timestep_shift=float(cfg.train_timestep_shift),
        )

    def save_deploy_assets(self, output_dir: str) -> None:
        """Copy the Qwen3-VL config / processor / tokenizer files (no weights) into ``<output_dir>/text_encoder/``.

        Args:
            output_dir: str, checkpoint dir. Files come from wherever this backbone read them (bundle or ckpt).

        Returns:
            None. An existing ``text_encoder/`` is left as it is.
        """
        dst = os.path.join(output_dir, _TEXT_ENCODER_SUBDIR)
        if os.path.isdir(dst):
            return
        shutil.copytree(self._text_encoder_dir, dst, ignore=shutil.ignore_patterns(*_WEIGHT_PATTERNS))

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

    # ---------------------------------------------------------------- video IO
    @torch.no_grad()
    def encode_pixels(self, video: Tensor) -> Tensor:
        """``(N, 3, T, H, W)`` in [-1, 1] -> normalized latents ``(N, 96, 1 + (T-1)//4, H/32, W/32)``.

        A single frame is encoded alone (native ``encode_task`` / ``encode_frame``: the 2-D kernel path).
        A clip is black-padded at the end, encoded and cropped back (native ``packing.encode_videos``,
        the DROID training path): its temporal NATTEN kernel is 5, so short clips such as our 1+4 future
        cannot run unpadded. Native pads to the 45-frame chunk; we pad only to ``_ENCODER_MIN_FRAMES``.
        The encoder is causal in time (per-frame patch embed, causal NATTEN, front-aligned
        TemporalMerging), so trailing pad frames never reach the kept latents; only the cost changes.
        Clips longer than one chunk still go through the native chunked encode.
        Raises ValueError for a clip with T > 1 and T != 1 (mod 4) (the data layout must provide 1 + 4k frames).
        """
        from openwam.model.video_backbone.flux.vae import padded_chunk_length

        dtype = next(self.vae.parameters()).dtype
        n, c, t, hp, wp = video.shape
        if t > 1:
            # The VAE takes T = 1 (mod 4) only (video_vae.VideoVAE); other lengths change how TemporalMerging
            # pairs frames, so refuse them instead of silently padding: the data layout has to be fixed.
            if (t - 1) % _TEMPORAL_COMPRESSION:
                raise ValueError(
                    f"flux3_action_base: a clip must have T = 1 (mod {_TEMPORAL_COMPRESSION}) frames, got T={t}"
                )
            chunk = self.vae.chunk_size_frames
            total = max(t, _ENCODER_MIN_FRAMES) if t <= chunk else padded_chunk_length(t, chunk)
            pad = torch.full((n, c, total - t, hp, wp), -1.0, dtype=video.dtype, device=video.device)
            video = torch.cat([video, pad], dim=2)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=video.device.type == "cuda"):
            lat = self.vae.encode(video.to(dtype), chunked_encode=video.shape[2] > self.vae.chunk_size_frames)
        return lat[:, :, : 1 + (t - 1) // _TEMPORAL_COMPRESSION].clone()

    @torch.no_grad()
    def decode_latents(self, latents: Tensor) -> Tensor:
        """Normalized latents ``(N, 96, T_lat, h, w)`` -> ``(N, 3, 4 * T_lat - 3, 32h, 32w)`` pixels in [-1, 1].

        The decoder's temporal NATTEN kernel spans 5 latents, so a shorter clip (our 1+4 future = 2 latents)
        is padded by repeating its last latent and the extra frames are cropped. The kernel is not causal,
        so the last decoded frames are approximate; fine for visualization, not a reconstruction metric.
        """
        t = latents.shape[2]
        if t < _DECODER_TEMPORAL_KERNEL:
            pad = latents[:, :, -1:].expand(-1, -1, _DECODER_TEMPORAL_KERNEL - t, -1, -1)
            latents = torch.cat([latents, pad], dim=2)
        dtype = next(self.vae.parameters()).dtype
        # Same autocast as encode_pixels; ViTNormInference.decode denormalizes and clamps to [-1, 1].
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=latents.device.type == "cuda"):
            video = self.vae.decode(latents.to(dtype))
        return video[:, :, : _TEMPORAL_COMPRESSION * (t - 1) + 1]

    # ---------------------------------------------------------------- text
    @torch.no_grad()
    def _qwen_entries(self, prompts: list) -> list:
        """Online Qwen3-VL encode (the default ``text_lookup``).

        Args:
            prompts: list[str] of B captions.

        Returns:
            list of B contexts ``(1, L_i, 20480)`` on ``self.device``, ``L_i`` each caption's own bucket length.
        """
        return [ctx.to(self.device) for ctx in self.text_encoder.encode(list(prompts))]

    @torch.no_grad()
    def build_text_cache(self, prompts: list) -> None:
        """Encode every training prompt (and the "" dropout caption) once, then park Qwen on the CPU.

        Args:
            prompts: list[str] of every training prompt.

        Returns:
            None. ``text_lookup`` becomes a PromptCache (sorted unique prompts + "", 16 per forward).
        """
        self.text_lookup = PromptCache(self.text_lookup, sorted(set(prompts) | {""}), batch_size=16)
        self.text_encoder.to("cpu")
        torch.cuda.empty_cache()

    def _caption_dropout(self, prompts: list) -> list:
        """prompts: B str. Returns: B prompts, "" for caption-dropped ones (p = text_dropout_p, training mode).

        Native caption dropout: the empty caption through the same chat template. Transitional: Task 12 moves
        caption dropout into the dataset.
        """
        if self.training and self._text_dropout_p > 0:
            keep = torch.rand(len(prompts)) >= self._text_dropout_p
            prompts = [p if k else "" for p, k in zip(prompts, keep.tolist())]
        return prompts

    def encode_text(self, prompts: list, *, grid, num_frames: int) -> TextCondition:
        """Captions -> zero-padded Qwen3-VL context; each caption keeps its own bucket.

        Args:
            prompts: list[str] of B captions. grid / num_frames: unused.

        Returns:
            TextCondition: context (B, L_max, 20480) zero-padded to the longest bucket, context_mask (B, L_max)
            = position < the caption's bucket length.
        """
        ctxs = self.text_lookup(self._caption_dropout(list(prompts)))
        lens = torch.tensor([c.shape[1] for c in ctxs], device=self.device)
        ctx = torch.zeros(len(ctxs), int(lens.max()), ctxs[0].shape[-1], device=self.device, dtype=self.dtype)
        for i, c in enumerate(ctxs):
            ctx[i, : c.shape[1]] = c[0].to(device=self.device, dtype=self.dtype)
        positions = torch.arange(ctx.shape[1], device=ctx.device).unsqueeze(0)
        return TextCondition(context=ctx, context_mask=positions < lens.unsqueeze(1))

    # ---------------------------------------------------------------- block loop
    def prepare(
        self,
        *,
        latents: Tensor,
        timestep: Tensor,
        text: TextCondition,
        layout,
        proprio: Tensor,
        use_gradient_checkpointing: bool,
        use_gradient_checkpointing_offload: bool,
    ) -> BlockLoopState:
        """Embed T / F / H / P, build per-token modulations and RoPE, stash per-segment metadata.

        Args:
            latents: (B, 96, T, h, w); the first ``layout.num_hist_latents`` are the clean history (H stream).
            timestep: (B,) = 1000 t.
            text: TextCondition of the text alone (flux runs with ``proprio_input=backbone_stream``, so the
                architecture appends no proprio token); context_mask gives each caption's bucket length.
            layout: ``cfg.common.layout`` (clean prefix length = the H stream).
            proprio: (B, H, state_dim) normalized proprio frames, embedded as the P stream.
            use_gradient_checkpointing(_offload): bool loop flags.

        Returns:
            BlockLoopState.
        """
        dit = self.dit
        b, _, t_lat, h, w = latents.shape
        n_hist = int(layout.num_hist_latents)
        ctx = text.context
        text_len = int(ctx.shape[1])
        # Each caption's own bucket length (the mask is a prefix of True).
        seq_lens = text.context_mask.sum(dim=1)
        # One P token per history frame.
        if tuple(proprio.shape) != (b, int(layout.num_proprio_tokens), self.state_dim):
            raise ValueError(
                f"flux proprio must be (B, {layout.num_proprio_tokens}, {self.state_dim}), got {tuple(proprio.shape)}"
            )
        prop = proprio
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
        t = (timestep.float() / 1000.0).to(dev)
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
            "text_valid": seq_lens.to(dev),
        }
        return BlockLoopState(
            hidden_states=hidden,
            time_mod=torch.zeros((), device=dev, dtype=dtype),
            rope_freqs=torch.zeros((), device=dev),
            context=ctx,
            grid_frames=t_lat,
            grid_height=h,
            grid_width=w,
            use_gradient_checkpointing=bool(use_gradient_checkpointing),
            use_gradient_checkpointing_offload=bool(use_gradient_checkpointing_offload),
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

    def build_joint_attention_masks(self, frame_mask: Tensor, state: BlockLoopState, s_action: int) -> list[Tensor]:
        """Per-layer masks over ``[T | F | H | P | A]`` (see module docstring and ``build_flux_layer_masks``).

        Args:
            frame_mask: (T + 1, T + 1) bool frame mask. state: BlockLoopState (segment lengths, text lengths).
            s_action: int, action tokens.

        Returns:
            list of ``num_layers`` masks.
        """
        n_hist = int(state.extras["n_hist"])
        if int(state.grid_frames) != int(frame_mask.shape[0]) - 1:
            raise ValueError(
                f"the video stream has {state.grid_frames} latent frames, layout says {frame_mask.shape[0] - 1}"
            )
        return build_flux_layer_masks(
            frame_mask,
            state.extras["seg_lens"],
            int(s_action),
            state.extras["text_valid"],
            n_hist=n_hist,
            num_mode_layers=_N_MODE,
            num_layers=self.num_layers,
            device=state.hidden_states.device,
            strict_clean_rows=self.strict_clean_rows,
        )

    def run_block(self, block_id: int, state: BlockLoopState, attn_mask: Tensor) -> BlockLoopState:
        """Video-only layer (no action tokens).

        Args:
            block_id: int. state: BlockLoopState. attn_mask: this layer's mask (``s_action = 0``).

        Returns:
            ``state`` with ``hidden_states`` updated.
        """
        q, k, v, post = self.pre_attn_at_layer(block_id, state)
        heads = self.num_heads
        split = lambda t: rearrange(t, "b s (n d) -> b n s d", n=heads)  # noqa: E731
        out = F.scaled_dot_product_attention(split(q), split(k), split(v), attn_mask=attn_mask)
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
def _flux_dit(geometry: DictConfig, action_cond_dim: int, dtype: torch.dtype) -> JointSingleSeq:
    """The trunk module with streams video / video_cond / action_cond (random init; caller loads weights).

    Args:
        geometry: the yaml ``dit`` block (JointSingleSeqParams fields other than the stream wiring).
        action_cond_dim: int, P input width.
        dtype: parameter dtype (set as the default dtype while building).

    Returns:
        JointSingleSeq.
    """
    params = JointSingleSeqParams(
        in_channels={"video": _LATENT_CHANNELS, "video_cond": _LATENT_CHANNELS, _P_STREAM: int(action_cond_dim)},
        sequence={"x_video": "video", "x_video_cond": "video_cond", f"x_{_P_STREAM}": _P_STREAM},
        **OmegaConf.to_container(geometry),
    )
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        return JointSingleSeq(params)
    finally:
        torch.set_default_dtype(prev)


def build_flux_dit(
    path: str, geometry: DictConfig, *, action_cond_dim: int, head_seed: int = 0, dtype: torch.dtype = torch.bfloat16
) -> JointSingleSeq:
    """JointSingleSeq with streams video / video_cond / action_cond, loaded from the action-base trunk.

    ``action_prediction_cond`` mode blocks + modulations are routed to ``action_cond`` (flux wiring); the
    ``action_cond`` input Linear is fresh (xavier, seeded; the trunk's ee50 / gaming heads are 50 / 64 wide)
    and its unused final layer zero. The trunk's action-prediction stream and heads are dropped.

    Args:
        path: flux-3-action-base.safetensors.
        geometry: the yaml ``dit`` block the trunk is built from.
        action_cond_dim: int, P input width.
        head_seed: int, seed of the fresh input Linear.
        dtype: parameter dtype.

    Returns:
        JointSingleSeq in eval mode on CPU. Raises ValueError when the file does not match ``geometry``.
    """
    from safetensors import safe_open

    model = _flux_dit(geometry, action_cond_dim, dtype)
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
    # A geometry the file does not have shows up as missing / unexpected keys or a shape mismatch.
    expected = model.state_dict()
    bad_shapes = [k for k, v in sd.items() if k in expected and tuple(v.shape) != tuple(expected[k].shape)]
    if bad_shapes:
        raise ValueError(f"flux trunk does not match the yaml dit={dict(geometry)}: shape mismatch on {bad_shapes[:6]}")
    res = model.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=False)
    fresh = {f"emb_in.{_P_STREAM}.weight", f"final_layer.{_P_STREAM}.linear.weight",
             f"final_layer.{_P_STREAM}.adaLN_modulation.1.weight"}
    if set(res.missing_keys) - fresh or res.unexpected_keys:
        raise ValueError(f"flux trunk does not match the yaml dit={dict(geometry)}: missing "
                         f"{sorted(set(res.missing_keys) - fresh)[:6]}, unexpected {res.unexpected_keys[:6]}")
    # Reference init of a fresh embodiment head (wiring.fresh_head_state_dict): xavier input, zero output.
    gen = torch.Generator().manual_seed(head_seed)
    weight = torch.empty(model.hidden_size, int(action_cond_dim))
    nn.init.xavier_uniform_(weight, generator=gen)
    with torch.no_grad():
        model.emb_in[_P_STREAM].weight.copy_(weight.to(dtype))
        for p in model.final_layer[_P_STREAM].parameters():
            p.zero_()
    model.eval()
    return model


def _flux_vae_params():
    """Returns: the ViTNormInferenceParams OpenWAM runs the flux VAE with (no compile, looped 8-frame decode)."""
    from openwam.model.video_backbone.flux.vae import ViTNormInferenceParams

    return ViTNormInferenceParams(use_compile=False, chunked_encode=False, decoder_max_t=8)


def build_flux_vae(path: str) -> nn.Module:
    """Frozen ViTNorm video VAE (``ViTNormInference``, bf16) from ``video_vae.safetensors``.

    Args:
        path: str, video_vae.safetensors.

    Returns:
        ViTNormInference in eval mode, every parameter frozen.
    """
    from safetensors.torch import load_file

    from openwam.model.video_backbone.flux.vae import ViTNormInference

    vae = ViTNormInference(_flux_vae_params())
    vae.load_state_dict(load_file(path, device="cpu"))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


__all__ = [
    "FluxActionVideoBackbone",
    "build_flux_dit",
    "build_flux_vae",
]
