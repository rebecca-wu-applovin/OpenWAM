"""GE-Base-fast (Genie Envisioner, AgiBot) video backbone — 07_sparse_history_backbones.

Components (all GE-native):
  * DiT: GE's multi-view LTX transformer (28 x 2048, 32 heads x 64, patch 1),
    weights ``GE_base_fast_v0.1.safetensors``; vendored in ``ge/transformer.py``.
  * VAE: LTX-Video ``AutoencoderKLLTXVideo`` (8x temporal causal, 32x spatial,
    128 ch), latents normalized with the VAE's ``latents_mean/std``.
  * Text: LTX-Video T5-XXL encoder, 128 tokens. Caption dropout is the dataset's
    (``dataloader.caption_dropout_p``; GE's 6% is set by the GE launch script).

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

Construction (everything from ``cfg = cfg.model.video_backbone``):
  * :meth:`GEBaseFastVideoBackbone.from_pretrained`: the DiT from the yaml ``dit`` block + the GE weights file,
    VAE / T5 / tokenizer with their libraries' ``from_pretrained`` from ``ltx_path``;
  * :meth:`GEBaseFastVideoBackbone.from_checkpoint_dir`: meta shells (DiT from the yaml, VAE / T5 from the
    ``config.json`` files :meth:`save_deploy_assets` copied into ``<ckpt>/ltx/``), tokenizer from the same copy.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import List, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from openwam.model.video_backbone.base import BlockLoopState, TextCondition, VideoBackbone
from openwam.model.video_backbone.common.weights import load_strict
from openwam.model.video_backbone.ge.scheduler import GEFlowScheduler

logger = logging.getLogger(__name__)

_GE_TEMPORAL_COMPRESSION = 8
_GE_SPATIAL_COMPRESSION = 32
_GE_LATENT_CHANNELS = 128
_GE_FRAME_RATE = 30.0  # GE's forward_pass hard-codes frame_rate=30 for the RoPE time scale


# LTX-Video files a checkpoint needs to rebuild the frozen VAE / T5 / tokenizer without the weights
# (copied into <ckpt>/ltx/ by save_deploy_assets, paths relative to ltx_path).
_CKPT_ASSET_SUBDIR = "ltx"
_LTX_ASSETS = (os.path.join("vae", "config.json"), os.path.join("text_encoder", "config.json"), "tokenizer")


class GEBaseFastVideoBackbone(VideoBackbone):
    """GE multi-view LTX transformer + LTX VAE + T5 behind the VideoBackbone ABC."""

    # Prompts per T5 forward while build_text_cache fills the cache.
    _text_cache_batch_size = 8

    # ================================================================
    # Construction
    # ================================================================

    def __init__(
        self,
        *,
        dit,
        vae,
        tokenizer,
        text_encoder,
        assets_dir: str,
        shift_video: float = 1.0,
        n_view: int = 2,
        text_max_length: int = 128,
        vae_sample: bool = True,
    ) -> None:
        """Wrap built modules; use :meth:`from_pretrained` / :meth:`from_checkpoint_dir` instead.

        Args:
            dit: LTXVideoTransformer3DModel. vae: AutoencoderKLLTXVideo. tokenizer: T5Tokenizer. text_encoder: T5.
            assets_dir: str, LTX-Video layout dir the VAE / T5 configs and tokenizer were read from
                (``ltx_path`` or ``<ckpt>/ltx``); save_deploy_assets copies from it.
            shift_video: float, deploy-grid α-shift (GE trains on the shift-1 grid).
            n_view: int, camera tiles stacked in the canvas. text_max_length: int, T5 tokens.
            vae_sample: bool, sample the VAE posterior.
        """
        super().__init__()
        # Flat named children so the model yaml freeze list hits vae / text_encoder.
        self.dit = dit
        self.vae = vae
        self.text_encoder = text_encoder
        self._tokenizer = tokenizer
        self._assets_dir = assets_dir
        cfg = dit.config
        self._dim = int(cfg.num_attention_heads * cfg.attention_head_dim)
        self._num_layers = int(cfg.num_layers)
        self._num_heads = int(cfg.num_attention_heads)
        self._head_dim = int(cfg.attention_head_dim)
        self._n_view = int(n_view)
        self._text_max_length = int(text_max_length)
        self._vae_sample = bool(vae_sample)
        # GE trains with diffusers' default FlowMatchEulerDiscreteScheduler (shift 1): with the yaml's
        # shift_video 1.0 the adapter's 1000-step grid sigma_i = 1 - i/1000 is the same grid (used by deploy);
        # its sample_train is GE's native training-noise draw.
        self._shift_video = float(shift_video)
        self._scheduler = GEFlowScheduler(shift_video=self._shift_video)
        self._temporal_compression = _GE_TEMPORAL_COMPRESSION
        self._causal_temporal = True
        # Text slot read by encode_text: the online T5 here, a PromptCache after build_text_cache.
        self.text_lookup = self._t5_entries
        self._mask_cache: dict = {}

    @classmethod
    def from_pretrained(cls, cfg: DictConfig) -> "GEBaseFastVideoBackbone":
        """Build from the pretrained GE / LTX weights.

        Args:
            cfg: ``cfg.model.video_backbone``: ``dit`` geometry, ``model_path`` + ``weights_file`` (GE DiT),
                ``ltx_path`` (LTX-Video dir: vae/, text_encoder/, tokenizer/) and the scalar knobs.

        Returns:
            GEBaseFastVideoBackbone on CPU in bf16. Raises ValueError when the DiT weights do not match the
            yaml ``dit`` block.
        """
        from diffusers import AutoencoderKLLTXVideo
        from safetensors.torch import load_file
        from transformers import T5EncoderModel, T5Tokenizer

        from openwam.model.video_backbone.ge.dit import LTXVideoTransformer3DModel

        ltx_path = str(cfg.ltx_path)
        # Real (random) init before the load: its RNG draws are part of the training seed stream.
        dit = LTXVideoTransformer3DModel(**OmegaConf.to_container(cfg.dit))
        state = load_file(os.path.join(str(cfg.model_path), str(cfg.weights_file)))
        try:
            load_strict(dit, state, assign=False)  # 718 tensors, no action expert
        except ValueError as err:
            raise ValueError(f"GE DiT weights do not match the yaml dit={dict(cfg.dit)}: {err}") from err
        dit.to(torch.bfloat16)
        # The frozen LTX-Video VAE / T5 / tokenizer come from their libraries' own loaders.
        vae = AutoencoderKLLTXVideo.from_pretrained(ltx_path, subfolder="vae", torch_dtype=torch.bfloat16)
        tokenizer = T5Tokenizer.from_pretrained(ltx_path, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(ltx_path, subfolder="text_encoder", torch_dtype=torch.bfloat16)
        return cls._assemble(cfg, dit, vae, tokenizer, text_encoder, ltx_path)

    @classmethod
    def from_checkpoint_dir(cls, cfg: DictConfig, ckpt_dir: str) -> "GEBaseFastVideoBackbone":
        """Build the empty skeleton of a saved checkpoint; the caller loads its weights.

        Args:
            cfg: the checkpoint's ``cfg.model.video_backbone`` (DiT geometry + scalar knobs).
            ckpt_dir: str, checkpoint dir holding ``ltx/{vae,text_encoder}/config.json`` and ``ltx/tokenizer/``.

        Returns:
            GEBaseFastVideoBackbone whose parameters are all on ``meta`` (bf16).
        """
        from accelerate import init_empty_weights
        from diffusers import AutoencoderKLLTXVideo
        from transformers import T5Config, T5EncoderModel, T5Tokenizer

        from openwam.model.video_backbone.ge.dit import LTXVideoTransformer3DModel

        # The VAE / T5 configs and the tokenizer were copied into <ckpt>/ltx/ by save_deploy_assets.
        assets = os.path.join(ckpt_dir, _CKPT_ASSET_SUBDIR)
        if not os.path.isdir(assets):
            raise FileNotFoundError(f"GE assets not found at {assets} (written by save_deploy_assets)")
        vae_config = AutoencoderKLLTXVideo.load_config(assets, subfolder="vae")
        t5_config = T5Config.from_pretrained(assets, subfolder="text_encoder")
        # Buffers stay real (include_buffers=False): the persistent ones are overwritten by the load anyway.
        with init_empty_weights(include_buffers=False):
            dit = LTXVideoTransformer3DModel(**OmegaConf.to_container(cfg.dit))
            vae = AutoencoderKLLTXVideo.from_config(vae_config)
            text_encoder = T5EncoderModel(t5_config)
        # Tie shared / encoder.embed_tokens as from_pretrained does (one tensor; the strict load keeps the tie).
        text_encoder.tie_weights()
        tokenizer = T5Tokenizer.from_pretrained(assets, subfolder="tokenizer")
        modules = [m.to(dtype=torch.bfloat16) for m in (dit, vae, text_encoder)]
        return cls._assemble(cfg, modules[0], modules[1], tokenizer, modules[2], assets)

    @classmethod
    def _assemble(
        cls, cfg: DictConfig, dit, vae, tokenizer, text_encoder, assets_dir: str
    ) -> "GEBaseFastVideoBackbone":
        """Shared tail of both constructors: wrap the modules with the yaml's scalar knobs.

        Args:
            cfg: ``cfg.model.video_backbone``. dit / vae / tokenizer / text_encoder: built components.
            assets_dir: str, the LTX-Video layout dir they were read from.

        Returns:
            GEBaseFastVideoBackbone.
        """
        return cls(
            dit=dit,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            assets_dir=assets_dir,
            shift_video=float(cfg.shift_video),
            n_view=int(cfg.n_view),
            text_max_length=int(cfg.text_max_length),
            vae_sample=bool(cfg.vae_sample),
        )

    # ================================================================
    # Geometry
    # ================================================================

    @property
    def dim(self) -> int:
        """DiT hidden width, heads * head_dim (2048)."""
        return self._dim

    @property
    def num_layers(self) -> int:
        """Number of transformer blocks (28)."""
        return self._num_layers

    @property
    def num_heads(self) -> int:
        """Attention heads per block (32)."""
        return self._num_heads

    @property
    def head_dim(self) -> int:
        """Per-head width (64)."""
        return self._head_dim

    @property
    def text_dim(self) -> int:
        """T5-XXL hidden width (the context the framework appends proprio to)."""
        return int(self.dit.config.caption_channels)

    @property
    def scheduler(self):
        """GEFlowScheduler: the shared grid for deploy, GE's native training-noise law for ``sample_train``."""
        return self._scheduler

    # ================================================================
    # Video and text IO
    # ================================================================

    def encode_pixels(self, pixels: Tensor) -> Tensor:
        """Each view tile of each canvas clip encoded as its own causal clip.

        Args:
            pixels: (N, 3, T, V*h_px, W) canvas pixels in [-1, 1], T = 1 + 8k.

        Returns: (N, 128, 1 + (T-1)//8, V*h, w) latents, views stacked along height.
        """
        # One VAE batch of N*V tiles, clip-major (the base encode_frames' N = B*T 1-frame clips: sample, frame, view).
        lat = self._vae_encode(self._split_views(pixels))
        return rearrange(lat, "(b v) c t h w -> b c t (v h) w", b=pixels.shape[0])

    def decode_latents(self, latents: Tensor) -> Tensor:
        """latents: (N, 128, T, V*h, w) normalized. Returns: (N, 3, 1 + 8 (T-1), V*h_px, W) canvas pixels, views stacked."""
        vae = self.vae
        lat = rearrange(latents, "b c t (v h) w -> (b v) c t h w", v=self._n_view)
        mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(lat.device, lat.dtype)
        std = vae.latents_std.view(1, -1, 1, 1, 1).to(lat.device, lat.dtype)
        lat = lat * std / float(vae.config.scaling_factor) + mean
        video = vae.decode(lat.to(next(vae.parameters()).dtype)).sample  # (N*V, 3, T, h, w)
        return rearrange(video, "(b v) c t h w -> b c t (v h) w", v=self._n_view)

    def _split_views(self, frames: Tensor) -> Tensor:
        """frames: (B, 3, T, V*h_px, W) canvas. Returns: (B*V, 3, T, h_px, W) per-view clips (top tile = view 0)."""
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

    def encode_text(self, prompts: list, *, grid, num_frames: int) -> TextCondition:
        """T5 context of B prompts (through ``text_lookup``: online or cached).

        Args:
            prompts: list[str] of B prompts. grid / num_frames: unused.

        Returns:
            TextCondition: context (B, 128, 4096), context_mask (B, 128) the tokenizer attention mask, extra {}.
        """
        entries = self.text_lookup(list(prompts))
        emb = torch.cat([e[0] for e in entries], dim=0)
        mask = torch.cat([e[1] for e in entries], dim=0)
        return TextCondition(context=emb, context_mask=mask, extra={})

    def _t5_entries(self, prompts: List[str]) -> list:
        """Online T5 encode (the default ``text_lookup``).

        Args:
            prompts: list[str] of B prompts.

        Returns:
            list of B entries ``(embedding (1, 128, 4096), mask (1, 128) bool)`` on ``self.device``.
        """
        emb, mask = self._t5(prompts)
        return [(emb[j : j + 1], mask[j : j + 1]) for j in range(len(prompts))]

    @torch.no_grad()
    def _t5(self, prompts: List[str]) -> Tuple[Tensor, Tensor]:
        """GE ``_encode_prompt``: T5 over a padded batch.

        Args:
            prompts: list[str] of B prompts.

        Returns:
            ((B, 128, 4096) embeddings in the backbone dtype, (B, 128) bool attention mask), on ``self.device``.
        """
        # Pad / truncate to text_max_length tokens; T5 runs wherever it is parked (GPU, or CPU after the cache).
        tok = self._tokenizer(
            prompts,
            padding="max_length",
            max_length=self._text_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        dev = next(self.text_encoder.parameters()).device
        emb = self.text_encoder(tok.input_ids.to(dev))[0]
        return emb.to(device=self.device, dtype=self.dtype), tok.attention_mask.bool().to(self.device)

    # ================================================================
    # Block loop
    # ================================================================

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
        """Patchify (patch 1: one token per latent pixel), per-token time / view modulation, RoPE, text.

        Args:
            latents: (B, 128, T, V*h, w). timestep: (B,) in [0, 1000].
            text: TextCondition, context (B, L, 4096) (text, + proprio appended by the framework), context_mask (B, L).
            layout: ``cfg.common.layout``; the first ``num_hist_latents`` latents are clean (their tokens get t = 0,
                GE pixel_wise_timestep).
            proprio: (B, H, D) normalized proprio; unused (context tokens). use_gradient_checkpointing(_offload):
                bool loop flags.

        Returns:
            BlockLoopState.
        """
        dit = self.dit
        b = latents.shape[0]
        t_lat, vh, vw = self._grid(latents)
        n_view = self._n_view
        tpf = n_view * vh * vw
        n_clean = int(layout.num_hist_latents)

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
        context = dit.caption_projection(text.context.to(hidden.dtype))
        cmask = text.context_mask.to(device=context.device, dtype=torch.bool)

        return BlockLoopState(
            hidden_states=hidden,
            time_mod=temb,
            rope_freqs=torch.zeros((), device=hidden.device),
            context=context,
            context_mask=cmask,
            grid_frames=t_lat,
            grid_height=n_view * vh,
            grid_width=vw,
            use_gradient_checkpointing=bool(use_gradient_checkpointing),
            use_gradient_checkpointing_offload=bool(use_gradient_checkpointing_offload),
            extras={"cos": cos, "sin": sin, "emb_t": emb_t, "view_h": vh},
        )

    def run_block(self, block_id: int, state: BlockLoopState, attn_mask: Tensor) -> BlockLoopState:
        """Video-only block (no action stream).

        Args:
            block_id: int. state: BlockLoopState. attn_mask: (S, S) bool mask of this layer.

        Returns:
            ``state`` with ``hidden_states`` updated.
        """

        def _body(x):
            """x: (B, S, dim) hidden states. Returns: (B, S, dim) after this block (checkpointable closure)."""
            # A state copy carrying x, so the checkpointed closure has no side effect on ``state``.
            local = BlockLoopState(**{**state.__dict__, "hidden_states": x})
            q, k, v, post = self.pre_attn_at_layer(block_id, local)
            # (B, S, H*D) -> (B, H, S, D) for SDPA, and back.
            n = self._num_heads
            q, k, v = (t.unflatten(2, (n, -1)).transpose(1, 2) for t in (q, k, v))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask).transpose(1, 2).flatten(2, 3)
            return self.post_attn_at_layer(block_id, local, out, post).hidden_states

        # Checkpoint only when training with gradients; inference runs the body directly.
        if state.use_gradient_checkpointing and torch.is_grad_enabled():
            state.hidden_states = torch.utils.checkpoint.checkpoint(_body, state.hidden_states, use_reentrant=False)
        else:
            state.hidden_states = _body(state.hidden_states)
        return state

    def finalize(self, state: BlockLoopState) -> Tensor:
        """Final AdaLN (time + view embedding) + proj_out.

        Args:
            state: BlockLoopState after the last block (``extras["emb_t"]`` from :meth:`prepare`).

        Returns:
            (B, 128, T, V*h, w) predicted velocity.
        """
        dit = self.dit
        ss = dit.scale_shift_table[None, None] + state.extras["emb_t"][:, :, None]
        shift, scale = ss[:, :, 0], ss[:, :, 1]
        out = dit.proj_out(dit.norm_out(state.hidden_states) * (1 + scale) + shift)
        return rearrange(out, "b (t h w) c -> b c t h w", t=state.grid_frames, h=state.grid_height, w=state.grid_width)

    def pre_attn_at_layer(self, layer_id: int, state: BlockLoopState) -> Tuple[Tensor, Tensor, Tensor, dict]:
        """norm1 + AdaLN → attn1 q/k/v (RMS-normed across heads, RoPE'd).

        Args:
            layer_id: int. state: BlockLoopState.

        Returns:
            q, k, v each (B, S, 2048), and the post dict (gates / MLP modulation) :meth:`post_attn_at_layer` reads.
        """
        from openwam.model.video_backbone.ge.dit import apply_rotary_emb

        block = self.dit.transformer_blocks[layer_id]
        # Per-token AdaLN (time + view), then attn1's QK RMS-norm and RoPE (each view on its own grid).
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._ada(layer_id, state)
        x = block.norm1(state.hidden_states) * (1 + scale_msa) + shift_msa
        attn = block.attn1
        rope = (state.extras["cos"], state.extras["sin"])
        q = apply_rotary_emb(attn.norm_q(attn.to_q(x)), rope)
        k = apply_rotary_emb(attn.norm_k(attn.to_k(x)), rope)
        v = attn.to_v(x)
        post = {"gate_msa": gate_msa, "shift_mlp": shift_mlp, "scale_mlp": scale_mlp, "gate_mlp": gate_mlp}
        return q, k, v, post

    def post_attn_at_layer(
        self, layer_id: int, state: BlockLoopState, attn_out: Tensor, post_state: dict
    ) -> BlockLoopState:
        """attn1 out-proj + gate, text cross-attn (attn2, no pre-norm, as GE), gated FFN.

        Args:
            layer_id: int. state: BlockLoopState. attn_out: (B, S, 2048) attention output. post_state: from
                :meth:`pre_attn_at_layer`.

        Returns:
            ``state`` with ``hidden_states`` updated.
        """
        block = self.dit.transformer_blocks[layer_id]
        x = state.hidden_states + block.attn1.to_out[1](block.attn1.to_out[0](attn_out)) * post_state["gate_msa"]
        x = x + self._cross_attn(block.attn2, x, state.context, state.context_mask)
        h = block.norm2(x) * (1 + post_state["scale_mlp"]) + post_state["shift_mlp"]
        state.hidden_states = x + block.ff(h) * post_state["gate_mlp"]
        return state

    def _grid(self, latents: Tensor) -> Tuple[int, int, int]:
        """latents: (B, C, T, V*h, w). Returns: (T, h_view, w); raises ValueError when V*h does not split into V views."""
        t, vh, w = latents.shape[2:]
        if vh % self._n_view:
            raise ValueError(f"latent height {vh} does not split into {self._n_view} views")
        return t, vh // self._n_view, w

    def _ada(self, layer_id: int, state: BlockLoopState):
        """layer_id: int. state: BlockLoopState (``time_mod`` (B, S, 6*dim)). Returns: six AdaLN tensors, each (B, S, dim)."""
        block = self.dit.transformer_blocks[layer_id]
        b, s = state.hidden_states.shape[:2]
        ada = block.scale_shift_table[None, None] + state.time_mod.reshape(b, s, 6, -1)
        return ada.unbind(dim=2)

    def _cross_attn(self, attn, x: Tensor, context: Tensor, mask: Tensor) -> Tensor:
        """All views' queries of a sample attend that sample's text (+ proprio) tokens.

        Args:
            attn: the block's attn2. x: (B, S, dim) queries. context: (B, L, dim) projected text.
            mask: (B, L) bool, True = attend.

        Returns:
            (B, S, dim) cross-attention output (after out-proj).
        """
        heads = attn.heads
        q = attn.norm_q(attn.to_q(x)).unflatten(2, (heads, -1)).transpose(1, 2)
        k = attn.norm_k(attn.to_k(context)).unflatten(2, (heads, -1)).transpose(1, 2)
        v = attn.to_v(context).unflatten(2, (heads, -1)).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
        out = out.transpose(1, 2).flatten(2, 3).to(q.dtype)
        return attn.to_out[1](attn.to_out[0](out))

    def build_joint_attention_masks(self, frame_mask: Tensor, state: BlockLoopState, s_action: int) -> list:
        """One (S_v + S_a)^2 bool mask per layer: the frame mask with GE's per-view / cross-view block pattern.

        Args:
            frame_mask: (T + 1, T + 1) bool frame mask. state: BlockLoopState (grid, ``view_h``). s_action: int.

        Returns:
            list of ``num_layers`` masks (cached per grid / action length / device).
        """
        from openwam.model.video_backbone.base import frame_tokens
        from openwam.model.video_backbone.ge.masks import build_ge_joint_masks

        frame_tokens(frame_mask, state, int(state.hidden_states.shape[1]))  # latent count must match the layout
        # The frame mask is fixed per architecture: the grid / action length / device identify the masks.
        key = (
            state.grid_frames,
            state.extras["view_h"],
            state.grid_width,
            int(s_action),
            str(state.hidden_states.device),
        )
        if key not in self._mask_cache:
            self._mask_cache[key] = build_ge_joint_masks(
                frame_mask,
                n_view=self._n_view,
                view_h=state.extras["view_h"],
                view_w=state.grid_width,
                s_action=int(s_action),
                num_layers=self._num_layers,
                device=state.hidden_states.device,
            )
        return self._mask_cache[key]

    # ================================================================
    # Deploy assets
    # ================================================================

    def save_deploy_assets(self, output_dir: str) -> None:
        """Copy the VAE / T5 ``config.json`` and the tokenizer into ``<output_dir>/ltx/`` (from_checkpoint_dir reads them).

        Args:
            output_dir: str, checkpoint dir. Files come from wherever this backbone read them (ltx_path or a ckpt).

        Returns:
            None. Files already present are left as they are.
        """
        # Per asset: skip what a previous save already copied.
        for rel in _LTX_ASSETS:
            src = os.path.join(self._assets_dir, rel)
            dst = os.path.join(output_dir, _CKPT_ASSET_SUBDIR, rel)
            if os.path.exists(dst):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            # The tokenizer is a directory of side files; the two configs are single files.
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)


__all__ = ["GEBaseFastVideoBackbone"]
