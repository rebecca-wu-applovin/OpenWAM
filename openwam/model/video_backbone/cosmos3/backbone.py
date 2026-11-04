"""Cosmos3-Edge video backbone.

Implements the :class:`VideoBackbone` contract on NVIDIA's Cosmos3-Edge unified
dual-stream transformer (vendored in ``cosmos3/_vendor/``) — flat named children
(``self.dit`` / ``self.vae``) mirroring the other families. The heavy batched
block-loop lives in ``cosmos3/dit.py``; prompt tokenization + mRoPE
packing in ``cosmos3/text.py``.

Construction (everything from ``cfg = cfg.model.video_backbone``; Edge and Nano differ only in the yaml):
  - :meth:`Cosmos3EdgeVideoBackbone.from_pretrained`: the diffusers-style bundle at ``model_path``
    (``transformer/`` + ``vae/`` + ``text_tokenizer/``), checked against the yaml ``dit`` / ``vae`` blocks;
  - :meth:`Cosmos3EdgeVideoBackbone.from_checkpoint_dir`: meta shells built from those yaml blocks, tokenizer
    from the checkpoint's ``text_tokenizer/`` (copied there by :meth:`save_deploy_assets`).
``diffusers`` / ``transformers`` are imported inside the constructors, so this module stays CPU-CI safe.

The und (text) stream is causal, frozen, and computationally independent of the gen (video)
stream, so :meth:`encode_text` runs the whole und tower once
under ``no_grad`` and caches the per-layer gen-facing K/V; the und final hidden
(2048-wide) doubles as the ``context`` tensor the action stream consumes
(``text_dim == 2048``). There is no cross-attention and no external text
encoder; timestep conditioning is additive on noisy-frame tokens only.

Supported architectures: ``dual_system`` / {``joint_cross_attn``,
``joint_self_attn``}. ``joint_self_attn`` rides MoT via the und-prefix-KV
declaration + GQA KV-expand.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any, List, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from openwam.model.common.flow_scheduler import FlowScheduler
from openwam.model.video_backbone.base import BlockLoopState, TextCondition, VideoBackbone
from openwam.model.video_backbone.cosmos3 import dit as cosmos3_dit
from openwam.model.video_backbone.cosmos3 import text as cosmos3_text

logger = logging.getLogger(__name__)

# Cosmos3-Edge native geometry (transformer/config.json; VAE = Wan2.2-TI2V).
_COSMOS3_TEMPORAL_COMPRESSION: int = 4
_COSMOS3_SPATIAL_COMPRESSION: int = 16
_COSMOS3_CAUSAL_TEMPORAL: bool = True

_COSMOS3_INSTALL_HINT = (
    "cosmos3 needs `diffusers>=0.38` (for AutoencoderKLWan and the vendored transformer's support imports) "
    "and `transformers` for the tokenizer. Install with: pip install 'diffusers>=0.38,<1'."
)
# Tokenizer dir: ``text_tokenizer/`` in the bundle and, copied by save_deploy_assets, in a checkpoint dir.
_TOKENIZER_SUBDIR = "text_tokenizer"
# The tokenizer's structural files (the only non-weight assets a checkpoint needs; dit / vae come from the yaml).
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")
# und (text) pathway children frozen at build: they run under no_grad in the backbone's preprocess.
_UND_PATHWAY_LAYER_CHILDREN = ("input_layernorm", "post_attention_layernorm", "mlp")
_UND_PATHWAY_ATTN_CHILDREN = ("to_q", "to_k", "to_v", "to_out", "norm_q", "norm_k", "k_norm_und_for_gen")


def _vendor():
    """Lazy-import the vendored transformer module (pulls diffusers). Returns: the module."""
    try:
        from openwam.model.video_backbone.cosmos3._vendor import transformer_cosmos3
    except ImportError as exc:  # diffusers missing or too old
        raise ImportError(_COSMOS3_INSTALL_HINT) from exc
    return transformer_cosmos3


def _check_cfg(cfg: DictConfig) -> None:
    """Reject backbone yaml values the code cannot honour, before any heavy import.

    Args:
        cfg: ``cfg.model.video_backbone``.

    Returns:
        None. Raises ValueError / NotImplementedError naming the bad key.
    """
    # The clip fps scales the prompt template and the vision mRoPE time axis.
    if float(cfg.fps) <= 0.0:
        raise ValueError(f"cosmos3 fps must be > 0, got {cfg.fps!r}.")
    # Room for the chat-template tokens around the prompt.
    if int(cfg.max_text_tokens) < 8:
        raise ValueError(f"cosmos3 max_text_tokens must be >= 8, got {cfg.max_text_tokens}.")


def _norm_cfg_value(value):
    """Comparable form of a config value: lists recursed, floats rounded to 12 digits (yaml vs json parsing)."""
    if isinstance(value, (list, tuple)):
        return [_norm_cfg_value(v) for v in value]
    if isinstance(value, float):
        return round(value, 12)
    return value


def _check_against_yaml(config, declared: DictConfig, name: str) -> None:
    """Compare a module loaded from the bundle with the yaml block it must equal.

    Args:
        config: the loaded module's ``.config`` (diffusers FrozenDict).
        declared: the yaml block (``cfg.dit`` / ``cfg.vae``) the checkpoint shell is built from.
        name: that block's key, for the message.

    Returns:
        None. Raises ValueError listing every differing key: weightless fields (rope theta, timestep scale,
        mRoPE margins, ...) change behavior without changing any tensor shape, so a strict load cannot catch them.
    """
    # Only the keys the yaml declares are compared; the bundle may carry more.
    mismatches = {
        key: (config.get(key), value)
        for key, value in OmegaConf.to_container(declared).items()
        if _norm_cfg_value(config.get(key)) != _norm_cfg_value(value)
    }
    if mismatches:
        raise ValueError(
            f"cosmos3 bundle {name} config differs from the yaml {name} block on {mismatches} (bundle, yaml)"
        )


def _freeze_und_pathway(net) -> int:
    """requires_grad_(False) on the und half + shared text embedding.

    Args:
        net: Cosmos3OmniTransformer.

    Returns:
        int, number of frozen parameter elements.
    """
    frozen = 0
    # Shared text embedding and the und final norm.
    for module in [net.embed_tokens, net.norm]:
        for p in module.parameters():
            p.requires_grad_(False)
            frozen += p.numel()
    # Per layer: the und norms / MLP, then the und attention projections.
    for layer in net.layers:
        for name in _UND_PATHWAY_LAYER_CHILDREN:
            for p in getattr(layer, name).parameters():
                p.requires_grad_(False)
                frozen += p.numel()
        # The vendor sets k_norm_und_for_gen to None unless its config enables it (Edge); every other child exists.
        for name in _UND_PATHWAY_ATTN_CHILDREN:
            child = getattr(layer.self_attn, name)
            if child is not None:
                for p in child.parameters():
                    p.requires_grad_(False)
                    frozen += p.numel()
    return frozen


def _drop_lm_head(net) -> None:
    """Remove ``lm_head`` (268M params, unused by the diffusion forward) so it never enters checkpoints.

    Args:
        net: Cosmos3OmniTransformer (modified in place).

    Returns:
        None.
    """
    delattr(net, "lm_head")
    object.__setattr__(net, "lm_head", None)


def _freeze_unused_native_heads(net) -> int:
    """Freeze the Omni action/sound heads OpenWAM's video-backbone path never calls.

    Its action stream is the external ActionDiT. They stay registered children (riding the checkpoint
    unchanged keeps a future native-action variant loadable) but, frozen, they occupy no optimizer state.

    Args:
        net: Cosmos3OmniTransformer.

    Returns:
        int, number of frozen parameter elements.
    """
    # The vendor builds each head only when its config switch is on.
    heads = []
    if net.config.action_gen:
        heads += [net.action_proj_in, net.action_proj_out, net.action_modality_embed]
    if net.config.sound_gen:
        heads += [net.audio_proj_in, net.audio_proj_out, net.audio_modality_embed]
    frozen = 0
    for head in heads:
        # A module's parameters, or the bare modality-embedding parameter itself.
        for p in head.parameters() if isinstance(head, torch.nn.Module) else [head]:
            p.requires_grad_(False)
            frozen += p.numel()
    return frozen


class Cosmos3EdgeVideoBackbone(VideoBackbone):
    """Wrap the Cosmos3-Edge generator + Wan2.2 VAE + tokenizer behind the ABC."""

    # ================================================================
    # Construction
    # ================================================================

    def __init__(
        self,
        *,
        net: Any,
        vae: Any,
        tokenizer: Any,
        tokenizer_dir: str,
        dim: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        context_dim: int,
        latents_mean: Tensor,
        latents_std: Tensor,
        shift_video: float,
        train_timestep: DictConfig,
        use_system_prompt: bool,
        prompt_templates: bool,
        duration_template: bool,
        clip_fps: float,
        max_text_tokens: int,
    ) -> None:
        """Wrap built modules; use :meth:`from_pretrained` / :meth:`from_checkpoint_dir` instead.

        Args:
            net: Cosmos3OmniTransformer (und + gen streams). vae: AutoencoderKLWan.
            tokenizer: PreTrainedTokenizerFast. tokenizer_dir: str, the dir it was read from.
            dim / num_layers / num_heads / head_dim: int gen-stream geometry (yaml ``dit``).
            context_dim: int, und hidden size (the action stream's context width).
            latents_mean / latents_std: (48,) VAE latent normalization (yaml ``vae.latents_mean`` / ``latents_std``).
            shift_video: float, α-shift of the scheduler built here. train_timestep: DictConfig, its training law.
            use_system_prompt / prompt_templates / duration_template: bool prompt knobs.
            clip_fps: float, assumed clip frame rate (yaml ``fps``). max_text_tokens: int.
        """
        super().__init__()
        # --- Flat named children (freeze list hits video_backbone.vae) ---
        self.dit = net
        self.vae = vae
        self._tokenizer = tokenizer
        # Where the tokenizer was read from (bundle or checkpoint); save_deploy_assets copies from here.
        self._tokenizer_dir = tokenizer_dir

        # (1, 48, 1, 1, 1) fp32, broadcast over (N, 48, T, h, w) latents.
        self._latents_mean = latents_mean.float().view(1, -1, 1, 1, 1)
        self._latents_std = latents_std.float().view(1, -1, 1, 1, 1)

        self._dim = int(dim)
        self._num_layers = int(num_layers)
        self._num_heads = int(num_heads)
        self._head_dim = int(head_dim)
        self._context_dim = int(context_dim)
        # Grid draw, uniform weights. Deploy rides the native linear flow ramp + shift with Euler steps, not the
        # bundle's karras UniPC grid (see diffusers#14272).
        self._scheduler = FlowScheduler(shift=shift_video, train_timestep=train_timestep)

        self._temporal_compression = _COSMOS3_TEMPORAL_COMPRESSION
        self._causal_temporal = _COSMOS3_CAUSAL_TEMPORAL

        self._use_system_prompt = bool(use_system_prompt)
        self._prompt_templates = bool(prompt_templates)
        self._duration_template = bool(duration_template)
        # Clip frame rate. Nothing in the dataloader stack reports one, so this
        # is a configured assumption rather than a measurement (yaml ``fps``).
        self._clip_fps = float(clip_fps)
        self._max_text_tokens = int(max_text_tokens)
        self._pristine_inv_freq = self._capture_pristine_inv_freq()

    @classmethod
    def from_pretrained(cls, cfg: DictConfig, assets: DictConfig) -> "Cosmos3EdgeVideoBackbone":
        """Build from the diffusers-style bundle at ``assets.model_path``.

        Args:
            cfg: ``cfg.model.video_backbone``: the ``dit`` / ``vae`` blocks the bundle must match, prompt knobs,
                ``shift_video``.
            assets: ``cfg.assets.video_backbone``: ``model_path`` (the bundle root).

        Returns:
            Cosmos3EdgeVideoBackbone on CPU (bf16; the bundle keeps ``time_embedder`` fp32). Raises ValueError
            when the bundle's transformer / VAE config differs from the yaml.
        """
        _check_cfg(cfg)
        # Heavy imports only now: the module stays importable on CPU-only CI.
        vendor = _vendor()
        try:
            from diffusers import AutoencoderKLWan
        except ImportError as exc:
            raise ImportError(_COSMOS3_INSTALL_HINT) from exc
        model_path = str(assets.model_path)
        if not os.path.isdir(os.path.join(model_path, "transformer")):
            raise FileNotFoundError(
                f"cosmos3 model_path has no transformer/ subfolder: {model_path} (expected a diffusers-style bundle)."
            )
        # Each module is checked against its yaml block (weightless fields included) right after loading.
        net = vendor.Cosmos3OmniTransformer.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        _check_against_yaml(net.config, cfg.dit, "dit")
        vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.bfloat16)
        _check_against_yaml(vae.config, cfg.vae, "vae")
        return cls._assemble(cfg, net, vae, os.path.join(model_path, _TOKENIZER_SUBDIR))

    @classmethod
    def from_checkpoint_dir(cls, cfg: DictConfig, ckpt_dir: str) -> "Cosmos3EdgeVideoBackbone":
        """Build the empty skeleton of a saved checkpoint from the yaml ``dit`` / ``vae`` blocks.

        Args:
            cfg: the checkpoint's ``cfg.model.video_backbone``.
            ckpt_dir: str, checkpoint dir holding ``text_tokenizer/`` (save_deploy_assets).

        Returns:
            Cosmos3EdgeVideoBackbone whose parameters are on ``meta`` (bf16). The rotary ``inv_freq`` buffer
            is real (non-persistent, so no checkpoint carries it).
        """
        _check_cfg(cfg)
        from accelerate import init_empty_weights

        vendor = _vendor()
        try:
            from diffusers import AutoencoderKLWan
        except ImportError as exc:
            raise ImportError(_COSMOS3_INSTALL_HINT) from exc
        # The tokenizer is the only asset read from the checkpoint dir; geometry comes from the yaml.
        tokenizer_dir = os.path.join(ckpt_dir, _TOKENIZER_SUBDIR)
        if not os.path.isdir(tokenizer_dir):
            raise FileNotFoundError(f"cosmos3 tokenizer not found at {tokenizer_dir} (written by save_deploy_assets)")
        with init_empty_weights():
            net = vendor.Cosmos3OmniTransformer(**OmegaConf.to_container(cfg.dit))
        # Cast while on meta (free), so a resume shell materializes bf16 like a finetune load.
        net = net.to(dtype=torch.bfloat16)
        # The only non-persistent buffer: fully determined by config, recomputed in fp32.
        inv_freq = cosmos3_dit.rotary_inv_freq(int(cfg.dit.head_dim), float(cfg.dit.rope_theta))
        net.rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)
        # Buffers are never in the state dict: any still on meta would stay garbage after the load.
        leftover = [n for n, b in net.named_buffers() if b.is_meta]
        if leftover:
            raise RuntimeError(f"cosmos3 checkpoint shell has unmaterialized meta buffers: {leftover}")
        with init_empty_weights():
            vae = AutoencoderKLWan(**OmegaConf.to_container(cfg.vae)).to(dtype=torch.bfloat16)
        leftover = [n for n, b in vae.named_buffers() if b.is_meta]
        if leftover:
            raise RuntimeError(f"cosmos3 VAE checkpoint shell has unmaterialized meta buffers: {leftover}")
        return cls._assemble(cfg, net, vae, tokenizer_dir)

    @classmethod
    def _assemble(cls, cfg: DictConfig, net, vae, tokenizer_dir: str) -> "Cosmos3EdgeVideoBackbone":
        """Shared tail of both constructors: drop / freeze the unused parts, read the tokenizer, wrap.

        Args:
            cfg: ``cfg.model.video_backbone``. net: Cosmos3OmniTransformer. vae: AutoencoderKLWan (real or meta).
            tokenizer_dir: str, ``text_tokenizer/`` dir to read.

        Returns:
            Cosmos3EdgeVideoBackbone.
        """
        from transformers import PreTrainedTokenizerFast

        _drop_lm_head(net)
        # Frozen in every mode: finetune / resume build from a ckpt dir too and must not train the und tower.
        n_heads_frozen = _freeze_unused_native_heads(net)
        if n_heads_frozen:
            logger.info("cosmos3: froze unused native action/sound heads (%.1fM params)", n_heads_frozen / 1e6)
        n_frozen = _freeze_und_pathway(net)
        logger.info("cosmos3: froze und pathway + embeddings (%.1fM params)", n_frozen / 1e6)
        # The VAE is frozen too (only encodes / decodes under no_grad).
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
        return cls(
            net=net,
            vae=vae,
            tokenizer=tokenizer,
            tokenizer_dir=tokenizer_dir,
            dim=int(cfg.dit.hidden_size),
            num_layers=int(cfg.dit.num_hidden_layers),
            num_heads=int(cfg.dit.num_attention_heads),
            head_dim=int(cfg.dit.head_dim),
            # The action stream's context is the und final hidden, so its width is the hidden size.
            context_dim=int(cfg.dit.hidden_size),
            latents_mean=torch.tensor(list(cfg.vae.latents_mean), dtype=torch.float32),
            latents_std=torch.tensor(list(cfg.vae.latents_std), dtype=torch.float32),
            shift_video=float(cfg.shift_video),
            train_timestep=cfg.train_timestep,
            use_system_prompt=bool(cfg.use_system_prompt),
            prompt_templates=bool(cfg.prompt_templates),
            duration_template=bool(cfg.prompt_duration_template),
            clip_fps=float(cfg.fps),
            max_text_tokens=int(cfg.max_text_tokens),
        )

    def _capture_pristine_inv_freq(self) -> Tensor:
        """Keep an fp32 copy of the rotary table that no dtype cast can reach.

        Held as a plain attribute, deliberately: ``nn.Module._apply`` walks
        parameters and registered buffers, so a plain tensor survives every
        ``.to(dtype=)`` / ``.bfloat16()`` performed on the module — including
        DeepSpeed's ``self.module.bfloat16()`` inside ``_configure_distributed_model``,
        which casts floating-point *buffers* and runs between the trainer's two
        ``set_dtype_device`` calls. Snapshotting inside ``set_dtype_device``
        cannot survive that: by the second call the buffer is already bf16, and
        upcasting a rounded value recovers the dtype, not the bits.

        Falls back to recomputing from config when the live buffer is already
        non-fp32 at construction (a caller that pre-cast the net), since a
        rounded snapshot would be worse than the exact closed form.
        """
        inv_freq = self.dit.rotary_emb.inv_freq
        # The normal case: copy the fp32 table as it is.
        if inv_freq.dtype == torch.float32:
            return inv_freq.detach().to(device="cpu", copy=True)
        # Pre-cast net: recompute the exact table from the config.
        head_dim, rope_theta = self.dit.config.head_dim, self.dit.config.rope_theta
        logger.warning(
            "cosmos3_edge: rotary inv_freq was already %s at build; recomputing the fp32 table from config.",
            inv_freq.dtype,
        )
        return cosmos3_dit.rotary_inv_freq(int(head_dim), float(rope_theta))

    def set_dtype_device(self, dtype: torch.dtype, device: torch.device) -> None:
        """Move to ``(dtype, device)``, then restore the two pieces that must stay fp32.

        Args:
            dtype: torch.dtype of the compute. device: torch.device.

        Returns:
            None. The rotary ``inv_freq`` comes back from the build-time fp32 copy; ``time_embedder`` is pinned fp32.
        """
        super().set_dtype_device(dtype, device)
        self._latents_mean = self._latents_mean.to(device=device)
        self._latents_std = self._latents_std.to(device=device)

        # Restore the rotary table from the build-time fp32 copy. It is NOT
        # snapshotted here: this method runs twice on the training path
        # (``OpenWAMTrainer.__init__`` and ``OpenWAMTrainer.prepare_accelerate``) with ``accelerator.prepare()`` in
        # between, and DeepSpeed's ``_configure_distributed_model`` calls
        # ``self.module.bfloat16()``, which casts floating-point buffers. A
        # snapshot taken at the second call is therefore already rounded, and
        # upcasting it back recovers the dtype but not the bits.
        #
        # Why it matters: bf16 costs ~0.37% relative precision on inv_freq,
        # which the vision stream's temporal offset (und_len +
        # unified_3d_mrope_temporal_modality_margin = 15000) multiplies into a
        # phase error large enough to scramble cos/sin (cosine ~0.77 against the
        # fp32 phases, 6.6 rad max) while leaving the text positions intact.
        # Cosmos3VLTextRotaryEmbedding runs the position matmul in fp32 with
        # autocast disabled, but that protects the product, not the table.
        self.dit.rotary_emb.register_buffer("inv_freq", self._pristine_inv_freq.to(device=device), persistent=False)

        # The timestep-embedding MLP. Upstream lists it in
        # ``_keep_in_fp32_modules``, and this pin holds wherever OpenWAM owns
        # the dtype: deploy, eager training, and any non-ZeRO path. It does NOT
        # survive DeepSpeed ZeRO — ``_update_model_bit16_weights`` rebinds each
        # ``p.data`` back into the flattened bf16 group after a step — so under
        # ZeRO the MLP runs bf16 from step 1. That is a real train/deploy
        # difference, bounded to this 256→2048 MLP: the sinusoid itself is
        # computed by the parameter-free ``time_proj`` on an fp32 input in both
        # cases (see cosmos3_dit.prepare_block_loop), so the frequency content
        # is identical and only the projection's arithmetic precision differs.
        # Left pinned rather than dropped because the paths where it does hold
        # are the ones that serve the checkpoint.
        self.dit.time_embedder.to(dtype=torch.float32)

    # ================================================================
    # Geometry
    # ================================================================

    @property
    def dim(self) -> int:
        """Gen-stream hidden width (yaml ``dit.hidden_size``)."""
        return self._dim

    @property
    def num_layers(self) -> int:
        """Number of transformer layers (yaml ``dit.num_hidden_layers``)."""
        return self._num_layers

    @property
    def num_heads(self) -> int:
        """Query heads per layer (yaml ``dit.num_attention_heads``)."""
        return self._num_heads

    @property
    def head_dim(self) -> int:
        """Per-head width (yaml ``dit.head_dim``)."""
        return self._head_dim

    @property
    def text_dim(self) -> int:
        """Context width for the action stream = und hidden size (2048)."""
        return self._context_dim

    # ================================================================
    # Video and text IO
    # ================================================================

    def encode_pixels(self, pixels: Tensor) -> Tensor:
        """Pixel clips -> normalized latents (Wan2.2 VAE posterior mode, upstream ``sample_mode="argmax"``).

        Args:
            pixels: (N, 3, T, H, W) in [-1, 1], T = 1 + 4k.

        Returns:
            (N, 48, 1 + (T-1)//4, H/16, W/16) latents ``(mu - mean) / std`` (fp32 math, cast back to the VAE dtype).
        """
        vae = self.vae
        vae_dtype = next(vae.parameters()).dtype
        posterior = vae.encode(pixels.to(device=self.device, dtype=vae_dtype)).latent_dist
        raw = posterior.mode()
        mean = self._latents_mean.to(raw.device)
        std = self._latents_std.to(raw.device)
        return ((raw.float() - mean) / std).to(raw.dtype)

    def decode_latents(self, latents: Tensor) -> Tensor:
        """latents: (N, 48, T_lat, h, w) normalized. Returns: (N, 3, 1 + 4 (T_lat-1), 16h, 16w) VAE pixels."""
        vae = self.vae
        vae_dtype = next(vae.parameters()).dtype
        z = self._unnormalize_latents(latents).to(dtype=vae_dtype)
        return vae.decode(z).sample

    def _unnormalize_latents(self, latents: Tensor) -> Tensor:
        """latents: (N, 48, T, h, w) normalized. Returns: the raw VAE latents ``x * std + mean`` (fp32 math)."""
        mean = self._latents_mean.to(latents.device)
        std = self._latents_std.to(latents.device)
        return (latents.float() * std + mean).to(latents.dtype)

    @torch.no_grad()
    def encode_text(self, prompts: list, *, grid, num_frames: int) -> TextCondition:
        """Templated prompts through the frozen und tower (online; cosmos3 text depends on the clip, no cache).

        Args:
            prompts: list[str] of B prompts.
            grid: (T_lat, h, w) latent grid; the template's resolution is (16h, 16w) and the vision mRoPE
                positions cover this grid.
            num_frames: int, pixel frames of the equivalent contiguous clip (template duration; 1 = image).

        Returns:
            TextCondition: context (B, L, 2048) und final hidden, context_mask (B, L) = position < the prompt's
            token count; extra ``und_kv`` (per-layer gen-facing K/V), ``und_mask`` ((B, L) bool, or None when no
            prompt is padded) and ``vision_positions`` [3, 1, N].
        """
        t_lat, h, w = (int(v) for v in grid)
        # The template states the pixel size of the latent grid (16x spatial compression).
        enc = self._encode_prompts(
            list(prompts),
            num_frames=int(num_frames),
            height=h * _COSMOS3_SPATIAL_COMPRESSION,
            width=w * _COSMOS3_SPATIAL_COMPRESSION,
        )
        context, seq_lens = enc["context"], enc["seq_lens"]
        positions = torch.arange(context.shape[1], device=context.device).unsqueeze(0)
        # Family-private: only cosmos3's prepare reads the und bundle.
        extra = {
            "und_mask": enc["context_mask"],
            "und_kv": enc["und_kv"],
            "vision_positions": self._vision_positions(enc["und_len_padded"], (t_lat, h, w)),
        }
        return TextCondition(
            context=context, context_mask=positions < seq_lens.to(context.device).unsqueeze(1), extra=extra
        )

    def _encode_prompts(self, prompts: List[str], *, num_frames: int, height: int, width: int) -> dict:
        """Tokenize (templated at the yaml clip fps) and run the frozen und tower once.

        Args:
            prompts: list[str] of B prompts. num_frames: int, pixel frames of the clip (1 = image template).
            height / width: int, pixel size the template states.

        Returns:
            dict: ``context`` (B, L, 2048) und final hidden, ``context_mask`` (B, L) bool or None (no padding),
            ``seq_lens`` (B,), ``und_kv`` per-layer gen-facing K/V, ``und_len_padded`` int L.
        """
        tokenizer = self._tokenizer
        texts = prompts
        # Optional resolution / duration template around each prompt (upstream's prompt format).
        if self._prompt_templates:
            texts = [
                cosmos3_text.apply_prompt_templates(
                    p,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    fps=self._clip_fps,
                    add_duration_template=self._duration_template,
                )
                for p in prompts
            ]
        # Chat-format token ids per prompt (unpadded), then right-padded to one batch.
        ids = [
            cosmos3_text.tokenize_prompt(
                tokenizer,
                t,
                use_system_prompt=self._use_system_prompt,
                is_image=num_frames == 1,
                max_length=self._max_text_tokens,
            )
            for t in texts
        ]
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        input_ids, und_mask, seq_lens = cosmos3_text.pad_und_batch(ids, pad_token_id=int(pad_id))
        input_ids = input_ids.to(self.device)
        # Whether any prompt is padded is host-side knowledge (Python list
        # lengths), so decide it here and hand ``None`` down when there is no
        # padding — that is what keeps SDPA on a fused kernel and keeps the
        # MoT loop free of a ``.all()`` device sync. Every B=1 request
        # and any uniform-length batch takes this path.
        und_mask = und_mask.to(self.device) if len(set(len(i) for i in ids)) > 1 else None

        # Text mRoPE positions, then the frozen und tower once (context + per-layer K/V for the gen stream).
        net = self.dit
        float_pos = bool(net.config.enable_fps_modulation)
        text_pos = cosmos3_text.text_mrope_positions(input_ids.shape[1], float_positions=float_pos)
        cos_und, sin_und = cosmos3_dit.compute_rotary(net, text_pos.unsqueeze(1), self.device, self.dtype)
        context, und_kv = cosmos3_dit.run_und_tower(net, input_ids, und_mask, cos_und, sin_und)
        return {
            "context": context,
            "context_mask": und_mask,
            "seq_lens": seq_lens.to(self.device),
            "und_kv": und_kv,
            "und_len_padded": int(input_ids.shape[1]),
        }

    def _vision_positions(self, und_len_padded: int, latent_grid: Tuple[int, int, int]) -> Tensor:
        """Vision mRoPE positions of the latent grid after ``und_len_padded`` text tokens, at the yaml clip fps.

        Args:
            und_len_padded: int, padded text length L. latent_grid: (T_lat, h, w).

        Returns:
            (3, 1, N) positions (N = vision tokens), broadcast over the batch.
        """
        net = self.dit
        # Token grid after patchify; the positions start after the text block plus the modality margin.
        grid = cosmos3_text.patch_grid(*latent_grid, int(net.config.latent_patch_size))
        float_pos = bool(net.config.enable_fps_modulation)
        _, vision_pos = cosmos3_text.build_joint_positions(
            und_len_padded,
            grid,
            modality_margin=int(net.config.unified_3d_mrope_temporal_modality_margin),
            fps=self._clip_fps if float_pos else None,
            base_fps=float(net.config.base_fps),
            temporal_compression_factor=self._temporal_compression,
            float_positions=float_pos,
            reset_spatial_indices=bool(net.config.unified_3d_mrope_reset_spatial_ids),
        )
        return vision_pos.unsqueeze(1)  # [3, 1, N] — broadcast over batch

    def build_text_cache(self, prompts: list) -> None:
        """Not possible: the prompt templates and vision positions need the clip's latent grid.

        Args:
            prompts: list[str] (unused).

        Returns:
            Never; raises ValueError (cosmos3 yamls set ``training.text_cache: false``).
        """
        raise ValueError(
            "cosmos3 encodes text per clip geometry and cannot cache it; set training.text_cache=false"
        )

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
        """Pack the gen stream (``cosmos3_dit.prepare_block_loop``) with the und bundle from ``text.extra``.

        Args:
            latents: (B, 48, T_lat, h, w); the first ``layout.num_hist_latents`` get no timestep embedding.
            timestep: (B,). text: TextCondition from :meth:`encode_text` (context may carry proprio tokens).
            layout: ``cfg.common.layout`` (clean prefix length). proprio: (B, H, D); unused (context tokens).
            use_gradient_checkpointing(_offload): bool loop flags.

        Returns:
            BlockLoopState.
        """
        # The und bundle (K/V, mask, vision positions) was built by encode_text; the clean prefix gets no t-embedding.
        return cosmos3_dit.prepare_block_loop(
            self.dit,
            latents=latents,
            timestep=timestep,
            context=text.context,
            context_mask=text.context_mask,
            und_mask=text.extra["und_mask"],
            und_kv=text.extra["und_kv"],
            vision_positions=text.extra["vision_positions"],
            num_clean_prefix_frames=int(layout.num_hist_latents),
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )

    def run_block(self, block_id: int, state: BlockLoopState, attn_mask: Tensor) -> BlockLoopState:
        """One gen block on the video stream alone.

        Args:
            block_id: int. state: BlockLoopState.
            attn_mask: this layer's mask from :meth:`build_joint_attention_masks` (und prefix columns first); the
                gen block reads its und columns from ``und_mask`` itself, so only the gen columns are passed on.

        Returns:
            ``state`` with ``hidden_states`` updated.
        """
        gen_mask = attn_mask[..., int(state.prefix_kv_len) :]
        return cosmos3_dit.run_block(self.dit, block_id, state, gen_mask=gen_mask)

    def finalize(self, state: BlockLoopState) -> Tensor:
        """state: BlockLoopState after the last block. Returns: (B, 48, T_lat, h, w) predicted velocity."""
        return cosmos3_dit.finalize_block_loop(self.dit, state)

    def pre_attn_at_layer(self, layer_id: int, state: BlockLoopState) -> Tuple[Tensor, Tensor, Tensor, dict]:
        """Gen block first half (``cosmos3_dit.state_pre_attn``): und prefix K/V prepended, GQA K/V expanded.

        Args:
            layer_id: int. state: BlockLoopState.

        Returns:
            q (B, S, H*D), k / v (B, P + S, H*D) with the P und prefix keys first, and the post dict.
        """
        return cosmos3_dit.state_pre_attn(self.dit, layer_id, state)

    def post_attn_at_layer(
        self, layer_id: int, state: BlockLoopState, attn_out: Tensor, post_state: dict
    ) -> BlockLoopState:
        """Gen block second half (``cosmos3_dit.state_post_attn``).

        Args:
            layer_id: int (unused: the layer reference rides ``post_state``). state: BlockLoopState.
            attn_out: (B, S, H*D) attention output of the video tokens. post_state: from :meth:`pre_attn_at_layer`.

        Returns:
            ``state`` with ``hidden_states`` updated.
        """
        _ = layer_id  # layer reference rides post_state
        return cosmos3_dit.state_post_attn(state, attn_out, post_state)

    def build_joint_attention_masks(self, frame_mask: Tensor, state: BlockLoopState, s_action: int) -> list:
        """The shared frame-major expansion with the cached und keys prepended (every row may read them).

        Args:
            frame_mask: (T + 1, T + 1) bool frame mask. state: BlockLoopState (``prefix_kv_len`` / ``prefix_kv_mask``).
            s_action: int, action tokens.

        Returns:
            list of ``num_layers`` masks (S, P + S), or (B, 1, S, P + S) when the und text is padded per sample.
        """
        from openwam.model.video_backbone.cosmos3.masks import widen_mask_for_prefix_kv

        mask = super().build_joint_attention_masks(frame_mask, state, s_action)[0]
        mask = widen_mask_for_prefix_kv(mask, int(state.prefix_kv_len), state.prefix_kv_mask)
        return [mask] * self.num_layers

    # ================================================================
    # Deploy assets
    # ================================================================

    def save_deploy_assets(self, output_dir: str) -> None:
        """Copy the tokenizer's structural files into ``<output_dir>/text_tokenizer/`` (from_checkpoint_dir reads them).

        Args:
            output_dir: str, checkpoint dir. Files come from wherever this backbone read its tokenizer.

        Returns:
            None. Only missing files are copied: repeated saves are idempotent and a partial copy left by an
            interrupted save is completed on the next one.
        """
        dst = os.path.join(output_dir, _TOKENIZER_SUBDIR)
        os.makedirs(dst, exist_ok=True)
        for name in _TOKENIZER_FILES:
            src = os.path.join(self._tokenizer_dir, name)
            if os.path.isfile(src) and not os.path.exists(os.path.join(dst, name)):
                shutil.copy2(src, os.path.join(dst, name))
