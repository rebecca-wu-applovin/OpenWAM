"""Wan :class:`VideoBackbone`: Wan2.2-TI2V-5B and the backbones that are only another yaml of it.

:class:`WanBackbone` is the one Wan class; ``wan22_ti2v_5b``, ``wan22_ti2v_tiny`` (debug) and
``longlive2_robot_s`` (LongLive2.0-Robot-S DiT weights, 128-token prompts) all build it.

Construction (both classmethods, everything decided by ``cfg = cfg.model.video_backbone``):
  - :meth:`WanBackbone.from_pretrained`: the DiT / VAE / umT5 skeletons are built from the yaml
    geometry with parameters on ``meta``, then filled from the upstream files with
    :func:`load_strict` (``init: pretrained``), or randomly initialized (``init: random``).
  - :meth:`WanBackbone.from_checkpoint_dir`: the same skeletons left on ``meta`` (the caller loads the
    checkpoint), tokenizer from the checkpoint dir's copy written by :meth:`save_deploy_assets`.

Modules are named children (clean ``dit.*`` / ``vae.*`` / ``text_encoder.*`` keys); scheduler,
tokenizer and division factors are plain attributes. External code reaches them only through the
ABC methods.
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
from typing import Callable, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from openwam.model.compile_options import (
    compile_enabled,
    section_enabled,
    torch_compile_kwargs,
    wan_blocks_compile_cfg,
)
from openwam.model.grad_checkpoint import gradient_checkpoint_forward
from openwam.model.video_backbone.base import BlockLoopState, TextCondition, VideoBackbone
from openwam.model.video_backbone.common.weights import load_strict, read_weights
from openwam.model.video_backbone.wan import dit_forward as wan_dit_forward
from openwam.model.video_backbone.wan.dit import WanModel
from openwam.model.video_backbone.wan.scheduler import FlowMatchScheduler
from openwam.model.video_backbone.wan.text_encoder import HuggingfaceTokenizer, WanTextEncoder
from openwam.model.video_backbone.wan.vae import WanVideoVAE38

logger = logging.getLogger(__name__)

# Where save_deploy_assets puts the umT5 tokenizer inside a checkpoint dir (from_checkpoint_dir reads it back).
CKPT_TOKENIZER_SUBDIR = os.path.join("tokenizer", "google", "umt5-xxl")
# Tokenizer padding length; every encode call passes its own max_length (text_max_length).
_TOKENIZER_SEQ_LEN = 512


def _build_modules(cfg: DictConfig) -> Tuple[WanModel, WanVideoVAE38, WanTextEncoder]:
    """Instantiate the three Wan modules from the yaml geometry (on whatever device context the caller set).

    Args:
        cfg: ``cfg.model.video_backbone`` with the ``dit`` / ``vae`` / ``text_encoder`` geometry blocks.

    Returns:
        (dit, vae, text_encoder). Built in this order, which fixes the RNG draws of a random init.
    """
    dit = WanModel(**OmegaConf.to_container(cfg.dit))
    vae = WanVideoVAE38(z_dim=int(cfg.vae.z_dim), dim=int(cfg.vae.dim))
    text_encoder = WanTextEncoder(**OmegaConf.to_container(cfg.text_encoder))
    return dit, vae, text_encoder


def _build_meta_modules(cfg: DictConfig) -> Tuple[WanModel, WanVideoVAE38, WanTextEncoder]:
    """Same as :func:`_build_modules` but every parameter is on ``meta`` (no storage, no init).

    Args:
        cfg: ``cfg.model.video_backbone``.

    Returns:
        (dit, vae, text_encoder) skeletons. Buffers and plain tensor attributes (DiT RoPE ``freqs``,
        VAE ``mean`` / ``std``) are real: they are not in the state dict, so no load would fill them.
    """
    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False):
        return _build_modules(cfg)


def _strip_prefix(state: dict, prefix: str, path: str) -> dict:
    """Remove ``prefix`` from every key of ``state``.

    Args:
        state: dict[str, Tensor] read from ``path``.
        prefix: the yaml ``dit_weights_prefix`` ("" leaves keys unchanged).
        path: the file the state came from (error message only).

    Returns:
        dict[str, Tensor] with stripped keys. Raises ValueError if any key lacks the prefix.
    """
    bad = [k for k in state if not k.startswith(prefix)]
    if bad:
        raise ValueError(f"{path}: {len(bad)} keys without the dit_weights_prefix {prefix!r}, e.g. {bad[:3]}")
    return {k[len(prefix) :]: v for k, v in state.items()}


def _load_component(module: nn.Module, state: dict, geometry: DictConfig, name: str) -> None:
    """Strictly fill one meta skeleton, naming the yaml geometry block when the weights disagree with it.

    Args:
        module: the skeleton built from ``geometry``.
        state: dict[str, Tensor] of its weights.
        geometry: the yaml block the skeleton was built from (``cfg.dit`` / ``cfg.vae`` / ``cfg.text_encoder``).
        name: that block's key, for the message.

    Returns:
        None. Raises ValueError (with the declared geometry) on any key / shape mismatch.
    """
    try:
        load_strict(module, state, assign=True)
    except ValueError as err:
        raise ValueError(f"{name} weights do not match the yaml geometry {name}={dict(geometry)}: {err}") from err


class WanBackbone(VideoBackbone):
    """Exposes the Wan modules through the VideoBackbone interface.

    Modules registered as named children (clean ``dit.*`` / ``vae.*`` keys);
    scheduler / tokenizer / division factors are plain attributes. Build it with
    :meth:`from_pretrained` or :meth:`from_checkpoint_dir`.
    """

    # Prompts per umT5 forward while build_text_cache fills the cache.
    _text_cache_batch_size = 8

    # ================================================================
    # Construction
    # ================================================================

    def __init__(
        self,
        *,
        dit: nn.Module,
        vae: nn.Module,
        text_encoder: nn.Module,
        tokenizer,
        tokenizer_dir: str,
        shift_video: float,
        text_max_length: int,
    ):
        """Wrap already-built modules. Use :meth:`from_pretrained` / :meth:`from_checkpoint_dir` instead.

        Args:
            dit: WanModel. vae: WanVideoVAE38. text_encoder: WanTextEncoder.
            tokenizer: callable umT5 tokenizer (HuggingfaceTokenizer).
            tokenizer_dir: str, the dir ``tokenizer`` was read from (save_deploy_assets copies it).
            shift_video: float, α-shift of the video scheduler (single source of truth behind the property).
            text_max_length: int, umT5 tokens per prompt (512 Wan, 128 LongLive).
        """
        super().__init__()
        self.dit = dit
        self.vae = vae
        self.text_encoder = text_encoder
        self._tokenizer = tokenizer
        self._tokenizer_dir = tokenizer_dir
        self._scheduler = FlowMatchScheduler()
        self._device = torch.device("cuda")
        self._dtype = torch.bfloat16
        self._shift_video = float(shift_video)
        self.text_max_length = int(text_max_length)
        # Wan native contract: VAE 4× temporal compression + causal first-frame token.
        self._temporal_compression, self._causal_temporal = 4, True
        self._wan_blocks_compile_enabled = False
        self._wan_blocks_compile_kwargs: dict | None = None
        self._compiled_wan_blocks: dict[int, Callable[..., Tensor]] = {}
        # Text slot read by encode_text: the online umT5 here, a PromptCache after build_text_cache.
        self.text_lookup = self._t5_entries

    @classmethod
    def from_pretrained(cls, cfg: DictConfig) -> "WanBackbone":
        """Build from upstream weights (``init: pretrained``) or a random init (``init: random``).

        Args:
            cfg: ``cfg.model.video_backbone``. Reads the ``dit`` / ``vae`` / ``text_encoder`` geometry,
                ``init``, ``model_path`` + ``weights`` (file names), ``dit_weights`` /
                ``dit_weights_prefix`` (DiT file override, LongLive), ``tokenizer_path``,
                ``shift_video``, ``text_max_length``.

        Returns:
            WanBackbone on CPU, every module in bf16 and eval mode. Raises ValueError when a weight file
            does not match the yaml geometry.
        """
        if cfg.init == "random":
            # Debug backbone: real tensors, default init, no file read.
            dit, vae, text_encoder = _build_modules(cfg)
        elif cfg.init == "pretrained":
            dit, vae, text_encoder = _build_meta_modules(cfg)
            # DiT: the model_path shards, or the dit_weights file (LongLive model.pt) instead of them.
            if cfg.dit_weights is None:
                dit_files = sorted(glob.glob(os.path.join(cfg.model_path, cfg.weights.dit)))
            else:
                dit_files = [str(cfg.dit_weights)]
            if not dit_files:
                raise FileNotFoundError(f"no DiT weights match {os.path.join(cfg.model_path, cfg.weights.dit)}")
            logger.info("Wan DiT weights: %s", dit_files)
            dit_state = _strip_prefix(read_weights(dit_files), str(cfg.dit_weights_prefix), str(dit_files))
            _load_component(dit, dit_state, cfg.dit, "dit")
            # Cast now so the fp32 shards (~20 GB) are freed before the next file is read.
            del dit_state
            dit.to(dtype=torch.bfloat16)
            # Upstream VAE keys are the inner VideoVAE38_'s; the wrapper holds it as ``model``.
            vae_state = read_weights([os.path.join(cfg.model_path, cfg.weights.vae)])
            _load_component(vae, {f"model.{k}": v for k, v in vae_state.items()}, cfg.vae, "vae")
            te_state = read_weights([os.path.join(cfg.model_path, cfg.weights.text_encoder)])
            _load_component(text_encoder, te_state, cfg.text_encoder, "text_encoder")
        else:
            raise ValueError(f"video_backbone.init must be 'pretrained' or 'random', got {cfg.init!r}")
        return cls._assemble(cfg, dit, vae, text_encoder, str(cfg.tokenizer_path))

    @classmethod
    def from_checkpoint_dir(cls, cfg: DictConfig, ckpt_dir: str) -> "WanBackbone":
        """Build the empty skeleton of a saved OpenWAM checkpoint; the caller loads its weights.

        Args:
            cfg: the checkpoint's ``cfg.model.video_backbone`` (geometry, ``shift_video``, ``text_max_length``).
            ckpt_dir: str, checkpoint dir holding ``tokenizer/google/umt5-xxl`` (save_deploy_assets).

        Returns:
            WanBackbone whose parameters are all on ``meta`` (bf16), with the checkpoint's tokenizer.
            Fill it with a strict ``assign=True`` load (or ``materialize_meta`` + an accelerate state load).
        """
        tokenizer_dir = os.path.join(ckpt_dir, CKPT_TOKENIZER_SUBDIR)
        if not os.path.isdir(tokenizer_dir):
            raise FileNotFoundError(f"umT5 tokenizer not found at {tokenizer_dir} (written by save_deploy_assets)")
        dit, vae, text_encoder = _build_meta_modules(cfg)
        return cls._assemble(cfg, dit, vae, text_encoder, tokenizer_dir)

    @classmethod
    def _assemble(cls, cfg: DictConfig, dit, vae, text_encoder, tokenizer_dir: str) -> "WanBackbone":
        """Shared tail of both constructors: bf16 + eval, tokenizer, scalar cfg values.

        Args:
            cfg: ``cfg.model.video_backbone``. dit / vae / text_encoder: built modules (real or meta).
            tokenizer_dir: str, umT5 tokenizer dir to read.

        Returns:
            WanBackbone.
        """
        # Every module bf16 + eval (meta shells cast for free); the tokenizer pads to 512, encode truncates.
        modules = [m.to(dtype=torch.bfloat16).eval() for m in (dit, vae, text_encoder)]
        tokenizer = HuggingfaceTokenizer(name=tokenizer_dir, seq_len=_TOKENIZER_SEQ_LEN, clean="whitespace")
        return cls(
            dit=modules[0],
            vae=modules[1],
            text_encoder=modules[2],
            tokenizer=tokenizer,
            tokenizer_dir=tokenizer_dir,
            shift_video=float(cfg.shift_video),
            text_max_length=int(cfg.text_max_length),
        )

    # ================================================================
    # Geometry
    # ================================================================

    @property
    def _dit(self):
        """The WanModel (alias of ``self.dit`` the geometry properties read)."""
        return self.dit

    @property
    def dim(self) -> int:
        """DiT hidden width (3072 for TI2V-5B)."""
        return int(self._dit.dim)

    @property
    def num_layers(self) -> int:
        """Number of DiT blocks."""
        return len(self._dit.blocks)

    @property
    def num_heads(self) -> int:
        """Self-attention heads per block."""
        return int(self._dit.blocks[0].num_heads)

    @property
    def head_dim(self) -> int:
        """Per-head width, dim / num_heads."""
        return int(self._dit.dim) // self.num_heads

    @property
    def text_dim(self) -> int:
        """Raw text-context width the DiT's text embedding takes (= yaml ``dit.text_dim``)."""
        return int(self._dit.text_embedding[0].in_features)

    @property
    def scheduler(self):
        """The Wan :class:`FlowMatchScheduler` (bell-weighted 1000-step training grid, shift_video at deploy)."""
        return self._scheduler

    # ================================================================
    # Video and text IO
    # ================================================================

    def encode_pixels(self, pixels: Tensor) -> Tensor:
        """pixels: (N, 3, T, H, W) in [-1, 1]. Returns: (N, 48, 1 + (T-1)//4, H/16, W/16) normalized VAE38 latents."""
        # The base flow hands float32 pixels; the VAE runs in the backbone dtype.
        return self.vae.batch_encode(pixels.to(self.dtype), device=pixels.device)

    def decode_latents(self, latents: Tensor) -> Tensor:
        """latents: (N, 48, T_lat, h, w). Returns: (N, 3, 1 + 4 (T_lat-1), 16h, 16w) pixels (tiled VAE decode)."""
        return self.vae.decode(latents.to(self.device), device=self.device, tiled=True)

    def encode_text(self, prompts: list, *, grid, num_frames: int) -> TextCondition:
        """umT5 context of B prompts (through ``text_lookup``: online or cached).

        Args:
            prompts: list[str] of B prompts. grid / num_frames: unused (Wan's text does not depend on the clip).

        Returns:
            TextCondition: context (B, L, D), context_mask (B, L) = position < the prompt's token count, extra {}.
        """
        entries = self.text_lookup(prompts)
        # Stack the per-prompt entries; the mask is "position < that prompt's token count".
        context = torch.cat([e[0] for e in entries], dim=0)
        seq_lens = torch.cat([e[1] for e in entries], dim=0).to(device=context.device)
        positions = torch.arange(context.shape[1], device=context.device).unsqueeze(0)
        return TextCondition(context=context, context_mask=positions < seq_lens.unsqueeze(1), extra={})

    @torch.no_grad()
    def _t5_entries(self, prompts: list) -> list:
        """Online umT5 encode (the default ``text_lookup``).

        Args:
            prompts: list[str] of B prompts.

        Returns:
            list of B entries ``(context (1, L, D), seq_lens (1,) long)`` on ``self.device``, L = text_max_length;
            context is zero past each prompt's length.
        """
        # Pad every prompt to text_max_length tokens; the mask marks the real ones.
        ids, mask = self._tokenizer(
            prompts,
            return_mask=True,
            add_special_tokens=True,
            max_length=self.text_max_length,
            padding="max_length",
            truncation=True,
        )
        ids, mask = ids.to(self.device), mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        # Zero the padding positions so cached and online contexts agree.
        for i, v in enumerate(seq_lens):
            context[i, v:] = 0
        return [(context[j : j + 1], seq_lens[j : j + 1]) for j in range(len(prompts))]

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
        """Pre-loop: text embedding, patchify, per-token time modulation, RoPE.

        Args:
            latents: (B, z, T_lat, h, w); the first ``layout.num_hist_latents`` are clean (per-token t = 0).
            timestep: (B,) video timestep. text: TextCondition (context (B, L, D_text), context_mask (B, L)).
            layout: ``cfg.common.layout`` (``num_hist_latents`` clean prefix, ``num_latents`` = T_lat).
            proprio: (B, H, D) normalized proprio; unused (Wan reads proprio as context tokens).
            use_gradient_checkpointing / use_gradient_checkpointing_offload: bool loop flags.

        Returns:
            BlockLoopState of the patchified sequence.
        """
        dit = self.dit
        # Text: project the raw context; the mask must stay (B, L) bool aligned with it.
        context = dit.text_embedding(text.context)
        context_mask = text.context_mask.to(device=context.device, dtype=torch.bool)
        if context_mask.ndim != 2:
            raise ValueError(f"context_mask must be 2D [B, L], got shape {tuple(context_mask.shape)}")
        if context_mask.shape[0] != context.shape[0] or context_mask.shape[1] != context.shape[1]:
            raise ValueError(
                f"context_mask shape must match context [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
            )

        # latents and text always share the batch size B (the architecture builds both per sample).
        hidden_states = dit.patchify(latents)

        grid_frames, grid_height, grid_width = hidden_states.shape[2:]
        # Per-token time embedding / modulation: the clean history frames at t = 0, the rest at timestep.
        time_embed, time_modulation = wan_dit_forward.build_time_modulation(
            dit, timestep, int(layout.num_hist_latents), int(layout.num_latents), grid_height * grid_width
        )
        hidden_states = rearrange(hidden_states, "b c f h w -> b (f h w) c").contiguous()
        # 3-D RoPE: the (frame, height, width) frequency tables broadcast over the grid, one row per token.
        freqs = (
            torch.cat(
                [
                    dit.freqs[0][:grid_frames]
                    .view(grid_frames, 1, 1, -1)
                    .expand(grid_frames, grid_height, grid_width, -1),
                    dit.freqs[1][:grid_height]
                    .view(1, grid_height, 1, -1)
                    .expand(grid_frames, grid_height, grid_width, -1),
                    dit.freqs[2][:grid_width]
                    .view(1, 1, grid_width, -1)
                    .expand(grid_frames, grid_height, grid_width, -1),
                ],
                dim=-1,
            )
            .reshape(grid_frames * grid_height * grid_width, 1, -1)
            .to(hidden_states.device)
        )

        # Wan keeps the DiT and the head's time embedding in extras for run_block / finalize.
        extras = {
            "dit": dit,
            "time_embed": time_embed,  # Wan head time embedding; consumed in finalize()
        }
        return BlockLoopState(
            hidden_states=hidden_states,
            time_mod=time_modulation,
            rope_freqs=freqs,
            context=context,
            context_mask=context_mask,
            grid_frames=grid_frames,
            grid_height=grid_height,
            grid_width=grid_width,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            extras=extras,
        )

    def run_block(self, block_id: int, state: BlockLoopState, attn_mask: Tensor) -> BlockLoopState:
        """One Wan DiT block on the video stream alone (compiled when deploy compile is on, else eager /
        gradient-checkpointed).

        Args:
            block_id: int. state: BlockLoopState. attn_mask: (S, S) bool self-attention mask of this layer.

        Returns:
            ``state`` with ``hidden_states`` updated.
        """
        block = state.extras["dit"].blocks[block_id]
        # Compile only the deploy (no checkpointing) path; checkpointing wraps whichever callable runs.
        checkpointing = state.use_gradient_checkpointing or state.use_gradient_checkpointing_offload
        if self._wan_blocks_compile_enabled and not checkpointing:
            block = self._compiled_wan_block(block_id, block)
        # DiTBlock(x, context, t_mod, freqs, cross-attn mask, self-attn mask).
        state.hidden_states = gradient_checkpoint_forward(
            block,
            state.use_gradient_checkpointing,
            state.use_gradient_checkpointing_offload,
            state.hidden_states,
            state.context,
            state.time_mod,
            state.rope_freqs,
            self._cross_attn_mask(state),
            attn_mask,
        )
        return state

    def finalize(self, state: BlockLoopState):
        """Wan DiT head (per-token ``time_embed`` (B, S, dim)) + unpatchify.

        Args:
            state: BlockLoopState after the last block (``extras["time_embed"]`` from :meth:`prepare`).

        Returns:
            (B, 48, T_lat, h, w) predicted velocity.
        """
        dit = state.extras["dit"]
        hidden_states = dit.head(state.hidden_states, state.extras["time_embed"])

        hidden_states = dit.unpatchify(hidden_states, (state.grid_frames, state.grid_height, state.grid_width))
        return hidden_states

    def pre_attn_at_layer(self, layer_id: int, state: BlockLoopState) -> Tuple[Tensor, Tensor, Tensor, tuple]:
        """First half of a Wan DiT block (``DiTBlock.pre_attn``), for DualSystemMoTDriver's mixed attention.

        Args:
            layer_id: int. state: BlockLoopState.

        Returns:
            q, k, v (B, S, dim) each and the post tuple :meth:`post_attn_at_layer` consumes.
        """
        block = state.extras["dit"].blocks[layer_id]
        return block.pre_attn(state.hidden_states, state.time_mod, state.rope_freqs)

    def post_attn_at_layer(
        self, layer_id: int, state: BlockLoopState, attn_out: Tensor, post_state: tuple
    ) -> BlockLoopState:
        """Second half of a Wan DiT block (``DiTBlock.post_attn``): gate -> cross-attn -> FFN.

        Args:
            layer_id: int. state: BlockLoopState. attn_out: (B, S, dim) unprojected attention output of the video
                slice. post_state: the tuple :meth:`pre_attn_at_layer` returned.

        Returns:
            ``state`` with ``hidden_states`` updated.
        """
        block = state.extras["dit"].blocks[layer_id]
        state.hidden_states = block.post_attn(attn_out, post_state, state.context, self._cross_attn_mask(state))
        return state

    @staticmethod
    def _cross_attn_mask(state: BlockLoopState) -> Tensor:
        """Text cross-attention mask: each video query row reads the same context keys.

        Args:
            state: BlockLoopState with ``context_mask`` (B, L) bool and ``hidden_states`` (B, S, dim).

        Returns:
            (B, 1, S, L) bool (an expanded view).
        """
        return state.context_mask.unsqueeze(1).expand(-1, state.hidden_states.shape[1], -1).unsqueeze(1)

    def apply_compile_optimizations(self, compile_cfg) -> None:
        """Enable lazy per-layer Wan ``DiTBlock.forward`` compile for deploy.

        Args:
            compile_cfg: the deploy compile block; its ``wan_blocks`` section holds the ``torch.compile`` kwargs.

        Returns:
            None. :meth:`run_block` then compiles each block on first use (never while checkpointing).
        """
        # Reset first: a disabled config leaves the backbone eager.
        self._compiled_wan_blocks.clear()
        self._wan_blocks_compile_enabled = False
        self._wan_blocks_compile_kwargs = None

        if not compile_enabled(compile_cfg, default=False, strict=True):
            return

        # The wan_blocks section may switch the per-block compile off on its own.
        section = wan_blocks_compile_cfg(compile_cfg)
        if not section_enabled(section, default=True):
            logger.info("Wan block compile disabled by config; running eager.")
            return

        kwargs = torch_compile_kwargs(section, default_mode="default")
        self._wan_blocks_compile_kwargs = kwargs
        self._wan_blocks_compile_enabled = True
        logger.info("Enabled lazy Wan block compile with torch.compile kwargs=%s", kwargs)

    def _compiled_wan_block(self, block_id: int, block: nn.Module) -> Callable[..., Tensor]:
        """Lazily ``torch.compile`` one DiTBlock (deploy compile). block_id: int. block: DiTBlock.

        Returns: the compiled callable, cached per block id.
        """
        compiled = self._compiled_wan_blocks.get(block_id)
        if compiled is None:
            compiled = torch.compile(block, **(self._wan_blocks_compile_kwargs or {}))
            self._compiled_wan_blocks[block_id] = compiled
        return compiled

    # ================================================================
    # Deploy assets
    # ================================================================

    def save_deploy_assets(self, output_dir: str) -> None:
        """Copy the umT5 tokenizer into the checkpoint dir, where :meth:`from_checkpoint_dir` reads it.

        Args:
            output_dir: str, checkpoint dir; the tokenizer lands in ``<output_dir>/tokenizer/google/umt5-xxl``.
                It is copied from wherever this backbone read it (upstream dir, or the source checkpoint).

        Returns:
            None. A tokenizer already present in ``output_dir`` is left as is.
        """
        dst = os.path.join(output_dir, CKPT_TOKENIZER_SUBDIR)
        if os.path.isdir(dst):
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(self._tokenizer_dir, dst)
        logger.info("Copied umT5 tokenizer %s -> %s", self._tokenizer_dir, dst)


__all__ = ["WanBackbone"]
