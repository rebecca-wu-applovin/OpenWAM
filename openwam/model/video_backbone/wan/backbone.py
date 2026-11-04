"""Shared Wan :class:`VideoBackbone` base plus the concrete Wan subclass.

:class:`WanBase` holds every piece of behavior common to the Wan family (DiT
forward, TI2V first-frame conditioning, deploy). The concrete backbone adds only
its construction:
  - :class:`Wan22Ti2v` — Wan2.2-TI2V-5B.

Lives in ``wan/backbone.py``. Owns the Wan
modules (DiT/VAE/text encoder/tokenizer) — modules as named
children, scheduler/tokenizer/division factors as plain attributes; external
code reaches them only through the ABC methods.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from openwam.model.compile_options import (
    compile_enabled,
    section_enabled,
    torch_compile_kwargs,
    wan_blocks_compile_cfg,
)
from openwam.model.video_backbone.base import BlockLoopState, VideoBackbone
from openwam.model.video_backbone.wan import conditioning as wan_conditioning
from openwam.model.video_backbone.wan import dit_forward as wan_dit_forward
from openwam.model.video_backbone.wan import encode as wan_encode
from openwam.model.video_backbone.wan import loader
from openwam.model.video_backbone.wan.dit import modulate, rope_apply
from openwam.model.video_backbone.wan.preprocess import (
    check_resize_height_width,
)
from openwam.model.video_backbone.common.checkpoint import gradient_checkpoint_forward

logger = logging.getLogger(__name__)


class WanBase(VideoBackbone):
    """Exposes the Wan modules through the VideoBackbone interface.

    Modules registered as named children (clean ``dit.*`` / ``vae.*`` keys);
    scheduler / tokenizer / division factors are plain attributes. Concrete
    subclasses construct via their own ``from_pretrained(source)``.
    """

    # ================================================================
    # Construction
    # ================================================================

    def __init__(self, holder, *, shift_video=None, text_dim: Optional[int] = None):
        """Internal constructor. Use a subclass ``from_pretrained()`` instead.

        ``shift_video`` is the optional Esser α-shift on the video scheduler,
        stored as the single source of truth behind the ABC property. ``None``
        keeps the scheduler template default (Wan = 5.0).
        """
        super().__init__()
        # ``holder`` is a transient carrier: drain its sub-modules + non-Module
        # state into self, then let it go out of scope. Nothing reads it after.
        # Optional sub-modules declared up front so the attribute always exists
        # (the loop below only setattr's the ones the holder actually carries).
        self.vae = None
        # Promote sub-modules to named children so state_dict uses clean prefixes.
        for _name in ("dit", "vae", "text_encoder"):
            _mod = getattr(holder, _name, None)
            if _mod is not None:
                # nn.Module → named child; non-Module (test mocks) → plain attr,
                # same ``self.<name>`` access resolves on both.
                setattr(self, _name, _mod)
        # Backbone-owned non-Module state.
        self._scheduler = getattr(holder, "scheduler", None)
        self._tokenizer = getattr(holder, "tokenizer", None)
        self._height_division_factor = getattr(holder, "height_division_factor", None)
        self._width_division_factor = getattr(holder, "width_division_factor", None)
        self._time_division_factor = getattr(holder, "time_division_factor", None)
        self._time_division_remainder = getattr(holder, "time_division_remainder", None)
        self._device = torch.device("cuda")
        self._dtype = torch.bfloat16
        self._shift_video = None if shift_video is None else float(shift_video)
        actual_text_dim = loader.infer_text_dim(getattr(holder, "dit", None))
        self._text_dim = actual_text_dim if text_dim is None else int(text_dim)
        # Wan native contract: DiT patch (1,2,2); VAE 4× temporal compression +
        # causal first-frame token.
        self._dit_patch_size = (1, 2, 2)
        self._temporal_compression, self._causal_temporal = 4, True
        self._wan_blocks_compile_enabled = False
        self._wan_blocks_compile_kwargs: dict | None = None
        self._compiled_wan_blocks: dict[int, Callable[..., Tensor]] = {}
        # prompt -> (context (1, 512, D), seq_lens (1,)) once build_text_cache ran; None = encode online.
        self._text_cache: Optional[dict] = None

    # ================================================================
    # Internal properties
    # ================================================================

    @property
    def _dit(self):
        return self.dit

    @property
    def needs_first_frame_skip(self) -> bool:
        """``True``: Wan is TI2V-only, so ``latent[0]`` is a clean conditioning
        frame (per-token t=0 on frame-0 tokens) excluded from the diffusion loss.
        """
        return True

    # ================================================================
    # ABC: Properties (6) — device/dtype inherited from VideoBackbone
    # ================================================================

    @property
    def dim(self) -> int:
        return int(self._dit.dim)

    @property
    def num_layers(self) -> int:
        return len(self._dit.blocks)

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def num_heads(self) -> int:
        return int(self._dit.blocks[0].num_heads)

    @property
    def head_dim(self) -> int:
        return int(self._dit.dim) // self.num_heads

    @property
    def text_dim(self) -> Optional[int]:
        return getattr(self, "_text_dim", None)

    # umT5 token budget per prompt (Wan native 512; LongLive / Long-WAM uses 128).
    text_max_length: int = 512

    @property
    def video_attention_mask_mode(self) -> str:
        """Video self-attention mask mode for joint MoT mask construction.

        Modes: ``bidirectional`` (full v↔v, default), ``per_frame_causal``
        (token-level causal block-diagonal), ``first_frame_causal`` (first frame
        sees only itself, later frames see all). Explicit override wins, else
        the DiT's value, else ``bidirectional``.
        """
        explicit = getattr(self, "_video_attention_mask_mode", None)
        if explicit is not None:
            return explicit
        return getattr(self._dit, "video_attention_mask_mode", "bidirectional")

    @video_attention_mask_mode.setter
    def video_attention_mask_mode(self, mode: str) -> None:
        self._video_attention_mask_mode = mode

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the video↔video block of the joint MoT attention mask
        (``True`` = attend to). Layout matches FastWAM's equivalent.
        """
        mode = self.video_attention_mask_mode
        if mode == "bidirectional":
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

        if mode == "per_frame_causal":
            if video_seq_len % video_tokens_per_frame != 0:
                raise ValueError(
                    "video_seq_len must be divisible by video_tokens_per_frame in 'per_frame_causal' mode, "
                    f"got {video_seq_len} and {video_tokens_per_frame}"
                )
            num_video_frames = video_seq_len // video_tokens_per_frame
            frame_causal = torch.tril(torch.ones((num_video_frames, num_video_frames), dtype=torch.bool, device=device))
            return frame_causal.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(
                video_tokens_per_frame, dim=1
            )

        if mode == "first_frame_causal":
            video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
            # First-frame rows attend only to first-frame keys; later rows stay True.
            video_mask[:first_frame_tokens, first_frame_tokens:] = False
            return video_mask

        raise ValueError(
            f"Unsupported video_attention_mask_mode '{mode}'. "
            "Choose from: bidirectional, per_frame_causal, first_frame_causal."
        )

    def apply_compile_optimizations(self, compile_cfg) -> None:
        """Enable lazy per-layer Wan ``DiTBlock.forward`` compile for deploy."""

        self._compiled_wan_blocks.clear()
        self._wan_blocks_compile_enabled = False
        self._wan_blocks_compile_kwargs = None

        if not compile_enabled(compile_cfg, default=False, strict=True):
            return

        section = wan_blocks_compile_cfg(compile_cfg)
        if not section_enabled(section, default=True):
            logger.info("Wan block compile disabled by config; running eager.")
            return

        kwargs = torch_compile_kwargs(section, default_mode="default")
        self._wan_blocks_compile_kwargs = kwargs
        self._wan_blocks_compile_enabled = True
        logger.info("Enabled lazy Wan block compile with torch.compile kwargs=%s", kwargs)

    def _compiled_wan_block(self, block_id: int, block: nn.Module) -> Callable[..., Tensor]:
        compiled = self._compiled_wan_blocks.get(block_id)
        if compiled is None:
            compiled = torch.compile(block, **(self._wan_blocks_compile_kwargs or {}))
            self._compiled_wan_blocks[block_id] = compiled
        return compiled

    def _run_wan_block(
        self,
        block_id: int,
        block: nn.Module,
        state: BlockLoopState,
        block_context_mask: Optional[Tensor],
        attn_mask: Optional[Tensor],
    ) -> Tensor:
        compile_allowed = (
            self._wan_blocks_compile_enabled
            and not state.use_gradient_checkpointing
            and not state.use_gradient_checkpointing_offload
        )
        if compile_allowed:
            try:
                compiled = self._compiled_wan_block(block_id, block)
                if attn_mask is not None:
                    return compiled(
                        state.hidden_states,
                        state.context,
                        state.time_mod,
                        state.rope_freqs,
                        block_context_mask,
                        attn_mask,
                    )
                return compiled(
                    state.hidden_states,
                    state.context,
                    state.time_mod,
                    state.rope_freqs,
                    block_context_mask,
                )
            except Exception as exc:
                self._compiled_wan_blocks.clear()
                self._wan_blocks_compile_enabled = False
                logger.warning("Wan block torch.compile failed at block %s; falling back to eager: %s", block_id, exc)

        if attn_mask is not None:
            return gradient_checkpoint_forward(
                block,
                state.use_gradient_checkpointing,
                state.use_gradient_checkpointing_offload,
                state.hidden_states,
                state.context,
                state.time_mod,
                state.rope_freqs,
                block_context_mask,
                attn_mask,
            )

        return gradient_checkpoint_forward(
            block,
            state.use_gradient_checkpointing,
            state.use_gradient_checkpointing_offload,
            state.hidden_states,
            state.context,
            state.time_mod,
            state.rope_freqs,
            block_context_mask,
        )

    # ================================================================
    # ABC: Three-step execution (3)
    # ================================================================

    def prepare(self, **kw) -> BlockLoopState:
        dit = self.dit
        latents = kw["latents"]
        timestep = kw["timestep"]
        context = kw["context"]
        context_mask = kw.get("context_mask")
        seq_lens = kw.get("seq_lens")
        fuse_vae_embedding_in_latents = kw.get("fuse_vae_embedding_in_latents", False)
        num_clean_prefix_frames = kw.get("num_clean_prefix_frames", 0)
        use_gradient_checkpointing = kw.get("use_gradient_checkpointing", False)
        use_gradient_checkpointing_offload = kw.get("use_gradient_checkpointing_offload", False)
        force_per_token_t_mod = bool(kw.get("force_per_token_t_mod", False))

        time_embed, time_modulation = wan_dit_forward.build_time_modulation(
            dit,
            timestep,
            latents,
            patch_size=self._dit_patch_size,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            force_per_token_t_mod=force_per_token_t_mod,
            num_clean_prefix_frames=num_clean_prefix_frames,
            zero_clean_prefix_t_mod=bool(kw.get("zero_clean_prefix_t_mod", False)),
            has_first_frame_latents=kw.get("first_frame_latents") is not None,
        )

        context = dit.text_embedding(context)
        if context_mask is None:
            if seq_lens is not None:
                seq_lens = seq_lens.to(device=context.device)
                positions = torch.arange(context.shape[1], device=context.device).unsqueeze(0)
                context_mask = positions < seq_lens.unsqueeze(1)
        else:
            context_mask = context_mask.to(device=context.device, dtype=torch.bool)
            if context_mask.ndim != 2:
                raise ValueError(f"context_mask must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != context.shape[0] or context_mask.shape[1] != context.shape[1]:
                raise ValueError(
                    f"context_mask shape must match context [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
                )

        hidden_states = latents
        if hidden_states.shape[0] != context.shape[0]:
            hidden_states = torch.concat([hidden_states] * context.shape[0], dim=0)
        if timestep.shape[0] != context.shape[0]:
            timestep = torch.concat([timestep] * context.shape[0], dim=0)

        hidden_states = dit.patchify(hidden_states)

        grid_frames, grid_height, grid_width = hidden_states.shape[2:]
        hidden_states = rearrange(hidden_states, "b c f h w -> b (f h w) c").contiguous()
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

    def run_block(self, block_id: int, state: BlockLoopState) -> BlockLoopState:
        dit = state.extras["dit"]
        block = dit.blocks[block_id]
        attn_mask = state.extras.get("shared_attention_mask")
        context_mask = state.context_mask

        block_context_mask = (
            context_mask.unsqueeze(1).expand(-1, state.hidden_states.shape[1], -1) if context_mask is not None else None
        )
        if attn_mask is not None:
            if block_context_mask is None:
                block_context_mask = (
                    torch.ones(
                        (state.context.shape[0], state.context.shape[1]),
                        dtype=torch.bool,
                        device=state.context.device,
                    )
                    .unsqueeze(1)
                    .expand(-1, state.hidden_states.shape[1], -1)
                )
            state.hidden_states = self._run_wan_block(block_id, block, state, block_context_mask, attn_mask)
            return state

        state.hidden_states = self._run_wan_block(block_id, block, state, block_context_mask, None)
        return state

    def pre_attn_at_layer(self, layer_id: int, state: BlockLoopState) -> Tuple[Tensor, Tensor, Tensor, dict]:
        """First half of a Wan DiT block (norm1 + AdaLN + Q/K/V + RoPE), up to
        the attention call. Lets DualSystemMoTDriver pull video-side Q/K/V before the
        mixed attention; pairs with :meth:`post_attn_at_layer`.
        """
        q, k, v, post_tuple = self.pre_attn_at_layer_for_compile(layer_id, state)
        residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = post_tuple
        block = state.extras["dit"].blocks[layer_id]
        post_state = {
            "block": block,
            "residual_x": residual_x,
            "gate_msa": gate_msa,
            "shift_mlp": shift_mlp,
            "scale_mlp": scale_mlp,
            "gate_mlp": gate_mlp,
        }
        return q, k, v, post_state

    def pre_attn_at_layer_for_compile(
        self, layer_id: int, state: BlockLoopState
    ) -> Tuple[Tensor, Tensor, Tensor, tuple[Tensor, ...]]:
        """Compile-friendly Wan pre-attention half using a tensor tuple post-state."""
        block = state.extras["dit"].blocks[layer_id]

        time_modulation = state.time_mod
        has_seq = time_modulation.dim() == 4
        chunk_dim = 2 if has_seq else 1
        chunks = (
            block.modulation.to(dtype=time_modulation.dtype, device=time_modulation.device) + time_modulation
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            chunks = tuple(c.squeeze(2) for c in chunks)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks

        residual_x = state.hidden_states
        attn_input = modulate(block.norm1(state.hidden_states), shift_msa, scale_msa)

        self_attn = block.self_attn
        q = self_attn.norm_q(self_attn.q(attn_input))
        k = self_attn.norm_k(self_attn.k(attn_input))
        v = self_attn.v(attn_input)
        q = rope_apply(q, state.rope_freqs, self_attn.num_heads)
        k = rope_apply(k, state.rope_freqs, self_attn.num_heads)

        post_state = (residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        return q, k, v, post_state

    def post_attn_at_layer(
        self, layer_id: int, state: BlockLoopState, attn_out: Tensor, post_state: dict
    ) -> BlockLoopState:
        """Second half of a Wan DiT block: gate → cross-attn → FFN.
        ``attn_out`` is the unprojected (pre ``self_attn.o``) attention
        output for the video slice of the joint mixed attention.
        """
        if isinstance(post_state, dict):
            post_state = (
                post_state["residual_x"],
                post_state["gate_msa"],
                post_state["shift_mlp"],
                post_state["scale_mlp"],
                post_state["gate_mlp"],
            )
        return self.post_attn_at_layer_for_compile(layer_id, state, attn_out, post_state)

    def post_attn_at_layer_for_compile(
        self, layer_id: int, state: BlockLoopState, attn_out: Tensor, post_state: tuple[Tensor, ...]
    ) -> BlockLoopState:
        """Compile-friendly Wan post-attention half consuming a tensor tuple."""
        block = state.extras["dit"].blocks[layer_id]
        self_attn = block.self_attn
        residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = post_state

        hidden_states = block.gate(residual_x, gate_msa, self_attn.o(attn_out))
        context_mask = None
        if state.context_mask is not None:
            context_mask = state.context_mask.unsqueeze(1).expand(-1, hidden_states.shape[1], -1).unsqueeze(1)
        hidden_states = hidden_states + block.cross_attn(
            block.norm3(hidden_states), state.context, ctx_mask=context_mask
        )
        mlp_input = modulate(block.norm2(hidden_states), shift_mlp, scale_mlp)
        hidden_states = block.gate(hidden_states, gate_mlp, block.ffn(mlp_input))
        state.hidden_states = hidden_states

        return state

    def finalize(self, state: BlockLoopState):
        """Wan DiT head + unpatchify. Returns ``(B, z_dim, F, H, W)``."""
        dit = state.extras["dit"]
        head = dit.head
        time_embed = state.extras["time_embed"]
        head_time_embed = time_embed if time_embed.dim() == 3 else time_embed.unsqueeze(1)

        hidden_states = head(state.hidden_states, head_time_embed)

        hidden_states = dit.unpatchify(hidden_states, (state.grid_frames, state.grid_height, state.grid_width))
        return hidden_states

    # ================================================================
    # Text-embedding cache (training: fixed prompt set, frozen T5)
    # ================================================================

    @torch.no_grad()
    def build_text_cache(self, prompts: list, *, batch_size: int = 8, offload_device: str = "cpu") -> None:
        """Encode every training prompt once, then park the text encoder off the GPU.

        Args:
            prompts: list[str], every prompt training can draw (dataset ``all_prompts()``).
            batch_size: int, prompts per T5 forward while building.
            offload_device: where the frozen T5 lives afterwards. It stays a registered
                child, so ``state_dict`` / checkpoints keep the same ``text_encoder.*`` keys.

        Returns: None. Sets ``self._text_cache`` = {prompt: (context (1, 512, D), seq_lens (1,))}
            on the GPU; ``_encode_text`` then reads it and raises on unknown prompts.
        """
        cache = {}
        for i in range(0, len(prompts), batch_size):
            chunk = list(prompts[i : i + batch_size])
            context, seq_lens = wan_encode.encode_text(
                chunk,
                tokenizer=self._tokenizer,
                text_encoder=self.text_encoder,
                device=self.device,
                max_length=self.text_max_length,
            )
            # Per-prompt (1, ...) slices; clone so each entry owns its storage.
            for j, p in enumerate(chunk):
                cache[p] = (context[j : j + 1].clone(), seq_lens[j : j + 1].clone())
        self._text_cache = cache
        self.text_encoder.to(offload_device)
        torch.cuda.empty_cache()

    def _encode_text(self, prompts: list) -> Tuple[Tensor, Tensor]:
        """Text context for a batch: cache lookup if built, else online T5.

        Args: prompts: list[str] of length B.
        Returns: context (B, 512, D) on self.device, seq_lens (B,) long.
        """
        if self._text_cache is None:
            return wan_encode.encode_text(
                prompts,
                tokenizer=self._tokenizer,
                text_encoder=self.text_encoder,
                device=self.device,
                max_length=self.text_max_length,
            )
        missing = [p for p in prompts if p not in self._text_cache]
        if missing:
            # The encoder is on CPU now; a new prompt means all_prompts() missed a source.
            raise KeyError(f"text cache has no entry for prompts {missing[:3]} (cache size {len(self._text_cache)})")
        # torch.cat copies, so callers may modify the batch in place.
        context = torch.cat([self._text_cache[p][0] for p in prompts], dim=0)
        seq_lens = torch.cat([self._text_cache[p][1] for p in prompts], dim=0)
        return context, seq_lens

    # ================================================================
    # ABC: Unified preprocessing (1)
    # ================================================================

    def _sparse_pixels(self, clip) -> Tensor:
        """clip: list of T PIL frames. Returns: (1, 3, T, H_px, W_px) pixels in [-1, 1] on the device."""
        return wan_encode.preprocess_video(clip, dtype=self.dtype, device=self.device)

    def preprocess_sparse_condition(self, history_frames, text, *, num_future_frames: int) -> dict:
        """Sparse-history conditioning: H history frames encoded one by one + the text context.

        Args:
            history_frames: list of B clips, each H PIL frames (oldest first).
            text: list of B prompts.
            num_future_frames: int F (1 + 4k), the future frames the clip covers.

        Returns:
            preprocess dict without ``input_latents``; ``first_frame_latents`` (B, z, H, h, w) holds the H clean
            history latents and ``num_clean_prefix_frames`` = H (per-token t=0 on them, loss skips them).
        """
        h, f = len(history_frames[0]), int(num_future_frames)
        height, width, _ = check_resize_height_width(
            history_frames[0][0].size[1],
            history_frames[0][0].size[0],
            f,
            height_division_factor=self._height_division_factor,
            width_division_factor=self._width_division_factor,
            time_division_factor=self._time_division_factor,
            time_division_remainder=self._time_division_remainder,
        )
        if (f - 1) % self._time_division_factor:
            raise ValueError(f"future frames must be 1 + {self._time_division_factor}k, got {f}")
        context, seq_lens = self._encode_text(text)
        # History: (B*H, 3, 1, H_px, W_px) → B*H single-frame latents → (B, z, H, h, w).
        hist = torch.cat([self._sparse_pixels(clip).transpose(1, 2).flatten(0, 1).unsqueeze(2) for clip in history_frames])
        hist_lat = wan_encode.encode_video(hist, vae=self.vae)
        hist_lat = hist_lat.unflatten(0, (len(history_frames), h)).squeeze(3).transpose(1, 2)
        return {
            "context": context,
            "seq_lens": seq_lens,
            "height": height,
            "width": width,
            "num_frames": h + f,
            "fuse_vae_embedding_in_latents": True,
            "num_clean_prefix_frames": h,
            "first_frame_latents": hist_lat.to(dtype=self.dtype, device=self.device),
        }

    def encode_sparse_future(self, future_frames) -> Tensor:
        """future_frames: list of B clips of F PIL frames. Returns: (B, z, 1 + (F-1)//4, h, w), one causal clip each."""
        fut = torch.cat([self._sparse_pixels(clip) for clip in future_frames], dim=0)
        fut_lat = wan_encode.encode_video(fut, vae=self.vae)
        return fut_lat.to(dtype=self.dtype, device=self.device)

    def _preprocess_sparse_history(self, frames, text, num_history_frames: int) -> dict:
        """Train inputs for the sparse-history layout: the shared condition + the future clip as target.

        Args:
            frames: list of B clips of H + F PIL frames (history first).
            text: list of B prompts.
            num_history_frames: int, H.

        Returns:
            :meth:`preprocess_sparse_condition` dict plus ``input_latents`` (B, z, H + T_fut, h, w).
        """
        h = int(num_history_frames)
        cond = self.preprocess_sparse_condition(
            [clip[:h] for clip in frames], text, num_future_frames=len(frames[0]) - h
        )
        fut_lat = self.encode_sparse_future([clip[h:] for clip in frames])
        return {**cond, "input_latents": torch.cat([cond["first_frame_latents"], fut_lat], dim=2)}

    def preprocess_input_for_train(self, *, frames=None, text=None, **kw) -> dict:
        """Unified train preprocessing: raw data (``frames``/``text``, optional
        ``ref_images`` in kw) → the denoising-loop input dict.
        ``num_history_frames`` > 0 switches to the sparse-history layout.
        """
        if int(kw.get("num_history_frames", 0) or 0) > 0:
            return self._preprocess_sparse_history(frames, text, kw["num_history_frames"])
        device = self.device
        dtype = self.dtype

        height, width, num_frames = check_resize_height_width(
            frames[0][0].size[1],
            frames[0][0].size[0],
            len(frames[0]),
            height_division_factor=self._height_division_factor,
            width_division_factor=self._width_division_factor,
            time_division_factor=self._time_division_factor,
            time_division_remainder=self._time_division_remainder,
        )

        context, seq_lens = self._encode_text(text)

        all_input_videos = []
        for clip_frames in frames:
            all_input_videos.append(wan_encode.preprocess_video(clip_frames, dtype=self.dtype, device=self.device))
        stacked_inputs = torch.cat(all_input_videos, dim=0)
        input_latents = wan_encode.encode_video(stacked_inputs, vae=self.vae)
        input_latents = input_latents.to(dtype=dtype, device=device)

        # TI2V first-frame conditioning: ``latent[0]`` of the already-encoded video
        # latents is the clean reference (no second VAE call, no prepend).
        # ``base.compute_loss`` clean-replaces ``latents[:, :, 0:1]`` with it on every
        # step and ``_compute_video_loss`` trims frame 0. Only when a reference
        # frame is actually provided.
        ref_images = kw.get("ref_images")
        has_ref = ref_images is not None and ref_images[0] is not None

        return {
            "input_latents": input_latents,
            "context": context,
            "seq_lens": seq_lens,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "fuse_vae_embedding_in_latents": has_ref,
            "num_clean_prefix_frames": 0,
            "first_frame_latents": input_latents[:, :, 0:1].clone() if has_ref else None,
        }

    # ================================================================
    # ABC: Sub-module access (1)
    # ================================================================

    def get_submodule(self, name: str) -> nn.Module | None:
        # Non-Module names (tokenizer / scheduler) resolve to None.
        mod = getattr(self, name, None)
        return mod if isinstance(mod, nn.Module) else None

    # ================================================================
    # ABC: Decoding (1)
    # ================================================================

    def decode_video(self, latents: Tensor, *, tiled: bool = True) -> list:
        video_tensor = wan_encode.decode_latents(
            latents, vae=self.vae, device=self.device, tiled=tiled
        )
        return wan_encode.latents_to_frames(video_tensor)

    # ================================================================
    # Self-contained checkpoint: specs into config + artifacts into dir
    # ================================================================

    def save_deploy_assets(self, output_dir: str, cfg) -> None:
        """Make this backbone's checkpoint slice self-contained in one pass:
        merge the Wan component + tokenizer specs into ``cfg`` and copy the Wan
        tokenizer into ``output_dir`` (single tokenizer-layout source, no
        duplication).
        """
        from openwam.model.video_backbone.wan.component_specs import save_video_backbone_deploy_assets

        save_video_backbone_deploy_assets(output_dir, cfg)

    # ================================================================
    # ABC: Deploy-input preprocessing (override)
    # ================================================================

    def preprocess_input_for_inference(
        self,
        *,
        prompt: str,
        first_frame_image=None,
        num_frames: int = 49,
        height: int = 384,
        width: int = 320,
        seed: int = 42,
        num_inference_steps: int = 50,
        shift: float = 5.0,
        tiled: bool = True,
        prompt_embed_cache: Optional[dict] = None,
        **kw,
    ) -> dict:
        """Build the inference denoising-loop input dict from explicit kwargs,
        via explicit backbone helpers (no unit-runner).

        Only the text embedding is cached (prompt-keyed, seed/dim-independent);
        noise / first_frame_latents are rebuilt every call.
        ``**kw`` swallows fields other backbones consume (CFG) that Wan
        ignores — Wan does no CFG at inference.
        """
        # Wan native tiling grid (other backbones may differ).
        tile_size = (30, 52)
        tile_stride = (15, 26)

        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, shift=shift)

        # ShapeChecker: snap to a model-valid grid; noise uses these.
        height, width, num_frames = check_resize_height_width(
            height,
            width,
            num_frames,
            height_division_factor=self._height_division_factor,
            width_division_factor=self._width_division_factor,
            time_division_factor=self._time_division_factor,
            time_division_remainder=self._time_division_remainder,
        )

        if self._text_cache is not None:
            # Training-time rollout (vis): the text encoder is parked on CPU, use the cache.
            context, seq_lens = self._encode_text([prompt])
        else:
            context, seq_lens = wan_encode.encode_text_for_inference(
                prompt,
                prompt_embed_cache=prompt_embed_cache,
                tokenizer=self._tokenizer,
                text_encoder=self.text_encoder,
                device=self.device,
                max_length=self.text_max_length,
            )

        inputs_shared = {
            "end_image": None,
            "input_video": None,
            "denoising_strength": 1.0,
            "control_video": None,
            "reference_image": None,
            "seed": seed,
            "rand_device": "cpu",
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": 1.0,
            "cfg_merge": False,
            "sigma_shift": shift,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "input_audio": None,
        }
        inputs_shared["context"] = context
        inputs_shared["seq_lens"] = seq_lens
        inputs_shared["prompt"] = prompt
        inputs_shared["num_inference_steps"] = num_inference_steps

        # Deploy ``input_video`` is always None, so ``latents`` == ``noise``
        # (same tensor under both keys, as base.generate expects).
        noise = wan_conditioning.build_deploy_noise(
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            rand_device="cpu",
            vae=self.vae,
            dtype=self.dtype,
            device=self.device,
        )
        inputs_shared["noise"] = noise
        inputs_shared["latents"] = noise

        wan_conditioning.finalize_ti2v_first_frame_latents(
            inputs_shared,
            first_frame_image,
            vae=self.vae,
            dtype=self.dtype,
            device=self.device,
        )
        return inputs_shared


class Wan22Ti2v(WanBase):
    """Wan2.2-TI2V-5B backbone (native VAE)."""

    @classmethod
    def from_pretrained(cls, source, *, text_dim: Optional[int] = None, **kw) -> "Wan22Ti2v":
        """Build a Wan22Ti2v from a source.

        Sources: ``DictConfig`` (full Hydra cfg → loader), ``str`` dir path /
        ``dict`` with ``model_path`` (lightweight build), else an already-built
        component holder. Construction returns a transient holder that
        ``__init__`` drains into the backbone.
        """
        holder = loader.build_holder(source, **kw)
        # Resolve optional cfg-side ``shift_video`` here (not in __init__)
        # because the cfg shape depends on the ``source`` type.
        shift_video_cfg = loader.resolve_cfg_shift_video(source)

        return cls(holder, shift_video=shift_video_cfg, text_dim=text_dim)


__all__ = ["WanBase", "Wan22Ti2v"]
