"""Contract between the WAM architecture and a concrete video backbone.

Layering: only the architecture talks to the backbone through this contract;
train/deploy go through the architecture, never the backbone object directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from openwam.model.video_backbone.common.text_cache import PromptCache


@dataclass
class BlockLoopState:
    """Mutable state flowing through prepare → run_block → finalize.

    Architecture may read/write ``hidden_states`` / ``time_mod`` / ``rope_freqs``
    / ``context``; ``grid_*`` and ``extras`` are backbone-owned (read-only to it).
    """

    # Architecture may read/write
    hidden_states: Tensor  # (B, L, dim)
    time_mod: Tensor  # (B, 6, dim) or (B, L, 6, dim) per-token
    rope_freqs: Tensor
    context: Tensor  # text cross-attention embedding
    context_mask: Optional[Tensor] = None  # (B, L_context) bool, True = attend

    # Patch grid for unpatchify
    grid_frames: int = 0
    grid_height: int = 0
    grid_width: int = 0

    # Prefix K/V tokens the backbone prepends to its per-layer keys/values but
    # NOT to its queries (e.g. Cosmos3's cached und text stream). The MoT driver
    # widens the joint mask by this many leading key columns — visible to every
    # query row, gated per-sample by ``prefix_kv_mask`` (True = attend). Zero
    # for backbones whose K and Q sequences coincide (Wan).
    prefix_kv_len: int = 0
    prefix_kv_mask: Optional[Tensor] = None  # (B, prefix_kv_len) bool

    # Loop config threaded from prepare() into run_block()
    use_gradient_checkpointing: bool = False
    use_gradient_checkpointing_offload: bool = False

    # Backbone-private escape hatch (Wan also stashes dit/time_embed here)
    extras: dict = field(default_factory=dict)


@dataclass
class TextCondition:
    """Text conditioning of one batch, as :meth:`VideoBackbone.encode_text` returns it.

    ``extra`` is family-private (cosmos3: ``und_kv`` / ``und_mask`` / ``vision_positions``):
    only that family's ``prepare`` reads it; the architecture, joint and mask code never do.
    """

    context: Tensor  # (B, L, D_text)
    context_mask: Tensor  # (B, L) bool, True = attend
    extra: dict = field(default_factory=dict)


class VideoBackbone(ABC, nn.Module):
    """Architecture ↔ video backbone contract — the architecture's private helper.

    Inherits ``nn.Module`` so named children (``self.dit`` / ``self.vae`` / ...)
    are moved by :meth:`set_dtype_device` and serialized into the state_dict.

    Every family's ``__init__`` sets the three fields declared below (``_shift_video``, ``_causal_temporal``,
    ``_temporal_compression``); the properties read them directly.
    """

    # Device / dtype before the first set_dtype_device (which overrides them per instance).
    _device: torch.device = torch.device("cuda")
    _dtype: torch.dtype = torch.bfloat16
    # Set by every family's __init__ (no default: a family that forgets them fails on first read).
    _shift_video: float
    _causal_temporal: bool
    _temporal_compression: int
    # Prompts per text-encoder forward while build_text_cache fills the cache (class attribute of each caching family).
    _text_cache_batch_size: int

    # ================================================================
    # Construction
    # ================================================================

    @classmethod
    @abstractmethod
    def from_pretrained(cls, cfg) -> "VideoBackbone":
        """Build from the upstream weights.

        Args:
            cfg: ``cfg.model.video_backbone`` (DictConfig): module geometry, ``model_path`` and the family's knobs.

        Returns:
            VideoBackbone with every weight loaded (CPU). Raises ValueError when the weights do not match the
            yaml geometry.
        """

    @classmethod
    @abstractmethod
    def from_checkpoint_dir(cls, cfg, ckpt_dir: str) -> "VideoBackbone":
        """Build the empty skeleton of a saved OpenWAM checkpoint; the caller fills its weights.

        Args:
            cfg: the checkpoint's ``cfg.model.video_backbone`` (DictConfig); geometry comes from it.
            ckpt_dir: str, checkpoint dir holding the non-weight assets :meth:`save_deploy_assets` copied
                (tokenizer / processor / third-party ``config.json``).

        Returns:
            VideoBackbone whose parameters are on ``meta`` (no weight file is read). Fill it with a strict
            ``assign=True`` load, or ``materialize_meta`` + an accelerate state load (resume).
        """

    def set_dtype_device(self, dtype: torch.dtype, device: torch.device) -> None:
        """Move every registered child to ``(dtype, device)``; overrides call ``super()`` first.

        Args:
            dtype: torch.dtype of the compute (bf16 in training and deploy).
            device: torch.device to run on.

        Returns:
            None. Must NOT ``.eval()``: trainable submodules stay in train mode.
        """
        self._dtype = dtype
        self._device = device
        self.to(dtype=dtype, device=device)

    # ================================================================
    # Geometry
    # ================================================================

    @property
    @abstractmethod
    def dim(self) -> int:
        """Hidden dim of the video DiT."""

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Number of DiT blocks."""

    @property
    @abstractmethod
    def num_heads(self) -> int:
        """Attention heads per block."""

    @property
    @abstractmethod
    def head_dim(self) -> int:
        """Per-head attention dim."""

    @property
    @abstractmethod
    def text_dim(self) -> int:
        """Per-token raw text-context width (the context the framework appends proprio tokens to)."""

    @property
    @abstractmethod
    def scheduler(self):
        """Flow-matching scheduler: ``set_timesteps`` / ``timesteps`` / ``sigmas`` / ``step`` for deploy, and
        ``sample_train(batch_size, *, device, dtype) -> (timesteps (B,), sigmas (B,), loss_weights (B,) fp32)``
        for the training noise (the grid draw, or a family's native law)."""

    @property
    def temporal_compression(self) -> int:
        """``T_pixel / T_lat``; set ``self._temporal_compression`` in ``__init__``."""
        return self._temporal_compression

    @property
    def device(self) -> torch.device:
        """torch.device the backbone runs on (set by :meth:`set_dtype_device`)."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """torch.dtype of the backbone's compute (set by :meth:`set_dtype_device`)."""
        return self._dtype

    @property
    def shift_video(self) -> float:
        """α-shift for the video scheduler (single source of truth for train + inference)."""
        return self._shift_video

    @property
    def causal_temporal(self) -> bool:
        """Whether the first frame is encoded into its own standalone latent token."""
        return self._causal_temporal

    # ================================================================
    # Video and text IO (families implement the first three; the shared flow below runs on top of them)
    # ================================================================

    @abstractmethod
    def encode_pixels(self, pixels: Tensor) -> Tensor:
        """Encode pixel clips with the family's VAE.

        Args:
            pixels: (N, 3, T, H, W) in [-1, 1] (:meth:`to_pixels` output), T = 1 + temporal_compression * k.

        Returns:
            (N, z, 1 + (T-1) / temporal_compression, h, w) normalized latents, each clip encoded causally.
        """

    @abstractmethod
    def decode_latents(self, latents: Tensor) -> Tensor:
        """Decode latents with the family's VAE.

        Args:
            latents: (N, z, T_lat, h, w) normalized latents.

        Returns:
            (N, 3, T, H, W) pixels in [-1, 1].
        """

    @abstractmethod
    def encode_text(self, prompts: list, *, grid: Tuple[int, int, int], num_frames: int) -> TextCondition:
        """Encode B prompts.

        Args:
            prompts: list[str] of B prompts.
            grid: (T_lat, h, w), the latent grid the text conditions (cosmos3 needs it; others ignore it).
            num_frames: int, pixel frames of the equivalent contiguous clip, 1 + temporal_compression * (T_lat - 1)
                (cosmos3's prompt template; others ignore it).

        Returns:
            TextCondition with context (B, L, D_text), context_mask (B, L) bool and the family's ``extra``.
        """

    @torch.no_grad()
    def build_text_cache(self, prompts: list) -> None:
        """Encode every prompt once with the (already loaded) text encoder, then park the encoder on the CPU.

        Builds a :class:`~openwam.model.video_backbone.common.text_cache.PromptCache` from the family's
        ``text_lookup`` slot (the online encoder), ``_text_cache_batch_size`` prompts per encoder forward, swaps it
        into that slot and moves ``text_encoder`` to the CPU (still a registered child: state_dict keys unchanged).
        ``encode_text`` reads ``text_lookup`` either way. A family that cannot cache (cosmos3) overrides this to raise.

        Args:
            prompts: list[str], every prompt training can draw (dataset ``all_prompts()``); the cache adds the ""
                of caption-dropped samples itself.

        Returns:
            None.
        """
        self.text_lookup = PromptCache(self.text_lookup, prompts, batch_size=self._text_cache_batch_size)
        # Free the encoder's GPU memory; its weights stay in the module tree for checkpointing.
        self.text_encoder.to("cpu")
        torch.cuda.empty_cache()

    # Shared preprocessing flow (implemented once here; families only encode). One latent layout
    # (cfg.common.layout): past frames (per_frame | clip) + the current frame t alone, both clean, then the future:
    # the clip [t, future] encoded as one with its latent 0 (= t) dropped.
    def to_pixels(self, clips: list) -> Tensor:
        """PIL clips -> pixels.

        Args:
            clips: list of B clips, each T same-size PIL frames.

        Returns:
            (B, 3, T, H, W) float32 in [-1, 1] on ``self.device`` (uint8 -> LUT, see ``pil_clips_to_tensor``).
        """
        from openwam.model.video_backbone.common.pixels import pil_clips_to_tensor

        return pil_clips_to_tensor(clips, device=self.device)

    def encode_frames(self, pixels: Tensor) -> Tensor:
        """Encode every frame alone (one :meth:`encode_pixels` call over all B*T 1-frame clips).

        Args:
            pixels: (B, 3, T, H, W) in [-1, 1].

        Returns:
            (B, z, T, h, w) latents, latent t of sample b = frame t of sample b encoded alone.
        """
        b, t = pixels.shape[0], pixels.shape[2]
        # (B*T, 3, 1, H, W), sample-major: frame t of sample b is row b*T + t.
        frames = pixels.transpose(1, 2).flatten(0, 1).unsqueeze(2)
        return self.encode_pixels(frames).unflatten(0, (b, t)).squeeze(3).transpose(1, 2)

    def encode_clip(self, pixels: Tensor) -> Tensor:
        """Encode each sample's frames as one causal clip.

        Args:
            pixels: (B, 3, T, H, W) in [-1, 1], T = 1 + temporal_compression * k.

        Returns:
            (B, z, 1 + (T-1) / temporal_compression, h, w) latents. Raises ValueError for any other T.
        """
        tc, t = int(self.temporal_compression), int(pixels.shape[2])
        if t < 1 or (t - 1) % tc:
            raise ValueError(f"{type(self).__name__}: a causal clip must be 1 + {tc}k frames, got {t}")
        return self.encode_pixels(pixels)

    @staticmethod
    def _batch_prompts(prompts, batch_size: int) -> list:
        """prompts: one str (shared by the batch) or B str. batch_size: B. Returns: list of B prompts."""
        prompts = [prompts] * batch_size if isinstance(prompts, str) else list(prompts)
        if len(prompts) != batch_size:
            raise ValueError(f"got {len(prompts)} prompts for a batch of {batch_size} clips")
        return prompts

    def _encode_history(self, pixels: Tensor, layout) -> Tensor:
        """The clean latents of the H history frames: the past frames, then the current frame alone.

        Args:
            pixels: (B, 3, H, Hp, Wp) history pixels in [-1, 1], oldest first, the current frame t last.
            layout: ``cfg.common.layout``; ``history_encoding`` picks how the H - 1 past frames are encoded.

        Returns:
            (B, z, num_hist_latents, h, w) in the backbone dtype. Raises ValueError when the frame or latent
            count disagrees with the layout.
        """
        h = int(layout.num_hist_frames)
        if pixels.shape[2] != h:
            raise ValueError(f"layout has {h} history frames, got {pixels.shape[2]}")
        parts = []
        if int(layout.num_past_latents) > 0:
            # Contiguous slices: the VAE sees the same memory layout as a separately built clip.
            past = pixels[:, :, : h - 1].contiguous()
            # The only place the past-frame encoding is chosen.
            parts.append(self.encode_frames(past) if layout.history_encoding == "per_frame" else self.encode_clip(past))
        parts.append(self.encode_frames(pixels[:, :, h - 1 :].contiguous()))
        hist = torch.cat(parts, dim=2).to(device=self.device, dtype=self.dtype)
        if hist.shape[2] != int(layout.num_hist_latents):
            raise ValueError(f"history encoded to {hist.shape[2]} latents, layout says {layout.num_hist_latents}")
        return hist

    def _encode_layout_text(self, prompts: list, layout, latent_hw) -> TextCondition:
        """Text of the whole latent clip.

        Args:
            prompts: list of B str. layout: ``cfg.common.layout``. latent_hw: (h, w) latent grid.

        Returns:
            TextCondition for the (num_latents, h, w) grid and the equivalent contiguous clip of
            1 + tc * (num_latents - 1) frames.
        """
        t_lat = int(layout.num_latents)
        return self.encode_text(
            prompts, grid=(t_lat, *latent_hw), num_frames=1 + int(self.temporal_compression) * (t_lat - 1)
        )

    @torch.no_grad()
    def preprocess_condition(self, history: list, prompts, layout) -> Tuple[Tensor, TextCondition]:
        """Deploy condition: the H history frames as clean latents, and the text.

        Args:
            history: list of B clips, each H PIL frames (oldest first, the current frame t last).
            prompts: one str or B str.
            layout: ``cfg.common.layout``.

        Returns:
            (hist_latents (B, z, num_hist_latents, h, w) in the backbone dtype, TextCondition for the
            (num_latents, h, w) grid).
        """
        prompts = self._batch_prompts(prompts, len(history))
        hist = self._encode_history(self.to_pixels(history), layout)
        return hist, self._encode_layout_text(prompts, layout, hist.shape[3:])

    @torch.no_grad()
    def preprocess_train(self, clips: list, prompts, layout) -> Tuple[Tensor, TextCondition]:
        """Training inputs: the full latent clip and the text.

        Args:
            clips: list of B clips, each H + F PIL frames (H history frames, the current frame t last, then F
                future frames).
            prompts: one str or B str.
            layout: ``cfg.common.layout``.

        Returns:
            latents (B, z, num_latents, h, w): ``cat(hist_latents, future)``, the first ``num_hist_latents`` the
            clean condition (the same code as :meth:`preprocess_condition`), the future = the clip [t, future]
            encoded as one with its latent 0 dropped; TextCondition for the (num_latents, h, w) grid.
        """
        prompts = self._batch_prompts(prompts, len(clips))
        pixels = self.to_pixels(clips)
        h, f = int(layout.num_hist_frames), int(layout.num_future_frames)
        if pixels.shape[2] != h + f:
            raise ValueError(f"layout has {h} + {f} frames, got clips of {pixels.shape[2]}")
        hist = self._encode_history(pixels[:, :, :h], layout)
        # The future latents see t while being encoded (Wan TI2V pretraining / contiguous-video encode).
        future = self.encode_clip(pixels[:, :, h - 1 :].contiguous())[:, :, 1:]
        latents = torch.cat([hist, future.to(device=self.device, dtype=self.dtype)], dim=2)
        if latents.shape[2] != int(layout.num_latents):
            raise ValueError(f"clip encoded to {latents.shape[2]} latents, layout says {layout.num_latents}")
        return latents, self._encode_layout_text(prompts, layout, latents.shape[3:])

    @torch.no_grad()
    def decode_future(self, latents: Tensor, layout) -> list:
        """Decode the future of one latent clip.

        Args:
            latents: (1, z, num_latents, h, w) normalized latents of the whole clip.
            layout: ``cfg.common.layout``.

        Returns:
            list of F PIL frames: the segment [current frame, future] decoded as one clip, frame 0 (t) dropped.
            Raises ValueError for a batch B != 1 or a latent count other than ``layout.num_latents``.
        """
        if latents.shape[0] != 1:
            raise ValueError(f"decode_future decodes one clip, got a batch of {latents.shape[0]}")
        if latents.shape[2] != int(layout.num_latents):
            raise ValueError(f"got {latents.shape[2]} latents, layout says {layout.num_latents}")
        from openwam.model.video_backbone.common.pixels import tensor_to_pil_frames

        segment = latents[:, :, int(layout.num_hist_latents) - 1 :]
        # One pixel -> uint8 conversion for every family (clamp, (x + 1) * 127.5, truncate).
        return tensor_to_pil_frames(self.decode_latents(segment))[1:]

    # ================================================================
    # Block loop (prepare / run_block / finalize required; MoT split, masks and compile optional)
    # ================================================================

    @abstractmethod
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
        """Pre-loop: patchify / freqs / time_mod / text. Returns a BlockLoopState.

        Args:
            latents: (B, z, T_lat, h, w) latents, T_lat = ``layout.num_latents``; the first
                ``layout.num_hist_latents`` are clean (t = 0).
            timestep: (B,) video timestep.
            text: TextCondition (the architecture may have appended proprio tokens to context / context_mask).
            layout: ``cfg.common.layout`` (DictConfig); read for the clean prefix length.
            proprio: (B, H, D) normalized proprio, always passed; only a ``proprio_input=backbone_stream`` family
                (flux) embeds it in its own stream, the others ignore it (they read the context's proprio tokens).
            use_gradient_checkpointing / use_gradient_checkpointing_offload: bool loop flags.
        """

    @abstractmethod
    def run_block(self, block_id: int, state: BlockLoopState, attn_mask: Tensor) -> BlockLoopState:
        """Run a single DiT block on the video stream alone (no action tokens).

        Args:
            block_id: int, layer index.
            state: BlockLoopState from :meth:`prepare`.
            attn_mask: bool token mask of this layer, ``build_joint_attention_masks(frame_mask, state, 0)[block_id]``.

        Returns:
            BlockLoopState with the layer applied. Gradient checkpointing is transparent to the architecture.
        """

    @abstractmethod
    def finalize(self, state: BlockLoopState) -> Tensor:
        """Post-loop: head + unpatchify.

        Args:
            state: BlockLoopState after the last block.

        Returns:
            (B, z, T_lat, h, w) predicted velocity (noise - x0) of the whole latent clip.
        """

    def pre_attn_at_layer(self, layer_id: int, state: BlockLoopState) -> Tuple[Tensor, Tensor, Tensor, dict]:
        """Block first half (norm + modulate + Q/K/V + RoPE, no attention), for the MoT driver's joint attention.

        Args:
            layer_id: int, block index. state: BlockLoopState of the video stream.

        Returns:
            (q, k, v) each (B, S, num_heads * head_dim), and ``post_state`` that :meth:`post_attn_at_layer` consumes.
            Default: raises NotImplementedError (a family without the split cannot run ``joint_self_attn``).
        """
        raise NotImplementedError(f"{type(self).__name__} does not support joint self-attention.")

    def post_attn_at_layer(
        self, layer_id: int, state: BlockLoopState, attn_out: Tensor, post_state: dict
    ) -> BlockLoopState:
        """Block second half from :meth:`pre_attn_at_layer`: gate → cross-attn → FFN.

        Args:
            layer_id: int, block index. state: BlockLoopState. attn_out: (B, S, num_heads * head_dim) attention
                output of the video tokens (before the output projection). post_state: from :meth:`pre_attn_at_layer`.

        Returns:
            ``state`` with ``hidden_states`` updated. Default: raises NotImplementedError.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support joint self-attention.")

    def build_joint_attention_masks(self, frame_mask: Tensor, state: BlockLoopState, s_action: int) -> list:
        """Token-level masks of the joint ``[video tokens, action tokens]`` sequence, one per layer.

        Default: every latent frame is ``tokens_per_frame`` consecutive tokens (frame-major), the action block
        ``s_action`` tokens after them; all layers share one mask. GE (views), flux (text / proprio tokens in the
        sequence) and cosmos3 (und prefix keys) override.

        Args:
            frame_mask: (T + 1, T + 1) bool from ``build_frame_mask`` (last row / column = action block).
            state: BlockLoopState of the video stream (its token count and ``grid_frames``).
            s_action: int, action tokens (0 for the video-only loop).

        Returns:
            list of ``num_layers`` (S_v + s_action, S_v + s_action) bool masks, True = attend.
        """
        s_video = int(state.hidden_states.shape[1])
        tpf = frame_tokens(frame_mask, state, s_video)
        mask = expand_frame_mask(frame_mask, [tpf] * (frame_mask.shape[0] - 1), s_action, state.hidden_states.device)
        return [mask] * self.num_layers

    def apply_compile_optimizations(self, compile_cfg) -> None:
        """Deploy-time compile hook.

        Args:
            compile_cfg: the deploy ``optimization.compile`` block (see ``openwam.model.compile_options``).

        Returns:
            None. Default: eager execution, nothing to compile.
        """
        return None

    # ================================================================
    # Deploy assets
    # ================================================================

    def save_deploy_assets(self, output_dir: str) -> None:
        """Copy this backbone's non-weight assets into a checkpoint dir, where :meth:`from_checkpoint_dir` reads them.

        Args:
            output_dir: str, the checkpoint (run) dir.

        Returns:
            None. Only copies files (tokenizer / processor / third-party ``config.json``); never writes cfg.
            Default: nothing to copy.
        """


def frame_tokens(frame_mask: Tensor, state: BlockLoopState, s_video: int) -> int:
    """Tokens per latent frame of a frame-major video stream.

    Args:
        frame_mask: (T + 1, T + 1) frame mask. state: BlockLoopState (``grid_frames``). s_video: int, video tokens.

    Returns:
        int S_v / T. Raises ValueError when the stream does not hold T = layout.num_latents latent frames.
    """
    t = int(frame_mask.shape[0]) - 1
    if int(state.grid_frames) != t or s_video % t:
        raise ValueError(f"the video stream has {state.grid_frames} latent frames ({s_video} tokens), layout says {t}")
    return s_video // t


def expand_frame_mask(frame_mask: Tensor, frame_of_tokens, s_action: int, device) -> Tensor:
    """Expand a frame mask to tokens.

    Args:
        frame_mask: (T + 1, T + 1) bool, last row / column = action block.
        frame_of_tokens: list of T ints, tokens of each latent frame in sequence order (frame-major).
        s_action: int, action tokens after the video tokens.
        device: torch device of the result.

    Returns:
        (sum(frame_of_tokens) + s_action, same) bool: token i inherits the row / column of its frame.
    """
    counts = [int(n) for n in frame_of_tokens] + [int(s_action)]
    # output_size from python ints: no data-dependent shape inside a compiled MoT loop.
    index = torch.repeat_interleave(
        torch.arange(len(counts), device=device), torch.tensor(counts, device=device), output_size=sum(counts)
    )
    fm = frame_mask.to(device=device, dtype=torch.bool)
    return fm[index][:, index]


__all__ = [
    "BlockLoopState",
    "TextCondition",
    "VideoBackbone",
    "expand_frame_mask",
    "frame_tokens",
]
