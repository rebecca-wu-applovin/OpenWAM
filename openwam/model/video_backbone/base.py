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

    ``extra`` is family-private (cosmos3: ``und_kv`` / ``und_mask`` / ``vision_positions``; flux: ``text_len``):
    only that family's ``prepare`` reads it; the architecture, joint and mask code never do.
    """

    context: Tensor  # (B, L, D_text)
    context_mask: Tensor  # (B, L) bool, True = attend
    extra: dict = field(default_factory=dict)


class VideoBackbone(ABC, nn.Module):
    """Architecture ↔ video backbone contract — the architecture's private helper.

    Inherits ``nn.Module`` so named children (``self.dit`` / ``self.vae`` / ...)
    are moved by :meth:`set_dtype_device` and serialized into the state_dict.
    """

    # ================================================================
    # Required: structural metadata
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
    def scheduler(self):
        """Flow-matching scheduler; must support set_timesteps / timesteps / sigmas."""

    @property
    def dit_patch_size(self) -> Tuple[int, int, int]:
        """DiT ``(T, H, W)`` patch size; set ``self._dit_patch_size`` in ``__init__``."""
        return self._dit_patch_size

    @property
    def temporal_compression(self) -> int:
        """``T_pixel / T_lat``; set ``self._temporal_compression`` in ``__init__``."""
        return self._temporal_compression

    # ================================================================
    # Required: construction
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

    # ================================================================
    # Required: video and text IO (the only encoding code a family writes)
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

    # ================================================================
    # Required: three-step execution
    # ================================================================

    @abstractmethod
    def prepare(
        self,
        *,
        latents: Tensor,
        timestep: Tensor,
        text: TextCondition,
        num_clean_latents: int,
        raw_proprio: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> BlockLoopState:
        """Pre-loop: patchify / freqs / time_mod / text. Returns a BlockLoopState.

        Args:
            latents: (B, z, T_lat, h, w) latents; the first ``num_clean_latents`` are clean (t = 0).
            timestep: (B,) video timestep.
            text: TextCondition (the architecture may have appended proprio tokens to context / context_mask).
            num_clean_latents: int, clean prefix latents.
            raw_proprio: (B, N, state_dim) raw proprio for a family that embeds it in its own stream (flux), else None.
            use_gradient_checkpointing / use_gradient_checkpointing_offload: bool loop flags.
        """

    @abstractmethod
    def run_block(self, block_id: int, state: BlockLoopState) -> BlockLoopState:
        """Run a single DiT block. Gradient checkpointing is transparent to the architecture."""

    @abstractmethod
    def finalize(self, state: BlockLoopState) -> Tensor:
        """Post-loop: head + SP gather + unpatchify. Returns ``(B, C, T, H, W)``."""

    # ================================================================
    # Optional metadata (defaults)
    # ================================================================

    @property
    def device(self) -> torch.device:
        return getattr(self, "_device", torch.device("cuda"))

    @property
    def dtype(self) -> torch.dtype:
        return getattr(self, "_dtype", torch.bfloat16)

    @property
    def shift_video(self) -> Optional[float]:
        """α-shift for the video scheduler (single source of truth for train +
        inference). ``None`` falls back to the scheduler default (Wan = 5.0)."""
        return getattr(self, "_shift_video", None)

    @property
    def text_dim(self) -> Optional[int]:
        """Per-token raw text/context embedding dim. ``None`` keeps the 4096 fallback."""
        return None

    @property
    def causal_temporal(self) -> bool:
        """Whether the first frame is encoded into its own standalone latent token."""
        return getattr(self, "_causal_temporal", True)

    @property
    def video_attention_mask_mode(self) -> str:
        """v↔v mask mode for joint MoT. Default bidirectional; causal backbones override."""
        return "bidirectional"

    def build_video_to_video_mask(
        self, video_seq_len: int, video_tokens_per_frame: int, device: torch.device
    ) -> Tensor:
        """Build the v↔v attention-mask block; mode from :attr:`video_attention_mask_mode`."""
        if self.video_attention_mask_mode != "bidirectional":
            raise NotImplementedError(
                f"{type(self).__name__} does not implement build_video_to_video_mask "
                f"for mode '{self.video_attention_mask_mode}'."
            )
        return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

    # ================================================================
    # Shared preprocessing flow (implemented once here; families only encode)
    # ================================================================
    # Two transitional layouts until the latent layout comes from cfg (Task 8): sparse history (H > 0 frames each
    # encoded alone + the future as one clip) and first frame (H = 0: the whole clip as one causal clip, its
    # latent 0 is the clean condition).

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
            (B, z, 1 + (T-1) / temporal_compression, h, w) latents. Raises ValueError unless T = 1 (mod tc).
        """
        self.num_future_latents(pixels.shape[2])  # validates T = 1 + tc * k
        return self.encode_pixels(pixels)

    @staticmethod
    def _batch_prompts(prompts, batch_size: int) -> list:
        """prompts: one str (shared by the batch) or B str. batch_size: B. Returns: list of B prompts."""
        prompts = [prompts] * batch_size if isinstance(prompts, str) else list(prompts)
        if len(prompts) != batch_size:
            raise ValueError(f"got {len(prompts)} prompts for a batch of {batch_size} clips")
        return prompts

    @torch.no_grad()
    def preprocess_condition(
        self, history: list, prompts, *, num_frames: int, num_latents: int
    ) -> Tuple[Tensor, TextCondition]:
        """Deploy condition: H history frames, each encoded alone, and the text.

        Args:
            history: list of B clips, each H PIL frames (oldest first, frame t last).
            prompts: one str or B str.
            num_frames: int, pixel frames of the equivalent contiguous clip the text describes.
            num_latents: int T_lat, latents of the whole clip that will be denoised (H clean + the future).

        Returns:
            (hist_latents (B, z, H, h, w) in the backbone dtype, TextCondition for the (T_lat, h, w) grid).
        """
        prompts = self._batch_prompts(prompts, len(history))
        hist = self.encode_frames(self.to_pixels(history)).to(device=self.device, dtype=self.dtype)
        text = self.encode_text(prompts, grid=(int(num_latents), *hist.shape[3:]), num_frames=int(num_frames))
        return hist, text

    @torch.no_grad()
    def preprocess_train(self, clips: list, prompts, *, num_hist_frames: int) -> Tuple[Tensor, Tensor, TextCondition]:
        """Training inputs: clean prefix latents, the full latent clip and the text.

        Args:
            clips: list of B clips, each T PIL frames (sparse history: H history frames first, then F future).
            prompts: one str or B str.
            num_hist_frames: int H; > 0: sparse history (each history frame alone + the future as one clip);
                0: first frame (the whole clip as one causal clip).

        Returns:
            clean_latents (B, z, N_clean, h, w): the clean prefix (sparse: the H history latents; first frame:
                latent 0 of the clip); latents (B, z, T_lat, h, w) with ``latents[:, :, :N_clean] == clean_latents``;
                TextCondition for the (T_lat, h, w) grid.
        """
        prompts = self._batch_prompts(prompts, len(clips))
        pixels = self.to_pixels(clips)
        h = int(num_hist_frames)
        if h > 0:
            # Contiguous slices: the VAE sees the same memory layout as a separately built clip.
            clean = self.encode_frames(pixels[:, :, :h].contiguous()).to(device=self.device, dtype=self.dtype)
            future = self.encode_clip(pixels[:, :, h:].contiguous()).to(device=self.device, dtype=self.dtype)
            latents = torch.cat([clean, future], dim=2)
        else:
            latents = self.encode_clip(pixels).to(device=self.device, dtype=self.dtype)
            clean = latents[:, :, :1].clone()
        t_lat = int(latents.shape[2])
        text = self.encode_text(
            prompts, grid=(t_lat, *latents.shape[3:]), num_frames=1 + int(self.temporal_compression) * (t_lat - 1)
        )
        return clean, latents, text

    def num_future_latents(self, num_future_frames: int) -> int:
        """Latent frames of a causally encoded F-frame clip.

        Args:
            num_future_frames: int F.

        Returns: int 1 + (F - 1) / temporal_compression. Raises ValueError unless F = 1 (mod temporal_compression).
        """
        tc = int(self.temporal_compression)
        f = int(num_future_frames)
        if f < 1 or (f - 1) % tc:
            raise ValueError(f"{type(self).__name__}: future frames must be 1 + {tc}k, got {f}")
        return 1 + (f - 1) // tc

    @torch.no_grad()
    def decode_video(self, latents: Tensor) -> list:
        """Decode one latent clip to PIL frames.

        Args:
            latents: (1, z, T_lat, h, w) normalized latents.

        Returns:
            list of T PIL frames. Raises ValueError for a batch B != 1.
        """
        from openwam.model.video_backbone.common.pixels import tensor_to_pil_frames

        if latents.shape[0] != 1:
            raise ValueError(f"decode_video decodes one clip, got a batch of {latents.shape[0]}")
        return tensor_to_pil_frames(self.decode_latents(latents).float().clamp(-1, 1))

    # ================================================================
    # Optional: deploy compile
    # ================================================================

    def apply_compile_optimizations(self, compile_cfg) -> None:
        """Optional deploy-time compile hook; backbones default to eager execution."""
        return None

    # ================================================================
    # Optional: joint self-attention split (default raise)
    # ================================================================

    def pre_attn_at_layer(self, layer_id: int, state: BlockLoopState) -> Tuple[Tensor, Tensor, Tensor, dict]:
        """Block first half (norm + modulate + Q/K/V + RoPE, no attention). Returns
        ``(q, k, v, post_state)``; ``post_state`` feeds :meth:`post_attn_at_layer`."""
        raise NotImplementedError(f"{type(self).__name__} does not support joint self-attention.")

    def post_attn_at_layer(
        self, layer_id: int, state: BlockLoopState, attn_out: Tensor, post_state: dict
    ) -> BlockLoopState:
        """Block second half from :meth:`pre_attn_at_layer`: gate → cross-attn → FFN."""
        raise NotImplementedError(f"{type(self).__name__} does not support joint self-attention.")

    # ================================================================
    # Lifecycle: device/dtype (default moves all registered children)
    # ================================================================

    def set_dtype_device(self, dtype: torch.dtype, device: torch.device) -> None:
        """Move everything to ``(dtype, device)``. Overrides call ``super()`` first.
        Must NOT ``.eval()`` — trainable submodules stay in train mode."""
        self._dtype = dtype
        self._device = device
        self.to(dtype=dtype, device=device)

    # ================================================================
    # Optional deploy-asset hook (orchestrated by the architecture)
    # ================================================================

    def save_deploy_assets(self, output_dir: str) -> None:
        """Copy this backbone's non-weight assets into a checkpoint dir, where :meth:`from_checkpoint_dir` reads them.

        Args:
            output_dir: str, the checkpoint (run) dir.

        Returns:
            None. Only copies files (tokenizer / processor / third-party ``config.json``); never writes cfg.
            Default: nothing to copy.
        """

    # ================================================================
    # Text cache (switched by cfg.model.video_backbone.text_cache)
    # ================================================================

    def build_text_cache(self, prompts: list) -> None:
        """Encode every prompt once with the (already loaded) text encoder, then park the encoder on the CPU.

        A caching family builds a :class:`~openwam.model.video_backbone.common.text_cache.PromptCache` from its
        ``text_lookup`` slot (the online encoder), swaps it into that slot and moves the text encoder to the CPU
        (still a registered child: state_dict keys unchanged). ``encode_text`` reads ``text_lookup`` either way.

        Args:
            prompts: list[str], the dataset's full prompt set.

        Returns:
            None. Default: the family cannot cache its text; its yaml must say ``text_cache: false``.
        """
        raise NotImplementedError(f"{type(self).__name__} has no text cache; set model.video_backbone.text_cache=false")


__all__ = [
    "BlockLoopState",
    "TextCondition",
    "VideoBackbone",
]
