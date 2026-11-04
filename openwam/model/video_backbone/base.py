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
    # Required: construction + training preprocessing
    # ================================================================

    @classmethod
    @abstractmethod
    def from_pretrained(cls, source, **kw) -> "VideoBackbone":
        """Build from pretrained weights. ``source``: path / config / specs / pipe object."""

    @abstractmethod
    def preprocess_input_for_train(self, *, frames=None, text=None, **kw) -> dict:
        """Raw training data → tensor dict with at least ``input_latents`` /
        ``context`` / ``seq_lens``. Unconsumed kwargs are dropped via ``**kw``."""

    # ================================================================
    # Required: three-step execution
    # ================================================================

    @abstractmethod
    def prepare(self, **pipeline_inputs) -> BlockLoopState:
        """Pre-loop: patchify / freqs / time_mod / SP. Returns a BlockLoopState."""

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
    def needs_first_frame_skip(self) -> bool:
        """Whether ``latent[0]`` is unconditionally a conditioning frame (Wan I2V)."""
        return False

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
    # Optional: sparse-history layout (H single-frame history latents + one future clip)
    # ================================================================
    # Training builds its inputs as condition + future; deploy builds the same condition from
    # its own observation history and denoises the future from noise.

    def preprocess_sparse_condition(self, history_frames, text, *, num_future_frames: int) -> dict:
        """Conditioning inputs of the sparse-history layout: everything the forward needs except the future.

        Args:
            history_frames: list of B clips, each H PIL frames (oldest first, frame t last).
            text: list of B prompts.
            num_future_frames: int F, future frames the latents will cover (1 + temporal_compression * k).

        Returns: forward kwargs dict with ``first_frame_latents`` (B, z, H, h, w) clean history latents
            (each frame encoded alone), ``num_clean_prefix_frames`` = H, the text context entries, and the
            clip geometry (``num_frames`` = H + F, ``height``, ``width``). No ``input_latents``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support the sparse-history layout.")

    def encode_sparse_future(self, future_frames) -> Tensor:
        """Clean latents of the sparse layout's future clip (training target).

        Args:
            future_frames: list of B clips, each F PIL frames.

        Returns: (B, z, num_future_latents(F), h, w) latents, F frames encoded as one causal clip.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support the sparse-history layout.")

    def num_future_latents(self, num_future_frames: int) -> int:
        """Latent frames of a causally encoded F-frame future clip.

        Args:
            num_future_frames: int F.

        Returns: int 1 + (F - 1) / temporal_compression. Raises ValueError unless F = 1 (mod temporal_compression).
        """
        tc = int(self.temporal_compression)
        f = int(num_future_frames)
        if f < 1 or (f - 1) % tc:
            raise ValueError(f"{type(self).__name__}: future frames must be 1 + {tc}k, got {f}")
        return 1 + (f - 1) // tc

    # ================================================================
    # Optional: deploy preprocessing, decode, and compile
    # ================================================================

    def preprocess_input_for_inference(self, **kw) -> dict:
        """Deploy-time input prep → dict ready for the inference denoising loop.

        Producers pass explicit kwargs (prompt / first_frame_image /
        num_frames / height / width / seed / num_inference_steps / shift / tiled /
        prompt_embed_cache); backbones declare what they consume and
        let ``**kw`` swallow the rest."""
        raise NotImplementedError(f"{type(self).__name__} does not support deploy inference.")

    def decode_video(self, latents: Tensor, *, tiled: bool = True) -> list:
        """Latent ``(B, C, T, H, W)`` → PIL frames. Irreversible encoders omit it."""
        raise NotImplementedError(f"{type(self).__name__} does not support decode_video.")

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

    def save_deploy_assets(self, output_dir: str, cfg) -> None:
        """Make this backbone's slice of the checkpoint self-contained.

        Both halves of self-containment in one place: (1) write the backbone's
        component/tokenizer reconstruction specs into ``cfg`` (so deploy rebuilds
        the module skeletons from ``config.yaml`` without the training-time
        ``model_path``), and (2) copy its non-weight artifact files (tokenizer /
        processor side files) into ``output_dir``. Runs before
        the architecture writes ``config.yaml``. Default no-op."""


__all__ = [
    "BlockLoopState",
    "VideoBackbone",
]
