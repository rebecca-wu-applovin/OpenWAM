"""DualSystem joint cross-attention architecture.

Bridge-collection mode: the video DiT runs to completion; hidden states
at the configured ``bridge_layers`` are captured along the way and feed
a separate ActionDiT via cross-attention. ``detach_bridge=True`` blocks
action gradients from flowing back into the video DiT.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.architectures.base import BaseWAMArchitecture, TrainOptions
from openwam.model.architectures.registry import _cfg_get, register_architecture
from openwam.model.architectures.utils.common import resolve_bridge_layers
from openwam.model.compile_options import (
    compile_enabled,
    cross_attn_compile_cfg,
    section_enabled,
    torch_compile_kwargs,
)
from openwam.model.video_backbone.base import TextCondition
from openwam.model.video_backbone.registry import LoadMode

logger = logging.getLogger(__name__)


def _cross_attn_options(cfg) -> dict:
    return {"detach_bridge": bool(_cfg_get(cfg, "detach_bridge", False))}


@register_architecture(
    "dual_system_cross_attn",
    framework="dual_system",
    variant="joint_cross_attn",
    options_from_cfg=_cross_attn_options,
)
class DualSystemCrossAttnArchitecture(BaseWAMArchitecture):
    """DualSystem with bridge cross-attention.

    Action processing happens **after** the video DiT block loop completes:
    captured per-block bridge features feed the ActionDiT's cross-attention
    layers. The video DiT is not aware of the action stream.
    """

    def __init__(self, cfg: dict, *, layout, mode: LoadMode, ckpt_dir: Optional[str], train_options: TrainOptions):
        """Arguments as in :class:`BaseWAMArchitecture`; ``cfg`` also carries the ActionDiT geometry."""
        super().__init__(cfg, layout=layout, mode=mode, ckpt_dir=ckpt_dir, train_options=train_options)
        self._compiled_action_forward: Optional[Callable[..., Tensor]] = None
        # Action-side geometry defaults to the video backbone's, mirroring ``joint_self_attn`` (cross_attn does
        # not need num_heads / head_dim parity, but the defaults keep the two variants apples-to-apples);
        # explicit cfg values win for ablations.
        cfg = {
            "num_dit_layers": self.video_backbone.num_layers,
            "video_dim": self.video_backbone.dim,
            "num_heads": self.video_backbone.num_heads,
            "attn_head_dim": self.video_backbone.head_dim,
            **cfg,
        }
        bl = resolve_bridge_layers(cfg)
        # The ActionDiT reads one bridge per layer: each must be a video DiT layer the forward actually runs.
        num_video_layers = self.video_backbone.num_layers
        if not bl or not all(0 <= b < num_video_layers for b in bl):
            raise ValueError(
                f"joint_cross_attn: bridge_layers {list(bl)} must be a non-empty subset of the video DiT's "
                f"layers 0..{num_video_layers - 1}"
            )
        video_dim = int(cfg["video_dim"])
        # Default the action-side raw context width to the loaded backbone's
        # text_dim (Wan T5-XXL=4096); explicit cfg/CLI still wins.
        text_dim = int(cfg.get("text_dim", self.video_backbone.text_dim))
        self._init_proprio_context(cfg, text_dim=text_dim)
        self._detach_bridge = bool(cfg.get("detach_bridge", False))

        # Mirror joint_self_attn's heterogeneous-hidden support: when
        # ``attn_head_dim`` is supplied explicitly, the action residual hidden
        # dim (``dim``) is allowed to differ from ``num_heads * attn_head_dim``
        # — Q/K/V project across the gap. Falls back to ``dim // num_heads``
        # for back-compat with older same-width cross_attn configs.
        action_dim_hidden = int(cfg.get("dim", 768))
        num_heads = int(cfg["num_heads"])
        attn_head_dim = cfg.get("attn_head_dim")
        if attn_head_dim is not None:
            attn_head_dim = int(attn_head_dim)

        self.action_backbone = ActionDiT(
            action_dim=int(cfg["action_dim"]),
            dim=action_dim_hidden,
            ffn_dim=int(cfg.get("ffn_dim", 3072)),
            num_heads=num_heads,
            num_layers=len(bl),
            video_dim=video_dim,
            bridge_layers=bl,
            variant="joint_cross_attn",
            attn_head_dim=attn_head_dim,
            text_dim=text_dim,
            shift_action=float(cfg["shift_action"]),
        )

    @property
    def detach_bridge(self) -> bool:
        return self._detach_bridge

    def apply_compile_optimizations(self, compile_cfg) -> None:
        """Compile the tensor-only action cross-attention helper when requested."""

        super().apply_compile_optimizations(compile_cfg)
        self._compiled_action_forward = None
        if not compile_enabled(compile_cfg, default=False, strict=True):
            return

        section = cross_attn_compile_cfg(compile_cfg)
        if not section_enabled(section, default=True):
            logger.info("cross-attn action compile disabled by config; running eager.")
            return

        kwargs = torch_compile_kwargs(section, default_mode="reduce-overhead")
        # A compile setup failure propagates: no silent eager fallback.
        self._compiled_action_forward = torch.compile(self.action_backbone.forward_with_bridge_tuple, **kwargs)
        logger.info("Enabled cross-attn action compile with torch.compile kwargs=%s", kwargs)

    def _predict_actions_from_bridges(
        self,
        noisy_actions: Tensor,
        bridges: dict[int, Tensor],
        action_timestep: Tensor,
        *,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> Tensor:
        """Predict actions from collected bridge tensors, using compile when active."""

        ab = self.action_backbone
        compiled_forward = self._compiled_action_forward
        if compiled_forward is not None and not use_gradient_checkpointing and not use_gradient_checkpointing_offload:
            bridge_tuple = ab.bridge_tuple_from_dict(bridges)
            # A failure of the compiled forward propagates: no silent eager fallback.
            torch.compiler.cudagraph_mark_step_begin()
            return compiled_forward(
                noisy_actions,
                bridge_tuple,
                action_timestep,
                context=context,
                context_mask=context_mask,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )

        return ab(
            noisy_actions,
            bridges,
            action_timestep,
            context=context,
            context_mask=context_mask,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )

    def forward(
        self,
        noisy_actions: Tensor,
        action_timestep: Tensor,
        *,
        latents: Tensor,
        timestep: Tensor,
        text: TextCondition,
        proprio: Tensor,
        proprio_mask: Tensor,
        use_gradient_checkpointing: bool,
        use_gradient_checkpointing_offload: bool,
    ) -> Tuple[Tensor, Tensor]:
        """Joint forward; arguments as in :meth:`BaseWAMArchitecture.forward`. Returns (video pred, action pred)."""
        video_backbone = self.video_backbone
        action_backbone = self.action_backbone

        # Each stream's context (proprio tokens per cfg proprio_input).
        video_text, action_text = self.build_contexts(text, proprio, proprio_mask)
        vstate = video_backbone.prepare(
            latents=latents,
            timestep=timestep,
            text=video_text,
            layout=self.layout,
            proprio=proprio,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )

        # The video's own self-attention reads the frame mask's video block (no action tokens in this stream).
        masks = video_backbone.build_joint_attention_masks(self.frame_mask, vstate, 0)
        bridge_set = frozenset(action_backbone.bridge_layers)
        bridges: dict[int, Tensor] = {}
        detach_bridge = self._detach_bridge

        for block_id in range(video_backbone.num_layers):
            vstate = video_backbone.run_block(block_id, vstate, masks[block_id])
            if block_id in bridge_set:
                bridge = vstate.hidden_states
                bridges[block_id] = bridge.detach() if detach_bridge else bridge

        # Every bridge layer is a video layer (checked at construction), so the bridge set is complete here.
        video_pred = video_backbone.finalize(vstate)
        action_pred = self._predict_actions_from_bridges(
            noisy_actions,
            bridges,
            action_timestep,
            context=action_text.context,
            context_mask=action_text.context_mask,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        return video_pred, action_pred


__all__ = ["DualSystemCrossAttnArchitecture"]
