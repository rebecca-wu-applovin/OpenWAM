"""DualSystem joint cross-attention architecture.

Bridge-collection mode: the video DiT runs to completion; hidden states
at the ``bridge_interval``-selected layers are captured along the way and feed
a separate ActionDiT via cross-attention. ``detach_bridge=True`` blocks
action gradients from flowing back into the video DiT.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from omegaconf import DictConfig
from torch import Tensor

from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.architectures.base import BaseWAMArchitecture, TrainOptions
from openwam.model.architectures.registry import register_architecture
from openwam.model.architectures.utils.common import bridge_layers
from openwam.model.video_backbone.base import TextCondition
from openwam.model.video_backbone.registry import LoadMode

logger = logging.getLogger(__name__)


@register_architecture("dual_system_cross_attn", framework="dual_system", variant="joint_cross_attn")
class DualSystemCrossAttnArchitecture(BaseWAMArchitecture):
    """DualSystem with bridge cross-attention.

    Action processing happens **after** the video DiT block loop completes:
    captured per-block bridge features feed the ActionDiT's cross-attention
    layers. The video DiT is not aware of the action stream.
    """

    def __init__(
        self,
        cfg: DictConfig,
        *,
        mode: LoadMode,
        assets: Optional[DictConfig],
        ckpt_dir: Optional[str],
        train_options: TrainOptions,
    ):
        """Arguments as in :class:`BaseWAMArchitecture`; ``cfg.action_backbone`` sizes the ActionDiT."""
        super().__init__(cfg, mode=mode, assets=assets, ckpt_dir=ckpt_dir, train_options=train_options)
        ab, geometry = cfg.action_backbone, cfg.video_backbone.geometry
        # One ActionDiT block per bridged video layer; geometry is the yaml block _build_video_backbone checked,
        # so every id in range(0, num_layers, interval) is a layer the video forward runs.
        bl = bridge_layers(int(geometry.num_layers), int(cfg.architecture.bridge_interval))
        self._init_proprio_context(
            enabled=bool(cfg.architecture.use_proprioception),
            state_dim=int(cfg.state_dim),
            text_dim=int(geometry.text_dim),
        )
        self._detach_bridge = bool(cfg.architecture.detach_bridge)

        # The action residual width (dim) may differ from num_heads * head_dim: Q/K/V project across the gap.
        self.action_backbone = ActionDiT.from_cfg(
            ab, action_dim=int(cfg.action_dim), bridge_layers=bl, variant="joint_cross_attn"
        )

    @property
    def detach_bridge(self) -> bool:
        return self._detach_bridge

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
        """Predict actions from collected bridge tensors.

        Args:
            noisy_actions: (B, T_a, A) noisy action latents.
            bridges: {video layer id: (B, L_v, video_dim) bridge hidden state}.
            action_timestep: (B,) action timestep.
            context / context_mask: (B, L, text_dim) action-side context and its (B, L) mask, or None.
            use_gradient_checkpointing / use_gradient_checkpointing_offload: bool training checkpoint switches.

        Returns: (B, T_a, A) action prediction.
        """
        return self.action_backbone(
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

        # Every bridge layer is a video layer (range over geometry.num_layers), so the bridge set is complete here.
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
