"""DualSystem joint cross-attention architecture.

Bridge-collection mode: the video DiT runs to completion; hidden states
at the ``bridge_interval``-selected layers are captured along the way and feed
a separate ActionDiT via cross-attention. ``detach_bridge=True`` blocks
action gradients from flowing back into the video DiT.
"""

from __future__ import annotations

from typing import Optional, Tuple

from omegaconf import DictConfig
from torch import Tensor

from openwam.model.action_backbone.separate_action_dit import CrossAttnActionDiT
from openwam.model.architectures.base import BaseWAMArchitecture, TrainOptions
from openwam.model.architectures.registry import register_architecture
from openwam.model.video_backbone.base import TextCondition
from openwam.model.video_backbone.registry import LoadMode


def select_bridge_layers(num_video_layers: int, bridge_interval: int) -> tuple[int, ...]:
    """Video DiT layers whose hidden state feeds an ActionDiT block (one block each, ascending).

    Args:
        num_video_layers: int, the video backbone's layer count (yaml ``geometry.num_layers``).
        bridge_interval: int >= 1 (checked by resolve_config), stride over the video layers (1 = every layer).

    Returns: tuple of ascending layer ids ``range(0, num_video_layers, bridge_interval)``.
    """
    return tuple(range(0, num_video_layers, bridge_interval))


@register_architecture(framework="dual_system", variant="joint_cross_attn")
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
        # One ActionDiT block per bridged video layer; geometry is the yaml block _build_video_backbone checks,
        # so every id in range(0, num_layers, interval) is a layer the video forward runs.
        num_video_layers = int(cfg.video_backbone.geometry.num_layers)
        bridge_ids = select_bridge_layers(num_video_layers, int(cfg.architecture.bridge_interval))
        super().__init__(
            cfg, mode=mode, assets=assets, ckpt_dir=ckpt_dir, train_options=train_options,
            action_dit_cls=CrossAttnActionDiT, num_action_layers=len(bridge_ids),
        )
        self.bridge_layers = bridge_ids
        self._detach_bridge = bool(cfg.architecture.detach_bridge)

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
        vstate, action_text = self._prepare_video(
            latents, timestep, text, proprio, proprio_mask,
            use_gradient_checkpointing, use_gradient_checkpointing_offload,
        )

        # The video's own self-attention reads the frame mask's video block (no action tokens in this stream).
        masks = video_backbone.build_joint_attention_masks(self.frame_mask, vstate, 0)
        bridge_set = frozenset(self.bridge_layers)
        # Collected in ascending layer order = ActionDiT block order.
        bridges: list[Tensor] = []
        for block_id in range(video_backbone.num_layers):
            vstate = video_backbone.run_block(block_id, vstate, masks[block_id])
            if block_id in bridge_set:
                bridge = vstate.hidden_states
                bridges.append(bridge.detach() if self._detach_bridge else bridge)

        video_pred = video_backbone.finalize(vstate)
        action_pred = self.action_backbone(
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
