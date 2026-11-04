"""DualSystem joint self-attention architecture.

True joint attention (MMDiT / FastWAM MoT style): at every transformer
layer, the video and action backbones each compute Q/K/V independently
through ``pre_attn_at_layer``; :class:`DualSystemMoTDriver` concatenates the
two modalities, runs a single mixed self-attention, splits the result,
and feeds each slice back through ``post_attn_at_layer``.

The driver lives on ``self._mot_driver`` as a plain Python object with
no parameters; ``forward`` delegates the per-layer loop to it.
"""

from __future__ import annotations

from typing import Optional, Tuple

from omegaconf import DictConfig
from torch import Tensor

from openwam.model.action_backbone.separate_action_dit import MoTActionDiT
from openwam.model.architectures.base import BaseWAMArchitecture, TrainOptions
from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver
from openwam.model.architectures.registry import register_architecture
from openwam.model.video_backbone.base import TextCondition
from openwam.model.video_backbone.registry import LoadMode


@register_architecture(framework="dual_system", variant="joint_self_attn")
class DualSystemSelfAttnArchitecture(BaseWAMArchitecture):
    """DualSystem with true joint self-attention (FastWAM MoT pattern)."""

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
        # One action block per video layer (one mixed attention each); geometry is the yaml block
        # _build_video_backbone checks against the built module.
        super().__init__(
            cfg, mode=mode, assets=assets, ckpt_dir=ckpt_dir, train_options=train_options,
            action_dit_cls=MoTActionDiT, num_action_layers=int(cfg.video_backbone.geometry.num_layers),
        )
        # The driver pairs the two backbones; it validates their layer / head geometry parity.
        self._mot_driver = DualSystemMoTDriver(
            self.video_backbone,
            self.action_backbone,
            frame_mask=self.frame_mask,
            mot_checkpoint_mixed_attn=train_options.mot_checkpoint_mixed_attn,
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
        vstate, action_text = self._prepare_video(
            latents, timestep, text, proprio, proprio_mask,
            use_gradient_checkpointing, use_gradient_checkpointing_offload,
        )
        astate = self.action_backbone.prepare_state(
            noisy_actions,
            action_timestep,
            context=action_text.context,
            context_mask=action_text.context_mask,
        )
        # The driver checkpoints each layer itself (the action state is built once, outside the loop).
        vstate, astate = self._mot_driver.run_joint_loop(
            vstate,
            astate,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        return self.video_backbone.finalize(vstate), self.action_backbone.extract_prediction(astate)


__all__ = ["DualSystemSelfAttnArchitecture"]
