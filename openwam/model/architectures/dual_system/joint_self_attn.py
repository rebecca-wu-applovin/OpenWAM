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

import logging
from typing import Optional, Tuple

from omegaconf import DictConfig
from torch import Tensor

from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.architectures.base import BaseWAMArchitecture, TrainOptions
from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver
from openwam.model.architectures.registry import register_architecture
from openwam.model.architectures.utils.common import bridge_layers
from openwam.model.video_backbone.base import TextCondition
from openwam.model.video_backbone.registry import LoadMode

logger = logging.getLogger(__name__)


@register_architecture(
    "dual_system_self_attn",
    framework="dual_system",
    variant="joint_self_attn",
)
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
        super().__init__(cfg, mode=mode, assets=assets, ckpt_dir=ckpt_dir, train_options=train_options)
        self._mot_driver: DualSystemMoTDriver | None = None
        ab, geometry = cfg.action_backbone, cfg.video_backbone.geometry
        # One ActionDiT block per selected video layer; geometry is the yaml block _build_video_backbone checked.
        bl = bridge_layers(int(geometry.num_layers), int(cfg.architecture.bridge_interval))
        self._init_proprio_context(
            enabled=bool(cfg.architecture.use_proprioception),
            state_dim=int(cfg.state_dim),
            text_dim=int(geometry.text_dim),
        )

        # FastWAM-Joint compat: the action residual width (dim) may differ from video_dim; the MoT driver
        # only requires num_heads / head_dim parity (validated at DualSystemMoTDriver.__init__).
        self.action_backbone = ActionDiT.from_cfg(
            ab, action_dim=int(cfg.action_dim), bridge_layers=bl, variant="joint_self_attn"
        )

        # The MoT driver pairs the two backbones (both built by now).
        self._mot_checkpoint_mixed_attn = bool(train_options.mot_checkpoint_mixed_attn)
        self.build_mot_driver()

    def build_mot_driver(self) -> DualSystemMoTDriver:
        """Construct the :class:`DualSystemMoTDriver` from the current backbones (re-callable).

        Returns:
            DualSystemMoTDriver, also stored as ``self._mot_driver``.
        """
        self._mot_driver = DualSystemMoTDriver(
            self.video_backbone,
            self.action_backbone,
            frame_mask=self.frame_mask,
            mot_checkpoint_mixed_attn=self._mot_checkpoint_mixed_attn,
        )
        return self._mot_driver

    @property
    def mot_driver(self) -> DualSystemMoTDriver | None:
        """The MoT joint-attention driver (None if the architecture wasn't fully built)."""
        return self._mot_driver

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

        driver = self._mot_driver
        astate = action_backbone.prepare_state(
            noisy_actions,
            action_timestep,
            context=action_text.context,
            context_mask=action_text.context_mask,
        )
        # The driver checkpoints each layer itself (the action state is built once, outside the loop).
        vstate, astate = driver.run_joint_loop(
            vstate,
            astate,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        return video_backbone.finalize(vstate), action_backbone.extract_prediction(astate)


__all__ = ["DualSystemSelfAttnArchitecture"]
