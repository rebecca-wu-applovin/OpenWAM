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
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.architectures.base import BaseWAMArchitecture, TrainOptions
from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver
from openwam.model.architectures.registry import register_architecture
from openwam.model.architectures.utils.common import resolve_bridge_layers
from openwam.model.compile_options import (
    compile_enabled,
    section_enabled,
    self_attn_compile_cfg,
    torch_compile_kwargs,
)
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

    def __init__(self, cfg: dict, *, layout, mode: LoadMode, ckpt_dir: Optional[str], train_options: TrainOptions):
        """Arguments as in :class:`BaseWAMArchitecture`; ``cfg`` also carries the ActionDiT geometry."""
        super().__init__(cfg, layout=layout, mode=mode, ckpt_dir=ckpt_dir, train_options=train_options)
        self._mot_driver: DualSystemMoTDriver | None = None
        self._compiled_mot_run_joint_loop: Optional[Callable[..., Tuple[object, object]]] = None
        # ActionDiT geometry defaults to the video backbone's; explicit cfg values win.
        cfg = {
            "num_dit_layers": self.video_backbone.num_layers,
            "video_dim": self.video_backbone.dim,
            "num_heads": self.video_backbone.num_heads,
            "attn_head_dim": self.video_backbone.head_dim,
            **cfg,
        }
        bl = resolve_bridge_layers(cfg)
        video_dim = int(cfg["video_dim"])
        # Default the action-side raw context width to the loaded backbone's
        # text_dim (Wan T5-XXL=4096); explicit cfg/CLI still wins.
        text_dim = int(cfg.get("text_dim", self.video_backbone.text_dim))
        self._init_proprio_context(cfg, text_dim=text_dim)

        # FastWAM-Joint compat: action residual hidden_dim may differ from
        # video_dim. The MoT driver only requires num_heads / attn_head_dim
        # parity (validated at DualSystemMoTDriver.__init__).
        action_dim_hidden = int(cfg.get("dim", 1024))
        num_heads = int(cfg.get("num_heads", 24))
        attn_head_dim = int(cfg.get("attn_head_dim", video_dim // num_heads))

        self.action_backbone = ActionDiT(
            action_dim=int(cfg["action_dim"]),
            dim=action_dim_hidden,
            ffn_dim=int(cfg.get("ffn_dim", 4 * action_dim_hidden)),
            num_heads=num_heads,
            num_layers=len(bl),
            video_dim=video_dim,
            bridge_layers=bl,
            variant="joint_self_attn",
            attn_head_dim=attn_head_dim,
            text_dim=text_dim,
            shift_action=float(cfg["shift_action"]),
        )

        # The MoT driver pairs the two backbones (both built by now).
        self._mot_checkpoint_mixed_attn = bool(cfg.get("mot_checkpoint_mixed_attn", True))
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

    def apply_compile_optimizations(self, compile_cfg) -> None:
        """Compile the eval-time MoT joint loop when requested."""

        super().apply_compile_optimizations(compile_cfg)
        self._compiled_mot_run_joint_loop = None
        if not compile_enabled(compile_cfg, default=False, strict=True):
            return

        section = self_attn_compile_cfg(compile_cfg)
        if not section_enabled(section, default=True):
            logger.info("dual self-attn MoT compile disabled by config; running eager.")
            return
        driver = self._mot_driver

        def _run_joint_loop(vstate, astate):
            return driver.run_joint_loop(
                vstate,
                astate,
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
            )

        kwargs = torch_compile_kwargs(section, default_mode="reduce-overhead")
        # A compile setup failure propagates: no silent eager fallback.
        self._compiled_mot_run_joint_loop = torch.compile(_run_joint_loop, **kwargs)
        logger.info("Enabled dual self-attn MoT compile with torch.compile kwargs=%s", kwargs)

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
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        compiled_loop = self._compiled_mot_run_joint_loop
        if compiled_loop is not None and not use_gradient_checkpointing and not use_gradient_checkpointing_offload:
            # A failure of the compiled loop propagates: no silent eager fallback.
            vstate, astate = compiled_loop(vstate, astate)
        else:
            vstate, astate = driver.run_joint_loop(
                vstate,
                astate,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
        return video_backbone.finalize(vstate), action_backbone.extract_prediction(astate)


__all__ = ["DualSystemSelfAttnArchitecture"]
