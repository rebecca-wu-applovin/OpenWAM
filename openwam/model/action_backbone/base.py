"""Action-stream backbone ABC.

:class:`ActionDiTBackbone` — for the DualSystem ActionDiT: a standalone action
transformer driven either by bridge cross-attention (``forward``) or by the MoT
joint-attention loop (``prepare_state`` / ``pre_attn_at_layer`` /
``post_attn_at_layer`` / ``extract_prediction``). Declares that contract as
abstract.

It inherits ``nn.Module, ABC`` directly — concrete subclasses inherit so the
state_dict lives at ``action_backbone.<param>`` with no wrapping prefix.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
from torch import nn

from openwam.model.common.flow_scheduler import FlowScheduler

if TYPE_CHECKING:
    from openwam.model.architectures.base import ActionState


class ActionDiTBackbone(nn.Module, ABC):
    """ABC for standalone DualSystem action transformers (ActionDiT).

    The action stream is a separate transformer that couples to the video stream
    either by bridge cross-attention (``forward``) or by the MoT joint
    self-attention loop. Both coupling paths plus the geometry the MoT driver
    validates against the video backbone are declared abstract here.
    """

    # Set by the subclass __init__ (no default: a subclass that forgets them fails on first read).
    _bridge_layers: tuple[int, ...]

    def __init__(self, scheduler: FlowScheduler):
        """Args:
            scheduler: FlowScheduler of the action stream (yaml ``shift_action`` + ``train_timestep``); its ``shift``
                is the only copy of the action α-shift, read by training and inference alike.
        """
        super().__init__()
        self.scheduler = scheduler

    @property
    def bridge_layers(self) -> tuple[int, ...]:
        """Video DiT blocks where the action stream couples to the video stream: the architecture iterates
        the video block loop and cross-attends at these block ids."""
        return self._bridge_layers

    def set_dtype_device(self, dtype, device) -> None:
        """Move action backbone params/buffers to (dtype, device)."""
        self.to(dtype=dtype, device=device)

    def save_deploy_assets(self, output_dir: str) -> None:
        """Default no-op: action weights are fully captured by the safetensors
        checkpoint, no external artifacts to copy. Part of the architecture's
        deploy-asset hook contract.

        Args:
            output_dir: str, the checkpoint dir (unused).

        Returns:
            None.
        """

    @property
    def uses_proprioception(self) -> bool:
        """Whether this backbone consumes a ``proprio`` input. Default False."""
        return False

    @property
    @abstractmethod
    def num_heads(self) -> int: ...

    @property
    @abstractmethod
    def head_dim(self) -> int: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...

    @abstractmethod
    def forward(
        self,
        action_tokens: torch.Tensor,
        bridges: Dict[int, torch.Tensor],
        timestep: torch.Tensor,
        *,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> torch.Tensor:
        """joint_cross_attn coupling: standalone action prediction from per-layer bridges.

        Args:
            action_tokens: (B, T_action, action_dim) noisy action sequence.
            bridges: {block_id: (B, T_video, video_dim)} video hidden state captured at every
                ``bridge_layers`` block.
            timestep: (B,) or (1,) action diffusion timestep.
            context: (B, L, text_dim) raw text/proprio context, or None.
            context_mask: (B, L) bool, True = attend, or None.
            use_gradient_checkpointing / use_gradient_checkpointing_offload: bool, per-block checkpointing.

        Returns:
            (B, T_action, action_dim) predicted action noise.
        """

    @abstractmethod
    def prepare_state(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> ActionState:
        """MoT coupling: embed the actions and build the per-layer state the driver threads through the blocks.

        Args:
            noisy_actions: (B, T_action, action_dim) noisy action sequence.
            timestep: (B,) or (1,) action diffusion timestep.
            context: (B, L, text_dim) raw text/proprio context, or None.
            context_mask: (B, L) bool, True = attend, or None.
            use_gradient_checkpointing / use_gradient_checkpointing_offload: bool (the driver checkpoints itself).

        Returns:
            ActionState whose ``payload`` holds the (B, T_action, dim) action hidden state and its modulation.
        """

    @abstractmethod
    def pre_attn_at_layer(
        self, layer_id: int, astate: ActionState
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """MoT coupling: first half of an action block (norm + AdaLN + Q/K/V + RoPE).

        Args:
            layer_id: int, block index.
            astate: ActionState from :meth:`prepare_state` (or the previous layer).

        Returns:
            (q, k, v, post_state): q / k / v (B, T_action, num_heads * head_dim), ready to join the video Q/K/V;
            post_state, the dict :meth:`post_attn_at_layer` consumes.
        """

    @abstractmethod
    def post_attn_at_layer(
        self,
        layer_id: int,
        astate: ActionState,
        attn_out: torch.Tensor,
        post_state: dict,
    ) -> ActionState:
        """MoT coupling: second half of an action block (gate + context cross-attn + FFN).

        Args:
            layer_id: int, block index.
            astate: ActionState of this layer.
            attn_out: (B, T_action, num_heads * head_dim) unprojected joint-attention output of the action rows.
            post_state: dict returned by :meth:`pre_attn_at_layer` for this layer.

        Returns:
            ActionState with the payload's action hidden state advanced by one block.
        """

    @abstractmethod
    def extract_prediction(self, astate: ActionState) -> torch.Tensor:
        """MoT coupling: decode the final action hidden state into a prediction.

        Args:
            astate: ActionState after the last block.

        Returns:
            (B, T_action, action_dim) predicted action noise.
        """


__all__ = ["ActionDiTBackbone"]
