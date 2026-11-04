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

import torch
from torch import nn

from openwam.model.action_backbone.scheduler import ActionScheduler


class ActionDiTBackbone(nn.Module, ABC):
    """ABC for standalone DualSystem action transformers (ActionDiT).

    The action stream is a separate transformer that couples to the video stream
    either by bridge cross-attention (``forward``) or by the MoT joint
    self-attention loop. Both coupling paths plus the geometry the MoT driver
    validates against the video backbone are declared abstract here.
    """

    def __init__(self):
        super().__init__()
        self.scheduler = ActionScheduler()
        self._shift_action = None

    @property
    def shift_action(self):
        """Optional α-shift for the action scheduler — single source of truth read by
        the architecture for both training and inference. ``None`` (the default) falls
        back to the scheduler template; concrete backbones may set ``self._shift_action``
        in ``__init__``. Symmetric to ``VideoBackbone.shift_video``."""
        return self._shift_action

    @property
    def bridge_layers(self) -> tuple[int, ...]:
        """Video DiT blocks where the action stream couples to the video stream.
        Default none. ActionDiT overrides with its bridge cross-attn layers — the
        architecture iterates the video block loop and cross-attends at these
        block ids."""
        return getattr(self, "_bridge_layers", ())

    def apply_expert(self, layer_id: int, x_action, t_mod):
        """ActionDiT couples by cross-attention, not expert FFN — never called.
        Kept for a uniform coupling contract; raises if reached."""
        raise NotImplementedError(f"{type(self).__name__} has no expert FFN (couples by bridge cross-attention).")

    def set_dtype_device(self, dtype, device) -> None:
        """Move action backbone params/buffers to (dtype, device)."""
        self.to(dtype=dtype, device=device)

    def save_deploy_assets(self, output_dir: str, cfg) -> None:
        """Default no-op: action weights are fully captured by the safetensors
        checkpoint, no external artifacts to copy. Part of the architecture's
        deploy-asset hook contract."""

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
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """joint_cross_attn coupling: standalone action prediction from per-layer bridges."""
        raise NotImplementedError

    @abstractmethod
    def prepare_state(self, *args, **kwargs):
        """MoT coupling: build the per-layer action state consumed by the driver."""
        raise NotImplementedError

    @abstractmethod
    def pre_attn_at_layer(self, *args, **kwargs):
        """MoT coupling: first half of an action block (norm + AdaLN + Q/K/V + RoPE)."""
        raise NotImplementedError

    @abstractmethod
    def post_attn_at_layer(self, *args, **kwargs):
        """MoT coupling: second half of an action block (gate + context cross-attn + FFN)."""
        raise NotImplementedError

    @abstractmethod
    def extract_prediction(self, *args, **kwargs) -> torch.Tensor:
        """MoT coupling: decode the final action hidden state into a prediction."""
        raise NotImplementedError


__all__ = ["ActionDiTBackbone"]
