"""Video backbone package for OpenWAM.

This package is the public facade. The pieces live in dedicated modules:
  - ``base`` — :class:`VideoBackbone` ABC + :class:`BlockLoopState`
  - ``registry``          — the registry dict + ``register_video_backbone`` /
                            ``build_video_backbone``
This file re-exports them and registers the built-in Wan backbone at the bottom::

    from openwam.model.video_backbone import build_video_backbone
    backbone = build_video_backbone("wan22_ti2v_5b", cfg)

Adding a new backbone:
1. Subclass :class:`VideoBackbone` (implement prepare/run_block/finalize).
2. Register it at the bottom of this file via ``register_video_backbone("name")(YourClass)``.
3. Set ``video_backbone.name: your_name`` in the model config yaml.
"""

from __future__ import annotations

from openwam.model.video_backbone.base import BlockLoopState, VideoBackbone
from openwam.model.video_backbone.registry import (
    _VIDEO_BACKBONE_REGISTRY,
    build_video_backbone,
    register_video_backbone,
)

__all__ = [
    "BlockLoopState",
    "VideoBackbone",
    "Wan22Ti2v",
    "Cosmos3EdgeVideoBackbone",
    "build_video_backbone",
    "register_video_backbone",
    "_VIDEO_BACKBONE_REGISTRY",
]

# Built-in registrations (kept at the bottom so the implementation can import
# from ``registry`` / ``base`` without circular issues).
from openwam.model.video_backbone.wan.backbone import Wan22Ti2v  # noqa: E402

register_video_backbone("wan22_ti2v_5b")(Wan22Ti2v)

# Cosmos3 backbones (Edge 4B / Nano 16B share one class; the variant is picked
# by cfg.name in cosmos3/pipeline_builder._COSMOS3_VARIANTS). CPU-only-CI safe —
# the vendored transformer (which pulls ``diffusers``) is deferred to ``from_pretrained``.
from openwam.model.video_backbone.cosmos3.backbone import Cosmos3EdgeVideoBackbone  # noqa: E402

register_video_backbone("cosmos3_edge")(Cosmos3EdgeVideoBackbone)
register_video_backbone("cosmos3_nano")(Cosmos3EdgeVideoBackbone)

# LongLive2.0-Robot-S: Wan2.2-TI2V-5B structure with the LongLive DiT weights (07).
from openwam.model.video_backbone.wan.longlive import LongLiveRobotS  # noqa: E402

register_video_backbone("longlive2_robot_s")(LongLiveRobotS)

# GE-Base-fast (Genie Envisioner): multi-view LTX DiT + LTX VAE / T5 (07). CPU-only-CI safe —
# diffusers / transformers are imported inside from_pretrained.
from openwam.model.video_backbone.ge.backbone import GEBaseFastVideoBackbone  # noqa: E402

register_video_backbone("ge_base_fast")(GEBaseFastVideoBackbone)

# flux-3-action-base video trunk (07). CPU-only-CI safe — NATTEN (VAE) and transformers (Qwen)
# are imported inside from_pretrained.
from openwam.model.video_backbone.flux import FluxActionVideoBackbone  # noqa: E402

register_video_backbone("flux3_action_base")(FluxActionVideoBackbone)
