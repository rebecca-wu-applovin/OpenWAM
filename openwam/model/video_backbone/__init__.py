"""Video backbone package for OpenWAM.

This package is the public facade. The pieces live in dedicated modules:
  - ``base`` — :class:`VideoBackbone` ABC + :class:`BlockLoopState` + :class:`TextCondition`
  - ``registry``          — the registry dict + ``register_video_backbone`` /
                            ``build_video_backbone`` / ``LoadMode``
This file re-exports them and registers the built-in backbones at the bottom::

    from openwam.model.video_backbone import LoadMode, build_video_backbone
    backbone = build_video_backbone(cfg.model.video_backbone, LoadMode.PRETRAINED, None)

Adding a new backbone = implement the 14 abstract members of :class:`VideoBackbone` + write one yaml:
1. ``<family>/backbone.py``: subclass :class:`VideoBackbone` and implement its 14 abstract members (construction
   ``from_pretrained`` / ``from_checkpoint_dir``; geometry ``dim`` / ``num_layers`` / ``num_heads`` / ``head_dim`` /
   ``text_dim`` / ``scheduler``; IO ``encode_pixels`` / ``decode_latents`` / ``encode_text``; block loop
   ``prepare`` / ``run_block`` / ``finalize``). The shared preprocessing flow and masks come from the base.
2. Register it at the bottom of this file via ``register_video_backbone("name")(YourClass)``.
3. Write ``configs/model/video_backbone/<name>.yaml`` (``name: <name>`` + the module geometry).
See ``assets/openwam_usage_docs/architecture-extension.md`` ("Add a video DiT backbone").
"""

from __future__ import annotations

from openwam.model.video_backbone.base import BlockLoopState, TextCondition, VideoBackbone
from openwam.model.video_backbone.registry import (
    _VIDEO_BACKBONE_REGISTRY,
    LoadMode,
    build_video_backbone,
    register_video_backbone,
)

__all__ = [
    "BlockLoopState",
    "LoadMode",
    "TextCondition",
    "VideoBackbone",
    "WanBackbone",
    "Cosmos3EdgeVideoBackbone",
    "build_video_backbone",
    "register_video_backbone",
    "_VIDEO_BACKBONE_REGISTRY",
]

# Built-in registrations (kept at the bottom so the implementation can import
# from ``registry`` / ``base`` without circular issues).
# Wan family: one class, the variants are yamls (wan22_ti2v_tiny keeps name wan22_ti2v_5b;
# longlive2_robot_s = Wan2.2-TI2V-5B with the LongLive2.0-Robot-S DiT weights).
from openwam.model.video_backbone.wan.backbone import WanBackbone  # noqa: E402

register_video_backbone("wan22_ti2v_5b")(WanBackbone)
register_video_backbone("longlive2_robot_s")(WanBackbone)

# Cosmos3 backbones (Edge 4B / Nano 16B share one class; the variant is the yaml's dit / vae
# geometry). CPU-only-CI safe — the vendored transformer (which pulls ``diffusers``) is imported
# inside the constructors.
from openwam.model.video_backbone.cosmos3.backbone import Cosmos3EdgeVideoBackbone  # noqa: E402

register_video_backbone("cosmos3_edge")(Cosmos3EdgeVideoBackbone)
register_video_backbone("cosmos3_nano")(Cosmos3EdgeVideoBackbone)

# GE-Base-fast (Genie Envisioner): multi-view LTX DiT + LTX VAE / T5 (07). CPU-only-CI safe —
# diffusers / transformers are imported inside the constructors.
from openwam.model.video_backbone.ge.backbone import GEBaseFastVideoBackbone  # noqa: E402

register_video_backbone("ge_base_fast")(GEBaseFastVideoBackbone)

# flux-3-action-base video trunk (07). CPU-only-CI safe — NATTEN (VAE) and transformers (Qwen)
# are imported inside the constructors.
from openwam.model.video_backbone.flux import FluxActionVideoBackbone  # noqa: E402

register_video_backbone("flux3_action_base")(FluxActionVideoBackbone)
