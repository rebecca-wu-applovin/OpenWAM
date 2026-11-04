"""Video backbone registry + factory.

Decorator-based registration, mirroring ``openwam/model/architectures/registry.py`` for
architectures. A backbone class registers under one or more names (the Wan class backs several
yamls); ``build_video_backbone`` is the factory the architecture calls for training and deploy.

Every backbone has exactly two constructors, picked by the explicit :class:`LoadMode` the entry point
(trainer / deploy loader) decided once:
  - ``PRETRAINED`` -> ``cls.from_pretrained(cfg, assets)``: the upstream weights at the ``assets`` paths (fresh training);
  - ``CHECKPOINT`` -> ``cls.from_checkpoint_dir(cfg, ckpt_dir)``: an empty shell of one of our checkpoints
    (deploy, finetune and resume alike; the caller fills the weights).
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Type

from omegaconf import DictConfig

from openwam.model.video_backbone.base import VideoBackbone

_VIDEO_BACKBONE_REGISTRY: Dict[str, Type[VideoBackbone]] = {}


class LoadMode(Enum):
    """Where a backbone's weights come from: upstream pretrained files, or one of our checkpoint dirs."""

    PRETRAINED = "pretrained"
    CHECKPOINT = "checkpoint"


def register_video_backbone(name: str):
    """Decorator registering a VideoBackbone class under ``name``.

    Args:
        name: str registry key (the yaml's ``video_backbone.name``); one class may register several names.

    Returns:
        the decorator; it returns the class unchanged and raises ValueError for a name already taken.
    """

    def _wrap(cls: Type[VideoBackbone]) -> Type[VideoBackbone]:
        """cls: the VideoBackbone subclass. Returns: cls, now registered under ``name``."""
        if name in _VIDEO_BACKBONE_REGISTRY:
            raise ValueError(f"Video backbone '{name}' already registered")
        _VIDEO_BACKBONE_REGISTRY[name] = cls
        return cls

    return _wrap


def build_video_backbone(
    cfg: DictConfig, mode: LoadMode, assets: Optional[DictConfig], ckpt_dir: Optional[str]
) -> VideoBackbone:
    """Build the backbone ``cfg.name`` names, with the constructor ``mode`` selects.

    Args:
        cfg: ``cfg.model.video_backbone`` (DictConfig, what the model is); ``cfg.name`` is the registry key.
        mode: LoadMode, decided by the entry point (trainer: PRETRAINED without a ckpt, CHECKPOINT for
            finetune / resume; deploy: always CHECKPOINT).
        assets: ``cfg.assets.video_backbone`` (DictConfig: upstream paths + load mode) for PRETRAINED, None for
            CHECKPOINT (the checkpoint dir holds everything).
        ckpt_dir: str checkpoint dir for CHECKPOINT, None for PRETRAINED.

    Returns:
        VideoBackbone: loaded (PRETRAINED) or a meta shell to be filled by the caller (CHECKPOINT).
        Raises KeyError for an unknown name, ValueError when ``assets`` / ``ckpt_dir`` do not fit ``mode``.
    """
    name = cfg.name
    if name not in _VIDEO_BACKBONE_REGISTRY:
        available = ", ".join(sorted(_VIDEO_BACKBONE_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown video backbone '{name}'. Available: {available}")
    cls = _VIDEO_BACKBONE_REGISTRY[name]
    if mode is LoadMode.PRETRAINED:
        # Upstream files: the paths come from assets, never from a checkpoint.
        # The `is None` checks validate the args against the explicit mode; they never infer the mode.
        if assets is None or ckpt_dir is not None:
            raise ValueError(
                f"{name}: LoadMode.PRETRAINED reads the upstream weights at assets.video_backbone; "
                f"got assets={'set' if assets is not None else None}, ckpt_dir={ckpt_dir!r}"
            )
        return cls.from_pretrained(cfg, assets)
    if mode is LoadMode.CHECKPOINT:
        # Checkpoint shell: everything it reads lives in ckpt_dir; upstream paths would be ignored.
        # As above, the `is None` checks only reject args that do not fit the mode.
        if ckpt_dir is None or assets is not None:
            raise ValueError(
                f"{name}: LoadMode.CHECKPOINT reads only ckpt_dir (the checkpoint's assets live there); "
                f"got ckpt_dir={ckpt_dir!r}, assets={'set' if assets is not None else None}"
            )
        return cls.from_checkpoint_dir(cfg, ckpt_dir)
    raise ValueError(f"unknown LoadMode {mode!r}")
