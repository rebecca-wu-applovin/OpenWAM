"""Deploy-side inference engine interface and the per-request denoise schedule.

``BaseInferenceEngine`` declares the engine interface; the production engine is
:class:`openwam.deploy.sparse_engine.SparseHistoryInferenceEngine` (every checkpoint: the first-frame layout is
the H = 1 history). :func:`build_schedule` turns the deploy config + per-request overrides into the
``(t_video, t_action)`` schedule the architecture's denoising loop runs.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from openwam.deploy.denoise_schedule import make_schedule
from openwam.model.architectures.base import BaseWAMArchitecture

logger = logging.getLogger(__name__)


def build_schedule(inf_cfg, architecture, conditions: dict):
    """Denoise schedule of one request: deploy config, per-request overrides, the checkpoint's stream shifts.

    Args:
        inf_cfg: ``cfg.inference`` (``denoise_mode`` / ``denoise_steps`` / async controls).
        architecture: the WAM architecture (``video_scheduler`` / ``action_scheduler`` / backbone shifts).
        conditions: request dict; ``denoise_mode`` / ``denoise_steps`` / ``lead_modality`` /
            ``variance_shift_alpha`` / ``linear_offset`` / ``shift`` / ``shift_video`` override the config.

    Returns: (schedule list of (t_video, t_action), denoise_steps int, action shift float).
    """
    denoise_mode = conditions.get("denoise_mode", getattr(inf_cfg, "denoise_mode", "sync"))
    denoise_steps = conditions.get("denoise_steps", inf_cfg.denoise_steps)
    lead_modality = conditions.get("lead_modality", getattr(inf_cfg, "lead_modality", "video"))
    variance_shift_alpha = conditions.get("variance_shift_alpha", getattr(inf_cfg, "variance_shift_alpha", 1.0))
    linear_offset = conditions.get("linear_offset", getattr(inf_cfg, "linear_offset", 0.0))
    # Single source of truth for each stream's α-shift is the backbone
    # property — ``action_backbone.shift_action`` and
    # ``video_backbone.shift_video`` — set via the model yaml and saved in
    # the checkpoint, so the training sigma buffer (set by
    # ``init_training_schedulers`` via the same properties) and the
    # inference trajectory match. Deploy carries no independent shift knob.
    # ``conditions[...]`` allows a per-request override (smoke tests /
    # ablations). ``getattr`` guards lightweight stub architectures.
    _ab = getattr(architecture, "action_backbone", None)
    shift = conditions.get("shift", getattr(_ab, "shift_action", None) if _ab is not None else None)
    if shift is None:
        shift = 5.0
    _vb = getattr(architecture, "video_backbone", None)
    shift_video = conditions.get(
        "shift_video",
        getattr(_vb, "shift_video", None) if _vb is not None else None,
    )

    schedule = make_schedule(
        denoise_mode,
        video_scheduler=architecture.video_scheduler,
        action_scheduler=architecture.action_scheduler,
        num_steps=denoise_steps,
        shift=shift,
        shift_video=shift_video,
        lead=lead_modality,
        alpha=variance_shift_alpha,
        offset=linear_offset,
    )
    return schedule, denoise_steps, shift


class BaseInferenceEngine(ABC):
    """Inference engine base class.

    Concrete engines implement ``generate`` which accepts observation
    conditions and produces video frames and/or action trajectories.

    Args:
        cfg: Hydra config.
        architecture: WAM architecture wrapping the action backbone.
        action_backbone: Optional action backbone reference for implementations that still expose one.
    """

    require_architecture = False

    def __init__(self, cfg, architecture: Optional[BaseWAMArchitecture] = None, action_backbone=None):
        if self.require_architecture and architecture is None:
            raise ValueError("architecture is required")
        self.cfg = cfg
        self.architecture = architecture
        self.action_backbone = action_backbone

    @abstractmethod
    def generate(self, conditions: dict) -> dict:
        """Generate video and/or actions from conditions.

        Args:
            conditions: dict with keys like ``prompt``, ``reference_image``,
                ``context_video``, ``seed``, etc.

        Returns:
            dict with ``video`` (Tensor) and ``actions`` (Tensor).
        """
        ...
