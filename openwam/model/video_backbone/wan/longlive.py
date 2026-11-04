"""LongLive2.0-Robot-S video backbone (NVlabs LongLive; 07_sparse_history_backbones).

Robot-S is a Wan2.2-TI2V-5B DiT further trained as an autoregressive robot-video
generator (I2V teacher forcing, not distilled). Its ``model.pt`` holds exactly the
825 Wan2.2 DiT tensors under a ``model.`` prefix, so the backbone is
:class:`Wan22Ti2v` (same VAE, umT5, DiT code) with the DiT weights swapped.

Following Long-WAM, two settings differ from the Wan arm and live in the yaml:
the umT5 prompt budget (``text_max_length: 128``) and the per-frame causal history
mask (``video_attention_mask_mode=history_per_frame_causal`` in the launcher).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from openwam.model.video_backbone.wan.backbone import Wan22Ti2v

logger = logging.getLogger(__name__)

_PREFIX = "model."


def load_longlive_dit_state(path: str) -> dict:
    """Read a LongLive generator ``model.pt`` and strip the ``model.`` prefix.

    Args:
        path: str, ``model.pt`` (FP32 state dict, 825 tensors for Robot-S).

    Returns:
        dict[str, Tensor] keyed like the Wan2.2 DiT state dict.
    """
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    bad = [k for k in sd if not k.startswith(_PREFIX)]
    if bad:
        raise ValueError(f"LongLive checkpoint {path}: {len(bad)} keys without the '{_PREFIX}' prefix, e.g. {bad[:3]}")
    return {k[len(_PREFIX) :]: v for k, v in sd.items()}


class LongLiveRobotS(Wan22Ti2v):
    """Wan2.2-TI2V-5B structure + LongLive2.0-Robot-S DiT weights."""

    @classmethod
    def from_pretrained(cls, source, *, text_dim: Optional[int] = None, **kw) -> "LongLiveRobotS":
        """Build the Wan2.2 backbone, then (training) load the Robot-S DiT weights strictly.

        Args:
            source: DictConfig model cfg (training: ``video_backbone.dit_weights_path`` is read)
                or a deploy source (weights then come from the OpenWAM checkpoint).
            text_dim / kw: as :meth:`Wan22Ti2v.from_pretrained`.

        Returns:
            LongLiveRobotS.
        """
        # Training passes the architecture params (a dict / DictConfig holding ``video_backbone``);
        # deploy passes a ``_source`` dir or components dict without it.
        vb_cfg = source.get("video_backbone") if hasattr(source, "get") and "video_backbone" in source else None
        if vb_cfg is not None and vb_cfg.get("_source") is not None:
            vb_cfg = None  # deploy rebuild: weights come from the OpenWAM checkpoint
        if vb_cfg is not None:
            # The mask mode lives in model.architecture (flattened into ``source``); check it before
            # the ~20GB load so a launch without the Long-WAM history mask fails fast, not silently.
            required = vb_cfg.get("required_video_attention_mask_mode")
            actual = source.get("video_attention_mask_mode")
            if required is not None and actual != required:
                raise ValueError(
                    f"longlive2_robot_s requires model.architecture.video_attention_mask_mode={required!r} "
                    f"(Long-WAM history mask), got {actual!r}"
                )
        bb = super().from_pretrained(source, text_dim=text_dim, **kw)
        if vb_cfg is not None:
            bb.text_max_length = int(vb_cfg.get("text_max_length", 128))
            weights = vb_cfg.get("dit_weights_path")
            if not weights:
                raise ValueError("longlive2_robot_s needs video_backbone.dit_weights_path (LongLive2.0-Robot-S model.pt)")
            state = load_longlive_dit_state(str(weights))
            dit = bb.dit
            dtype = next(dit.parameters()).dtype
            # strict: the key sets must match the Wan2.2 DiT exactly.
            dit.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=True)
            logger.info("LongLiveRobotS: loaded %d DiT tensors from %s", len(state), weights)
        return bb


__all__ = ["LongLiveRobotS", "load_longlive_dit_state"]
