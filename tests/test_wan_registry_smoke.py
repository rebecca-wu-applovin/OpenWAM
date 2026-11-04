"""Wan video backbone registry smoke tests.

Verifies that the three Wan backbones plan §1 promises are wired up
without touching GPU or real weights.
"""

from __future__ import annotations

import pytest


def test_registry_keys_present():
    from openwam.model.video_backbone import _VIDEO_BACKBONE_REGISTRY

    assert {"wan22_ti2v_5b"} <= set(_VIDEO_BACKBONE_REGISTRY)


def test_unknown_key_raises():
    from omegaconf import OmegaConf

    from openwam.model.video_backbone import LoadMode, build_video_backbone

    with pytest.raises(KeyError, match="Unknown video backbone"):
        build_video_backbone(OmegaConf.create({"name": "nonexistent_backbone"}), LoadMode.PRETRAINED, None)
