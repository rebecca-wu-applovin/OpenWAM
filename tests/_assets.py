"""Locate real model weights for GPU tests.

Each asset is resolved in this order:

1. its ``OPENWAM_<ASSET>`` environment variable,
2. the legacy variable names some tests used before the names were unified,
3. the directory ``scripts/download_assets/`` writes to by default.

So after downloading with the repo scripts nothing needs to be exported; point
the variable elsewhere only for weights stored outside ``assets/``. Tests skip
(or are deselected via ``-m "not gpu"``) when the resolved path does not exist.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VIDEO_BACKBONES = PROJECT_ROOT / "assets" / "video_backbone_ckpt"
# yam_sharpa LeRobot v3 sources (the training smoke tests read real episodes).
_DATASETS = Path("/root/projects/2610_policy_model/datasets")

# canonical env var -> (legacy env vars, default location written by the downloader)
_ASSETS: dict[str, tuple[tuple[str, ...], Path]] = {
    "OPENWAM_WAN22_TI2V_5B": (("WAN22_TI2V_5B", "WAN_TI2V_5B_PATH"), _VIDEO_BACKBONES / "Wan2.2-TI2V-5B"),
    "OPENWAM_COSMOS3_EDGE": (("COSMOS3_EDGE_ASSET_PATH",), _VIDEO_BACKBONES / "Cosmos3-Edge"),
    "OPENWAM_COSMOS3_NANO": (("COSMOS3_NANO_ASSET_PATH",), _VIDEO_BACKBONES / "Cosmos3-Nano"),
    "OPENWAM_YAM_SHARPA_SIMTELEOP": ((), _DATASETS / "simteleop_1004_fleet" / "dataset"),
    "OPENWAM_YAM_SHARPA_GPT": ((), _DATASETS / "gpt_fleet_depth" / "dataset"),
}


def asset_path(name: str) -> str:
    """Return the on-disk location for ``name`` (a key of ``_ASSETS``)."""
    legacy, default = _ASSETS[name]
    for var in (name, *legacy):
        value = os.environ.get(var)
        if value:
            return value
    return str(default)
