"""Tests for deployment-related changes.

Covers the mixed-precision source of truth.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 1. mixed_precision in accelerate yaml (single source of truth)
# ---------------------------------------------------------------------------


class TestTrainingMixedPrecision:
    """``cfg.training.mixed_precision`` is the sole source of truth.

    ``configs/accelerate`` was removed; both training (``_build_accelerator``)
    and deploy (``model_loader``) read ``training.mixed_precision`` directly.
    """

    def test_training_field_present(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "train.yaml")
        mp = OmegaConf.select(cfg, "training.mixed_precision")
        assert mp == "bf16", f"Expected 'bf16', got {mp!r}"

    def test_accelerate_group_removed(self):
        assert not (PROJECT_ROOT / "configs" / "accelerate").exists(), (
            "configs/accelerate should be gone; DeepSpeed settings now live in train.yaml"
        )
