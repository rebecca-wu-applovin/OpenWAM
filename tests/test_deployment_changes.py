"""Tests for deployment-related changes.

Covers the mixed-precision source of truth, the architecture compile hooks and the cudagraph step marks of the
denoising loop.
"""

import sys
from pathlib import Path
from unittest.mock import patch

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


# ---------------------------------------------------------------------------
# 5. architecture compile hooks — no broad default compile side effects
# ---------------------------------------------------------------------------


class TestArchitectureCompileFlags:
    """Verify compile mode routing without broad default compile side effects."""

    def test_base_architecture_does_not_broad_compile_backbones(self):
        from omegaconf import OmegaConf

        from tests.test_openwam_trainer import _make_tiny_arch

        arch = _make_tiny_arch()
        cfg = OmegaConf.create(
            {
                "enabled": False,
                "cross_attn": {"torch_mode": "reduce-overhead", "dynamic": False},
            }
        )

        with patch("torch.compile") as mock_compile:
            arch.apply_compile_optimizations(cfg)

        mock_compile.assert_not_called()

    def test_architecture_auto_compile_respects_disabled_fast_path(self):
        from omegaconf import OmegaConf

        from tests.test_openwam_trainer import _make_tiny_arch

        arch = _make_tiny_arch()
        cfg = OmegaConf.create({"enabled": True, "cross_attn": {"enabled": False}})

        with patch("torch.compile") as mock_compile:
            arch.apply_compile_optimizations(cfg)

        mock_compile.assert_not_called()


# ---------------------------------------------------------------------------
# 8. architecture denoising loop — cudagraph_mark_step_begin placement
# ---------------------------------------------------------------------------


class TestCudagraphMarkStepBegin:
    """Structural check: every architecture forward dispatch of the denoising loop
    must be preceded by ``torch.compiler.cudagraph_mark_step_begin()`` to
    prevent CUDA Graph tree from raising 'tensor output overwritten by
    subsequent run' when fixed-shape compile paths are active.

    The dispatch sites are in ``BaseWAMArchitecture._denoise_loop`` (behind ``sample``) and
    ``_forward_with_cfg`` in ``base.py``.
    """

    _DISPATCH_SUBSTRINGS = (
        "noise_pred, action_noise_pred = self.forward(",
        "noise_pred = vb.finalize(state)",
        # §15 — CFG forward dispatch sites inside _forward_with_cfg
        "merged_noise, merged_action = self.forward(",
        "cond_noise, cond_action = self.forward(",
        "uncond_noise, uncond_action = self.forward(",
    )

    def _source(self):
        # Only check the inference loop onwards, not compute_loss()
        src = (PROJECT_ROOT / "openwam" / "model" / "architectures" / "base.py").read_text()
        marker = "def _denoise_loop("
        idx = src.index(marker)
        return src[idx:]

    def test_mark_count_equals_dispatch_sites(self):
        src = self._source()
        mark_count = src.count("torch.compiler.cudagraph_mark_step_begin()")
        expected = sum(src.count(sub) for sub in self._DISPATCH_SUBSTRINGS)
        assert mark_count == expected, (
            f"Expected {expected} cudagraph_mark_step_begin() call(s), found {mark_count}. "
            "Add torch.compiler.cudagraph_mark_step_begin() before any new "
            "architecture forward dispatch."
        )

    def test_mark_appears_before_not_after(self):
        """The mark must appear on the line immediately before each dispatch site."""
        src = self._source()
        lines = src.splitlines()
        for i, line in enumerate(lines):
            if any(sub in line for sub in self._DISPATCH_SUBSTRINGS):
                prev = i - 1
                while prev >= 0 and (lines[prev].strip() == "" or lines[prev].lstrip().startswith("#")):
                    prev -= 1
                assert "cudagraph_mark_step_begin" in lines[prev], (
                    f"Line {i + 1}: dispatch not preceded by "
                    f"cudagraph_mark_step_begin(). Found instead: {lines[prev]!r}"
                )
