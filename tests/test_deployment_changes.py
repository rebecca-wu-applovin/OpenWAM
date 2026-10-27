"""Tests for deployment-related changes.

Covers mixed-precision save/load behavior, the configs/deploy.yaml layout, the compile-option
parsing the architectures use, the deploy dtype map, and generate()'s cudagraph marks.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party"))


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
# 2. configs/deploy.yaml — the `deploy` block (ckpt / server / data / policy / sampler)
# ---------------------------------------------------------------------------


class TestDeploymentYaml:
    def _load(self):
        from omegaconf import OmegaConf

        return OmegaConf.load(PROJECT_ROOT / "configs" / "deploy.yaml")

    def test_ckpt_and_steps_per_plan_are_required(self):
        """The served weights and K have no default: deploy.ckpt.* and deploy.policy.steps_per_plan are null."""
        cfg = self._load()
        assert cfg.ckpt.dir is None and cfg.ckpt.name is None
        assert cfg.policy.steps_per_plan is None

    def test_sampler_denoise_defaults(self):
        """The sampler defaults to the plain sync trajectory (async knobs at their no-op values)."""
        cfg = self._load()
        assert cfg.sampler.denoise_steps is not None
        assert cfg.sampler.denoise_mode == "sync"
        assert cfg.sampler.lead_modality == "video"
        assert cfg.sampler.variance_shift_alpha == 1.0
        assert cfg.sampler.linear_offset == 0.0
        assert cfg.sampler.decode_video is False

    def test_sampler_cfg_fields_absent(self):
        """CFG knobs stay out of deploy.yaml."""
        from omegaconf import OmegaConf

        cfg = self._load()
        assert OmegaConf.select(cfg, "sampler.cfg_scale") is None
        assert OmegaConf.select(cfg, "sampler.cfg_merge") is None

    def test_server_and_data_defaults_present(self):
        """The server address and the expected data shape are stated in the yaml."""
        cfg = self._load()
        assert cfg.server.port is not None and cfg.server.device is not None
        assert {"num_frames", "height", "width"} <= set(cfg.data)


# ---------------------------------------------------------------------------
# 4. compile_options.py + architecture compile routing
# ---------------------------------------------------------------------------


class TestCompileOptions:
    """Compile-section parsing and the architectures' fast-path compile routing (no torch.compile side effects)."""

    def _compile_options(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "compile_options", PROJECT_ROOT / "openwam" / "model" / "compile_options.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_mode_only_compile_sections_use_fast_path_defaults(self):
        compile_options = self._compile_options()

        self_cfg = {"self_attn": {}}
        self_section = compile_options.self_attn_compile_cfg(self_cfg)
        assert compile_options.section_enabled(self_section, default=False) is True
        assert compile_options.torch_compile_kwargs(self_section) == {
            "dynamic": False,
            "mode": "reduce-overhead",
        }

        cross_cfg = {"cross_attn": {}}
        cross_section = compile_options.cross_attn_compile_cfg(cross_cfg)
        assert compile_options.section_enabled(cross_section, default=False) is True
        assert compile_options.torch_compile_kwargs(cross_section) == {
            "dynamic": False,
            "mode": "reduce-overhead",
        }

        idm_cfg = {"idm": {}}
        idm_section = compile_options.idm_compile_cfg(idm_cfg)
        assert compile_options.section_enabled(idm_section, default=False) is True
        assert compile_options.torch_compile_kwargs(idm_section) == {
            "dynamic": False,
            "mode": "reduce-overhead",
        }
        assert compile_options.section_enabled(idm_section.video_loop, default=False) is True
        assert compile_options.section_enabled(idm_section.action_cache, default=False) is True

        tri_cfg = {"tri_system": {}}
        tri_section = compile_options.tri_system_compile_cfg(tri_cfg)
        assert compile_options.section_enabled(tri_section, default=False) is True
        assert compile_options.torch_compile_kwargs(tri_section) == {
            "dynamic": False,
            "mode": "reduce-overhead",
        }

    def test_compile_section_enabled_false_is_respected(self):
        compile_options = self._compile_options()

        self_cfg = {"self_attn": {"enabled": False}}
        self_section = compile_options.self_attn_compile_cfg(self_cfg)
        assert compile_options.section_enabled(self_section, default=True) is False

        cross_cfg = {"cross_attn": {"enabled": False}}
        cross_section = compile_options.cross_attn_compile_cfg(cross_cfg)
        assert compile_options.section_enabled(cross_section, default=True) is False

        idm_cfg = {"idm": {"enabled": False}}
        idm_section = compile_options.idm_compile_cfg(idm_cfg)
        assert compile_options.section_enabled(idm_section, default=True) is False
        assert compile_options.section_enabled(idm_section.video_loop, default=True) is False
        assert compile_options.section_enabled(idm_section.action_cache, default=True) is False

        tri_cfg = {"tri_system": {"enabled": False}}
        tri_section = compile_options.tri_system_compile_cfg(tri_cfg)
        assert compile_options.section_enabled(tri_section, default=True) is False

    def test_idm_compile_subsections_can_be_disabled_independently(self):
        compile_options = self._compile_options()

        idm_section = compile_options.idm_compile_cfg(
            {
                "idm": {
                    "video_loop": {"enabled": False},
                    "action_cache": {"enabled": True},
                }
            }
        )

        assert compile_options.section_enabled(idm_section, default=True) is True
        assert compile_options.section_enabled(idm_section.video_loop, default=True) is False
        assert compile_options.section_enabled(idm_section.action_cache, default=False) is True

    def test_compile_enabled_validation_rejects_non_bool_values(self):
        compile_options = self._compile_options()

        assert compile_options.normalize_compile_enabled("true") is True
        assert compile_options.normalize_compile_enabled("false") is False
        with pytest.raises(ValueError, match="Unknown compile enabled value"):
            compile_options.normalize_compile_enabled("default")
        with pytest.raises(ValueError, match="Unknown legacy compile mode"):
            compile_options.compile_enabled({"mode": "default"}, strict=True)
        with pytest.raises(ValueError, match="Unknown compile enabled value"):
            compile_options.normalize_compile_enabled("self_attn")
        with pytest.raises(ValueError, match="Unknown compile enabled value"):
            compile_options.normalize_compile_enabled("cross-attn")

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
# 6. model_loader.py — training.mixed_precision -> served dtype
# ---------------------------------------------------------------------------


class TestModelLoaderDtype:
    """model_loader.DTYPES: the dtype each training.mixed_precision value is served in."""

    def test_dtype_map(self):
        """bf16 / fp16 / no map to bfloat16 / float16 / float32."""
        from openwam.deploy.model_loader import DTYPES

        assert DTYPES == {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}

    def test_dtype_unknown_value_is_rejected(self):
        """No silent bf16 fallback: an unknown mixed_precision value is a KeyError."""
        from openwam.deploy.model_loader import DTYPES

        with pytest.raises(KeyError):
            DTYPES["unknown_value"]


# ---------------------------------------------------------------------------
# 8. architecture.generate() — cudagraph_mark_step_begin placement
# ---------------------------------------------------------------------------


class TestCudagraphMarkStepBegin:
    """Structural check: every architecture forward dispatch in generate()
    must be preceded by ``torch.compiler.cudagraph_mark_step_begin()`` to
    prevent CUDA Graph tree from raising 'tensor output overwritten by
    subsequent run' when fixed-shape compile paths are active.

    After the architecture refactor the dispatch site is in
    ``BaseWAMArchitecture.generate()`` in ``base.py``.
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
        # Only check generate() method, not compute_loss()
        src = (PROJECT_ROOT / "openwam" / "model" / "architectures" / "base.py").read_text()
        marker = "def generate("
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
