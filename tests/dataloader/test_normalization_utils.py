"""``utils.normalization``: the forward formulas, the on/off rule and the strict stats-file loader."""

from __future__ import annotations

import json

import numpy as np
import pytest

from openwam.dataloader.utils.normalization import (
    STAT_KEYS,
    apply_normalization,
    load_stats_file,
    normalization_on,
)


def _stats():
    return {
        "mean": np.array([5.0, 0.0], dtype=np.float32),
        "std": np.array([2.0, 0.5], dtype=np.float32),
        "min": np.array([0.0, -1.0], dtype=np.float32),
        "max": np.array([10.0, 1.0], dtype=np.float32),
    }


class TestApplyNormalization:
    def test_min_max_min_maps_to_neg_one(self):
        arr = np.array([[0.0, -1.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats(), "min-max")
        np.testing.assert_allclose(out, [[-1.0, -1.0]], atol=1e-6)

    def test_min_max_max_maps_to_pos_one(self):
        arr = np.array([[10.0, 1.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats(), "min-max")
        np.testing.assert_allclose(out, [[1.0, 1.0]], atol=1e-6)

    def test_min_max_mid_maps_to_zero(self):
        arr = np.array([[5.0, 0.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats(), "min-max")
        np.testing.assert_allclose(out, [[0.0, 0.0]], atol=1e-6)

    def test_z_score_mean_maps_to_zero(self):
        arr = np.array([[5.0, 0.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats(), "z-score")
        np.testing.assert_allclose(out, [[0.0, 0.0]], atol=1e-6)

    def test_z_score_one_std_maps_to_one(self):
        arr = np.array([[7.0, 0.5]], dtype=np.float32)
        out = apply_normalization(arr, _stats(), "z-score")
        np.testing.assert_allclose(out, [[1.0, 1.0]], atol=1e-6)

    def test_broadcast_across_leading_dims(self):
        arr = np.array([[0.0, -1.0], [10.0, 1.0], [5.0, 0.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats(), "min-max")
        np.testing.assert_allclose(
            out,
            [[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]],
            atol=1e-6,
        )


def _stats_with_quantile():
    return {
        "mean": np.array([0.0, 0.0], dtype=np.float32),
        "std": np.array([1.0, 1.0], dtype=np.float32),
        "min": np.array([-100.0, -50.0], dtype=np.float32),  # outliers
        "max": np.array([100.0, 50.0], dtype=np.float32),
        "q01": np.array([-1.0, -2.0], dtype=np.float32),
        "q99": np.array([1.0, 2.0], dtype=np.float32),
    }


class TestQuantileNormalization:
    def test_q01_maps_to_neg_one(self):
        arr = np.array([[-1.0, -2.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats_with_quantile(), "quantile")
        np.testing.assert_allclose(out, [[-1.0, -1.0]], atol=1e-6)

    def test_q99_maps_to_pos_one(self):
        arr = np.array([[1.0, 2.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats_with_quantile(), "quantile")
        np.testing.assert_allclose(out, [[1.0, 1.0]], atol=1e-6)

    def test_midpoint_maps_to_zero(self):
        arr = np.array([[0.0, 0.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats_with_quantile(), "quantile")
        np.testing.assert_allclose(out, [[0.0, 0.0]], atol=1e-6)

    def test_outlier_clipped_to_pos_one(self):
        # Value 100 is way beyond q99=1 → clipped to 1.0
        arr = np.array([[100.0, 50.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats_with_quantile(), "quantile")
        np.testing.assert_allclose(out, [[1.0, 1.0]], atol=1e-6)

    def test_outlier_clipped_to_neg_one(self):
        arr = np.array([[-100.0, -50.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats_with_quantile(), "quantile")
        np.testing.assert_allclose(out, [[-1.0, -1.0]], atol=1e-6)

    def test_quantile_within_bounds_no_clip(self):
        arr = np.array([[0.5, 1.0], [-0.5, -1.0]], dtype=np.float32)
        out = apply_normalization(arr, _stats_with_quantile(), "quantile")
        # No clipping for values within [q01, q99]
        expected = np.array([[0.5, 0.5], [-0.5, -0.5]], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-6)


@pytest.mark.parametrize("mode", [None, "none", "null", "weird-mode"])
def test_apply_normalization_rejects_a_non_mode(mode):
    """No silent pass-through: normalization off is decided by the caller, not by the mode string."""
    with pytest.raises(ValueError, match="normalize_mode"):
        apply_normalization(np.zeros((1, 2), np.float32), _stats_with_quantile(), mode)


@pytest.mark.parametrize("mode, on", [(None, False), ("min-max", True), ("z-score", True), ("quantile", True)])
def test_normalization_on_reads_the_config_value(mode, on):
    """null is off, each formula is on."""
    assert normalization_on(mode) is on


@pytest.mark.parametrize("mode", ["none", "null", "minmax"])
def test_normalization_on_rejects_any_other_string(mode):
    """'none' / 'null' strings are not aliases of yaml null: a typo raises instead of turning normalization off."""
    with pytest.raises(ValueError, match="normalize_mode"):
        normalization_on(mode)


def _write_stats(tmp_path, block: dict):
    """block: the fields of the one block ``fingertips_rel``. Returns: path of a .json stats file holding it."""
    path = tmp_path / "stats.json"
    path.write_text(json.dumps({"fingertips_rel": block}))
    return path


def _full_block(dim: int = 2) -> dict:
    """dim: vector length. Returns: a block with every STAT_KEYS field, as lists."""
    return {key: [float(i) for i in range(dim)] for key in STAT_KEYS}


def test_load_stats_file_returns_every_field_as_float32(tmp_path):
    """All six vectors come back as (D,) float32."""
    stats = load_stats_file(_write_stats(tmp_path, _full_block()), block="fingertips_rel", dim=2)
    assert set(stats) == set(STAT_KEYS)
    assert all(v.dtype == np.float32 and v.shape == (2,) for v in stats.values())


@pytest.mark.parametrize("missing", ["mean", "std", "q01"])
def test_load_stats_file_missing_field_raises(tmp_path, missing):
    """A block missing any field raises instead of filling a neutral default (a broken file must not train)."""
    block = _full_block()
    del block[missing]
    with pytest.raises(KeyError, match=missing):
        load_stats_file(_write_stats(tmp_path, block), block="fingertips_rel", dim=2)


def test_load_stats_file_missing_block_raises(tmp_path):
    """The block is looked up by its key; there is no flat-file fallback."""
    with pytest.raises(KeyError, match="unify_rel"):
        load_stats_file(_write_stats(tmp_path, _full_block()), block="unify_rel", dim=2)
