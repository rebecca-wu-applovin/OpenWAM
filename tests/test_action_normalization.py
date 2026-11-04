"""Deploy-side normalization: the training stats loader / formula, and its inverse ``invert_normalization``.

Deploy normalizes the proprio with training's own ``load_stats_file`` (``<action_mode>_state`` block) +
``apply_normalization`` and maps model actions back with ``invert_normalization`` (``<action_mode>`` block).
"""

import shutil

import numpy as np
import pytest
from omegaconf import OmegaConf

from openwam.dataloader.utils.normalization import apply_normalization, invert_normalization

NORM_STATS = "/root/projects/2610_policy_model/norm_stats"


def _stats(d: int, seed: int = 0) -> dict:
    """d: width; seed: RNG seed. Returns: a materialized float32 stats dict with distinct per-dim ranges."""
    rng = np.random.default_rng(seed)
    lo = rng.uniform(-3, 0, d).astype(np.float32)
    hi = (lo + rng.uniform(0.5, 4, d)).astype(np.float32)
    q01, q99 = (lo + 0.1 * (hi - lo)).astype(np.float32), (hi - 0.1 * (hi - lo)).astype(np.float32)
    return {"min": lo, "max": hi, "q01": q01, "q99": q99, "mean": ((lo + hi) / 2).astype(np.float32),
            "std": ((hi - lo) / 4).astype(np.float32)}


@pytest.mark.parametrize("mode", ["min-max", "quantile", "z-score"])
def test_invert_normalization_round_trips(mode):
    """Inside the clip band, invert(apply(x)) == x for every training mode."""
    s = _stats(48)
    lo, hi = (s["q01"], s["q99"]) if mode == "quantile" else (s["min"], s["max"])
    x = np.random.default_rng(1).uniform(lo, hi, (5, 48)).astype(np.float32)
    back = invert_normalization(apply_normalization(x, s, mode), s, mode)
    np.testing.assert_allclose(back, x, atol=1e-5)


@pytest.mark.parametrize("mode", [None, "none", "unknown"])
def test_invert_normalization_rejects_non_invertible_mode(mode):
    """A mode with no inverse formula raises instead of passing the normalized values through."""
    with pytest.raises(ValueError, match="invert_normalization"):
        invert_normalization(np.zeros((2, 48), np.float32), _stats(48), mode)


@pytest.mark.parametrize("mode, dim", [("fingertips_rel", 48), ("unify_rel", 62)])
def test_load_norm_stats_uses_state_and_action_blocks(tmp_path, mode, dim):
    """load_norm_stats returns (state block, action block) of the run's normalization_stats.npy, raw width."""
    from openwam.deploy.model_loader import load_norm_stats

    shutil.copyfile(f"{NORM_STATS}/yam_sharpa_{mode}.npy", tmp_path / "normalization_stats.npy")
    raw = np.load(tmp_path / "normalization_stats.npy", allow_pickle=True).item()
    dl = OmegaConf.create({"action_mode": mode, "normalize_mode": "min-max"})
    state, action = load_norm_stats(dl, str(tmp_path))
    np.testing.assert_array_equal(state["min"], np.asarray(raw[f"{mode}_state"]["min"], np.float32))
    np.testing.assert_array_equal(action["max"], np.asarray(raw[mode]["max"], np.float32))
    assert state["min"].shape == action["min"].shape == (dim,)


def test_load_norm_stats_missing_file_raises(tmp_path):
    """A run dir without normalization_stats.npy cannot be served."""
    from openwam.deploy.model_loader import load_norm_stats

    dl = OmegaConf.create({"action_mode": "fingertips_rel", "normalize_mode": "min-max"})
    with pytest.raises(FileNotFoundError):
        load_norm_stats(dl, str(tmp_path))
