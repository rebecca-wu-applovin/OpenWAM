"""Sparse-history layout of the yam_sharpa reader (07_sparse_history_backbones).

Checks against the legacy contiguous window of the SAME source: frame t, the
future rows, proprio and action must match row for row. Needs the local
simteleop dataset; skipped when it is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "dataloader" / "yam_sharpa.yaml"
HIST = [-80, -60, -40, -20, 0]
FUT4 = [5, 10, 15, 20, 25]
FUT8 = list(range(5, 46, 5))

pytestmark = pytest.mark.skipif(
    not Path("/root/projects/2610_policy_model/datasets/simteleop_1004/dataset").is_dir(),
    reason="local yam_sharpa dataset not available",
)


def _build(history=None, future=None, rows_ahead=0):
    """Build the readers (both sources, so the pooled stats file matches) without color jitter.

    Args:
        history / future: offset lists or None (legacy layout).
        rows_ahead: int, ``min_rows_ahead``.

    Returns:
        YamSharpaDataset of the simteleop source.
    """
    from openwam.dataloader.yam_sharpa import MultiYamSharpaDataset

    cfg = OmegaConf.load(CFG)
    cfg.height, cfg.width = 384, 224
    cfg.color_jitter = None
    cfg.history_offsets, cfg.future_offsets, cfg.min_rows_ahead = history, future, rows_ahead
    multi = MultiYamSharpaDataset.from_config(cfg)
    return multi.datasets[multi.names.index("simteleop")]


@pytest.fixture(scope="module")
def readers():
    return {
        "legacy": _build(),
        "fut4": _build(HIST, FUT4, rows_ahead=45),
        "fut8": _build(HIST, FUT8, rows_ahead=45),
    }


def _px(img):
    """PIL → (H, W, 3) int array."""
    return np.asarray(img, dtype=np.int16)


def test_shared_sampling_range(readers):
    # min_rows_ahead=45 gives the 4x and 8x layouts the same windows.
    assert len(readers["fut4"]) == len(readers["fut8"]) < len(readers["legacy"])


def test_shapes(readers):
    s4, s8 = readers["fut4"][0], readers["fut8"][0]
    assert len(s4["video"]) == 5 + 5 and len(s8["video"]) == 5 + 9
    assert s4["video"][0].size == (224, 384)
    assert tuple(s4["proprio"].shape) == (5, 48)
    assert tuple(s4["action"].shape) == (32, 48)
    assert s4["num_history_frames"] == 5 and s4["video_mask"] is None


def _legacy_index(sparse_reader, legacy_reader, idx):
    """Legacy window index with the same (episode, offset) as sparse window ``idx``.

    Both readers keep the same episode table; only the per-episode window counts differ.
    """
    ep = int(np.searchsorted(sparse_reader._cum_n_starts, idx, side="right") - 1)
    within = idx - int(sparse_reader._cum_n_starts[ep])
    return int(legacy_reader._cum_n_starts[ep]) + within


@pytest.mark.parametrize("k", [100, 2000, -1])
def test_rows_match_legacy_window(readers, k):
    # Same episode + offset in both readers (window stride 1, same valid_start).
    # k = -1 is the very last window (its row t+45 is the last row of an episode).
    k = k % len(readers["fut8"])
    legacy = readers["legacy"][_legacy_index(readers["fut8"], readers["legacy"], k)]
    sparse = readers["fut8"][k]
    # Frame t = last history frame = legacy frame 0.
    assert np.abs(_px(sparse["video"][4]) - _px(legacy["video"][0])).max() == 0
    # Row t+20 = future index 3 (offset 20) = legacy frame 5 (stride 4).
    assert np.abs(_px(sparse["video"][5 + 3]) - _px(legacy["video"][5])).max() == 0
    # Proprio of frame t and the full action chunk are unchanged.
    np.testing.assert_allclose(sparse["proprio"][4].numpy(), legacy["proprio"][0].numpy())
    np.testing.assert_allclose(sparse["action"].numpy(), legacy["action"].numpy())


def test_history_clamps_to_episode_start(readers):
    # Window 0 starts at the episode start: every history row clamps to frame t.
    s = readers["fut4"][0]
    for j in range(4):
        assert np.abs(_px(s["video"][j]) - _px(s["video"][4])).max() == 0
        np.testing.assert_allclose(s["proprio"][j].numpy(), s["proprio"][4].numpy())
    # Window 80 has a full history: frame t-4s differs from frame t.
    s = readers["fut4"][80]
    assert not np.allclose(s["proprio"][0].numpy(), s["proprio"][4].numpy())


def test_offset_validation():
    from openwam.dataloader.yam_sharpa import _validate_sparse_offsets

    assert _validate_sparse_offsets(None, None) == (None, None)
    with pytest.raises(ValueError):
        _validate_sparse_offsets([-20, -40, 0], FUT4)  # not increasing
    with pytest.raises(ValueError):
        _validate_sparse_offsets([-40, -20], FUT4)  # does not end at frame t
    with pytest.raises(ValueError):
        _validate_sparse_offsets(HIST, None)  # half-configured
