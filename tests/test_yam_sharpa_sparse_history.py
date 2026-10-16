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
    # Proprio of frame t (up to the rot6d round trip of the anchor-camera re-expression) and the
    # full action chunk are unchanged.
    np.testing.assert_allclose(sparse["proprio"][4].numpy(), legacy["proprio"][0].numpy(), atol=1e-6)
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


def test_history_proprio_in_anchor_camera():
    # A moving camera must not leak into the proprio: re-expressing each row's wrist from its
    # own camera into the frame-t camera recovers the same base-frame pose for every row.
    from openwam.dataloader.transforms.rotation import _matrix_to_rotation_6d
    from openwam.dataloader.utils.yam_sharpa_repr import encode_proprio_in_anchor_cam

    rng = np.random.default_rng(0)

    def rigid(angle, t):
        """(4, 4) rotation about z by ``angle`` plus translation ``t``."""
        c, s_ = np.cos(angle), np.sin(angle)
        m = np.eye(4)
        m[:3, :3] = [[c, -s_, 0], [s_, c, 0], [0, 0, 1]]
        m[:3, 3] = t
        return m

    # Fixed wrist poses in the base frame, seen by 3 differently placed cameras.
    base_from_wrist = [rigid(0.3, [0.1, 0.2, 0.3]), rigid(-0.5, [0.4, -0.1, 0.2])]
    cams = [rigid(a, rng.normal(size=3)) for a in (0.0, 0.4, -0.7)]
    state = np.zeros((3, 48))
    for i, cam in enumerate(cams):
        for o, bw in zip((0, 24), base_from_wrist):
            cw = np.linalg.inv(cam) @ bw  # wrist in this row's camera
            state[i, o : o + 3] = cw[:3, 3]
            state[i, o + 3 : o + 9] = _matrix_to_rotation_6d(cw[None, :3, :3])[0]
            state[i, o + 9 : o + 24] = rng.normal(size=15)  # fingertips (wrist frame): untouched
    cam16 = np.stack([c.reshape(16) for c in cams])
    out = encode_proprio_in_anchor_cam("fingertips_rel", state, None, cam16, cam16[2])
    # Every row now equals the anchor row (wrist static in base), fingertips unchanged.
    np.testing.assert_allclose(out[:, :9], np.repeat(out[2:3, :9], 3, 0), atol=1e-6)
    np.testing.assert_allclose(out[:, 24:33], np.repeat(out[2:3, 24:33], 3, 0), atol=1e-6)
    np.testing.assert_allclose(out[:, 9:24], state[:, 9:24], atol=1e-6)
    np.testing.assert_allclose(out[2], state[2], atol=1e-6)  # anchor row itself is unchanged
