"""History / future layout of the yam_sharpa reader.

Checks a 5-frame history reader against the 1-frame (first-frame) reader of the SAME source: both need rows
t .. t+32, so they have the same windows, and frame t, the future rows, proprio and action must match row for row.
Needs the local simteleop dataset; skipped when it is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "dataloader" / "yam_sharpa.yaml"
DATA = Path("/root/projects/2610_policy_model/datasets")
SIMTELEOP = DATA / "simteleop_1004_fleet" / "dataset"
GPT = DATA / "gpt_fleet_depth" / "dataset"
HIST = [-80, -60, -40, -20, 0]
FUT8 = [4, 8, 12, 16, 20, 24, 28, 32]

pytestmark = pytest.mark.skipif(not SIMTELEOP.is_dir(), reason="local yam_sharpa dataset not available")


def _build(history, future, **extra):
    """Build the simteleop reader without color jitter or normalization (raw values; no stats file needed).

    Args:
        history / future: offset lists.
        **extra: more dataloader keys (``history_sampling``, ``history_window``).

    Returns:
        YamSharpaDataset of the simteleop source.
    """
    from openwam.dataloader.yam_sharpa import MultiYamSharpaDataset

    cfg = OmegaConf.load(CFG)
    cfg.sources = {"simteleop": {"dataset_dir": str(SIMTELEOP), "weight": 1.0}}
    cfg.height, cfg.width = 384, 224
    cfg.color_jitter = None
    cfg.normalize_mode = None
    cfg.history_offsets, cfg.future_offsets = history, future
    for key, value in extra.items():
        cfg[key] = value
    multi = MultiYamSharpaDataset.from_config(cfg)
    return multi.datasets[multi.names.index("simteleop")]


@pytest.fixture(scope="module")
def readers():
    return {
        "h1": _build([0], FUT8),
        "h5": _build(HIST, FUT8),
        "rand5": _build(HIST, FUT8, history_sampling="random", history_window=100),
    }


def _get(reader, k):
    """Window ``k`` of a reader, drawn with a fixed per-sample RNG (fixed-history readers never use it)."""
    return reader.sample(k, np.random.default_rng([0, k]))


def _px(img):
    """PIL → (H, W, 3) int array."""
    return np.asarray(img, dtype=np.int16)


def test_sampling_range(readers):
    # Rows t .. t+32 must exist for every layout: the history length does not change the windows.
    assert readers["h1"]._min_window_len() == readers["h5"]._min_window_len() == 33
    assert len(readers["h1"]) == len(readers["h5"])


def test_shapes(readers):
    s1, s5 = _get(readers["h1"], 0), _get(readers["h5"], 0)
    assert len(s1["video"]) == 1 + 8 and len(s5["video"]) == 5 + 8
    assert s5["video"][0].size == (224, 384)
    assert tuple(s1["proprio"].shape) == (1, 48) and tuple(s5["proprio"].shape) == (5, 48)
    assert tuple(s5["action"].shape) == (32, 48)
    # One proprio mask row per history frame; the layout (not the sample) carries H.
    assert tuple(s5["proprio_mask"].shape) == (5, 48) and tuple(s1["proprio_mask"].shape) == (1, 48)
    assert set(s5) == {"video", "prompt", "action", "action_mask", "proprio", "proprio_mask"}


@pytest.mark.parametrize("k", [100, 2000, -1])
def test_rows_match_first_frame_layout(readers, k):
    # Same windows in both readers; k = -1 is the very last window (its row t+32 is the last row of an episode).
    k = k % len(readers["h5"])
    s1, s5 = _get(readers["h1"], k), _get(readers["h5"], k)
    # Frame t and every future frame are the same rows.
    for j in range(1 + 8):
        assert np.abs(_px(s5["video"][4 + j]) - _px(s1["video"][j])).max() == 0
    # Proprio of frame t (both in the frame-t camera) and the full action chunk are unchanged.
    np.testing.assert_allclose(s5["proprio"][4].numpy(), s1["proprio"][0].numpy(), atol=1e-6)
    np.testing.assert_allclose(s5["action"].numpy(), s1["action"].numpy())


def test_history_clamps_to_episode_start(readers):
    # Window 0 starts at the episode start: every history row clamps to frame t.
    s = _get(readers["h5"], 0)
    for j in range(4):
        assert np.abs(_px(s["video"][j]) - _px(s["video"][4])).max() == 0
        np.testing.assert_allclose(s["proprio"][j].numpy(), s["proprio"][4].numpy())
    # Window 80 has a full history: frame t-4s differs from frame t.
    s = _get(readers["h5"], 80)
    assert not np.allclose(s["proprio"][0].numpy(), s["proprio"][4].numpy())


def test_offset_validation():
    from openwam.dataloader.yam_sharpa import _validate_sparse_offsets

    assert [a.tolist() for a in _validate_sparse_offsets([0], FUT8)] == [[0], FUT8]
    with pytest.raises(ValueError):
        _validate_sparse_offsets(None, None)  # required
    with pytest.raises(ValueError):
        _validate_sparse_offsets([-20, -40, 0], FUT8)  # not increasing
    with pytest.raises(ValueError):
        _validate_sparse_offsets([-40, -20], FUT8)  # does not end at frame t
    with pytest.raises(ValueError):
        _validate_sparse_offsets(HIST, None)  # half-configured


def test_history_proprio_in_anchor_camera():
    # A moving camera must not leak into the proprio: re-expressing each row's wrist from its
    # own camera into the frame-t camera recovers the same base-frame pose for every row.
    from openwam.dataloader.utils.rotation import matrix_to_rotation_6d
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
            state[i, o + 3 : o + 9] = matrix_to_rotation_6d(cw[None, :3, :3])[0]
            state[i, o + 9 : o + 24] = rng.normal(size=15)  # fingertips (wrist frame): untouched
    cam16 = np.stack([c.reshape(16) for c in cams])
    out = encode_proprio_in_anchor_cam("fingertips_rel", state, None, cam16, cam16[2])
    # Every row now equals the anchor row (wrist static in base), fingertips unchanged.
    np.testing.assert_allclose(out[:, :9], np.repeat(out[2:3, :9], 3, 0), atol=1e-6)
    np.testing.assert_allclose(out[:, 24:33], np.repeat(out[2:3, 24:33], 3, 0), atol=1e-6)
    np.testing.assert_allclose(out[:, 9:24], state[:, 9:24], atol=1e-6)
    np.testing.assert_allclose(out[2], state[2], atol=1e-6)  # anchor row itself is unchanged


# ----- random history sampling (history_sampling=random) ------------------------------------------


def _with_offsets(monkeypatch, offsets):
    """Make the reader's random draw return ``offsets`` (a list of H ints)."""
    import openwam.dataloader.yam_sharpa as ys

    monkeypatch.setattr(ys, "sample_history_offsets", lambda h, w, rng: np.asarray(offsets, dtype=np.int64))


def test_random_history_matches_fixed_rows(readers, monkeypatch):
    # Drawing the fixed offsets reproduces the fixed reader exactly (same rows, same clamp rule).
    _with_offsets(monkeypatch, HIST)
    for k in (0, 80, 2000):
        r, f = _get(readers["rand5"], k), _get(readers["h5"], k)
        for a, b in zip(r["video"], f["video"]):
            assert np.abs(_px(a) - _px(b)).max() == 0
        np.testing.assert_array_equal(r["proprio"].numpy(), f["proprio"].numpy())
        np.testing.assert_array_equal(r["action"].numpy(), f["action"].numpy())


def test_random_history_rows_and_clamp(readers, monkeypatch):
    # Window 80 = row 80 of its episode: offset -100 clamps to the episode start (sample, then clamp),
    # the other drawn rows are read as is; frame t, future and action never change.
    _with_offsets(monkeypatch, [-100, -80, -33, -1, 0])
    r, f = _get(readers["rand5"], 80), _get(readers["h5"], 80)
    start = _get(readers["h5"], 0)  # window 0 = the episode start, all history = row 0
    assert np.abs(_px(r["video"][0]) - _px(start["video"][4])).max() == 0  # clamped to row 0
    assert np.abs(_px(r["video"][1]) - _px(f["video"][0])).max() == 0  # row t-80 = fixed history 0
    for j in (4, 5, 12):  # frame t, first and last future frame
        assert np.abs(_px(r["video"][j]) - _px(f["video"][j])).max() == 0
    np.testing.assert_array_equal(r["proprio"][1].numpy(), f["proprio"][0].numpy())
    np.testing.assert_array_equal(r["action"].numpy(), f["action"].numpy())
    assert not np.allclose(r["proprio"][2].numpy(), f["proprio"][2].numpy())  # row t-33 != t-40
