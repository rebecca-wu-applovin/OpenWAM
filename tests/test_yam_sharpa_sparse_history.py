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


def _build(history=None, future=None, rows_ahead=0, **extra):
    """Build the readers (both sources, so the pooled stats file matches) without color jitter.

    Args:
        history / future: offset lists or None (legacy layout).
        rows_ahead: int, ``min_rows_ahead``.
        **extra: more dataloader keys (e.g. ``history_sampling``, ``history_window``).

    Returns:
        YamSharpaDataset of the simteleop source.
    """
    from openwam.dataloader.yam_sharpa import MultiYamSharpaDataset

    cfg = OmegaConf.load(CFG)
    cfg.height, cfg.width = 384, 224
    cfg.color_jitter = None
    cfg.history_offsets, cfg.future_offsets, cfg.min_rows_ahead = history, future, rows_ahead
    for key, value in extra.items():
        cfg[key] = value
    multi = MultiYamSharpaDataset.from_config(cfg)
    return multi.datasets[multi.names.index("simteleop")]


@pytest.fixture(scope="module")
def readers():
    return {
        "legacy": _build(),
        "fut4": _build(HIST, FUT4, rows_ahead=45),
        "fut8": _build(HIST, FUT8, rows_ahead=45),
        "rand4": _build(HIST, FUT4, rows_ahead=45, history_sampling="random", history_window=100),
        "panel": _build(HIST, FUT4, rows_ahead=45, history_layout="panel"),
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


# ----- random history sampling (history_sampling=random) ------------------------------------------


def test_sample_history_offsets():
    import random

    from openwam.dataloader.utils.sparse_history import sample_history_offsets

    rng = random.Random(0)
    seen = set()
    for _ in range(2000):
        o = sample_history_offsets(5, 100, rng)
        # H offsets, strictly increasing, inside [-100, 0], ending at frame t.
        assert o.shape == (5,) and o[-1] == 0 and (np.diff(o) > 0).all() and o[0] >= -100
        seen.update(o[:-1].tolist())
    assert seen == set(range(-100, 0))  # every past row of the window gets drawn
    np.testing.assert_array_equal(sample_history_offsets(1, 100, rng), [0])  # H=1 = frame t only


def test_history_sampling_validation():
    from openwam.dataloader.yam_sharpa import _validate_history_sampling

    h = np.asarray(HIST)
    assert _validate_history_sampling("fixed", 0, None) == "fixed"
    assert _validate_history_sampling("random", 80, h) == "random"
    with pytest.raises(ValueError):
        _validate_history_sampling("uniform", 100, h)  # unknown mode
    with pytest.raises(ValueError):
        _validate_history_sampling("random", 100, None)  # needs the sparse layout
    with pytest.raises(ValueError):
        _validate_history_sampling("random", 60, h)  # fixed offsets reach 80 rows back


def _with_offsets(monkeypatch, offsets):
    """Make the reader's random draw return ``offsets`` (a list of H ints)."""
    import openwam.dataloader.yam_sharpa as ys

    monkeypatch.setattr(ys, "sample_history_offsets", lambda h, w: np.asarray(offsets, dtype=np.int64))


def test_random_history_matches_fixed_rows(readers, monkeypatch):
    # Drawing the fixed offsets reproduces the fixed reader exactly (same rows, same clamp rule).
    _with_offsets(monkeypatch, HIST)
    for k in (0, 80, 2000):
        r, f = readers["rand4"][k], readers["fut4"][k]
        for a, b in zip(r["video"], f["video"]):
            assert np.abs(_px(a) - _px(b)).max() == 0
        np.testing.assert_allclose(r["proprio"].numpy(), f["proprio"].numpy())
        np.testing.assert_allclose(r["action"].numpy(), f["action"].numpy())


def test_random_history_rows_and_clamp(readers, monkeypatch):
    # Window 80 = row 80 of its episode: offset -100 clamps to the episode start (sample, then clamp),
    # the other drawn rows are read as is; frame t, future and action never change.
    _with_offsets(monkeypatch, [-100, -80, -33, -1, 0])
    r, f = readers["rand4"][80], readers["fut4"][80]
    start = readers["fut4"][0]  # window 0 = the episode start, all history = row 0
    assert np.abs(_px(r["video"][0]) - _px(start["video"][4])).max() == 0  # clamped to row 0
    assert np.abs(_px(r["video"][1]) - _px(f["video"][0])).max() == 0  # row t-80 = fixed history 0
    for j in (4, 5, 9):  # frame t, first and last future frame
        assert np.abs(_px(r["video"][j]) - _px(f["video"][j])).max() == 0
    np.testing.assert_allclose(r["proprio"][1].numpy(), f["proprio"][0].numpy())
    np.testing.assert_allclose(r["action"].numpy(), f["action"].numpy())
    assert not np.allclose(r["proprio"][2].numpy(), f["proprio"][2].numpy())  # row t-33 != t-40


def test_random_history_train_split_only(readers):
    # The flag is resolved once from (history_sampling, split); fixed readers never randomize.
    assert readers["rand4"]._random_history and not readers["fut4"]._random_history


def test_panel_layout_tiles_the_fixed_history(readers):
    # history_layout=panel: video = [2x2 panel of the 4 past rows, frame t] + future; proprio keeps 5 rows.
    from openwam.dataloader.transforms.multiview import compose_history_panel

    for k in (0, 80, 2000):
        p, f = readers["panel"][k], readers["fut4"][k]
        assert p["num_history_frames"] == 2 and len(p["video"]) == 2 + len(FUT4)
        assert np.abs(_px(p["video"][0]) - _px(compose_history_panel(f["video"][:4]))).max() == 0
        for a, b in zip(p["video"][1:], f["video"][4:]):  # frame t and the future are untouched
            assert np.abs(_px(a) - _px(b)).max() == 0
        np.testing.assert_allclose(p["proprio"].numpy(), f["proprio"].numpy())  # (5, D)
        np.testing.assert_allclose(p["action"].numpy(), f["action"].numpy())
