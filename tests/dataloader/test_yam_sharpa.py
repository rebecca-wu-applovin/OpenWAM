"""YAM-duo + Sharpa-hand reader: representations, frames, layout, sampling, stats, deploy."""

from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf
from PIL import Image
from scipy.spatial.transform import Rotation

from openwam.dataloader.registry import DATASET_REGISTRY, build_dataset
from openwam.dataloader.utils import yam_sharpa_repr as R
from openwam.dataloader.utils.normalization import load_stats_file
from openwam.dataloader.utils.stats_computation.yam_sharpa_stats_computation import (
    compute_yam_sharpa_stats,
    save_stats,
)
from openwam.dataloader.yam_sharpa import MultiYamSharpaDataset

EGO = "observation.images.ego_view"
CHEST = "observation.images.chest_view"
CAM_COLORS = {EGO: (255, 0, 0), CHEST: (0, 0, 255)}
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "dataloader" / "yam_sharpa.yaml"


def _T(R_, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R_, p
    return T


def _synthetic_episode(length: int, rng: np.random.Generator) -> dict:
    """Consistent rows: world wrist poses + a MOVING camera, state = wrist in that frame's camera."""
    cams = [
        _T(Rotation.from_rotvec(0.05 * k * np.array([0, 1, 0])).as_matrix(), [0.01 * k, 0, 0.6]) for k in range(length)
    ]
    state = np.zeros((length, 48), np.float32)
    action = np.zeros((length, 48), np.float32)
    targets_world = []
    for k in range(length):
        row_t = []
        for o in (0, 24):
            meas = _T(Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix(), rng.normal(size=3))
            tgt = _T(Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix(), rng.normal(size=3))
            for vec, W in ((state, meas), (action, tgt)):
                local = np.linalg.inv(cams[k]) @ W  # wrist pose in this frame's camera
                vec[k, o : o + 3] = local[:3, 3]
                vec[k, o + 3 : o + 9] = np.concatenate([local[:3, 0], local[:3, 1]])
                vec[k, o + 9 : o + 24] = rng.normal(size=15) * 0.1
            row_t.append(tgt)
        targets_world.append(row_t)
    return {
        "state": state,
        "action": action,
        "joints": rng.normal(size=(length, 56)).astype(np.float32),
        "setpoints": rng.normal(size=(length, 56)).astype(np.float32),
        "cams": np.stack([c.reshape(-1) for c in cams]).astype(np.float32),
        "cam_mats": cams,
        "targets_world": targets_world,
    }


def _write_source(root: Path, lengths, seed: int = 0, splits: dict | None = None) -> list:
    """Write a synthetic LeRobot v3 source (one data shard, empty mp4 files).

    Args:
        root: Path, dataset dir to create. lengths: list[int], rows per episode. seed: int, data RNG seed.
        splits: optional ``info.json`` splits, e.g. ``{"val": "0:2"}`` (absent = every episode is train).

    Returns:
        list of per-episode dicts from :func:`_synthetic_episode`.
    """
    rng = np.random.default_rng(seed)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    for cam in (EGO, CHEST):
        (root / "videos" / cam / "chunk-000").mkdir(parents=True)
        (root / "videos" / cam / "chunk-000" / "file-000.mp4").write_bytes(b"")
    features = {
        "observation.state": {"shape": [48]},
        "action": {"shape": [48]},
        "joint_angles": {"shape": [56]},
        "joint_setpoints": {"shape": [56]},
        "camera_pose.ego_view": {"shape": [16]},
        EGO: {"dtype": "video"},
        CHEST: {"dtype": "video"},
        "task_index": {"dtype": "int64"},
    }
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 20,
                "features": features,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                **({"splits": splits} if splits else {}),
            }
        )
    )
    pd.DataFrame({"task_index": [0]}, index=["pick cup"]).to_parquet(root / "meta" / "tasks.parquet")
    eps, rows, start = [], [], 0
    for ei, L in enumerate(lengths):
        ep = _synthetic_episode(L, rng)
        eps.append(ep)
        for k in range(L):
            rows.append(
                {
                    "observation.state": ep["state"][k],
                    "action": ep["action"][k],
                    "joint_angles": ep["joints"][k],
                    "joint_setpoints": ep["setpoints"][k],
                    "camera_pose.ego_view": ep["cams"][k],
                    "task_index": 0,
                    "episode_index": ei,
                    "frame_index": k,
                }
            )
        row = {
            "episode_index": ei,
            "length": L,
            "dataset_from_index": start,
            "data/chunk_index": 0,
            "data/file_index": 0,
        }
        for cam in (EGO, CHEST):
            row[f"videos/{cam}/chunk_index"] = 0
            row[f"videos/{cam}/file_index"] = 0
        start += L
        pq.write_table(
            pa.Table.from_pandas(pd.DataFrame([row])),
            root / "meta" / "episodes" / "chunk-000" / f"file-{ei:03d}.parquet",
        )
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows)), root / "data" / "chunk-000" / "file-000.parquet")
    return eps


@contextmanager
def _mock_decoder():
    def _fake(path, frame_indices, h, w):
        cam = next(c for c in CAM_COLORS if c in str(path))
        return [Image.new("RGB", (w, h), CAM_COLORS[cam]) for _ in frame_indices]

    with patch("openwam.dataloader.yam_sharpa._decode_video_frames", side_effect=_fake):
        yield


# Color jitter off, at the strengths the yaml ships; tests that want it on pass {**_JITTER, "enabled": True}.
_JITTER = {"enabled": False, "brightness": 0.2, "contrast": 0.2, "saturation": 0.2}


def _cfg(sources: dict, **overrides):
    """A complete yam_sharpa dataloader config over synthetic sources (every key from_config requires).

    Args:
        sources: {name: (dataset_dir, weight)}. **overrides: top-level keys replacing the defaults below.

    Returns:
        DictConfig for ``build_dataset`` (= ``merge(common, dataloader)``: action_dim / state_dim default to the
        action_mode's width, fingertips 48, unify 80).
    """
    cfg = {
        "type": "yam_sharpa",
        "sources": {n: {"dataset_dir": str(d), "weight": w, "enabled": True} for n, (d, w) in sources.items()},
        "action_mode": "fingertips_abs",
        "fps": 20,  # must equal the synthetic source's meta/info.json fps
        "normalize_mode": None,
        "num_frames": 5,
        "history_offsets": [0],  # the current frame + 2 future frames (rows t, t+2, t+4)
        "future_offsets": [2, 4],
        "height": 480,
        "width": 320,
        "camera_layout": [EGO, CHEST],
        "history_sampling": "fixed",
        "history_window": 100,
        "caption_dropout_p": 0.0,
        "seed": 0,
        "window_stride": 1,
        "normalization_stats_path": None,
        "color_jitter": _JITTER,
    }
    cfg.update(overrides)
    width = 80 if str(cfg["action_mode"]).startswith("unify") else 48
    return OmegaConf.create({"action_dim": width, "state_dim": width, **cfg})


@pytest.fixture
def source(tmp_path):
    eps = _write_source(tmp_path / "a", [10, 7])
    return tmp_path / "a", eps


def _first_window(ds: MultiYamSharpaDataset) -> dict:
    with _mock_decoder():
        return ds.datasets[0].sample(0, np.random.default_rng(0))  # episode 0, anchor t=0


def _rot6d(Rm):
    return np.concatenate([Rm[:3, 0], Rm[:3, 1]])


# ── representation semantics ────────────────────────────────────────────────


def test_registry_includes_yam_sharpa():
    assert "yam_sharpa" in DATASET_REGISTRY


def test_fingertips_abs_moves_target_wrist_into_anchor_camera(source):
    root, eps = source
    ds = build_dataset(_cfg({"a": (root, 1.0)}), "train")
    s = _first_window(ds)
    ep = eps[0]
    assert s["action"].shape == (4, 48) and s["proprio"].shape == (1, 48)
    np.testing.assert_allclose(s["proprio"][0].numpy(), ep["state"][0], atol=1e-6)
    cam0_inv = np.linalg.inv(ep["cam_mats"][0])
    for k in range(4):
        for h, o in enumerate((0, 24)):
            W = cam0_inv @ ep["targets_world"][k][h]  # target wrist in the frame-0 camera
            np.testing.assert_allclose(s["action"][k, o : o + 3].numpy(), W[:3, 3], atol=1e-4)
            np.testing.assert_allclose(s["action"][k, o + 3 : o + 9].numpy(), _rot6d(W), atol=1e-4)
            np.testing.assert_allclose(
                s["action"][k, o + 9 : o + 24].numpy(), ep["action"][k, o + 9 : o + 24], atol=1e-6
            )


def test_fingertips_rel_is_delta_in_anchor_camera(source):
    root, eps = source
    ds = build_dataset(_cfg({"a": (root, 1.0)}, action_mode="fingertips_rel"), "train")
    s = _first_window(ds)
    ep = eps[0]
    cam0_inv = np.linalg.inv(ep["cam_mats"][0])
    for k in range(4):
        for h, o in enumerate((0, 24)):
            W = cam0_inv @ ep["targets_world"][k][h]
            p_t = ep["state"][0, o : o + 3]
            R_t = R.rotation_6d_to_matrix(ep["state"][0, o + 3 : o + 9][None])[0]
            np.testing.assert_allclose(s["action"][k, o : o + 3].numpy(), W[:3, 3] - p_t, atol=1e-4)
            np.testing.assert_allclose(s["action"][k, o + 3 : o + 9].numpy(), _rot6d(W[:3, :3] @ R_t.T), atol=1e-4)
            np.testing.assert_allclose(
                s["action"][k, o + 9 : o + 24].numpy(),
                ep["action"][k, o + 9 : o + 24] - ep["state"][0, o + 9 : o + 24],
                atol=1e-5,
            )
    # proprio is still the absolute state
    np.testing.assert_allclose(s["proprio"][0].numpy(), ep["state"][0], atol=1e-6)


@pytest.mark.parametrize("mode", ["unify_abs", "unify_rel"])
def test_unify_layout_and_hand_joints(source, mode):
    root, eps = source
    ds = build_dataset(_cfg({"a": (root, 1.0)}, action_mode=mode), "train")
    s = _first_window(ds)
    ep = eps[0]
    assert s["action"].shape == (4, 80) and s["proprio"].shape == (1, 80)
    valid = np.zeros(80, bool)
    valid[list(range(0, 9)) + list(range(10, 32)) + list(range(34, 43)) + list(range(44, 66))] = True
    np.testing.assert_array_equal(s["action_mask"][0].numpy(), valid)
    np.testing.assert_array_equal(s["proprio_mask"][0].numpy(), valid)
    for slots, src in ((slice(10, 32), slice(12, 34)), (slice(44, 66), slice(34, 56))):
        np.testing.assert_allclose(s["proprio"][0, slots].numpy(), ep["joints"][0, src], atol=1e-6)
        want = ep["setpoints"][:4, src] - (ep["joints"][0, src] if mode == "unify_rel" else 0.0)
        np.testing.assert_allclose(s["action"][:, slots].numpy(), want, atol=1e-5)
    # wrist slots carry the same values as the matching fingertips representation
    ref = build_dataset(_cfg({"a": (root, 1.0)}, action_mode=mode.replace("unify", "fingertips")), "train")
    r = _first_window(ref)
    np.testing.assert_allclose(s["action"][:, 0:9].numpy(), r["action"][:, 0:9].numpy(), atol=1e-6)
    np.testing.assert_allclose(s["action"][:, 34:43].numpy(), r["action"][:, 24:33].numpy(), atol=1e-6)


@pytest.mark.parametrize("mode", ["fingertips_rel", "unify_rel"])
def test_decode_inverts_relative_encoding(source, mode):
    _, eps = source
    ep = eps[0]
    n = 6
    kw = dict(
        anchor_state48=np.repeat(ep["state"][:1], n, 0),
        anchor_joint_angles56=np.repeat(ep["joints"][:1], n, 0),
        anchor_cam16=np.repeat(ep["cams"][:1], n, 0),
        target_action48=ep["action"][:n],
        target_setpoints56=ep["setpoints"][:n],
        target_cam16=ep["cams"][:n],
    )
    rel = R.encode_action_pairs(mode, **kw)
    proprio = R.encode_proprio(mode, ep["state"][0], ep["joints"][0])[0]
    absolute = R.encode_action_pairs(mode.replace("_rel", "_abs"), **kw)
    np.testing.assert_allclose(R.decode_action(mode, rel, proprio), absolute, atol=1e-5)
    np.testing.assert_array_equal(R.decode_action(mode.replace("_rel", "_abs"), absolute, proprio), absolute)


def test_unknown_mode_and_unify_keys_rejected(source):
    root, _ = source
    with pytest.raises(ValueError, match="unknown YamSharpa action_mode"):
        build_dataset(_cfg({"a": (root, 1.0)}, action_mode="joint"), "train")
    with pytest.raises(ValueError, match="derives unify_action"):
        build_dataset(_cfg({"a": (root, 1.0)}, unify_action=True), "train")


def test_from_config_rejects_width_mismatch(source):
    # common.action_dim / state_dim are checked against the rows the action_mode produces, before any reader.
    root, _ = source
    with pytest.raises(ValueError, match=r"common\.action_dim=80.*produces 48-wide"):
        build_dataset(_cfg({"a": (root, 1.0)}, action_mode="fingertips_rel", action_dim=80), "train")
    with pytest.raises(ValueError, match=r"common\.action_dim=48 / common\.state_dim=48 .*unify_rel.*80-wide"):
        build_dataset(_cfg({"a": (root, 1.0)}, action_mode="unify_rel", action_dim=48, state_dim=48), "train")


def test_fps_must_match_info_json(source):
    root, _ = source
    build_dataset(_cfg({"a": (root, 1.0)}, fps=20), "train")
    with pytest.raises(ValueError, match="meta/info.json records fps=20"):
        build_dataset(_cfg({"a": (root, 1.0)}, fps=30), "train")


@pytest.mark.parametrize("key", ["seed", "num_frames", "sources.a.weight", "sources.a.enabled", "color_jitter.saturation"])
def test_from_config_requires_every_key(source, key):
    # No hidden defaults: a config missing any key the dataset reads fails instead of silently using a value.
    root, _ = source
    # Jitter on, so its strengths are read too.
    cfg = OmegaConf.to_container(_cfg({"a": (root, 1.0)}, color_jitter={**_JITTER, "enabled": True}))
    *parents, leaf = key.split(".")
    node = cfg
    for part in parents:
        node = node[part]
    del node[leaf]
    with pytest.raises(Exception, match=leaf):
        build_dataset(OmegaConf.create(cfg), "train")


def test_config_keys_accept_fps():
    from openwam.dataloader.yam_sharpa import YamSharpaDataset

    assert "fps" in YamSharpaDataset.CONFIG_KEYS


def _all_samples(reader, seed: int = 0) -> list:
    """Every window of one source reader, each drawn with its own ``(seed, k)`` RNG (fake video decoder).

    Args:
        reader: YamSharpaDataset. seed: int, RNG seed.

    Returns:
        list of sample dicts, one per window index.
    """
    with _mock_decoder():
        return [reader.sample(k, np.random.default_rng([seed, k])) for k in range(len(reader))]


def test_preload_never_rereads(source):
    # The data shards are read once at construction; sampling afterwards never touches a parquet file.
    root, _ = source
    reader = build_dataset(_cfg({"a": (root, 1.0)}), "train").datasets[0]
    with patch("pyarrow.parquet.read_table", side_effect=AssertionError("re-read a shard")) as read:
        got = _all_samples(reader)
    read.assert_not_called()
    # Same samples, byte for byte, as a reader whose shard table is read straight from parquet.
    shard = root / "data" / "chunk-000" / "file-000.parquet"
    direct = copy.copy(reader)
    direct._data_tables = {(0, 0): pq.read_table(shard, columns=list(type(reader).NEEDED_COLS))}
    want = _all_samples(direct)
    assert len(got) == len(want) > 0
    for g, w in zip(got, want):
        assert g["prompt"] == w["prompt"]
        for k in ("action", "action_mask", "proprio", "proprio_mask"):
            assert g[k].numpy().tobytes() == w[k].numpy().tobytes(), k


# ── random history sampling (history_sampling=random) ───────────────────────

RAND_HIST = [-8, -4, 0]  # fixed (val / deploy) offsets of the random-history sources below


def _random_cfg(root: Path, **overrides):
    """Config of one long-episode source with random history (window 40 rows) over ``RAND_HIST``."""
    keys = {"history_offsets": RAND_HIST, "history_sampling": "random", "history_window": 40}
    return _cfg({"a": (root, 1.0)}, **{**keys, **overrides})


def test_random_history_offsets():
    from openwam.dataloader.utils.sparse_history import sample_history_offsets

    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(2000):
        o = sample_history_offsets(5, 100, rng)
        # H offsets, strictly increasing, inside [-100, 0], ending at frame t.
        assert o.shape == (5,) and o.dtype == np.int64
        assert o[-1] == 0 and (np.diff(o) > 0).all() and o[0] >= -100
        seen.update(o[:-1].tolist())
    assert seen == set(range(-100, 0))  # every past row of the window gets drawn
    np.testing.assert_array_equal(sample_history_offsets(1, 100, rng), [0])  # H=1: frame t only
    # The draw is a function of the generator state alone.
    a = sample_history_offsets(5, 100, np.random.default_rng([42, 3, 7]))
    np.testing.assert_array_equal(a, sample_history_offsets(5, 100, np.random.default_rng([42, 3, 7])))


def _offsets_by_index(ds, epoch: int, num_workers: int) -> list:
    """History offsets each mixture index drew, read through a DataLoader.

    Args:
        ds: MultiYamSharpaDataset with random history. epoch: int, ``set_epoch`` value.
        num_workers: int, DataLoader workers (forked, so the patches below reach them).

    Returns:
        list of (H,) offset lists, one per index in order.
    """
    import torch

    import openwam.dataloader.yam_sharpa as ys

    real_draw, real_sample, last = ys.sample_history_offsets, ys.YamSharpaDataset.sample, {}

    def _draw(*args):
        last["offsets"] = real_draw(*args)  # remember this process's latest draw
        return last["offsets"]

    def _sample(self, idx, rng):
        out = real_sample(self, idx, rng)
        out["_offsets"] = last.pop("offsets").tolist()  # the draw of this very sample
        return out

    ds.set_epoch(epoch)
    with patch.object(ys, "sample_history_offsets", _draw), patch.object(ys.YamSharpaDataset, "sample", _sample):
        with _mock_decoder():
            loader = torch.utils.data.DataLoader(
                ds, batch_size=None, num_workers=num_workers, multiprocessing_context="fork" if num_workers else None
            )
            return [s["_offsets"] for s in loader]


def test_random_history_deterministic(tmp_path):
    # Offsets come from the per-sample (seed, epoch, idx) RNG: worker count does not matter, the epoch does.
    _write_source(tmp_path / "a", [60, 60])
    ds = build_dataset(_random_cfg(tmp_path / "a"), "train")
    serial = _offsets_by_index(ds, epoch=0, num_workers=0)
    assert serial == _offsets_by_index(ds, epoch=0, num_workers=4)
    assert all(len(o) == 3 and o[-1] == 0 and o[0] >= -40 for o in serial)
    other = _offsets_by_index(ds, epoch=1, num_workers=0)
    assert sum(a != b for a, b in zip(serial, other)) > len(serial) // 2


def test_val_split_uses_fixed_offsets(tmp_path):
    # Random history is a train-split augmentation; val always reads history_offsets (the deploy layout).
    _write_source(tmp_path / "a", [60, 60], splits={"train": "0:2", "val": "0:2"})
    rand_val = build_dataset(_random_cfg(tmp_path / "a"), "val").datasets[0]
    fixed_val = build_dataset(_random_cfg(tmp_path / "a", history_sampling="fixed"), "val").datasets[0]
    with patch("openwam.dataloader.yam_sharpa.sample_history_offsets", side_effect=AssertionError("drew")) as draw:
        got = _all_samples(rand_val)
    draw.assert_not_called()
    want = _all_samples(fixed_val)
    for g, w in zip(got, want):
        assert g["proprio"].numpy().tobytes() == w["proprio"].numpy().tobytes()
    # The same config on the train split does draw.
    rand_train = build_dataset(_random_cfg(tmp_path / "a"), "train").datasets[0]
    stub = lambda h, w, rng: np.array([-9, -1, 0])  # noqa: E731
    with patch("openwam.dataloader.yam_sharpa.sample_history_offsets", wraps=stub) as draw:
        _all_samples(rand_train)
    assert draw.call_count == len(rand_train)


# ── caption dropout (dataloader.caption_dropout_p) ──────────────────────────


def _prompts_by_index(ds, epoch: int, num_workers: int) -> list:
    """(prompt, proprio bytes) of every mixture index, read through a DataLoader.

    Args:
        ds: MultiYamSharpaDataset. epoch: int, ``set_epoch`` value. num_workers: int, forked DataLoader workers.

    Returns:
        list of (str prompt, bytes of the (H, D) proprio), one per index in order.
    """
    import torch

    ds.set_epoch(epoch)
    with _mock_decoder():
        loader = torch.utils.data.DataLoader(
            ds, batch_size=None, num_workers=num_workers, multiprocessing_context="fork" if num_workers else None
        )
        return [(s["prompt"], s["proprio"].numpy().tobytes()) for s in loader]


def test_caption_dropout_deterministic(tmp_path):
    # The dropout draw comes from the per-sample (seed, epoch, idx) RNG, after the random history offsets.
    _write_source(tmp_path / "a", [60, 60])
    dropped = build_dataset(_random_cfg(tmp_path / "a", caption_dropout_p=0.5), "train")
    serial = _prompts_by_index(dropped, epoch=0, num_workers=0)
    assert serial == _prompts_by_index(dropped, epoch=0, num_workers=4)  # worker count does not matter
    n_empty = sum(p == "" for p, _ in serial)
    assert 0 < n_empty < len(serial)
    assert _prompts_by_index(dropped, epoch=1, num_workers=0) != serial  # the epoch does
    # p = 0: prompts unchanged; the rest of every sample (random history included) is the same either way.
    kept = _prompts_by_index(build_dataset(_random_cfg(tmp_path / "a"), "train"), epoch=0, num_workers=0)
    assert all(p != "" for p, _ in kept)
    for (p_drop, proprio_drop), (p_keep, proprio_keep) in zip(serial, kept):
        assert proprio_drop == proprio_keep and p_drop in ("", p_keep)
    # Val never drops (a train-split augmentation).
    _write_source(tmp_path / "b", [60, 60], splits={"train": "0:2", "val": "0:2"})
    val = build_dataset(_random_cfg(tmp_path / "b", caption_dropout_p=1.0), "val")
    assert all(p != "" for p, _ in _prompts_by_index(val, epoch=0, num_workers=0))


def _videos_by_index(ds, epoch: int, num_workers: int) -> list:
    """The video bytes of every mixture index, read through a DataLoader.

    Args:
        ds: MultiYamSharpaDataset. epoch: int, ``set_epoch`` value. num_workers: int, forked DataLoader workers.

    Returns:
        list of bytes (all frames of one sample concatenated), one per index in order.
    """
    import torch

    ds.set_epoch(epoch)
    with _mock_decoder():
        loader = torch.utils.data.DataLoader(
            ds, batch_size=None, num_workers=num_workers, multiprocessing_context="fork" if num_workers else None
        )
        return [b"".join(np.asarray(f).tobytes() for f in s["video"]) for s in loader]


def test_color_jitter_draws_from_the_sample_rng(tmp_path):
    # The jitter factors come from the per-sample (seed, epoch, idx) RNG, not python's global random: the samples
    # do not depend on the global RNG state or the worker count, and change with the epoch.
    import random

    _write_source(tmp_path / "a", [60, 60])
    ds = build_dataset(_random_cfg(tmp_path / "a", color_jitter={**_JITTER, "enabled": True}), "train")
    random.seed(0)
    serial = _videos_by_index(ds, epoch=0, num_workers=0)
    random.seed(1)
    assert serial == _videos_by_index(ds, epoch=0, num_workers=0)
    assert serial == _videos_by_index(ds, epoch=0, num_workers=4)
    assert _videos_by_index(ds, epoch=1, num_workers=0) != serial
    # Jitter is on: the frames differ from the un-jittered reader's.
    plain = build_dataset(_random_cfg(tmp_path / "a"), "train")
    assert _videos_by_index(plain, epoch=0, num_workers=0) != serial


def test_visualizer_sample_has_eval_semantics(tmp_path):
    # The training visualizer's fixed sample comes from eval_view(): no caption dropout, the fixed history offsets.
    from openwam.train.utils.visualization import TrainVisualizer

    _write_source(tmp_path / "a", [60, 60])
    train = build_dataset(_random_cfg(tmp_path / "a", caption_dropout_p=1.0), "train")
    fixed = build_dataset(_random_cfg(tmp_path / "a", history_sampling="fixed"), "train")
    view = train.eval_view()
    assert view.datasets[0]._data_tables is train.datasets[0]._data_tables  # shares the preloaded shards
    with _mock_decoder():
        with patch("openwam.dataloader.yam_sharpa.sample_history_offsets", side_effect=AssertionError("drew")):
            vis = TrainVisualizer(view, rank=0)
        want = fixed[vis.sample_index]
        assert train[vis.sample_index]["prompt"] == ""  # the training dataset itself is unchanged
    assert vis.sample["prompt"] == want["prompt"] != ""
    assert vis.sample["proprio"].numpy().tobytes() == want["proprio"].numpy().tobytes()


def test_eval_view_turns_every_train_augmentation_off(tmp_path):
    # eval_view applies the val split's rule (_select_augmentations(train=False)): no color jitter, the fixed
    # history offsets; the training dataset keeps its augmentations.
    _write_source(tmp_path / "a", [60, 60])
    reader = build_dataset(_random_cfg(tmp_path / "a", color_jitter={**_JITTER, "enabled": True}), "train").datasets[0]
    assert reader._color_jitter is not None
    assert reader._draw_history_offsets == reader._random_history_offsets
    view = reader.eval_view()
    assert view._color_jitter is None
    assert view._draw_history_offsets == view._fixed_history_offsets
    assert reader._color_jitter is not None and reader._draw_history_offsets == reader._random_history_offsets
    # Behaviour: the view's samples are those of an un-augmented reader over the same windows.
    plain = build_dataset(_random_cfg(tmp_path / "a", history_sampling="fixed"), "train").datasets[0]
    with patch("openwam.dataloader.yam_sharpa.sample_history_offsets", side_effect=AssertionError("drew")):
        got = _all_samples(view)
    for g, w in zip(got, _all_samples(plain)):
        assert [np.asarray(f).tobytes() for f in g["video"]] == [np.asarray(f).tobytes() for f in w["video"]]
        assert g["proprio"].numpy().tobytes() == w["proprio"].numpy().tobytes()


# ── video layout ────────────────────────────────────────────────────────────


def test_video_is_ego_over_chest(source):
    root, _ = source
    s = _first_window(build_dataset(_cfg({"a": (root, 1.0)}), "train"))
    assert len(s["video"]) == 3  # frame t + future rows t+2, t+4
    img = np.asarray(s["video"][0])
    assert img.shape == (480, 320, 3)
    assert (img[:240] == CAM_COLORS[EGO]).all() and (img[240:] == CAM_COLORS[CHEST]).all()


# ── probability sampling ────────────────────────────────────────────────────


def test_sampling_follows_weights_and_covers_every_window(tmp_path):
    _write_source(tmp_path / "a", [30, 30], seed=1)
    _write_source(tmp_path / "b", [30, 30], seed=2)
    ds = build_dataset(_cfg({"a": (tmp_path / "a", 0.8), "b": (tmp_path / "b", 0.2)}), "train")
    assert len(ds) == sum(len(d) for d in ds.datasets)
    np.testing.assert_allclose(ds.probabilities, [0.8, 0.2])

    picks = {"a": set(), "b": set()}
    counts = {"a": 0, "b": 0}
    n_b = len(ds.datasets[1])

    def _record(self, j, rng):
        picks[self._dataset_id].add(j)
        return {"_j": j}

    with patch.object(type(ds.datasets[0]), "sample", _record):
        for epoch in range(8):
            ds.set_epoch(epoch)
            for i in range(len(ds)):
                counts[ds[i]["source"]] += 1
    frac_b = counts["b"] / (counts["a"] + counts["b"])
    assert abs(frac_b - 0.2) < 0.03
    # the down-weighted source is not stuck on its first windows
    assert len(picks["b"]) > 0.9 * n_b

    ds.set_epoch(0)
    with patch.object(type(ds.datasets[0]), "sample", _record):
        a0 = [ds[i]["_j"] for i in range(50)]
        b0 = [ds[i]["_j"] for i in range(50)]
    assert a0 == b0  # deterministic per (seed, epoch, idx)


# ── normalization stats ─────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", R.ACTION_REPRS)
def test_stats_shapes_pinning_and_sharing(source, mode):
    root, _ = source
    payload = compute_yam_sharpa_stats([root], mode, num_frames=5, workers=1)
    act, st = payload[mode], payload[f"{mode}_state"]
    D = R.raw_dim(mode)
    for block in (act, st):
        assert all(block[k].shape == (D,) for k in ("mean", "std", "min", "max", "q01", "q99"))
        rd = R.rot6d_dims(mode)
        np.testing.assert_array_equal(block["min"][rd], -1.0)
        np.testing.assert_array_equal(block["max"][rd], 1.0)
    if R.is_relative(mode):
        assert not np.allclose(act["max"], st["max"])
    else:
        for k in act:
            np.testing.assert_array_equal(act[k], st[k])


def test_stats_num_frames_is_required(source, tmp_path):
    """The action horizon comes from the dataloader's num_frames: no default in the function or the CLI."""
    from openwam.dataloader.utils.stats_computation.yam_sharpa_stats_computation import main

    root, _ = source
    with pytest.raises(TypeError, match="num_frames"):
        compute_yam_sharpa_stats([root], "fingertips_rel", workers=1)
    with pytest.raises(SystemExit):
        main(["--dataset-dirs", str(root), "--action-mode", "fingertips_rel", "--output", str(tmp_path / "s.npy")])
    # The payload is the two blocks + the provenance the reader matches (no unread schema field).
    payload = compute_yam_sharpa_stats([root], "fingertips_rel", num_frames=5, workers=1)
    assert set(payload) == {"fingertips_rel", "fingertips_rel_state", "yam_sharpa_meta"}


def test_stats_cover_every_supervised_pair(source):
    """min/max equal a brute-force scan over every (t, t+k) the reader can supervise."""
    root, eps = source
    mode = "fingertips_rel"
    payload = compute_yam_sharpa_stats([root], mode, num_frames=5, workers=1)
    rows = []
    for ep in eps:
        L = len(ep["state"])
        for t in range(L):
            for k in range(min(4, L - t)):
                rows.append(
                    R.encode_action_pairs(
                        mode,
                        anchor_state48=ep["state"][t],
                        anchor_joint_angles56=ep["joints"][t],
                        anchor_cam16=ep["cams"][t],
                        target_action48=ep["action"][t + k],
                        target_setpoints56=ep["setpoints"][t + k],
                        target_cam16=ep["cams"][t + k],
                    )[0]
                )
    rows = np.stack(rows)
    keep = [i for i in range(48) if i not in R.rot6d_dims(mode)]
    np.testing.assert_allclose(payload[mode]["min"][keep], rows.min(0)[keep], atol=1e-6)
    np.testing.assert_allclose(payload[mode]["max"][keep], rows.max(0)[keep], atol=1e-6)


def test_from_config_builds_reuses_and_guards_stats(source, tmp_path):
    root, _ = source
    path = tmp_path / "stats" / "fr.npy"
    cfg = _cfg(
        {"a": (root, 1.0)}, action_mode="fingertips_rel", normalize_mode="min-max", normalization_stats_path=str(path)
    )
    ds = build_dataset(cfg, "train")
    assert path.is_file() and ds.normalization_stats_path == str(path)
    mtime = path.stat().st_mtime_ns
    s = _first_window(build_dataset(cfg, "train"))
    assert path.stat().st_mtime_ns == mtime  # reused, not rebuilt
    a = s["action"].numpy()
    assert (np.abs(a) <= 1.0 + 1e-6).all()
    # relative action and absolute proprio use separate blocks
    act = load_stats_file(path, block="fingertips_rel", dim=48)
    st = load_stats_file(path, block="fingertips_rel_state", dim=48)
    assert not np.allclose(act["max"], st["max"])
    # a file built for another configuration is never silently reused
    other = _cfg(
        {"a": (root, 1.0)}, action_mode="fingertips_abs", normalize_mode="min-max", normalization_stats_path=str(path)
    )
    with pytest.raises(ValueError, match="was built for"):
        build_dataset(other, "train")


def test_normalization_keeps_rot6d(source, tmp_path):
    root, eps = source
    path = tmp_path / "fa.npy"
    save_stats(compute_yam_sharpa_stats([root], "fingertips_abs", num_frames=5, workers=1), path)
    cfg = _cfg({"a": (root, 1.0)}, normalize_mode="min-max", normalization_stats_path=str(path))
    s = _first_window(build_dataset(cfg, "train"))
    raw = build_dataset(_cfg({"a": (root, 1.0)}), "train")
    r = _first_window(raw)
    for dims in (slice(3, 9), slice(27, 33)):
        np.testing.assert_allclose(s["action"][:, dims].numpy(), r["action"][:, dims].numpy(), atol=1e-6)
        np.testing.assert_allclose(s["proprio"][:, dims].numpy(), r["proprio"][:, dims].numpy(), atol=1e-6)
    assert not np.allclose(s["action"][:, 0:3].numpy(), r["action"][:, 0:3].numpy())


# ── config + deploy ─────────────────────────────────────────────────────────


def test_repo_config_resolves_stats_path_per_mode():
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.action_mode = "unify_rel"
    assert cfg.normalization_stats_path.endswith("yam_sharpa_unify_rel.npy")
    assert list(cfg.camera_layout) == [EGO, CHEST] and (cfg.height, cfg.width) == (480, 320)
    assert "unify_action" not in cfg


@pytest.mark.parametrize("mode", R.ACTION_REPRS)
def test_deploy_normalization_inverts_the_training_sample(source, tmp_path, mode):
    """Deploy's I/O functions against a training sample: the raw proprio normalizes to the sample's proprio, and the
    sample's normalized action maps back (gather, invert, decode) to the absolute targets."""
    from openwam.dataloader.utils.normalization import apply_normalization, invert_normalization
    from openwam.dataloader.utils.unify_action import UNIFY_DIM, map_to_unify, parse_unify_spec, unmap_from_unify
    from openwam.deploy.model_loader import load_norm_stats

    root, eps = source
    stats_path = tmp_path / "normalization_stats.npy"
    save_stats(compute_yam_sharpa_stats([root], mode, num_frames=5, workers=1), stats_path)
    state_stats, action_stats = load_norm_stats(
        OmegaConf.create({"action_mode": mode, "normalize_mode": "min-max"}), str(tmp_path)
    )
    train_cfg = _cfg({"a": (root, 1.0)}, action_mode=mode, normalize_mode="min-max", normalization_stats_path=str(stats_path))
    s = _first_window(build_dataset(train_cfg, "train"))
    ep = eps[0]
    raw_proprio = R.encode_proprio(mode, ep["state"][0], ep["joints"][0])[0]

    # proprio IN: normalize the raw width, then (unify) scatter to 80 — the reader's order.
    prop = apply_normalization(raw_proprio.astype(np.float32), state_stats, "min-max")
    actions = s["action"].numpy()
    if R.is_unify(mode):
        dst = parse_unify_spec(R.unify_action_map(mode), UNIFY_DIM)
        prop, _ = map_to_unify(prop, dst, UNIFY_DIM)
        actions = unmap_from_unify(actions, dst)
    np.testing.assert_allclose(prop, s["proprio"][0].numpy(), atol=1e-5)

    # action OUT: invert the normalization, then (relative modes) compose with the raw frame-t proprio.
    actions = invert_normalization(actions, action_stats, "min-max")
    if R.is_relative(mode):
        actions = R.decode_action(mode, actions, raw_proprio)
    abs_mode = mode.replace("_rel", "_abs")
    want = R.encode_action_pairs(
        abs_mode,
        anchor_state48=np.repeat(ep["state"][:1], 4, 0),
        anchor_joint_angles56=np.repeat(ep["joints"][:1], 4, 0),
        anchor_cam16=np.repeat(ep["cams"][:1], 4, 0),
        target_action48=ep["action"][:4],
        target_setpoints56=ep["setpoints"][:4],
        target_cam16=ep["cams"][:4],
    )
    np.testing.assert_allclose(actions, want, atol=1e-4)


def test_deploy_canvas_matches_training_resize():
    """Deploy canvas == training canvas: each camera LANCZOS-resized to its tile (as the decode does), then stacked."""
    from openwam.dataloader.transforms.multiview import assemble_vstack_layout
    from openwam.deploy.engine import EngineSpec, InferenceEngine

    spec = EngineSpec(ckpt="-", action_mode="fingertips_abs", action_dim=48, chunk=32,
                      steps_per_plan=20, action_fps=20, history_offsets=(0,), camera_layout=(EGO, CHEST))
    # observation() reads only the spec and the canvas size: no model, no stats.
    engine = InferenceEngine(None, None, spec, state_stats={}, action_stats={}, normalize_mode="min-max",
                             height=384, width=224, seed=42, decode_video=False)
    rng = np.random.default_rng(0)
    native = {c: rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for c in (EGO, CHEST)}
    obs = {**native, "observation.state": np.zeros(48), "camera_pose.ego_view": np.eye(4).reshape(-1)}
    deploy = np.asarray(engine.observation(obs)["canvas"])
    # Training: decode_video_frames resizes each camera to (192, 224) with LANCZOS, then assemble_vstack_layout.
    tiles = {c: Image.fromarray(im).resize((224, 192), Image.LANCZOS) for c, im in native.items()}
    train = np.asarray(assemble_vstack_layout(tiles, [EGO, CHEST], out_h=384, out_w=224))
    assert np.array_equal(deploy, train)


def test_vstack_layout_pastes_tiles_and_rejects_unresized_frames():
    """Tiles are pasted as they are (top to bottom); a frame not resized to its tile raises instead of cropping."""
    from openwam.dataloader.transforms.multiview import assemble_vstack_layout

    tiles = {EGO: Image.new("RGB", (224, 192), (255, 0, 0)), CHEST: Image.new("RGB", (224, 192), (0, 255, 0))}
    canvas = np.asarray(assemble_vstack_layout(tiles, [EGO, CHEST], out_h=384, out_w=224))
    assert (canvas[:192] == (255, 0, 0)).all() and (canvas[192:] == (0, 255, 0)).all()
    with pytest.raises(ValueError, match="expected"):
        assemble_vstack_layout({EGO: Image.new("RGB", (640, 480))}, [EGO, CHEST], out_h=384, out_w=224)
