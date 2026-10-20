"""YAM-duo + Sharpa-hand reader: representations, frames, layout, sampling, stats, deploy."""

from __future__ import annotations

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

from openwam.dataloader.registry import build_dataset, list_registered_datasets
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


def _write_source(root: Path, lengths, seed: int = 0) -> list:
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


def _cfg(sources: dict, **overrides):
    cfg = {
        "type": "yam_sharpa",
        "sources": {n: {"dataset_dir": str(d), "weight": w} for n, (d, w) in sources.items()},
        "action_mode": "fingertips_abs",
        "normalize_mode": None,
        "num_frames": 5,
        "video_stride": 2,
        "height": 480,
        "width": 320,
        "camera_layout": [EGO, CHEST],
        "seed": 0,
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


@pytest.fixture
def source(tmp_path):
    eps = _write_source(tmp_path / "a", [10, 7])
    return tmp_path / "a", eps


def _first_window(ds: MultiYamSharpaDataset) -> dict:
    with _mock_decoder():
        return ds.datasets[0][0]  # episode 0, anchor t=0


def _rot6d(Rm):
    return np.concatenate([Rm[:3, 0], Rm[:3, 1]])


# ── representation semantics ────────────────────────────────────────────────


def test_registry_includes_yam_sharpa():
    assert "yam_sharpa" in list_registered_datasets()


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
            R_t = R._rotation_6d_to_matrix(ep["state"][0, o + 3 : o + 9][None])[0]
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


# ── video layout ────────────────────────────────────────────────────────────


def test_video_is_ego_over_chest(source):
    root, _ = source
    s = _first_window(build_dataset(_cfg({"a": (root, 1.0)}), "train"))
    assert len(s["video"]) == 3  # num_frames=5, stride 2
    img = np.asarray(s["video"][0])
    assert img.shape == (480, 320, 3)
    assert (img[:240] == CAM_COLORS[EGO]).all() and (img[240:] == CAM_COLORS[CHEST]).all()
    assert s["first_frame_image"][0] is s["video"][0]


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

    def _record(self, j):
        picks[self._dataset_id].add(j)
        return {"_j": j}

    with patch.object(type(ds.datasets[0]), "__getitem__", _record):
        for epoch in range(8):
            ds.set_epoch(epoch)
            for i in range(len(ds)):
                counts[ds[i]["_dataset_name"]] += 1
    frac_b = counts["b"] / (counts["a"] + counts["b"])
    assert abs(frac_b - 0.2) < 0.03
    # the down-weighted source is not stuck on its first windows
    assert len(picks["b"]) > 0.9 * n_b

    ds.set_epoch(0)
    with patch.object(type(ds.datasets[0]), "__getitem__", _record):
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
    act = load_stats_file(path, action_mode="fingertips_rel", normalize_mode="min-max", dim=48)
    st = load_stats_file(path, action_mode="fingertips_rel_state", normalize_mode="min-max", dim=48)
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
    assert cfg.multiview_layout == "vstack"
    assert "unify_action" not in cfg


@pytest.mark.parametrize("mode", R.ACTION_REPRS)
def test_deploy_normalizer_and_decoder(source, tmp_path, mode):
    from openwam.deploy.model_loader import _build_action_decoder, _build_normalizer, _UnifyAwareNormalizer

    root, eps = source
    save_stats(compute_yam_sharpa_stats([root], mode, num_frames=5, workers=1), tmp_path / "normalization_stats.npy")
    cfg = OmegaConf.create({"dataloader": {"type": "yam_sharpa", "action_mode": mode, "normalize_mode": "min-max"}})
    norm = _build_normalizer(cfg, str(tmp_path))
    D = R.raw_dim(mode)
    if R.is_unify(mode):
        assert isinstance(norm, _UnifyAwareNormalizer)
        out = norm.unnormalize(np.zeros((4, 80), np.float32))
    else:
        out = norm.unnormalize(np.zeros((4, 48), np.float32))
    assert out.shape == (4, D)

    # train-time normalized sample → deploy unnormalize (+ decode) == absolute target
    stats_path = tmp_path / "normalization_stats.npy"
    train_cfg = _cfg(
        {"a": (root, 1.0)}, action_mode=mode, normalize_mode="min-max", normalization_stats_path=str(tmp_path / "s.npy")
    )
    save_stats(np.load(stats_path, allow_pickle=True).item(), tmp_path / "s.npy")
    s = _first_window(build_dataset(train_cfg, "train"))
    ep = eps[0]
    raw_proprio = R.encode_proprio(mode, ep["state"][0], ep["joints"][0])[0]
    np.testing.assert_allclose(norm.normalize(raw_proprio), s["proprio"][0].numpy(), atol=1e-5)
    actions = norm.unnormalize(s["action"].numpy())
    decoder = _build_action_decoder(cfg)
    assert (decoder is None) == (not R.is_relative(mode))
    if decoder is not None:
        actions = decoder(actions, raw_proprio)
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


def test_deploy_obs_preprocess_vstack():
    from openwam.deploy.obs_preprocess import ObsPreprocessor, ObsValidationError

    cfg = OmegaConf.create(
        {
            "dataloader": {
                "multiview": True,
                "multiview_layout": "vstack",
                "camera_layout": [EGO, CHEST],
                "height": 480,
                "width": 320,
            }
        }
    )
    pre = ObsPreprocessor.from_cfg(cfg)
    ego = Image.new("RGB", (640, 480), CAM_COLORS[EGO])
    chest = Image.new("RGB", (640, 480), CAM_COLORS[CHEST])
    for imgs in ({EGO: ego, CHEST: chest}, {"head_camera": ego, "left_wrist_camera": chest}):
        out = np.asarray(pre.preprocess({"images": dict(imgs), "state": [0.0] * 48})["image"])
        assert out.shape == (480, 320, 3)
        assert (out[:240] == CAM_COLORS[EGO]).all() and (out[240:] == CAM_COLORS[CHEST]).all()
    with pytest.raises(ObsValidationError, match="requires every camera"):
        pre.preprocess({"images": {EGO: ego}})


def test_deploy_obs_preprocess_vstack_matches_training_resize():
    """Deploy canvas == training canvas: each camera LANCZOS-resized to its tile (as the decode does), then stacked."""
    from openwam.dataloader.transforms.multiview import assemble_vstack_layout
    from openwam.deploy.obs_preprocess import ObsPreprocessor

    cfg = OmegaConf.create(
        {"dataloader": {"multiview": True, "multiview_layout": "vstack", "camera_layout": [EGO, CHEST],
                        "height": 384, "width": 224}}
    )
    rng = np.random.default_rng(0)
    native = {c: Image.fromarray(rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)) for c in (EGO, CHEST)}
    deploy = np.asarray(ObsPreprocessor.from_cfg(cfg).preprocess({"images": dict(native), "state": [0.0] * 48})["image"])
    # Training: decode_video_frames resizes each camera to (192, 224) with LANCZOS, then assemble_vstack_layout.
    tiles = {c: im.resize((224, 192), Image.LANCZOS) for c, im in native.items()}
    train = np.asarray(assemble_vstack_layout(tiles, [EGO, CHEST], out_h=384, out_w=224))
    assert np.array_equal(deploy, train)
