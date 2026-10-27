"""Sparse-history deploy (CPU, no weights): history buffer, per-episode sessions of the InferenceEngine with a stub
architecture, the startup checks of the deploy cfg, and the hand-target split of the returned actions."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from omegaconf.errors import ReadonlyConfigError
from scipy.spatial.transform import Rotation

from openwam.dataloader.utils import yam_sharpa_repr as R
from openwam.dataloader.utils.sparse_history import history_rows
from openwam.deploy.config import DEPLOY_YAML, build_deploy_cfg, check_deploy
from openwam.deploy.engine import EngineSpec, InferenceEngine
from openwam.deploy.history_buffer import ObsHistoryBuffer

OFFSETS = [-80, -60, -40, -20, 0]


# ── ObsHistoryBuffer ────────────────────────────────────────────────────────


def test_history_rows_clamps_to_episode_start():
    """history_rows clamps every offset to the episode's first row."""
    assert history_rows(150, OFFSETS, 0).tolist() == [70, 90, 110, 130, 150]
    assert history_rows(30, OFFSETS, 0).tolist() == [0, 0, 0, 10, 30]
    assert history_rows(30, OFFSETS, 25).tolist() == [25, 25, 25, 25, 30]


def test_buffer_window_matches_dataset_rule_over_a_long_episode():
    """Observation j = step j*K; the window at every plan equals history_rows of that step, mapped back to obs."""
    for k in (1, 4, 5, 10, 20):
        buf = ObsHistoryBuffer(OFFSETS, action_fps=20, steps_per_obs=k)
        for j in range(300 // k):
            buf.append({"step": j * k})
            assert [o["step"] for o in buf.window()] == history_rows(j * k, OFFSETS, 0).tolist()
        assert buf._obs.maxlen == 1 + 80 // k  # keeps only what offset -80 can reach


def test_buffer_rejects_k_that_misses_history_frames():
    """K must divide action_fps and every offset; otherwise a history frame falls between two observations."""
    for k in (3, 8, 40):
        with pytest.raises(ValueError):
            ObsHistoryBuffer(OFFSETS, action_fps=20, steps_per_obs=k)
    with pytest.raises(ValueError, match="multiple of K"):
        ObsHistoryBuffer([-30, -15, 0], action_fps=20, steps_per_obs=10)
    with pytest.raises(ValueError, match="increase"):
        ObsHistoryBuffer([-20, -40, 0], action_fps=20, steps_per_obs=10)


def test_buffer_reset_starts_a_new_episode():
    """reset() drops the episode: the next window repeats the first new observation."""
    buf = ObsHistoryBuffer(OFFSETS, action_fps=20, steps_per_obs=10)
    for j in range(20):
        buf.append({"step": j})
    buf.reset()
    buf.append({"step": "new"})
    assert [o["step"] for o in buf.window()] == ["new"] * 5
    with pytest.raises(RuntimeError):
        ObsHistoryBuffer(OFFSETS, action_fps=20, steps_per_obs=10).window()


# ── InferenceEngine / Session with a stub architecture ──────────────────────

EGO, CHEST = "observation.images.ego_view", "observation.images.chest_view"
CHUNK, K = 32, 20
# One absolute fingertips_abs action row: per hand wrist at the origin with identity rot6d, fingertips 0.
_ACTION_ROW = np.zeros(48)
_ACTION_ROW[[3, 7, 27, 31]] = 1.0


class _StubArchitecture:
    """No model: records the proprio of every plan's history and returns a constant identity-wrist action chunk."""

    action_dim = 48

    def __init__(self):
        self.seen = []  # per plan: history tags (proprio dim 0 = the tag the test put in state[0])
        self.seeds = []
        self.video_backbone = SimpleNamespace(num_future_latents=lambda num_future_frames: 1)

    def normalize_deploy_proprio(self, raw):
        """raw: (H, D) raw proprio. Returns: (H, D) float32 tensor (identity normalization)."""
        self.seen.append(np.asarray(raw)[:, 0].round().astype(int).tolist())
        return torch.from_numpy(np.asarray(raw, dtype=np.float32))

    def prepare_sparse_condition(self, frames, prompts, proprio, *, num_future_frames):
        """frames: [[H PIL canvases]]. Returns: cond with (1, z, H, h, w) first-frame latents."""
        assert len(frames[0]) == proprio.shape[1]  # one canvas per history row
        return {"first_frame_latents": torch.zeros(1, 2, len(frames[0]), 2, 2)}

    def sample(self, cond, schedule, *, latent_shape, action_shape, seed):
        """Returns: zero latents of latent_shape and an action_shape chunk of _ACTION_ROW."""
        self.seeds.append(seed)
        actions = torch.from_numpy(np.broadcast_to(_ACTION_ROW, action_shape).copy())
        return torch.zeros(latent_shape), actions


class _IdentityNormalizer:
    """unnormalize: (..., D) -> the same values as float64 numpy."""

    @staticmethod
    def unnormalize(x):
        return np.asarray(x, dtype=np.float64)


def _engine(history_offsets=(-40, -20, 0)) -> InferenceEngine:
    """An InferenceEngine over the stub architecture: fingertips_abs, K=20, fps=20, 2-camera vstack 64x32."""
    spec = EngineSpec(
        ckpt="stub/checkpoint_step_1.safetensors",
        action_mode="fingertips_abs",
        action_dim=48,
        hand_targets="tips",
        chunk=CHUNK,
        steps_per_plan=K,
        action_fps=20,
        history_offsets=tuple(history_offsets),
        history_layout="frames",
        camera_layout=(EGO, CHEST),
    )
    return InferenceEngine(
        _StubArchitecture(),
        _IdentityNormalizer(),
        schedule=None,
        spec=spec,
        height=64,
        width=32,
        num_future_frames=4,
        seed=42,
        decode_video=False,
    )


def _obs(tag: int) -> dict:
    """tag: int written into state[0] (identity camera: proprio dim 0 == tag). Returns: one client observation."""
    state = np.zeros(48)
    state[0] = tag
    state[[3, 7, 27, 31]] = 1.0  # identity wrist rotations
    img = np.zeros((48, 64, 3), np.uint8)
    return {EGO: img, CHEST: img, "observation.state": state, "camera_pose.ego_view": np.eye(4).reshape(-1), "task": "p"}


def test_two_sessions_of_one_engine_keep_independent_histories():
    """Interleaved plans of two sessions never mix episodes: each window is that session's own history rule."""
    engine = _engine()
    arch = engine.architecture
    a, b = engine.new_session(), engine.new_session(seed=7)
    a.plan(_obs(100))
    b.plan(_obs(200))
    a.plan(_obs(101))
    a.plan(_obs(102))
    b.plan(_obs(201))
    a.plan(_obs(103))
    # Offsets [-40, -20, 0] at K = 20 = one observation per offset step, clamped to each episode's start.
    assert arch.seen == [
        [100, 100, 100],
        [200, 200, 200],
        [100, 100, 101],
        [100, 101, 102],
        [200, 200, 201],
        [101, 102, 103],
    ]
    assert arch.seeds == [42, 7, 42, 42, 7, 42]  # default deploy seed vs the session's own


def test_session_plan_returns_k_actions_and_hand_targets():
    """A plan returns the first K absolute actions plus their wrist / fingertip split."""
    out = _engine().new_session().plan(_obs(0))
    assert out["actions"].shape == (K, 48)
    np.testing.assert_allclose(out["actions"], np.broadcast_to(_ACTION_ROW, (K, 48)))
    assert out["wrist"].shape == (K, 2, 4, 4) and out["tips"].shape == (K, 2, 5, 3)
    np.testing.assert_allclose(out["wrist"], np.broadcast_to(np.eye(4), (K, 2, 4, 4)))
    assert "video" not in out


# ── build_deploy_cfg / check_deploy ─────────────────────────────────────────

_CKPT = "checkpoint_step_1.safetensors"


def _ckpt_dir(tmp_path, **dataloader):
    """Write a small sparse-history checkpoint dir (config.yaml + empty checkpoint file).

    Args:
        tmp_path: pytest tmp dir.
        dataloader: overrides of the checkpoint's dataloader block (value None deletes the key).
    Returns:
        str, the run dir.
    """
    dl = {
        "type": "yam_sharpa",
        "multiview": True,
        "multiview_layout": "vstack",
        "normalize_mode": "min-max",
        "action_mode": "fingertips_abs",
        "history_offsets": [-40, -20, 0],
        "history_layout": "frames",
        "fps": 20,
        "num_frames": 33,
        "height": 384,
        "width": 224,
    }
    dl.update(dataloader)
    dl = {k: v for k, v in dl.items() if v is not None}
    OmegaConf.save(OmegaConf.create({"dataloader": dl}), tmp_path / "config.yaml")
    (tmp_path / _CKPT).write_bytes(b"")
    return str(tmp_path)


def _overrides(run_dir, *extra, k=20):
    """run_dir: str; extra: more dotlist entries; k: steps_per_plan. Returns: the deploy dotlist."""
    return [f"deploy.ckpt.dir={run_dir}", f"deploy.ckpt.name={_CKPT}", f"deploy.policy.steps_per_plan={k}", *extra]


def test_build_deploy_cfg_accepts_a_sparse_history_checkpoint(tmp_path):
    """The happy path: checkpoint cfg + one deploy block, frozen."""
    cfg = build_deploy_cfg(_overrides(_ckpt_dir(tmp_path)))
    assert cfg.dataloader.fps == 20 and cfg.deploy.policy.steps_per_plan == 20
    assert cfg.deploy.sampler.denoise_mode == "sync"
    with pytest.raises(ReadonlyConfigError):
        cfg.deploy.policy.seed = 0


def test_build_deploy_cfg_rejects_non_deploy_overrides(tmp_path):
    """The checkpoint's model / data config is fixed: only deploy.* overrides are allowed."""
    with pytest.raises(ValueError, match="must start with 'deploy.'"):
        build_deploy_cfg(_overrides(_ckpt_dir(tmp_path), "dataloader.fps=10"))


def test_build_deploy_cfg_requires_ckpt_and_file(tmp_path):
    """No ckpt dir / name, or a name with no file behind it, is rejected."""
    with pytest.raises(ValueError, match="deploy.ckpt.dir"):
        build_deploy_cfg(["deploy.policy.steps_per_plan=20"])
    run = _ckpt_dir(tmp_path)
    with pytest.raises(ValueError, match="no checkpoint file"):
        build_deploy_cfg(
            [
                f"deploy.ckpt.dir={run}",
                "deploy.ckpt.name=checkpoint_step_9.safetensors",
                "deploy.policy.steps_per_plan=20",
            ]
        )


@pytest.mark.parametrize("key", ["fps", "history_layout", "history_offsets"])
def test_build_deploy_cfg_requires_sparse_history_keys(tmp_path, key):
    """Checkpoints trained without fps / history_layout / history_offsets are rejected, never defaulted."""
    with pytest.raises(ValueError, match=f"dataloader.{key} is missing"):
        build_deploy_cfg(_overrides(_ckpt_dir(tmp_path, **{key: None})))


@pytest.mark.parametrize("key, value", [("num_frames", 49), ("height", 480), ("width", 320)])
def test_build_deploy_cfg_rejects_data_shape_mismatch(tmp_path, key, value):
    """deploy.data.* states the expected shape; it must equal the checkpoint's dataloader values."""
    with pytest.raises(ValueError, match=f"deploy.data.{key}"):
        build_deploy_cfg(_overrides(_ckpt_dir(tmp_path), f"deploy.data.{key}={value}"))


def test_build_deploy_cfg_rejects_k_that_misses_fps_or_an_offset(tmp_path):
    """K must divide fps (K=8 vs 20) and every history offset (K=10 vs -15)."""
    with pytest.raises(ValueError, match="must divide"):
        build_deploy_cfg(_overrides(_ckpt_dir(tmp_path), k=8))
    with pytest.raises(ValueError, match="must divide"):
        build_deploy_cfg(_overrides(_ckpt_dir(tmp_path, history_offsets=[-30, -15, 0]), k=10))


def test_build_deploy_cfg_rejects_k_above_chunk(tmp_path):
    """K cannot exceed the action chunk (num_frames - 1 = 32)."""
    with pytest.raises(ValueError, match=r"in \[1, 32\]"):
        build_deploy_cfg(_overrides(_ckpt_dir(tmp_path), k=40))


@pytest.mark.parametrize(
    "override, match",
    [
        ("deploy.sampler.denoise_mode=unsupported", "Unsupported denoise mode"),
        ("deploy.sampler.variance_shift_alpha=9.0", "denoise_mode='async'"),
    ],
)
def test_build_deploy_cfg_validates_sampler(tmp_path, override, match):
    """The sampler block is validated at startup (async-only knobs need denoise_mode=async)."""
    with pytest.raises(ValueError, match=match):
        build_deploy_cfg(_overrides(_ckpt_dir(tmp_path), override))


def test_check_deploy_rejects_non_yam_sharpa(tmp_path):
    """check_deploy on an assembled cfg: only yam_sharpa vstack checkpoints are served."""
    cfg = OmegaConf.merge(
        OmegaConf.load(f"{_ckpt_dir(tmp_path, type='robotwin')}/config.yaml"),
        {"deploy": OmegaConf.load(DEPLOY_YAML)},
        {"deploy": {"ckpt": {"dir": str(tmp_path), "name": _CKPT}, "policy": {"steps_per_plan": 20}}},
    )
    with pytest.raises(ValueError, match="yam_sharpa checkpoints only"):
        check_deploy(cfg)


# ── split_hand_targets ──────────────────────────────────────────────────────


def _rot6d(Rm):
    """Rm: (3, 3) rotation. Returns: (6,) rot6d = its first two columns."""
    return np.concatenate([Rm[:, 0], Rm[:, 1]])


def test_split_hand_targets_fingertips():
    """fingertips_*: wrist (K, 2, 4, 4) camera_from_wrist and tips (K, 2, 5, 3) in the order they were packed."""
    rng = np.random.default_rng(0)
    k = 3
    rots = Rotation.random(2, random_state=0).as_matrix()
    a = np.zeros((k, 48))
    pos = rng.normal(size=(k, 2, 3))
    tips = rng.normal(size=(k, 2, 5, 3))
    for side, o in enumerate((0, 24)):
        a[:, o : o + 3] = pos[:, side]
        a[:, o + 3 : o + 9] = _rot6d(rots[side])
        a[:, o + 9 : o + 24] = tips[:, side].reshape(k, 15)
    out = R.split_hand_targets("fingertips_rel", a)
    assert set(out) == {"wrist", "tips"}
    assert out["wrist"].shape == (k, 2, 4, 4) and out["tips"].shape == (k, 2, 5, 3)
    np.testing.assert_allclose(out["wrist"][:, :, :3, 3], pos)
    np.testing.assert_allclose(out["wrist"][:, :, :3, :3], np.broadcast_to(rots, (k, 2, 3, 3)), atol=1e-12)
    np.testing.assert_allclose(out["wrist"][:, :, 3], np.broadcast_to([0, 0, 0, 1], (k, 2, 4)))
    np.testing.assert_allclose(out["tips"], tips)


def test_split_hand_targets_unify():
    """unify_*: rot6d of identity -> eye(3), finger joints (K, 2, 22); a wrong-width chunk is rejected."""
    k = 2
    a = np.zeros((k, 62))
    fingers = np.arange(k * 2 * 22, dtype=np.float64).reshape(k, 2, 22)
    for side, o in enumerate((0, 31)):
        a[:, o : o + 3] = [side, 0.5, -0.5]
        a[:, o + 3 : o + 9] = [1, 0, 0, 0, 1, 0]  # rot6d of the identity
        a[:, o + 9 : o + 31] = fingers[:, side]
    out = R.split_hand_targets("unify_abs", a)
    assert set(out) == {"wrist", "fingers"}
    np.testing.assert_allclose(out["wrist"][:, :, :3, :3], np.broadcast_to(np.eye(3), (k, 2, 3, 3)))
    np.testing.assert_allclose(out["wrist"][:, 1, :3, 3], np.broadcast_to([1, 0.5, -0.5], (k, 3)))
    np.testing.assert_allclose(out["fingers"], fingers)
    with pytest.raises(ValueError, match=r"\(K, 62\)"):
        R.split_hand_targets("unify_abs", np.zeros((k, 48)))
