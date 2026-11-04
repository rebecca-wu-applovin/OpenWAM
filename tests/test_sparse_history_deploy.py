"""Sparse-history deploy (CPU, no weights): history buffer, per-episode sessions of the InferenceEngine with a stub
architecture, the startup checks of the deploy cfg, and the hand-target split of the returned actions."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from openwam.dataloader.utils.sparse_history import history_rows
from openwam.deploy.history_buffer import ObsHistoryBuffer

OFFSETS = [-80, -60, -40, -20, 0]


def test_history_rows_clamps_to_episode_start():
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
SCALE = 1000.0  # stub stats: min-max over [-SCALE, SCALE], so normalized = raw / SCALE (no clip for small tags)


class _StubArchitecture:
    """No model: records each plan's proprio tags / seed / mask and returns a constant normalized action chunk."""

    device, dtype = torch.device("cpu"), torch.float32

    def __init__(self, action_row: np.ndarray):
        self.action_row = action_row  # (A,) raw action every chunk row decodes to
        self.seen, self.seeds, self.masks = [], [], []

    def rollout(self, history, prompts, proprio, schedule, *, seed, active_action_mask):
        """history: [[H PIL]]; proprio: (1, H, A) normalized. Returns: (latents, (1, CHUNK, A) normalized actions)."""
        assert len(history[0]) == proprio.shape[1]  # one canvas per history row
        # proprio dim 0 = the tag the test put in state[0] (identity camera), normalized by 1 / SCALE.
        self.seen.append(torch.round(proprio[0, :, 0] * SCALE).int().tolist())
        self.seeds.append(seed)
        self.masks.append(active_action_mask)
        actions = torch.from_numpy(np.broadcast_to(self.action_row / SCALE, (1, CHUNK, len(self.action_row))).copy())
        return torch.zeros(1), actions


def _stats(d: int) -> dict:
    """d: raw width. Returns: min-max stats over [-SCALE, SCALE] (float32, as load_stats_file returns them)."""
    one = np.ones(d, np.float32)
    return {"min": -SCALE * one, "max": SCALE * one, "mean": 0 * one, "std": one, "q01": -one, "q99": one}


def _engine(action_mode="fingertips_abs", history_offsets=(-40, -20, 0)):
    """An InferenceEngine over the stub: K=20, fps=20, 2-camera vstack 64x32. Returns: InferenceEngine."""
    from openwam.dataloader.utils.yam_sharpa_repr import is_unify, raw_dim
    from openwam.deploy.engine import EngineSpec, InferenceEngine

    d = raw_dim(action_mode)
    model_dim = 80 if is_unify(action_mode) else d
    # Identity wrist rotations (rot6d [1,0,0,0,1,0] at the start of each hand block), everything else 0.
    row = np.zeros(model_dim)
    row[[3, 7, d // 2 + 3, d // 2 + 7]] = 1.0
    spec = EngineSpec(ckpt="stub/checkpoint_step_1.safetensors", action_mode=action_mode, action_dim=d,
                      chunk=CHUNK, steps_per_plan=K,
                      action_fps=20, history_offsets=tuple(history_offsets), camera_layout=(EGO, CHEST))
    return InferenceEngine(_StubArchitecture(row), None, spec, state_stats=_stats(d), action_stats=_stats(d),
                           normalize_mode="min-max", height=64, width=32, seed=42, decode_video=False)


def _obs(tag: int, **extra) -> dict:
    """tag: int written into state[0] (identity camera: proprio dim 0 == tag). Returns: one client observation."""
    state = np.zeros(48)
    state[0] = tag
    state[[3, 7, 27, 31]] = 1.0  # identity wrist rotations
    img = np.zeros((48, 64, 3), np.uint8)
    obs = {EGO: img, CHEST: img, "observation.state": state, "camera_pose.ego_view": np.eye(4).reshape(-1),
           "task": "p"}
    return {**obs, **extra}


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
    """A plan returns the first K absolute actions (unnormalized) plus their wrist / fingertip split."""
    out = _engine().new_session().plan(_obs(0))
    row = np.zeros(48)
    row[[3, 7, 27, 31]] = 1.0
    assert out["actions"].shape == (K, 48)
    np.testing.assert_allclose(out["actions"], np.broadcast_to(row, (K, 48)), atol=1e-6)
    hand = out["hand_targets"]
    assert set(hand) == {"wrist", "tips"}
    assert hand["wrist"].shape == (K, 2, 4, 4) and hand["tips"].shape == (K, 2, 5, 3)
    np.testing.assert_allclose(hand["wrist"], np.broadcast_to(np.eye(4), (K, 2, 4, 4)), atol=1e-6)
    assert "video" not in out


def test_failed_plan_ends_the_session():
    """A plan that raises leaves the episode unusable: the next plan raises too, so a retry never runs on a history
    that already holds the failed step's observation; a new session plans normally."""
    engine = _engine()
    session = engine.new_session()
    session.plan(_obs(100))
    bad = _obs(101)
    del bad[CHEST]
    with pytest.raises(ValueError, match="chest_view"):
        session.plan(bad)
    with pytest.raises(RuntimeError, match="failed"):
        session.plan(_obs(102))
    assert len(engine.architecture.seen) == 1  # the model never ran after the failure
    engine.new_session().plan(_obs(200))
    assert engine.architecture.seen[-1] == [200, 200, 200]


def test_new_session_none_seed_uses_deploy_seed():
    """new_session(None) takes deploy.policy.seed; an explicit seed wins."""
    engine = _engine()
    assert engine.new_session().seed == 42 and engine.new_session(7).seed == 7


def test_observation_missing_camera_raises():
    """Every camera of the checkpoint's layout must be in the observation."""
    obs = _obs(0)
    del obs[CHEST]
    with pytest.raises(ValueError, match="chest_view"):
        _engine().new_session().plan(obs)


def test_unify_obs_without_joint_angles_raises():
    """unify_* proprio needs the 56 joint angles: a clear error, not a KeyError deep inside."""
    with pytest.raises(ValueError, match="joint_angles"):
        _engine("unify_abs", history_offsets=(0,)).new_session().plan(_obs(0))


def test_unify_engine_passes_explicit_active_mask():
    """The unify engine hands rollout the (80,) mask of the 62 scattered dims; fingertips hands None (all active)."""
    unify = _engine("unify_abs", history_offsets=(0,))
    out = unify.new_session().plan(_obs(0, joint_angles=np.zeros(56)))
    mask = unify.architecture.masks[-1]
    assert mask.dtype == torch.bool and mask.shape == (80,) and int(mask.sum()) == 62
    assert out["actions"].shape == (K, 62) and set(out["hand_targets"]) == {"wrist", "fingers"}
    assert out["hand_targets"]["fingers"].shape == (K, 2, 22)
    tips = _engine()
    tips.new_session().plan(_obs(0))
    assert tips.architecture.masks == [None]


def _rot6d(Rm):
    """Rm: (3, 3) rotation. Returns: (6,) its first two columns, the rot6d the action vector stores."""
    return np.concatenate([Rm[:3, 0], Rm[:3, 1]])


def test_split_hand_targets_fingertips():
    """fingertips_*: wrist (K, 2, 4, 4) camera_from_wrist and tips (K, 2, 5, 3) in the order they were packed."""
    from scipy.spatial.transform import Rotation

    from openwam.dataloader.utils import yam_sharpa_repr as R

    rng = np.random.default_rng(0)
    k = 3
    rots = Rotation.random(2, random_state=0).as_matrix()
    a = np.zeros((k, 48))
    pos = rng.normal(size=(k, 2, 3))
    tips = rng.normal(size=(k, 2, 5, 3))
    for side, o in enumerate((0, 24)):
        a[:, o : o + 3] = pos[:, side]
        # A non-unit-length rot6d: the split must orthonormalize it back to the rotation.
        a[:, o + 3 : o + 9] = 2.0 * _rot6d(rots[side])
        a[:, o + 9 : o + 24] = tips[:, side].reshape(k, 15)
    out = R.split_hand_targets("fingertips_rel", a)
    assert set(out) == {"wrist", "tips"}
    assert out["wrist"].shape == (k, 2, 4, 4) and out["tips"].shape == (k, 2, 5, 3)
    np.testing.assert_allclose(out["wrist"][:, :, :3, 3], pos)
    np.testing.assert_allclose(out["wrist"][:, :, :3, :3], np.broadcast_to(rots, (k, 2, 3, 3)), atol=1e-12)
    np.testing.assert_array_equal(out["wrist"][:, :, 3], np.broadcast_to([0, 0, 0, 1], (k, 2, 4)))
    np.testing.assert_allclose(np.linalg.det(out["wrist"][:, :, :3, :3]), 1.0)
    np.testing.assert_allclose(out["tips"], tips)


def test_split_hand_targets_unify():
    """unify_*: rot6d of identity -> eye(3), finger joints (K, 2, 22); a wrong-width chunk is rejected."""
    from openwam.dataloader.utils import yam_sharpa_repr as R

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


# ── build_deploy_cfg / check_deploy ─────────────────────────────────────────

_CKPT = "checkpoint_step_1.safetensors"


def _ckpt_dir(tmp_path, *overrides):
    """Write a run dir as training leaves it: the resolved training cfg + an (empty) checkpoint file.

    Args:
        tmp_path: pytest tmp dir.
        overrides: Hydra overrides of ``configs/train.yaml`` on top of a fingertips H = 3 sparse-history run.
    Returns:
        str, the run dir.
    """
    import os

    from hydra import compose, initialize_config_dir

    from openwam.config import resolve_config

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = [
        "dataloader.action_mode=fingertips_abs",
        "common.action_dim=48",
        "common.state_dim=48",
        "dataloader.history_offsets=[-40,-20,0]",
        "dataloader.height=384",
        "dataloader.width=224",
    ]
    with initialize_config_dir(config_dir=os.path.join(repo, "configs"), version_base=None):
        cfg = resolve_config(compose("train", overrides=[*base, *overrides]))
    OmegaConf.save(cfg, tmp_path / "config.yaml")
    (tmp_path / _CKPT).write_bytes(b"")
    return str(tmp_path)


def _overrides(run_dir, *extra, k=20):
    """run_dir: str; extra: more dotlist entries; k: steps_per_plan. Returns: the deploy dotlist."""
    return [f"deploy.ckpt.dir={run_dir}", f"deploy.ckpt.name={_CKPT}", f"deploy.policy.steps_per_plan={k}", *extra]


def test_build_deploy_cfg_accepts_a_sparse_history_checkpoint(tmp_path):
    """The happy path: the resolved checkpoint cfg + one deploy block, frozen; seed defaults to 42."""
    from omegaconf.errors import ReadonlyConfigError

    from openwam.deploy.config import build_deploy_cfg

    cfg = build_deploy_cfg(_overrides(_ckpt_dir(tmp_path)))
    assert cfg.dataloader.fps == 20 and cfg.deploy.policy.steps_per_plan == 20
    assert cfg.deploy.policy.seed == 42 and cfg.deploy.sampler.denoise_mode == "sync"
    assert cfg.common.layout.action_horizon == 32
    with pytest.raises(ReadonlyConfigError):
        cfg.deploy.policy.seed = 0


def test_build_deploy_cfg_rejects_non_deploy_overrides(tmp_path):
    """The checkpoint's model / data config is fixed: only deploy.* overrides are allowed."""
    from openwam.deploy.config import build_deploy_cfg

    with pytest.raises(ValueError, match="must start with 'deploy.'"):
        build_deploy_cfg(_overrides(_ckpt_dir(tmp_path), "dataloader.fps=10"))


def test_build_deploy_cfg_requires_ckpt_and_file(tmp_path):
    """No ckpt dir / name, or a name with no file behind it, is rejected."""
    from openwam.deploy.config import build_deploy_cfg

    with pytest.raises(ValueError, match="deploy.ckpt.dir"):
        build_deploy_cfg(["deploy.policy.steps_per_plan=20"])
    run = _ckpt_dir(tmp_path)
    with pytest.raises(ValueError, match="no checkpoint file"):
        build_deploy_cfg([f"deploy.ckpt.dir={run}", "deploy.ckpt.name=checkpoint_step_9.safetensors",
                          "deploy.policy.steps_per_plan=20"])


def test_build_deploy_cfg_requires_fps(tmp_path):
    """A checkpoint trained without dataloader.fps is rejected, never defaulted."""
    from openwam.deploy.config import build_deploy_cfg

    run = _ckpt_dir(tmp_path)
    cfg = OmegaConf.load(f"{run}/config.yaml")
    del cfg.dataloader["fps"]
    OmegaConf.save(cfg, f"{run}/config.yaml")
    with pytest.raises(ValueError, match="dataloader.fps is missing"):
        build_deploy_cfg(_overrides(run))


def test_build_deploy_cfg_takes_the_canvas_from_the_checkpoint(tmp_path):
    """The canvas size and frame count are the checkpoint's own: a 480x320 run deploys with no extra override."""
    from openwam.deploy.config import build_deploy_cfg

    cfg = build_deploy_cfg(_overrides(_ckpt_dir(tmp_path, "dataloader.height=480", "dataloader.width=320")))
    assert (cfg.dataloader.height, cfg.dataloader.width) == (480, 320)
    assert "data" not in cfg.deploy


@pytest.mark.parametrize("override", ["deploy.data.height=480", "deploy.policy.step_per_plan=20"])
def test_build_deploy_cfg_rejects_unknown_deploy_keys(tmp_path, override):
    """A deploy.* key the deploy yaml does not define (a removed key, a typo) is an error, never silently ignored."""
    from openwam.deploy.config import build_deploy_cfg

    with pytest.raises(ValueError, match=override.split("=")[0].removeprefix("deploy.")):
        build_deploy_cfg(_overrides(_ckpt_dir(tmp_path), override))


def test_build_deploy_cfg_rejects_k_that_misses_fps_or_an_offset(tmp_path):
    """K must divide fps (K=15 vs 20) and every history offset (K=20 vs -30): the history buffer's own rule."""
    from openwam.deploy.config import build_deploy_cfg

    with pytest.raises(ValueError, match="must divide action_fps"):
        build_deploy_cfg(_overrides(_ckpt_dir(tmp_path), k=15))
    with pytest.raises(ValueError, match="must be a multiple of K"):
        build_deploy_cfg(_overrides(_ckpt_dir(tmp_path, "dataloader.history_offsets=[-30,-20,0]"), k=20))


def test_build_deploy_cfg_rejects_k_above_chunk(tmp_path):
    """K cannot exceed the action chunk (common.layout.action_horizon = 32)."""
    from openwam.deploy.config import build_deploy_cfg

    with pytest.raises(ValueError, match=r"in \[1, 32\]"):
        build_deploy_cfg(_overrides(_ckpt_dir(tmp_path), k=40))


def test_build_deploy_cfg_rejects_bool_k(tmp_path):
    """steps_per_plan=true is a typo, not K = 1 (a Python bool is an int)."""
    from openwam.deploy.config import build_deploy_cfg

    with pytest.raises(ValueError, match="must be an int"):
        build_deploy_cfg(_overrides(_ckpt_dir(tmp_path), k="true"))


@pytest.mark.parametrize(
    "override, match",
    [
        ("deploy.sampler.denoise_mode=foo", "Unsupported denoise mode"),
        ("deploy.sampler.variance_shift_alpha=9.0", "denoise_mode='async'"),
    ],
)
def test_build_deploy_cfg_validates_sampler(tmp_path, override, match):
    """The sampler block is validated at startup (async-only knobs need denoise_mode=async)."""
    from openwam.deploy.config import build_deploy_cfg

    with pytest.raises(ValueError, match=match):
        build_deploy_cfg(_overrides(_ckpt_dir(tmp_path), override))


@pytest.mark.parametrize(
    "dataloader, match",
    [({"type": "robotwin"}, "yam_sharpa checkpoints only"), ({"multiview_layout": "l_shape"}, "vstack canvas only")],
)
def test_check_deploy_rejects_non_yam_sharpa_vstack(tmp_path, dataloader, match):
    """check_deploy on an assembled cfg: only yam_sharpa vstack checkpoints are served."""
    from openwam.deploy.config import DEPLOY_YAML, check_deploy

    run = _ckpt_dir(tmp_path)
    cfg = OmegaConf.merge(
        OmegaConf.load(f"{run}/config.yaml"),
        {"dataloader": dataloader},
        {"deploy": OmegaConf.load(DEPLOY_YAML)},
        {"deploy": {"ckpt": {"dir": run, "name": _CKPT}, "policy": {"steps_per_plan": 20}}},
    )
    with pytest.raises(ValueError, match=match):
        check_deploy(cfg)
