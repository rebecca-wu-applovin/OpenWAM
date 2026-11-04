"""Sparse-history deploy plumbing (CPU, no weights): history buffer, per-step recording in WAMPolicy, engine choice."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from openwam.dataloader.utils.sparse_history import history_rows
from openwam.deploy.history_buffer import ObsHistoryBuffer
from openwam.deploy.policy import WAMPolicy

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


class _Engine:
    """Records generate() conditions; returns a 32-step chunk whose rows are their index."""

    def __init__(self):
        self.calls = []
        self.architecture = None

    def generate(self, conditions):
        self.calls.append(conditions)
        return {"actions": np.repeat(np.arange(32, dtype=np.float32)[:, None], 2, axis=1)}


def test_policy_plans_every_observation_and_returns_k_actions():
    """K = 10: every request is one plan with its own history, and returns the chunk's first 10 actions."""
    eng = _Engine()
    buf = ObsHistoryBuffer([-20, -10, 0], action_fps=20, steps_per_obs=10)
    policy = WAMPolicy(eng, OmegaConf.create({}), history=buf)
    for j in range(4):
        out = policy.predict_action({"obs": j, "prompt": "p"})
        assert out.shape == (10, 2) and out[:, 0].tolist() == list(range(10))
    assert [[o["obs"] for o in c["history"]] for c in eng.calls] == [[0, 0, 0], [0, 0, 1], [0, 1, 2], [1, 2, 3]]
    policy.reset()
    policy.predict_action({"obs": "a", "prompt": "p"})
    assert [o["obs"] for o in eng.calls[-1]["history"]] == ["a", "a", "a"]
    with pytest.raises(ValueError, match="sync"):
        WAMPolicy(eng, OmegaConf.create({}), execution_config={"mode": "async"}, history=buf)


def test_policy_without_history_sends_no_history():
    eng = _Engine()
    action = WAMPolicy(eng, OmegaConf.create({})).predict_action({"prompt": "p"})
    assert "history" not in eng.calls[0] and action.shape == (2,)


def test_server_serves_every_checkpoint_with_the_sparse_engine(monkeypatch):
    """build_server_from_config: every ckpt (first-frame = history_offsets [0]) -> sparse engine + buffered policy."""
    import openwam.deploy.model_loader as model_loader
    import openwam.deploy.server as server
    import openwam.deploy.sparse_engine as sparse_engine

    made = {}

    class _Sparse:
        def __init__(self, cfg, architecture, *, history_offsets):
            made["sparse"] = history_offsets

    monkeypatch.setattr(sparse_engine, "SparseHistoryInferenceEngine", _Sparse)
    for offsets in (OFFSETS, [0]):
        made.clear()
        ckpt_cfg = OmegaConf.create({"dataloader": {"history_offsets": offsets, "num_frames": 33}})
        monkeypatch.setattr(model_loader, "load_from_checkpoint_dir", lambda *a, _c=ckpt_cfg, **k: (_c, object()))
        srv = server.build_server_from_config(OmegaConf.create({"inference": {"denoise_steps": 10}}), "/ckpt")
        assert made == {"sparse": offsets}
        assert srv.history_offsets == offsets


def test_server_sparse_requires_k_and_fps():
    """A sparse server builds its buffer from inference.inference_horizon (K) and inference.action_fps."""
    from openwam.deploy.server import PolicyServer

    base = {"denoise_steps": 10, "inference_mode": "sync"}
    with pytest.raises(ValueError, match="inference_horizon"):
        PolicyServer(_Engine(), OmegaConf.create({"inference": base}), history_offsets=OFFSETS)._init_policy()
    with pytest.raises(ValueError, match="divide"):
        cfg = OmegaConf.create({"inference": {**base, "inference_horizon": 8, "action_fps": 20}})
        PolicyServer(_Engine(), cfg, history_offsets=OFFSETS)._init_policy()
    cfg = OmegaConf.create({"inference": {**base, "inference_horizon": 10, "action_fps": 20}})
    srv = PolicyServer(_Engine(), cfg, history_offsets=OFFSETS)
    srv._init_policy()
    assert srv._policy._history.steps_per_obs == 10
