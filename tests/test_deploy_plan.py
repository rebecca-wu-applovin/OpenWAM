"""The ``plan`` request and the PONG ``serving`` block (CPU, no weights): one observation -> the K actions to
execute before the next, for clients that observe every K steps themselves (sim-policy-eval)."""

import numpy as np
from omegaconf import OmegaConf

from openwam.deploy.policy import WAMPolicy


class _Engine:
    """Counts generate() calls; returns a 32-step chunk whose rows are their index."""

    def __init__(self):
        self.calls = 0
        self.architecture = None

    def generate(self, conditions):
        self.calls += 1
        return {"actions": np.arange(32, dtype=np.float32)[:, None].repeat(4, axis=1)}


def _cfg(**inference):
    return OmegaConf.create({"inference": {"denoise_steps": 10, "inference_mode": "sync", "num_frames": 33,
                                           "action_fps": 20, **inference}})


def test_plan_returns_the_first_k_actions_and_plans_every_call():
    engine = _Engine()
    policy = WAMPolicy(engine, _cfg(), execution_config={"inference_horizon": 20})
    for _ in range(3):
        actions = policy.plan_actions({"prompt": "p"})
        assert actions.shape == (20, 4) and actions[:, 0].tolist() == list(range(20))
    assert engine.calls == 3   # no receding-horizon buffer: every plan generates
    assert policy.plan_steps == 20


def test_plan_without_horizon_returns_the_whole_chunk():
    policy = WAMPolicy(_Engine(), _cfg(), execution_config={"inference_horizon": None})
    assert policy.plan_actions({"prompt": "p"}).shape == (32, 4)


def test_pong_serving_block_describes_the_plan():
    from openwam.deploy.server import PolicyServer

    srv = PolicyServer(_Engine(), _cfg(inference_horizon=20), ckpt="/runs/x/checkpoint_step_5.safetensors")
    assert "serving" not in srv._pong_payload()   # before the policy exists there is nothing to describe
    srv._policy = WAMPolicy(srv.engine, srv.cfg, execution_config={"inference_horizon": 20})
    assert srv._pong_payload()["serving"] == {"chunk": 32, "inference_horizon": 20, "action_fps": 20,
                                              "ckpt": "/runs/x/checkpoint_step_5.safetensors",
                                              "obs_state": "proprio", "history_offsets": None}
    sparse = PolicyServer(_Engine(), _cfg(inference_horizon=20), history_offsets=[-40, -20, 0])
    sparse._init_policy()
    assert sparse._pong_payload()["serving"]["obs_state"] == "observation.state"


def test_simeval_launcher_splits_the_checkpoint_path(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "simeval_serve", Path(__file__).resolve().parents[1] / "benchmarks" / "simeval" / "serve.py")
    serve = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serve)
    ckpt = tmp_path / "checkpoint_step_7.safetensors"
    ckpt.write_bytes(b"x")
    assert serve.server_argv(["--ckpt", str(ckpt), "--inference-horizon", "20", "--port", "9"]) == [
        "--ckpt-dir", str(tmp_path), "--ckpt-name", ckpt.name, "--inference-horizon", "20", "--port", "9"]
