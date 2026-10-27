"""benchmarks/simeval: the sim_policy_v1 server and its observation / action translation, against a fake engine
(no model, no GPU). The client side is the raw wire format of sim-policy-eval's SimPolicyClient."""

import socket
import threading
import time
from dataclasses import dataclass
from multiprocessing.connection import Client

import numpy as np
import pytest

from benchmarks.simeval.server.serve import serve
from benchmarks.simeval.sim_policy import AUTHKEY, engine_obs, sim_camera

EGO, CHEST = "observation.images.ego_view", "observation.images.chest_view"


@dataclass(frozen=True)
class _Spec:
    ckpt: str = "fake/checkpoint_step_1.safetensors"
    chunk: int = 32
    steps_per_plan: int = 2
    action_fps: int = 20
    camera_layout: tuple = (EGO, CHEST)
    hand_targets: str = "tips"


class _Session:
    """Returns its own history (the observations' state[0]) as the wrist translation, and its seed / task."""

    def __init__(self, seed):
        self.seed, self.seen = seed, []

    def plan(self, obs):
        if obs["observation.state"][1] < 0:
            raise ValueError("bad observation")
        self.seen.append(obs["observation.state"][0])
        wrist = np.tile(np.eye(4), (2, 2, 1, 1))
        wrist[0, 0, 0, 3] = len(self.seen)
        return {"wrist": wrist, "tips": np.full((2, 2, 5, 3), self.seed, float), "actions": np.zeros((2, 48)),
                "seen": list(self.seen), "task": obs["task"]}


class _Engine:
    spec = _Spec()

    def new_session(self, seed):
        return _Session(42 if seed is None else seed)


def _obs(x, bad=False):
    """A sim_policy_v1 observation whose observation.state[0] is x."""
    state = np.zeros(48)
    state[0], state[1] = x, -1.0 if bad else 0.0
    img = np.zeros((4, 6, 3), np.uint8)
    cams = {c: {"intrinsic_matrix": np.eye(3), "base_from_camera": np.eye(4)} for c in ("ego_view", "chest_view")}
    return {"step": 0, "time": 0.0, "images": {"ego_view": img, "chest_view": img}, "cameras": cams,
            "proprio": {"observation.state": state}, "joint_names": [f"j{i}" for i in range(56)],
            "joint_pos": np.zeros(56)}


def _call(conn, *msg):
    """One request on conn. Returns: the (status, value) reply."""
    conn.send(msg)
    assert conn.poll(5)
    return conn.recv()


@pytest.fixture(scope="module")
def port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
    threading.Thread(target=serve, args=(_Engine(), "127.0.0.1", p), daemon=True).start()
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", p), timeout=0.1).close()  # a bare probe the server must survive
            return p
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("server did not start")


def _connect(port):
    return Client(("127.0.0.1", port), authkey=AUTHKEY)


def test_spec_is_the_sim_policy_v1_spec(port):
    """The spec carries the fields SimPolicyClient checks, with simulator camera names."""
    with _connect(port) as c:
        status, spec = _call(c, "spec")
    assert status == "ok"
    assert spec == {"protocol": "sim_policy_v1", "chunk": 32, "execute_steps": 2, "action_rate_hz": 20.0,
                    "cameras": ["ego_view", "chest_view"], "action": "hand_targets",
                    "ckpt": "fake/checkpoint_step_1.safetensors"}


def test_connections_keep_separate_episodes(port):
    """Two connections interleave plans; each sees only its own history, seed and instruction."""
    a, b = _connect(port), _connect(port)
    _call(a, "reset", {"instruction": "task a", "seed": None})
    _call(b, "reset", {"instruction": "task b", "seed": 7})
    _call(a, "plan", _obs(1))
    _call(b, "plan", _obs(10))
    _, ra = _call(a, "plan", _obs(2))
    _, rb = _call(b, "plan", _obs(20))
    assert set(ra) == {"wrist", "tips"}  # only the hand_targets action crosses the wire
    assert ra["wrist"][0, 0, 0, 3] == 2 and rb["wrist"][0, 0, 0, 3] == 2
    assert ra["tips"].max() == 42 and rb["tips"].max() == 7
    a.close()
    b.close()


def test_errors_reach_the_client_and_keep_the_connection(port):
    """A failing plan replies ("error", ...) and the next request on the same connection still works."""
    with _connect(port) as c:
        _call(c, "reset", {"instruction": "t", "seed": None})
        status, msg = _call(c, "plan", _obs(1, bad=True))
        assert status == "error" and "bad observation" in msg
        assert _call(c, "plan", _obs(3))[0] == "ok"


def test_plan_before_reset_is_an_error(port):
    """No episode on the connection yet."""
    with _connect(port) as c:
        status, msg = _call(c, "plan", _obs(1))
    assert status == "error" and "before 'reset'" in msg


def test_engine_obs_maps_simulator_names_to_lerobot_features():
    """Images by camera_layout key, ego camera pose row-major, joints, and the reset's instruction as the task."""
    obs = _obs(5)
    pose = np.arange(16, dtype=float).reshape(4, 4)
    obs["cameras"]["ego_view"]["base_from_camera"] = pose
    out = engine_obs(obs, "pick", (EGO, CHEST))
    assert out[EGO] is obs["images"]["ego_view"] and out[CHEST] is obs["images"]["chest_view"]
    np.testing.assert_array_equal(out["camera_pose.ego_view"], pose.reshape(16))
    assert out["observation.state"][0] == 5 and out["joint_angles"].shape == (56,) and out["task"] == "pick"


def test_sim_camera_rejects_non_image_keys():
    """camera_layout keys must be observation.images.<camera>."""
    assert sim_camera(EGO) == "ego_view"
    with pytest.raises(ValueError):
        sim_camera("ego_view")
