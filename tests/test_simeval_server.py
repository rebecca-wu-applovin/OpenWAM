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
        hand = {"wrist": wrist, "tips": np.full((2, 2, 5, 3), self.seed, float)}
        return {"actions": np.zeros((2, 48)), "hand_targets": hand, "seen": list(self.seen), "task": obs["task"]}


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


def test_failed_plan_ends_the_episode_but_keeps_the_connection(port):
    """A failing plan replies ("error", ...); further plans of that episode are refused until a new reset, and the
    connection itself stays usable."""
    with _connect(port) as c:
        _call(c, "reset", {"instruction": "t", "seed": None})
        status, msg = _call(c, "plan", _obs(1, bad=True))
        assert status == "error" and "bad observation" in msg
        status, msg = _call(c, "plan", _obs(3))
        assert status == "error" and "reset" in msg
        assert _call(c, "spec")[0] == "ok"
        _call(c, "reset", {"instruction": "t", "seed": None})
        status, reply = _call(c, "plan", _obs(4))
        assert status == "ok" and reply["wrist"][0, 0, 0, 3] == 1  # a fresh history


def test_protocol_violation_ends_the_episode(port):
    """An observation that breaks sim_policy_v1 is an error before the engine sees it, and ends the episode too."""
    with _connect(port) as c:
        _call(c, "reset", {"instruction": "t", "seed": None})
        obs = _obs(1)
        obs["joint_pos"] = np.zeros(55)
        status, msg = _call(c, "plan", obs)
        assert status == "error" and "joint_pos" in msg
        assert _call(c, "plan", _obs(2))[0] == "error"


def _break(obs, path, value):
    """obs: a sim_policy_v1 observation; path: keys down to the entry; value: its replacement, or _DROP to delete it.
    Returns: obs, changed in place."""
    *parents, last = path
    node = obs
    for key in parents:
        node = node[key]
    if value is _DROP:
        del node[last]
    else:
        node[last] = value
    return obs


_DROP = object()


@pytest.mark.parametrize(
    "path, value, match",
    [
        (("joint_pos",), _DROP, "missing"),                                    # a protocol key is missing
        (("extra",), 1, "unexpected"),                                         # a key the protocol does not define
        (("images", "chest_view"), _DROP, "images"),                           # a camera of the spec is missing
        (("images", "left_view"), np.zeros((4, 6, 3), np.uint8), "images"),    # a camera the spec does not read
        (("images", "ego_view"), np.zeros((4, 6, 3), np.float32), "uint8"),    # wrong dtype
        (("images", "ego_view"), np.zeros((4, 6), np.uint8), r"\(H, W, 3\)"),  # wrong shape
        (("cameras", "ego_view"), _DROP, "ego_view"),                          # no pose camera
        (("cameras", "ego_view", "base_from_camera"), np.eye(3), r"\(4, 4\)"),
        (("proprio", "observation.state"), _DROP, "observation.state"),
        (("proprio", "observation.state"), np.zeros(47), r"\(48,\)"),
        (("joint_pos",), np.zeros(55), r"\(56,\)"),
    ],
)
def test_engine_obs_rejects_protocol_violations(path, value, match):
    """Every key the protocol defines must be there with its shape / dtype, and nothing else: a clear error at the
    door instead of a shape error (or a silent broadcast) inside the model."""
    with pytest.raises(ValueError, match=match):
        engine_obs(_break(_obs(0), path, value), "t", (EGO, CHEST))


def test_engine_obs_allows_what_the_protocol_leaves_open():
    """Extra rendered cameras' intrinsics / poses and extra proprio vectors are part of sim_policy_v1."""
    obs = _obs(0)
    obs["cameras"]["left_view"] = {"intrinsic_matrix": np.eye(3), "base_from_camera": np.eye(4)}
    obs["proprio"]["observation.velocity"] = np.zeros(3)
    assert engine_obs(obs, "t", (EGO, CHEST))["observation.state"].shape == (48,)


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


def test_bare_tcp_probe_is_ignored(port):
    """Port probes (connect + close, no handshake) never take the server down: a real client still gets the spec."""
    for _ in range(3):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
    with _connect(port) as c:
        assert _call(c, "spec")[0] == "ok"


def _safetensors(path, n_bytes: int, *, truncate: int = 0):
    """Write a one-tensor safetensors file of n_bytes data, minus ``truncate`` trailing bytes. Returns: path."""
    import json
    import struct

    header = json.dumps({"w": {"dtype": "U8", "shape": [n_bytes], "data_offsets": [0, n_bytes]}}).encode()
    blob = struct.pack("<Q", len(header)) + header + bytes(n_bytes)
    path.write_bytes(blob[: len(blob) - truncate])
    return path


def test_list_ckpts_skips_unsettled_and_truncated(tmp_path, capsys):
    """Only complete checkpoints that stopped changing are listed, by step, with the K override for serve.py."""
    import os

    from benchmarks.simeval.list_ckpts import complete_checkpoints, main

    assert complete_checkpoints(tmp_path, 60) == []  # no config.yaml yet: nothing, even if files exist
    (tmp_path / "config.yaml").write_text("{}")
    old = time.time() - 600
    for name, truncate in (("checkpoint_step_200.safetensors", 0), ("checkpoint_step_100.safetensors", 0),
                           ("checkpoint_step_300.safetensors", 5)):
        os.utime(_safetensors(tmp_path / name, 64, truncate=truncate), (old, old))
    _safetensors(tmp_path / "checkpoint_step_400.safetensors", 64)  # complete but written just now
    assert [s for s, _ in complete_checkpoints(tmp_path, 60)] == [100, 200]
    main([str(tmp_path), "--steps-per-plan", "20", "--settle-s", "60"])
    lines = capsys.readouterr().out.splitlines()
    assert lines == [f"{s} {(tmp_path / f'checkpoint_step_{s}.safetensors').resolve()} deploy.policy.steps_per_plan=20"
                     for s in (100, 200)]


def test_probe_that_sends_bytes_is_ignored(port):
    """An HTTP health check / port scanner sending garbage fails the handshake; the server keeps serving."""
    with socket.create_connection(("127.0.0.1", port), timeout=1) as s:
        s.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
        time.sleep(0.2)
    with _connect(port) as c:
        assert _call(c, "spec")[0] == "ok"


def test_simultaneous_clients_all_connect(port):
    """Eval workers start together: 32 connections opened at once must all be accepted (no SYN dropped by a full
    listen backlog, which a client sees as a connect timeout)."""
    results = []

    def connect():
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            results.append(True)
        except OSError:
            results.append(False)

    threads = [threading.Thread(target=connect) for _ in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(results) and len(results) == 32


def test_idle_connection_does_not_block_other_clients(port):
    """A connection that never answers the handshake must not stall the accept loop for everyone else."""
    idle = socket.create_connection(("127.0.0.1", port), timeout=1)
    try:
        time.sleep(0.2)
        with _connect(port) as c:
            assert _call(c, "spec")[0] == "ok"
    finally:
        idle.close()


def test_malformed_request_gets_an_error_reply(port):
    """A request that is not a (verb, *payload) tuple replies ("error", ...) and the connection stays usable."""
    with _connect(port) as c:
        status, msg = _call(c)  # empty tuple
        assert status == "error"
        status, msg = _call(c, "fly")
        assert status == "error" and "unknown request" in msg
        assert _call(c, "spec")[0] == "ok"
