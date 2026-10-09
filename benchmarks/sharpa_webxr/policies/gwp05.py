"""GWP-0.5 Sharpa EEF adapter: talks to policy_server.py over the giga-models ZMQ protocol.

Execution follows wam.cpp's GWP-0.5 closed loop (eval/sim/run_robotwin_client.py + ActionChunkExecutor):
observe -> predict a 32-step chunk -> execute its first ``execute_steps`` (default: all 32) -> re-observe; the
queue is cleared on every new chunk, no ensembling.

Inputs: ego_view + chest_view at 480x640 (the server composes the training T-layout), state = giga codec
state_encode of the measured wrist poses (ego-camera frame) + hand joints. Outputs: CODECS['giga'].decode anchored
at that state -> per-step wrist pose targets + absolute hand joints, sent to the env as "eef" actions.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from env import ObsConfig
from policies import Policy

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/sharpa_eef"))
import action_codecs as A  # noqa: E402

CODEC = A.CODECS["giga"]
SIDES = ("left", "right")


class GWP05Policy(Policy):
    name = "gwp05"
    action_space = "eef"
    eef_frame = "ego_view"
    obs = ObsConfig(cameras={"ego_view": (480, 640), "chest_view": (480, 640)}, state=("eef",))

    def __init__(self, port: int, host: str = "localhost", execute_steps: int = 32, timeout_ms: int = 600000):
        sys.path.insert(0, str(REPO / "third_party/giga-world-policy/third_party/giga-models/giga_models/sockets"))
        from client import RobotInferenceClient  # upstream ZeroMQ client (file import: no giga_models package deps)

        self.client = RobotInferenceClient(host=host, port=int(port), timeout_ms=int(timeout_ms))
        assert self.client.ping(), f"policy server on {host}:{port} not answering"
        self.execute_steps = int(execute_steps)

    def reset(self, task: str, episode_seed: int) -> None:
        self.task, self.seed, self.chunk_i, self.step = task, episode_seed, 0, 0
        self.requests, self.ckpt = [], None

    def act(self, obs: dict) -> list:
        state_c = obs["eef"]["canonical"]
        state62 = CODEC.state_encode(state_c)[0]
        t0 = time.perf_counter()
        resp = self.client.inference({"images": obs["images"], "state": state62, "task": self.task,
                                      "seed": self.seed + self.chunk_i})
        self.requests.append({"step": self.step, "rpc_ms": (time.perf_counter() - t0) * 1e3, **resp.get("timing", {})})
        self.ckpt = resp.get("ckpt")
        cmd = CODEC.decode(np.asarray(resp["action"])[: self.execute_steps], state=state_c)
        acts = []
        for t in range(len(cmd.pos)):
            a = {}
            for k, s in enumerate(SIDES):
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = cmd.rot[t, k], cmd.pos[t, k]
                a[s] = {"pose": T, "hand": cmd.hand[t, k]}
            acts.append(a)
        self.chunk_i += 1
        self.step += len(acts)
        return acts

    def episode_stats(self) -> dict:
        return {"execute_steps": self.execute_steps, "chunks": self.chunk_i, "requests": self.requests, "ckpt": self.ckpt}
