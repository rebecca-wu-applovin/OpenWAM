"""Shared contract between rollout.py (sim side, .venv_webxr) and the per-model policy servers (.venv).

Every server owns its model's whole I/O: image layout, state encoding, normalization, sampler, and the action
codec decode. The sim side only sends the measured canonical state and camera images and receives absolute
canonical wrist poses + hand joints, so the rollout loop and IK are identical for every model.

Endpoints (giga_models ZeroMQ RobotInferenceServer, torch-serialized dicts):
  "meta"      -> {"model", "codec", "chunk", "execute_steps", "obs_offsets", "control_hz", "upstream", "sampler",
                  optional "cameras": {name: [H, W]} (default ego_view + chest_view at 480x640)}
                 execute_steps: actions run before replanning, matched to the model's upstream deploy.
                 obs_offsets: rows (<= 0) at which the policy needs past observations, e.g. [-5, 0] for LDA-1B.
  "inference" <- {"obs": {offset: {"images": {"ego_view": uint8 [480,640,3], "chest_view": ...},
                                   "state": canonical dict (1 row)}},   # one entry per meta obs_offsets
                  "task": str, "seed": int, "reset": bool,                # reset=True on the first chunk
                  "prev_cmd": canonical dict (1 row) | None,             # last commanded pose (LDA anchor)
                  "executed": int | None}                                 # actions run since the last call
              -> {"pos": [H,2,3], "rot": [H,2,3,3], "hand": [H,2,22]  (absolute, ego-camera frame),
                  "raw": float32 [H, width] model output before decode, "timing": {...}, "ckpt": str}
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for p in (REPO / "scripts/sharpa_eef", REPO / "third_party/giga-world-policy/third_party/giga-models"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import canonical as C  # noqa: E402


def canon_to_dict(c: C.Canonical) -> dict:
    return {"pos": np.asarray(c.pos, np.float64), "rot": np.asarray(c.rot, np.float64),
            "hand": np.asarray(c.hand, np.float64), "frame": c.frame}


def dict_to_canon(d: dict | None) -> C.Canonical | None:
    if d is None:
        return None
    pos = np.asarray(d["pos"], np.float64)
    return C.Canonical(pos, np.asarray(d["rot"], np.float64), np.asarray(d["hand"], np.float64),
                       np.ones(len(pos), bool), frame=d.get("frame", "ego_camera"))


class BasePolicy:
    """Subclass: set the meta fields and implement ``act``. ``reset`` clears per-episode state."""

    model = ""
    codec = ""
    chunk = 0
    execute_steps = 0
    obs_offsets = (0,)
    control_hz = 20.0
    upstream = ""   # which upstream deploy path the settings copy
    sampler = {}    # e.g. {"name": "unipc", "order": 2, "steps": 30, "shift": 5.0, "cfg": 1.0}
    ckpt = ""

    def meta(self) -> dict:
        return {k: getattr(self, k) for k in ("model", "codec", "chunk", "execute_steps", "obs_offsets",
                                               "control_hz", "upstream", "sampler", "ckpt")}

    def reset(self) -> None:
        pass

    def act(self, obs: dict, task: str, seed: int, prev_cmd: C.Canonical | None) -> tuple[C.Canonical, np.ndarray]:
        """obs: {offset: {"images": {...}, "state": Canonical}} -> (absolute Canonical [H], raw [H, width])."""
        raise NotImplementedError

    def inference(self, data: dict) -> dict:
        t0 = time.perf_counter()
        if data.get("reset"):
            self.reset()
        obs = {int(k): {"images": v["images"], "state": dict_to_canon(v["state"])} for k, v in data["obs"].items()}
        cmd, raw = self.act(obs, data["task"], int(data.get("seed", 0)), dict_to_canon(data.get("prev_cmd")))
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass
        out = canon_to_dict(cmd)
        out.update(raw=np.asarray(raw, np.float32), ckpt=self.ckpt,
                   timing={"server_ms": (time.perf_counter() - t0) * 1e3})
        return out


def cap_gpu_memory(gib: float) -> None:
    """Cap this process's CUDA caching allocator so an OOM hits the eval server, not a co-located training job."""
    import torch
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, gib * 2**30 / total), 0)


def serve(policy: BasePolicy, port: int) -> None:
    from giga_models.sockets.server import RobotInferenceServer

    srv = RobotInferenceServer(policy, port=port)
    srv.register_endpoint("meta", policy.meta, requires_input=False)
    print(f"[serve] {policy.model} on :{port} meta={policy.meta()}", flush=True)
    srv.run()
