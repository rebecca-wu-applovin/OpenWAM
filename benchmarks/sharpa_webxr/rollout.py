"""Closed-loop rollout of a GWP-0.5 Sharpa policy server in the WebXR-Teleop sim (runs in .venv_webxr).

Loop follows wam.cpp's GWP-0.5 closed-loop evaluator (eval/sim/run_robotwin_client.py + ActionChunkExecutor):
synchronous observe -> predict chunk -> execute the first ``execute_steps`` -> re-observe; queue cleared on every
new chunk, no ensembling. Default executes the whole 32-step chunk (wam.cpp default = full 48-step chunk).

Per step: decoded wrist poses (ego-camera frame, giga codec anchored at the measured state) -> per-arm IK
(scripts/sharpa_eef/eef_kinematics.ik, seeded from the previous IK solution) -> 56-D absolute joint targets,
held 50 ms in the sim. Saves the full qpos trajectory so videos and stage signals are computed offline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts/sharpa_eef"))
from env import SharpaWebXREnv, base_from_camera  # noqa: E402

import action_codecs as A  # noqa: E402
import canonical as C  # noqa: E402
import eef_kinematics as K  # noqa: E402

TASKS = json.loads((HERE / "tasks.json").read_text())
SCENES_ROOT = REPO / "data/webxr_scenes"
MAX_STEPS_CAP = 1200  # 60 s; only the coin-toss scene's 1.5 x p90 (5718) exceeds it
CODEC = A.CODECS["giga"]
BFC_EGO = base_from_camera("ego_view")
CAM_FROM_ARM = C.camera_from_arm_base(BFC_EGO)  # {side: camera_from_<side>_yam_base}
ARM_FROM_CAM = {s: np.linalg.inv(T) for s, T in CAM_FROM_ARM.items()}


def seed_for(scene_id: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{scene_id}/{seed}".encode()).digest()[:4], "little")


class PolicyClient:
    def __init__(self, port: int, host: str = "localhost", timeout_ms: int = 600000):
        sys.path.insert(0, str(REPO / "third_party/giga-world-policy/third_party/giga-models/giga_models/sockets"))
        from client import RobotInferenceClient  # upstream ZeroMQ client (file import: no giga_models package deps)

        self.c = RobotInferenceClient(host=host, port=port, timeout_ms=timeout_ms)
        assert self.c.ping(), f"policy server on :{port} not answering"

    def __call__(self, images, state, task, seed):
        return self.c.inference({"images": images, "state": state, "task": task, "seed": seed})


def to_canonical(joints: np.ndarray, names: list[str]) -> C.Canonical:
    return C.from_joints(joints[None], names, frame="ego_camera", base_from_camera=BFC_EGO.ravel())


def chunk_to_joint_targets(cmd: C.Canonical, names: list[str], arm_seed: dict) -> tuple[np.ndarray, dict]:
    """Canonical [T] (ego frame) -> [T,56] joint targets in ``names`` order; IK seeded row by row."""
    cols = C.joint_columns(names)
    out = np.zeros((len(cmd.pos), len(names)))
    info = {"pos_err": np.zeros((len(cmd.pos), 2)), "rot_err": np.zeros((len(cmd.pos), 2)),
            "converged": np.zeros((len(cmd.pos), 2), bool)}
    for k, s in enumerate(C.SIDES):
        arm, hcols = cols[s]
        q = arm_seed[s]
        for t in range(len(cmd.pos)):
            Tc = np.eye(4)
            Tc[:3, :3], Tc[:3, 3] = cmd.rot[t, k], cmd.pos[t, k]
            qs, pe, re, ok = K.ik(s, (ARM_FROM_CAM[s] @ Tc)[None], q[None])
            q = qs[0]
            out[t, arm] = q
            out[t, hcols] = cmd.hand[t, k]
            info["pos_err"][t, k], info["rot_err"][t, k], info["converged"][t, k] = pe[0], re[0], ok[0]
        arm_seed[s] = q
    return out, info


def run_episode(env: SharpaWebXREnv, policy, scene_id: str, seed: int, out_dir: Path, execute_steps: int = 32,
                max_steps: int | None = None) -> dict:
    task = TASKS[scene_id]["task"]
    max_steps = max_steps or min(TASKS[scene_id]["max_steps"], MAX_STEPS_CAP)
    t0 = time.time()
    obs = env.reset(seed_for(scene_id, seed))
    names = env.joint_names
    cols = C.joint_columns(names)
    arm_seed = {s: obs["joints"][cols[s][0]].copy() for s in C.SIDES}
    qpos = [env.sim.data.qpos.copy()]
    targets, ik_pe, ik_re, ik_ok, requests = [], [], [], [], []
    step, chunk_i = 0, 0
    while step < max_steps:
        state_c = to_canonical(obs["joints"], names)
        state62 = CODEC.state_encode(state_c)[0]
        tr = time.perf_counter()
        resp = policy(obs["images"], state62, task, seed_for(scene_id, seed) + chunk_i)
        rpc_ms = (time.perf_counter() - tr) * 1e3
        cmd = CODEC.decode(np.asarray(resp["action"])[:execute_steps], state=state_c)
        q_tgt, info = chunk_to_joint_targets(cmd, names, arm_seed)
        requests.append({"step": step, "rpc_ms": rpc_ms, **resp.get("timing", {})})
        for t in range(len(q_tgt)):
            if step >= max_steps:
                break
            last = t == len(q_tgt) - 1 or step == max_steps - 1
            obs = env.step(q_tgt[t]) if last else (env.step_noobs(q_tgt[t]))
            qpos.append(env.sim.data.qpos.copy())
            targets.append(q_tgt[t])
            ik_pe.append(info["pos_err"][t]); ik_re.append(info["rot_err"][t]); ik_ok.append(info["converged"][t])
            step += 1
        chunk_i += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "trajectory.npz", qpos=np.asarray(qpos), targets=np.asarray(targets, np.float32),
                        ik_pos_err=np.asarray(ik_pe, np.float32), ik_rot_err=np.asarray(ik_re, np.float32),
                        ik_converged=np.asarray(ik_ok), joint_names=np.asarray(names))
    meta = {"scene_id": scene_id, "task": task, "seed": seed, "reset_seed": seed_for(scene_id, seed),
            "randomization": env.randomization, "steps": step, "max_steps": max_steps, "execute_steps": execute_steps,
            "chunks": chunk_i, "wall_s": time.time() - t0, "ckpt": resp.get("ckpt"), "requests": requests,
            "ik_converged_frac": float(np.mean(ik_ok)) if ik_ok else None,
            "ik_pos_err_p95_mm": float(np.quantile(np.asarray(ik_pe), 0.95) * 1e3) if ik_pe else None}
    (out_dir / "rollout.json").write_text(json.dumps(meta, indent=1))
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--seeds", default="0-9", help="e.g. 0-9 or 0,3,5")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--execute-steps", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=None)
    a = p.parse_args()
    seeds = (list(range(int(a.seeds.split("-")[0]), int(a.seeds.split("-")[1]) + 1)) if "-" in a.seeds
             else [int(s) for s in a.seeds.split(",")])
    policy = PolicyClient(a.port)
    env = SharpaWebXREnv(SCENES_ROOT / a.scene)
    for s in seeds:
        d = a.out / a.scene / f"seed_{s}"
        if (d / "rollout.json").exists():
            print(f"[skip] {d}", flush=True)
            continue
        m = run_episode(env, policy, a.scene, s, d, a.execute_steps, a.max_steps)
        print(f"[done] {a.scene} seed {s}: {m['steps']} steps, {m['chunks']} chunks, {m['wall_s']:.0f}s, "
              f"ik_conv {m['ik_converged_frac']:.3f}", flush=True)


if __name__ == "__main__":
    main()
