"""Closed-loop rollout of any policy server (policies/<model>_server.py) in the sim; runs in .venv_webxr.

Sim side of the policies/common.py contract, the same for every model:
- the server's "meta" says how it is deployed: ``execute_steps`` (actions run before replanning, matched to the
  model's upstream deploy; --execute-steps overrides) and ``obs_offsets`` (rows <= 0 it needs observations at, e.g.
  LDA-1B's t-5; rows before the episode start repeat the first observation);
- the env (env.py, action_space "eef") is asked to observe only at those rows, and executes the returned absolute
  wrist poses + hand joints (IK inside the env, seeded from the previous solve);
- loop: observe -> request a chunk -> execute its first execute_steps -> re-observe; the queue is cleared on every
  new chunk, no ensembling. Episode length: limits.limit(scene) = min(1.5 x p90, 20 s).
Saves the full qpos trajectory; signals, judge frames and videos are computed offline (postprocess.py, render_video.py).

usage: rollout.py --port <server port> --scene <id> --seeds 0-9 --out <dir> [--execute-steps N] [--max-steps N]
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
from env import IMAGE_HW, ObsConfig, SharpaWebXREnv  # noqa: E402
from limits import limit  # noqa: E402
from policies.common import canon_to_dict, dict_to_canon  # noqa: E402

TASKS = json.loads((HERE / "tasks.json").read_text())
SCENES_ROOT = REPO / "data/webxr_scenes"
SIDES = ("left", "right")


def seed_for(scene_id: str, seed: int) -> int:
    """Reset / policy-noise seed for (scene, seed): identical across checkpoints and models."""
    return int.from_bytes(hashlib.sha256(f"{scene_id}/{seed}".encode()).digest()[:4], "little")


class PolicyClient:
    def __init__(self, port: int, host: str = "localhost", timeout_ms: int = 600000):
        sys.path.insert(0, str(REPO / "third_party/giga-world-policy/third_party/giga-models/giga_models/sockets"))
        from client import RobotInferenceClient  # upstream ZeroMQ client (file import: no giga_models package deps)

        self.c = RobotInferenceClient(host=host, port=port, timeout_ms=timeout_ms)
        assert self.c.ping(), f"policy server on {host}:{port} not answering"
        self.meta = self.c.call_endpoint("meta", requires_input=False)

    def __call__(self, req: dict) -> dict:
        return self.c.inference(req)


def make_env(meta: dict, scene_id: str) -> SharpaWebXREnv:
    cams = meta.get("cameras") or {"ego_view": IMAGE_HW, "chest_view": IMAGE_HW}
    return SharpaWebXREnv(SCENES_ROOT / scene_id, obs=ObsConfig(cameras=cams, state=("eef",)),
                          action_space="eef", eef_frame="ego_view")


def _pack(obs: dict) -> dict:
    return {"images": obs["images"], "state": canon_to_dict(obs["eef"]["canonical"])}


def run_episode(env: SharpaWebXREnv, policy: PolicyClient, scene_id: str, seed: int, out_dir: Path,
                execute_steps: int, max_steps: int | None = None) -> dict:
    task = TASKS[scene_id]["task"]
    max_steps = max_steps or limit(scene_id)
    offsets = sorted(int(o) for o in policy.meta["obs_offsets"])
    t0 = time.time()
    record = {0: _pack(env.reset(seed_for(scene_id, seed)))}  # row -> observation the server may still ask for
    qpos, targets, ik_pe, ik_re, ik_ok, requests = [env.sim.data.qpos.copy()], [], [], [], [], []
    prev_cmd, executed, resp = None, None, {}
    step, chunk_i = 0, 0
    while step < max_steps:
        tr = time.perf_counter()
        resp = policy({"obs": {o: record[max(step + o, 0)] for o in offsets}, "task": task,
                       "seed": seed_for(scene_id, seed) + chunk_i, "reset": chunk_i == 0, "prev_cmd": prev_cmd,
                       "executed": executed})
        requests.append({"step": step, "rpc_ms": (time.perf_counter() - tr) * 1e3, **resp.get("timing", {})})
        cmd = dict_to_canon({k: np.asarray(resp[k])[:execute_steps] for k in ("pos", "rot", "hand")})
        need = {max(step + len(cmd.pos) + o, 0) for o in offsets}  # rows the next request will read
        record = {k: v for k, v in record.items() if k in need}
        step0 = step
        for t in range(len(cmd.pos)):
            if step >= max_steps:
                break
            a = {}
            for k, s in enumerate(SIDES):
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = cmd.rot[t, k], cmd.pos[t, k]
                a[s] = {"pose": T, "hand": cmd.hand[t, k]}
            obs, info = env.step(a, observe=step + 1 in need)
            if obs is not None:
                record[step + 1] = _pack(obs)
            qpos.append(env.sim.data.qpos.copy())
            targets.append(env.last_target.copy())
            ik_pe.append(info["ik_pos_err"]); ik_re.append(info["ik_rot_err"]); ik_ok.append(info["ik_converged"])
            step += 1
        executed = step - step0
        prev_cmd = canon_to_dict(cmd[executed - 1:executed])  # last commanded pose (LDA anchor)
        chunk_i += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "trajectory.npz", qpos=np.asarray(qpos), targets=np.asarray(targets, np.float32),
                        ik_pos_err=np.asarray(ik_pe, np.float32), ik_rot_err=np.asarray(ik_re, np.float32),
                        ik_converged=np.asarray(ik_ok), joint_names=np.asarray(env.joint_names))
    meta = {"scene_id": scene_id, "task": task, "seed": seed, "reset_seed": seed_for(scene_id, seed),
            "randomization": env.randomization, "steps": step, "max_steps": max_steps, "execute_steps": execute_steps,
            "chunks": chunk_i, "wall_s": time.time() - t0, "ckpt": resp.get("ckpt"), "policy_meta": policy.meta,
            "requests": requests, "ik_converged_frac": float(np.mean(ik_ok)) if ik_ok else None,
            "ik_pos_err_p95_mm": float(np.quantile(np.asarray(ik_pe), 0.95) * 1e3) if ik_pe else None}
    (out_dir / "rollout.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--seeds", default="0-9", help="e.g. 0-9 or 0,3,5")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--execute-steps", type=int, default=None, help="default: the server's meta (upstream-matched)")
    p.add_argument("--max-steps", type=int, default=None, help="default: limits.limit(scene)")
    a = p.parse_args()
    seeds = (list(range(int(a.seeds.split("-")[0]), int(a.seeds.split("-")[1]) + 1)) if "-" in a.seeds
             else [int(s) for s in a.seeds.split(",")])
    policy = PolicyClient(a.port)
    execute = a.execute_steps or int(policy.meta["execute_steps"])
    print(f"[rollout] {policy.meta.get('model')} execute_steps={execute} obs_offsets={policy.meta['obs_offsets']}",
          flush=True)
    env = make_env(policy.meta, a.scene)
    for s in seeds:
        d = a.out / a.scene / f"seed_{s}"
        if (d / "rollout.json").exists():
            print(f"[skip] {d}", flush=True)
            continue
        m = run_episode(env, policy, a.scene, s, d, execute, a.max_steps)
        print(f"[done] {a.scene} seed {s}: {m['steps']} steps, {m['chunks']} chunks, {m['wall_s']:.0f}s, "
              f"ik_conv {m['ik_converged_frac']:.3f}", flush=True)


if __name__ == "__main__":
    main()
