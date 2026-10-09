"""Generic closed-loop rollout: a policy adapter (policies/) drives the env; runs in .venv_webxr.

Loop: obs = env.reset(seed); repeat { actions = policy.act(obs); step each action, observing only after the last }
until max_steps. Everything model-specific (inputs, chunking, replanning, decoding) lives in the adapter; the env
only takes actions and returns the observations the adapter declared. Saves the full qpos trajectory, so videos
and stage signals are computed offline (postprocess.py, render_video.py).

usage: rollout.py --scene <id> --seeds 0-9 --out <dir> [--policy gwp05] [--port 11500] [--execute-steps 32]
                  [--policy-kw key=value ...]
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
import policies  # noqa: E402
from env import SharpaWebXREnv  # noqa: E402

TASKS = json.loads((HERE / "tasks.json").read_text())
SCENES_ROOT = REPO / "data/webxr_scenes"
MAX_STEPS_CAP = 1200  # 60 s; only the coin-toss scene's 1.5 x p90 (5718) exceeds it


def seed_for(scene_id: str, seed: int) -> int:
    """Reset / policy-noise seed for (scene, seed): identical across checkpoints and models."""
    return int.from_bytes(hashlib.sha256(f"{scene_id}/{seed}".encode()).digest()[:4], "little")


def make_env(policy: policies.Policy, scene_id: str) -> SharpaWebXREnv:
    return SharpaWebXREnv(SCENES_ROOT / scene_id, obs=policy.obs, action_space=policy.action_space,
                          eef_frame=policy.eef_frame)


def run_episode(env: SharpaWebXREnv, policy: policies.Policy, scene_id: str, seed: int, out_dir: Path,
                max_steps: int | None = None) -> dict:
    task = TASKS[scene_id]["task"]
    max_steps = max_steps or min(TASKS[scene_id]["max_steps"], MAX_STEPS_CAP)
    t0 = time.time()
    obs = env.reset(seed_for(scene_id, seed))
    policy.reset(task, seed_for(scene_id, seed))
    qpos, targets, ik_pe, ik_re, ik_ok = [env.sim.data.qpos.copy()], [], [], [], []
    step, calls = 0, 0
    while step < max_steps:
        actions = policy.act(obs)
        calls += 1
        if not actions:
            raise RuntimeError(f"{policy.name}.act returned no actions")
        for t, a in enumerate(actions):
            if step >= max_steps:
                break
            last = t == len(actions) - 1 or step == max_steps - 1
            obs_t, info = env.step(a, observe=last)
            if last:
                obs = obs_t
            qpos.append(env.sim.data.qpos.copy())
            targets.append(env.last_target.copy())
            if "ik_converged" in info:
                ik_pe.append(info["ik_pos_err"]); ik_re.append(info["ik_rot_err"]); ik_ok.append(info["ik_converged"])
            step += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "trajectory.npz", qpos=np.asarray(qpos), targets=np.asarray(targets, np.float32),
                        ik_pos_err=np.asarray(ik_pe, np.float32), ik_rot_err=np.asarray(ik_re, np.float32),
                        ik_converged=np.asarray(ik_ok), joint_names=np.asarray(env.joint_names))
    meta = {"scene_id": scene_id, "task": task, "seed": seed, "reset_seed": seed_for(scene_id, seed),
            "randomization": env.randomization, "steps": step, "max_steps": max_steps, "policy": policy.name,
            "action_space": env.action_space, "policy_calls": calls, "wall_s": time.time() - t0,
            "ik_converged_frac": float(np.mean(ik_ok)) if ik_ok else None,
            "ik_pos_err_p95_mm": float(np.quantile(np.asarray(ik_pe), 0.95) * 1e3) if ik_pe else None,
            **policy.episode_stats()}
    (out_dir / "rollout.json").write_text(json.dumps(meta, indent=1))
    return meta


def parse_kw(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--seeds", default="0-9", help="e.g. 0-9 or 0,3,5")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--policy", default="gwp05", choices=sorted(policies.REGISTRY))
    p.add_argument("--port", type=int, help="policy server port (gwp05)")
    p.add_argument("--execute-steps", type=int, help="gwp05: actions executed per prediction (default 32 = full chunk)")
    p.add_argument("--policy-kw", nargs="*", metavar="KEY=VALUE", help="extra adapter constructor arguments")
    p.add_argument("--max-steps", type=int, default=None)
    a = p.parse_args()
    seeds = (list(range(int(a.seeds.split("-")[0]), int(a.seeds.split("-")[1]) + 1)) if "-" in a.seeds
             else [int(s) for s in a.seeds.split(",")])
    kw = parse_kw(a.policy_kw)
    if a.port is not None:
        kw["port"] = a.port
    if a.execute_steps is not None:
        kw["execute_steps"] = a.execute_steps
    policy = policies.load(a.policy, **kw)
    env = make_env(policy, a.scene)
    for s in seeds:
        d = a.out / a.scene / f"seed_{s}"
        if (d / "rollout.json").exists():
            print(f"[skip] {d}", flush=True)
            continue
        m = run_episode(env, policy, a.scene, s, d, a.max_steps)
        ik = f", ik_conv {m['ik_converged_frac']:.3f}" if m["ik_converged_frac"] is not None else ""
        print(f"[done] {a.scene} seed {s}: {m['steps']} steps, {m['policy_calls']} policy calls, {m['wall_s']:.0f}s{ik}",
              flush=True)


if __name__ == "__main__":
    main()
