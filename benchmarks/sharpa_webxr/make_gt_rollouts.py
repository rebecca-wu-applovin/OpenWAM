"""Ground-truth "rollouts" from simteleop teleop recordings (gs://.../webxr-teleop/live/episodes): recorded qpos at
20 Hz (every 3rd 60 Hz row) in the rollout format, so postprocess/judge can be validated on known outcomes."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np

LIVE = "gs://foundational-research/webxr-teleop/live"
SRC = Path("assets/sharpa_full_meta/simteleop_1004_fleet/meta/source.json")

p = argparse.ArgumentParser()
p.add_argument("--scenes", nargs="+", required=True)
p.add_argument("--per-scene", type=int, default=2)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--split", type=Path, default=Path("assets/sharpa_full_meta/split_v1.json"))
a = p.parse_args()
eps = json.loads(SRC.read_text())["episodes"]
val = set(json.loads(a.split.read_text())["sources"]["simteleop_1004_fleet"]["val_episodes"])
for sc in a.scenes:
    cand = [e for e in eps if e["scene_id"] == sc]
    cand = sorted(cand, key=lambda e: (e["episode_index"] not in val, e["episode_index"]))[: a.per_scene]
    for e in cand:
        d = a.out / sc / f"gt_ep{e['episode_index']}"
        if (d / "rollout.json").exists():
            continue
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["gsutil", "-q", "cp", f"{LIVE}/{e['upload_prefix']}/trajectory.hdf5", tmp], check=True)
            g = h5py.File(Path(tmp) / "trajectory.hdf5")["data/demo_0"]
            q = np.concatenate([g["initial_state/qpos"][:][None], g["states/qpos"][2::3]])
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "trajectory.npz", qpos=q)
        (d / "rollout.json").write_text(json.dumps({"scene_id": sc, "seed": f"gt_ep{e['episode_index']}", "gt": True,
                                                    "episode_index": e["episode_index"], "val": e["episode_index"] in val,
                                                    "outcome": e.get("outcome"), "task": e["task"], "steps": len(q) - 1}))
        print(d, len(q), e.get("outcome"), "val" if e["episode_index"] in val else "train")
