"""Video only (no signals): ego | chest at 320x240 each, 10 fps, from trajectory.npz -> video.mp4.

Covers the first limits.limit(scene) steps (min(1.5 x p90, 20 s)), the same span the signals and judge score.
"""
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import SharpaWebXREnv  # noqa: E402
from limits import limit  # noqa: E402

SCENES = Path(__file__).resolve().parents[2] / "data/webxr_scenes"
for d in map(Path, sys.argv[1:]):
    if (d / "video.mp4").exists():
        continue
    meta = json.loads((d / "rollout.json").read_text())
    env = SharpaWebXREnv(SCENES / meta["scene_id"])
    q = np.load(d / "trajectory.npz")["qpos"][: limit(meta["scene_id"]) + 1]
    frames = []
    for t in range(0, len(q), 2):
        env.sim.data.qpos[:] = q[t]
        mujoco.mj_forward(env.sim.model, env.sim.data)
        frames.append(np.concatenate([env.render(v, (240, 320)) for v in ("ego_view", "chest_view")], axis=1))
    tmp = d / f"video.{os.getpid()}.tmp.mp4"  # atomic: concurrent renderers of the same dir never leave a torn file
    imageio.mimwrite(tmp, frames, fps=10, quality=7, macro_block_size=8)
    os.replace(tmp, d / "video.mp4")
    print("wrote", d / "video.mp4", len(frames), flush=True)
