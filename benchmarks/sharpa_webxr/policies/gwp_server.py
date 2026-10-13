"""GWP-0.5 policy server on the policies/common.py contract (runs in .venv).

Model and sampler are policy_server.GWPSharpaPolicy unchanged (upstream action-only loop: FlowMatchEuler shift 5,
10 steps, no CFG, prefix KV cache over state + reference frame). Upstream-matched replanning: upstream executes 30 of
its 48-action chunk (run_inference_openloop.sh REPLAN_STEPS=30, 30 Hz) = 62.5% = 1.0 s; ours executes 20 of 32
(20 Hz) = 62.5% = 1.0 s.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import BasePolicy, cap_gpu_memory, serve  # noqa: E402

import action_codecs as A  # noqa: E402


class GWPServer(BasePolicy):
    model = "gwp05"
    codec = "giga"
    chunk = 32
    execute_steps = 20
    obs_offsets = (0,)
    upstream = "giga-world-policy scripts/inference_openloop.py + run_inference_openloop.sh (REPLAN_STEPS=30 of 48)"

    def __init__(self, ckpt: str, num_steps: int = 10, prep_dir: str | None = None):
        from policy_server import PREP_DIR, GWPSharpaPolicy

        self.p = GWPSharpaPolicy(ckpt, num_steps=num_steps, prep_dir=Path(prep_dir) if prep_dir else PREP_DIR)
        self.ckpt = ckpt
        self.sampler = {"name": "flowmatch_euler", "steps": num_steps, "shift": 5.0, "cfg": None,
                        "video": "not generated (action-only, prefix KV cache)"}
        self.C = A.CODECS["giga"]

    def act(self, obs, task, seed, prev_cmd):
        o = obs[0]
        state62 = self.C.state_encode(o["state"])[0]
        raw = self.p.predict(o["images"], state62, task, seed)
        return self.C.decode(raw, state=o["state"]), raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=11700)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--max-gpu-gib", type=float, default=0, help="cap allocator (0 = no cap)")
    ap.add_argument("--prep-dir", default=None, help="training prep dir (norm_stats.json, t5_task_embeddings.pt)")
    a = ap.parse_args()
    if a.max_gpu_gib:
        cap_gpu_memory(a.max_gpu_gib)
    serve(GWPServer(a.ckpt, a.num_steps, a.prep_dir), a.port)


if __name__ == "__main__":
    main()
