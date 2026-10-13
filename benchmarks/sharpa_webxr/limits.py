"""Episode length limit, shared by the runner and every offline pass (signals, judge frames, videos).

limit(scene) = min(ceil(1.5 x p90 demo length), MAX_SECONDS), in 20 Hz control steps. Policies are causal, so a
rollout recorded with a longer limit is cut to this one offline without re-running it (the first N steps are
exactly what a run capped at N would have produced).
"""
from __future__ import annotations

import json
from pathlib import Path

CONTROL_HZ = 20
MAX_SECONDS = 20.0
TASKS = json.loads((Path(__file__).resolve().parent / "tasks.json").read_text())


def limit(scene_id: str, max_seconds: float = MAX_SECONDS) -> int:
    return min(int(TASKS[scene_id]["max_steps"]), int(round(max_seconds * CONTROL_HZ)))
