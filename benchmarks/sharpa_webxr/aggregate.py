"""Per-checkpoint stage rates for one sim-eval run dir (.sharpa_sim_eval/<run>/<ckpt_tag>/**/seed_k/signals.json).

usage: aggregate.py <run> [--tags t1 t2 ...] [--tsv out.tsv]
Prints reach / grasp / move / all-three (with 95% Wilson CI) / fallen / IK, overall and split by scenes that have
gpt_fleet_depth data vs simteleop-only scenes. Searches recursively, so sorted subfolders (e.g. gpt_fleet_depth/) count.
"""
import argparse
import glob
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = json.load(open(ROOT / "benchmarks/sharpa_webxr/tasks.json"))
STAGES = ("reach", "grasp", "move")


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load(run_dir: Path, tag: str):
    return [json.load(open(f)) for f in glob.glob(str(run_dir / tag / "**" / "seed_*" / "signals.json"), recursive=True)]


def summarize(sigs):
    n = len(sigs)
    if n == 0:
        return None
    full = sum(all(s["stage"].get(k) for k in STAGES) for s in sigs)
    lo, hi = wilson(full, n)
    out = {"n": n, **{k: sum(bool(s["stage"].get(k)) for s in sigs) / n for k in STAGES},
           "all3": full / n, "all3_lo": lo, "all3_hi": hi,
           "fallen": sum(bool(s.get("fallen_any")) for s in sigs) / n,
           "ik": sum(s.get("ik_converged_frac", 0) for s in sigs) / n}
    for name, want in (("gpt", True), ("teleop", False)):
        sub = [s for s in sigs if bool(TASKS[s["scene_id"]]["n_eps"].get("gpt_fleet_depth")) == want]
        out[f"all3_{name}"] = sum(all(s["stage"].get(k) for k in STAGES) for s in sub) / max(len(sub), 1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--tags", nargs="*")
    p.add_argument("--tsv", type=Path)
    a = p.parse_args()
    run_dir = ROOT / ".sharpa_sim_eval" / a.run
    tags = a.tags or sorted(d.name for d in run_dir.iterdir() if d.is_dir() and d.name != "g3")
    cols = ["n", "reach", "grasp", "move", "all3", "all3_lo", "all3_hi", "all3_gpt", "all3_teleop", "fallen", "ik"]
    rows = []
    print(f"{'tag':28s} " + " ".join(f"{c:>8s}" for c in cols))
    for t in tags:
        r = summarize(load(run_dir, t))
        if r is None:
            continue
        rows.append((t, r))
        print(f"{t:28s} " + " ".join(f"{r[c]:8d}" if c == "n" else f"{r[c]:8.3f}" for c in cols))
    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("tag\t" + "\t".join(cols) + "\n")
            for t, r in rows:
                f.write(t + "\t" + "\t".join(str(r[c]) for c in cols) + "\n")


if __name__ == "__main__":
    main()
