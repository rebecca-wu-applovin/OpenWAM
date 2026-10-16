"""Measure the hybrid grader (state prefix + vision judge) against the human labels of the teleop episodes whose task
ends in a judge stage. Needs judge_frames/ in each episode dir (postprocess.py --frames-only --max-steps 1000000 <dirs>)
and verifier.json (verify.py). The judge is called only where the state prefix passed, with the sim facts in its prompt,
exactly as score.py --grader hybrid does. Verdicts are cached per episode as judge_<model>_<rubric>.json.

usage: judge_labeled.py [--root data/webxr_episodes] [--model gpt-6-luna] [--rubric r3] [--workers 8] [--max-calls N]
Prints per scene: recall / specificity of the hybrid verdict, and of the judge alone on the prefix-passing episodes.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibrate import load_reviews  # noqa: E402
from score import client, judge_one  # noqa: E402
from verify import load_specs  # noqa: E402

POS, NEG = ("success", "recover"), ("failure",)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=HERE.parents[1] / "data/webxr_episodes")
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--rubric", default="r3", choices=["r2", "r3"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-calls", type=int, default=None)
    p.add_argument("--no-calls", action="store_true", help="only use cached verdicts")
    p.add_argument("--tsv", type=Path, default=None)
    a = p.parse_args()
    specs = load_specs()
    reviews = load_reviews(a.root)
    eps = []
    for scene, spec in specs.items():
        if not any(s["check"] == "judge" for s in spec["stages"]):
            continue
        for d in sorted((a.root / scene).glob("*/")):
            vf = d / "verifier.json"
            if not vf.exists():
                continue
            v = json.loads(vf.read_text())
            if "stages" not in v:
                continue
            key = f"{scene}/{d.name}"
            label = v.get("label")
            r = reviews.get(key)
            if r and r.get("decision") == "verifier":  # reviewer sided with the verifier: label corrected to the verdict reviewed
                label = "success" if r.get("verifier") == "PASS" else "failure"
            elif r and r.get("decision") in ("unsure", "both_wrong"):
                continue
            if label not in POS + NEG:
                continue
            eps.append((scene, d, v, label in POS))
    need = [(d, v) for _, d, v, _ in eps if v["sim_prefix_ok"] and (d / "judge_frames" / "COMPLETE").exists()]
    missing_frames = sum(1 for _, d, v, _ in eps if v["sim_prefix_ok"] and not (d / "judge_frames" / "COMPLETE").exists())
    out = lambda d: d / f"judge_{a.model}_{a.rubric}.json"  # noqa: E731
    calls = [(d, v) for d, v in need if not out(d).exists() or json.loads(out(d).read_text()).get("sim_context") != v["judge_context"]]
    if a.no_calls:
        calls = []
    if a.max_calls is not None:
        calls = calls[: a.max_calls]
    print(f"{len(eps)} labelled episodes in judge tasks; prefix passed {len(need) + missing_frames}; "
          f"{missing_frames} without judge frames; {len(calls)} judge calls to make", flush=True)
    if calls:
        c = client()
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(lambda dv: judge_one(c, a.model, dv[0], True, a.rubric, dv[1]["judge_context"]), calls))
    rows, per = [], {}
    for scene, d, v, pos in eps:
        j = json.loads(out(d).read_text()) if out(d).exists() else None
        if v["sim_prefix_ok"]:
            if j is None or "error" in j:
                verdict, src = None, "nojudge"
            else:
                verdict, src = bool(j["success"]), "judge"
        else:
            verdict, src = False, "gated"
        rows.append((scene, d.name, pos, verdict, src, (j or {}).get("rationale", "")))
        t = per.setdefault(scene, {"P": 0, "TP": 0, "N": 0, "TN": 0, "jP": 0, "jTP": 0, "jN": 0, "jTN": 0, "undecided": 0})
        if verdict is None:
            t["undecided"] += 1
            continue
        if pos:
            t["P"] += 1
            t["TP"] += verdict
        else:
            t["N"] += 1
            t["TN"] += (not verdict)
        if src == "judge":
            if pos:
                t["jP"] += 1
                t["jTP"] += verdict
            else:
                t["jN"] += 1
                t["jTN"] += (not verdict)
    f = lambda k, n: f"{k / n:.3f}" if n else "  -  "  # noqa: E731
    print(f"\n{'scene':10s} {'n_pos/neg':>10s}  hybrid recall  spec  | judge-only (prefix passed) recall  spec   | undecided")
    T = {k: 0 for k in next(iter(per.values()))}
    for scene, t in sorted(per.items()):
        for k in T:
            T[k] += t[k]
        print(f"{scene[:10]:10s} {t['P']:4d}/{t['N']:<5d}  {f(t['TP'], t['P']):>13s} {f(t['TN'], t['N']):>5s}  | {t['jP']:3d}/{t['jN']:<3d} "
              f"{f(t['jTP'], t['jP']):>22s} {f(t['jTN'], t['jN']):>5s}   | {t['undecided']}")
    print(f"{'ALL':10s} {T['P']:4d}/{T['N']:<5d}  {f(T['TP'], T['P']):>13s} {f(T['TN'], T['N']):>5s}  | {T['jP']:3d}/{T['jN']:<3d} "
          f"{f(T['jTP'], T['jP']):>22s} {f(T['jTN'], T['jN']):>5s}   | {T['undecided']}")
    if a.tsv:
        with open(a.tsv, "w") as fh:
            fh.write("scene\tepisode\thuman_positive\thybrid_verdict\tsource\trationale\n")
            for r in rows:
                fh.write("\t".join(str(x).replace("\t", " ").replace("\n", " ") for x in r) + "\n")
        print("wrote", a.tsv)


if __name__ == "__main__":
    main()
