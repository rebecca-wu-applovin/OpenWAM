"""Score every rollout under a folder with a VLM judge and write per-task metadata into that folder.

usage: score.py <folder> [--model gpt-6-luna] [--workers 8] [--limit N] [--force]
  <folder>  any directory containing rollout dirs (…/<scene>/seed_k with rollout.json), e.g.
            .sharpa_sim_eval/gwp05_full_v2/step005000_ema; searched recursively.

Per rollout:
- The judge sees the 16 keyframes (ego | chest) with the task prompt and grades reach / grasp / move / complete /
  success, using the same rubric as judge.py.
- The verdict is cached in <rollout>/judge_<model>.json; reruns skip it unless --force.

Graded score follows GWP-0.5 report Sec. 4: 0.25 each for reach, grasp, move and complete (judge stages); success is
the judge's binary call. Sim signals (signals.json) are carried alongside, with judge/sim agreement per stage.

Output, written into <folder>:
- scores_<model>.json: overall and per-task (scene) SR, graded score, stage rates and per-seed rows
- scores_<model>.tsv: one line per task

Credentials: OPENAI_API_KEY, or the file ~/.config/openai/key (mode 600).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import RUBRIC, SCHEMA, keyframes  # noqa: E402

STAGES = ("reach", "grasp", "move", "complete")
SIM_STAGES = ("reach", "grasp", "move")


def client() -> openai.OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    kf = Path.home() / ".config/openai/key"
    if not key and kf.exists():
        key = kf.read_text().strip()
    if not key:
        sys.exit("no OpenAI key: set OPENAI_API_KEY or write it to ~/.config/openai/key (chmod 600)")
    return openai.OpenAI(api_key=key, max_retries=6, timeout=300)


def judge_one(c: openai.OpenAI, model: str, d: Path, force: bool) -> dict:
    out = d / f"judge_{model}.json"
    if out.exists() and not force:
        return json.loads(out.read_text())
    meta = json.loads((d / "rollout.json").read_text())
    content = [{"type": "input_text", "text": f"Task: {meta['task']}\nRollout length: {meta['steps'] / 20:.1f} s."}]
    for t, b64 in keyframes(d):
        content += [{"type": "input_text", "text": f"t = {t:.1f} s"},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"}]
    content.append({"type": "input_text", "text": "Grade this rollout."})
    t0 = time.time()
    try:
        resp = c.responses.create(
            model=model,
            instructions=RUBRIC,
            input=[{"role": "user", "content": content}],
            text={"format": {"type": "json_schema", "name": "rollout_grade", "schema": SCHEMA, "strict": True}},
        )
        verdict = json.loads(resp.output_text)
        verdict.update({"model": resp.model, "response_id": resp.id,
                        "usage": {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}})
    except (openai.APIError, json.JSONDecodeError) as e:  # recorded, excluded from aggregates, retried on rerun
        return {"error": f"{type(e).__name__}: {e}"[:500]}
    verdict["latency_s"] = round(time.time() - t0, 2)
    out.write_text(json.dumps(verdict, indent=1))
    return verdict


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(c - h, 4), round(c + h, 4)


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r["judge"]]
    n = len(ok)
    s = {"n": len(rows), "n_scored": n, "n_errors": len(rows) - n}
    if not n:
        return s
    succ = sum(r["judge"]["success"] for r in ok)
    s.update({"success_rate": round(succ / n, 4), "success_ci95": wilson(succ, n),
              "graded": round(sum(r["graded"] for r in ok) / n, 4),
              **{f"judge_{k}": round(sum(r["judge"][k] for r in ok) / n, 4) for k in STAGES},
              **{f"sim_{k}": round(sum(bool(r["sim"].get(k)) for r in ok) / n, 4) for k in SIM_STAGES},
              **{f"agree_{k}": round(sum(r["judge"][k] == bool(r["sim"].get(k)) for r in ok) / n, 4) for k in SIM_STAGES}})
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="score only the first N rollouts (pilot / cost check)")
    p.add_argument("--force", action="store_true", help="re-judge rollouts that already have a verdict")
    a = p.parse_args()

    dirs = sorted(f.parent for f in a.folder.rglob("rollout.json")
                  if (f.parent / "keyframes").is_dir() or (f.parent / "video.mp4").exists())
    if a.limit:
        dirs = dirs[: a.limit]
    if not dirs:
        sys.exit(f"no rollouts with keyframes or video under {a.folder}")
    c = client()
    print(f"judging {len(dirs)} rollouts under {a.folder} with {a.model}", flush=True)

    rows = []
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (d, v) in enumerate(zip(dirs, ex.map(lambda d: judge_one(c, a.model, d, a.force), dirs), strict=True)):
            meta = json.loads((d / "rollout.json").read_text())
            sig = json.loads((d / "signals.json").read_text()) if (d / "signals.json").exists() else {}
            row = {"scene_id": meta["scene_id"], "task": meta["task"], "seed": meta["seed"], "dir": str(d.relative_to(a.folder)),
                   "steps": meta["steps"], "judge": v, "sim": sig.get("stage", {}), "fallen": sig.get("fallen_any")}
            if "error" not in v:
                row["graded"] = 0.25 * sum(bool(v[k]) for k in STAGES)
            rows.append(row)
            tag = v.get("error") or f"success={v['success']} graded={row['graded']:.2f} ({v['confidence']})"
            print(f"[{i + 1}/{len(dirs)}] {row['dir']}: {tag}", flush=True)

    tasks = {}
    for r in rows:
        tasks.setdefault(r["scene_id"], []).append(r)
    per_task = {sc: {"task": rs[0]["task"], **summarize(rs), "seeds": sorted(rs, key=lambda r: r["seed"])}
                for sc, rs in sorted(tasks.items())}
    usage = [r["judge"].get("usage", {}) for r in rows if "error" not in r["judge"]]
    out = {"folder": str(a.folder), "model": a.model, "scored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "rubric": "GWP-0.5 graded: 0.25 x (reach, grasp, move, complete) from the judge; success = judge binary",
           "tokens": {"input": sum(u.get("input", 0) for u in usage), "output": sum(u.get("output", 0) for u in usage)},
           "overall": summarize(rows),
           "task_mean": {"success_rate": round(sum(t.get("success_rate", 0) for t in per_task.values()) / len(per_task), 4),
                         "graded": round(sum(t.get("graded", 0) for t in per_task.values()) / len(per_task), 4)},
           "per_task": per_task}
    jp = a.folder / f"scores_{a.model}.json"
    jp.write_text(json.dumps(out, indent=1))
    cols = ["n", "n_scored", "success_rate", "graded", "judge_reach", "judge_grasp", "judge_move", "judge_complete",
            "sim_reach", "sim_grasp", "sim_move"]
    with open(a.folder / f"scores_{a.model}.tsv", "w") as f:
        f.write("scene_id\ttask\t" + "\t".join(cols) + "\n")
        for sc, t in per_task.items():
            f.write(f"{sc}\t{t['task']}\t" + "\t".join(str(t.get(k, "")) for k in cols) + "\n")
    o = out["overall"]
    print(f"\nwrote {jp}\noverall: n={o['n']} scored={o['n_scored']} errors={o['n_errors']} "
          f"SR={o.get('success_rate')} CI={o.get('success_ci95')} graded={o.get('graded')} tokens={out['tokens']}")


if __name__ == "__main__":
    main()
