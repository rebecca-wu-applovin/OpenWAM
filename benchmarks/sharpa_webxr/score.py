"""Score every rollout under a folder with a VLM judge and write per-task metadata into that folder.

usage: score.py <folder> [--model gpt-6-luna] [--rubric r2] [--workers 8] [--limit N] [--force]
                [--grader {judge,verifier,hybrid}] [--no-judge-calls] [--max-judge-calls N]
  <folder>  any directory containing rollout dirs (…/<scene>/seed_k with rollout.json), e.g.
            .sharpa_sim_eval/gwp05_full_v2/step005000_ema; searched recursively.

Per rollout:
- r2 (default): the judge sees judge_frames/ (24 timesteps x 2 cameras at 480x640, written by
  `postprocess.py --frames-only`) over the scored span (limits.limit: min(1.5 x p90, 20 s)) and grades
  reach / grasp / move / complete / success with RUBRIC_R2 (grasp and move need visible lift or carry).
- r3 (default): r2 plus a note that the simulator is rigid bodies only (no liquid, no deformables): the judge grades
  whether the motion was performed the way a person would, never whether a substance visibly moved.
- r1: the 16 combined 320x240 keyframes and judge.py's rubric (first version, kept for comparison).
- The verdict is cached in <rollout>/judge_<model>[_r2].json; reruns skip it unless --force.

Graded score follows GWP-0.5 report Sec. 4: 0.25 each for reach, grasp, move and complete (judge stages); success is
the judge's binary call. Sim signals (signals.json) are carried alongside, with judge/sim agreement per stage.

Output, written into <folder>:
- scores_<model>[_r2].json: overall and per-task (scene) SR, graded score, stage rates and per-seed rows
- scores_<model>[_r2].tsv: one line per task

Graders (--grader):
- judge (default): the VLM judge decides every rollout, as described above.
- verifier: no OpenAI calls. verify.py's state verifier (cut at limits.limit(scene) steps, the same window as the judge
  frames) decides sim-only scenes by sim_prefix_ok; scenes whose last stage needs a judge are ungradable (reported
  separately, not counted as failures); scenes marked `skip` in task_specs.json are skipped.
- hybrid (production): sim-only scenes as in verifier. needs_judge scenes: sim_prefix_ok false -> failure, judge not
  called ("gated"); true -> the judge decides, with the verifier's sim facts prepended to its prompt. Rollout verdicts go
  to <rollout>/grade_<grader>.json and the summary to <folder>/scores_<grader>.json/.tsv (SR over decided rollouts, by
  task and by category, plus counts per verdict source). --no-judge-calls reuses existing judge files instead of calling
  the API (a missing one leaves the rollout ungradable); --max-judge-calls N caps real API calls.

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

import base64

import openai

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import RUBRIC, SCHEMA, keyframes  # noqa: E402

RUBRIC_R2 = """You grade robot manipulation rollouts from a physics simulator. A bimanual robot (two arms with
five-fingered dexterous hands) is given a natural-language task. You see frames sampled uniformly in time from one
rollout. At each time you get two images from fixed cameras: the HEAD camera (looking down at the table from above
the robot) and the CHEST camera (looking forward across the table). Captions give the time in seconds.

Grade four stages, each true or false, from what the images show:
1. reach: a hand gets to the task object: touching it or within a few cm of it.
2. grasp: the hand holds the task object. True only if the object is visibly lifted off the table, or visibly
   travels together with the hand between frames while the fingers are closed around it (for a lid, door, handle or
   button: the hand is on it while it moves or is pressed). Fingers hovering over, touching, nudging or pushing the
   object is NOT a grasp.
3. move: the task object is moved purposefully toward the task goal by the robot (carried, rotated, poured, opened,
   closed), visible as a clear change of its position or state. Pushing it a little, knocking it over, or moving a
   different object is NOT move.
4. complete: the task's goal state is visible: e.g. the object rests in/on the target, the lid is closed, the cap is
   on. For tasks without a placement (shake, stir, toss, hit, grip and hold), the described action is clearly and
   fully performed with the right object.

success = the full task was accomplished at some time in the rollout. Be strict: when the images are ambiguous or
the object is too small to tell, answer false. A later stage normally requires the earlier ones; if not, say why in
the rationale. In the rationale, name the object you judged and what happened to it."""

RUBRIC_R3 = RUBRIC_R2.replace(
    """4. complete: the task's goal state is visible: e.g. the object rests in/on the target, the lid is closed, the cap is
   on. For tasks without a placement (shake, stir, toss, hit, grip and hold), the described action is clearly and
   fully performed with the right object.""",
    """4. complete: the task was carried out as far as a rigid-body simulator can show it: the described action was
   clearly and fully performed with the right object, at the right place, in the way a person would do it (see the
   note on the simulator below).""") + """

About the simulator: it contains rigid bodies only. There is no liquid, powder, food, dough, paint or any deformable
material, no button that moves, no light, no sound. Nothing can visibly transfer, fill, mix, break or change its
surface. Never answer false because no substance, residue or changed contents is visible: that is impossible here by
construction. Judge whether the robot performed the motion a person would perform for this task with the right object
in the right place: brushing = the eraser is moved back and forth over the plate while held; stirring = the spoon is
moved around inside the pot; scooping = the spoon is dragged along the plate and lifted; hitting = the hammer head
strikes the box; pressing a button = a fingertip presses on the camera where its button is. Still answer false for the
wrong object, the wrong place, a motion that only grazes or nudges, an object dropped or knocked over, or an action
that was merely begun.

When simulator state checks are given before the images, they are ground truth from the physics engine: a stage
reported there as satisfied did happen even if the images are unclear. Use the images to judge the manner and the
completeness of the motion, not to re-check those facts."""

RUBRIC_R4 = RUBRIC_R3.split("When simulator state checks are given")[0].replace(
    "success = the full task was accomplished at some time in the rollout.",
    "success = the full task was accomplished AND the episode ended well (see \"The whole episode counts\" below).") + """

The whole episode counts, not only its best moment. Answer success = false, even if the action itself was done, when:
- the task object is dropped, thrown, slips out of the hand, or falls (onto the table, off it, or into something)
  at any point after the action, instead of being held or put down deliberately. At the end it must be in the hand,
  or resting where a person would put it, set down on purpose (lowered by the hand, not released from a height);
- the handling is unstable: the object wobbles, flips or swings in the hand, is fumbled or caught again, is juggled
  between hands to keep it from falling, or the hand repeatedly loses it;
- another object (the pot, plate, box, drum, pan) is knocked over, tipped, flung or pushed far by the action.
Compare consecutive frames: an object that is in the hand at one time and lying on the table, tilted or far away at
the next, without a visible lowering motion, was dropped. Look closely at the last frames.

Simulator measurements may be given before the images. They are exact facts about contacts and motion (distances,
times, speeds) from the physics engine, measured with simple thresholds. They are NOT a verdict on the task: a "yes"
there only means a threshold was crossed. Use them as facts, and decide the task, its manner and its ending from the
images."""

RUBRICS = {"r2": RUBRIC_R2, "r3": RUBRIC_R3, "r4": RUBRIC_R4}
SIM_HEADERS = {"r4": "Simulator measurements (exact, but not a verdict on the task):"}
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


def frames_r2(d: Path) -> list[tuple[float, str, str]]:
    """(t, head_b64, chest_b64) from judge_frames/t<sec>_<view>.jpg."""
    fd = d / "judge_frames"
    if not (fd / "COMPLETE").exists():  # written last by postprocess.judge_frames; partial frames would hide the end
        raise FileNotFoundError(f"{fd}: incomplete judge frames; run postprocess.py --frames-only {d}")
    ts = sorted({f.name.split("_")[0] for f in fd.glob("t*_ego_view.jpg")})
    if not ts:
        raise FileNotFoundError(f"{fd}: no judge frames; run postprocess.py --frames-only {d}")
    b64 = lambda f: base64.standard_b64encode(f.read_bytes()).decode()  # noqa: E731
    return [(float(t[1:]), b64(fd / f"{t}_ego_view.jpg"), b64(fd / f"{t}_chest_view.jpg")) for t in ts]


SIM_CONTEXT_HEADER = "Simulator state checks (ground truth from the physics engine, trust these over the images):"


def judge_one(c: openai.OpenAI, model: str, d: Path, force: bool, rubric: str = "r2", sim_context: str | None = None) -> dict:
    out = d / (f"judge_{model}.json" if rubric == "r1" else f"judge_{model}_{rubric}.json")
    if out.exists() and not force:
        cached = json.loads(out.read_text())
        if sim_context is None or cached.get("sim_context") == sim_context:  # a verdict made without the sim facts is stale
            return cached
    meta = json.loads((d / "rollout.json").read_text())
    img = lambda b: {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b}", "detail": "high"}  # noqa: E731
    if rubric == "r1":
        instructions = RUBRIC
        content = [{"type": "input_text", "text": f"Task: {meta['task']}\nRollout length: {meta['steps'] / 20:.1f} s."}]
        for t, b64 in keyframes(d):
            content += [{"type": "input_text", "text": f"t = {t:.1f} s"},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"}]
    else:
        instructions = RUBRICS[rubric]
        try:
            fr = frames_r2(d)
        except FileNotFoundError as e:
            return {"error": str(e)}
        content = [{"type": "input_text", "text": f"Task: {meta['task']}\nScored span: {fr[-1][0]:.1f} s, "
                                                  f"{len(fr)} times x 2 cameras."}]
        if sim_context:
            content.append({"type": "input_text", "text": f"{SIM_HEADERS.get(rubric, SIM_CONTEXT_HEADER)}\n{sim_context.removeprefix('Simulator state checks: ')}"})
        for t, head, chest in fr:
            content += [{"type": "input_text", "text": f"t = {t:.1f} s: head camera, then chest camera"}, img(head), img(chest)]
    content.append({"type": "input_text", "text": "Grade this rollout."})
    t0 = time.time()
    try:
        resp = c.responses.create(
            model=model,
            instructions=instructions,
            input=[{"role": "user", "content": content}],
            text={"format": {"type": "json_schema", "name": "rollout_grade", "schema": SCHEMA, "strict": True}},
        )
        verdict = json.loads(resp.output_text)
        verdict.update({"model": resp.model, "response_id": resp.id,
                        "usage": {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}})
    except (openai.APIError, json.JSONDecodeError) as e:  # recorded, excluded from aggregates, retried on rerun
        return {"error": f"{type(e).__name__}: {e}"[:500]}
    verdict["latency_s"] = round(time.time() - t0, 2)
    if sim_context:
        verdict["sim_context"] = sim_context
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


SOURCES = ("verifier", "judge", "gated", "skipped", "ungradable")


def run_verifier(d: Path, meta: dict, specs: dict, envs: dict, force: bool) -> dict:
    """verifier.json for one rollout, cut at limits.limit(scene) like the judge frames; recomputed if missing, --force,
    or computed over another window. Returns the verifier dict, {"skipped": ...} or {"error": ...}."""
    import verify  # noqa: E402  (imports mujoco; only the verifier / hybrid graders need it)
    from limits import limit
    scene = meta["scene_id"]
    spec = specs.get(scene)
    if spec is None:
        return {"error": "no spec in task_specs.json"}
    if spec.get("skip"):
        return {"skipped": spec["skip"]}
    if meta.get("revision_match") is False:
        return {"error": "recorded on an older scene revision"}
    vf = d / "verifier.json"
    if vf.exists() and not force:
        v = json.loads(vf.read_text())
        if "error" in v or v.get("analyzed_steps") == min(limit(scene), meta["steps"]):
            return v
    if scene not in envs:
        envs.clear()  # one scene at a time: dirs are visited grouped by scene
        envs[scene] = verify.SharpaWebXREnv(verify.SCENES_ROOT / scene)
    try:
        return verify.verify(d, envs[scene], spec, limit(scene))
    except Exception as e:  # noqa: BLE001  one bad rollout must not stop the batch
        v = {"error": f"{type(e).__name__}: {e}"[:500]}
        vf.write_text(json.dumps(v))
        return v


def grade_record(grader: str, meta: dict, v: dict, source: str, verdict, j: dict | None = None, note: str | None = None) -> dict:
    rec = {"grader": grader, "scene_id": meta["scene_id"], "seed": meta["seed"], "verdict": verdict, "source": source,
           "sim_prefix_ok": v.get("sim_prefix_ok"), "needs_judge": v.get("needs_judge"),
           "stages": [(s["name"], s["achieved"]) for s in v.get("stages", [])],
           "why": next((s["why"] for s in v.get("stages", []) if s["why"]), []),
           "analyzed_steps": v.get("analyzed_steps"), "judge_context": v.get("judge_context")}
    if note:
        rec["note"] = note
    if j is not None:
        rec["judge"] = {k: j.get(k) for k in ("success", "confidence", "rationale", "model")}
        rec["judge"]["used_sim_context"] = bool(j.get("sim_context"))
    return rec


def grade_all(a, dirs: list[Path]) -> list[dict]:
    """Grade every rollout dir with --grader verifier|hybrid; one grade_<grader>.json per rollout dir."""
    from verify import load_specs
    specs, envs, rows, todo = load_specs(), {}, {}, []
    metas = {d: json.loads((d / "rollout.json").read_text()) for d in dirs}
    for i, d in enumerate(sorted(dirs, key=lambda d: (metas[d]["scene_id"], str(d)))):
        meta = metas[d]
        v = run_verifier(d, meta, specs, envs, a.force)
        if "skipped" in v:
            rows[d] = grade_record(a.grader, meta, {}, "skipped", None, note=f"skipped: {v['skipped']}")
        elif "error" in v:
            rows[d] = grade_record(a.grader, meta, {}, "ungradable", None, note=f"verifier error: {v['error']}")
        elif not v["needs_judge"]:
            rows[d] = grade_record(a.grader, meta, v, "verifier", bool(v["sim_prefix_ok"]))
        elif a.grader == "verifier":
            rows[d] = grade_record(a.grader, meta, v, "ungradable", None, note="needs judge (--grader verifier)")
        elif not v["sim_prefix_ok"]:
            rows[d] = grade_record(a.grader, meta, v, "gated", False, note="sim stages failed; judge not called")
        else:
            todo.append((d, meta, v))
        if i % 50 == 49:
            print(f"verified {i + 1}/{len(dirs)}", flush=True)
    judged = {}
    if todo:
        out = lambda d: d / f"judge_{a.model}_{a.rubric}.json"  # noqa: E731
        cached = {d: out(d) for d, _, _ in todo if out(d).exists() and not a.force}
        calls = [d for d, _, _ in todo if d not in cached or not cached[d].exists()]
        if a.no_judge_calls:
            calls = []
        elif not a.force:  # a cached verdict made without the sim facts is re-judged
            calls += [d for d, _, v in todo if d in cached and json.loads(cached[d].read_text()).get("sim_context") != v["judge_context"]]
        if a.max_judge_calls is not None:
            calls = calls[: a.max_judge_calls]
        c = client() if calls else None
        print(f"hybrid: {len(todo)} rollouts need the judge, {len(calls)} API calls, {len(todo) - len(calls)} reused/ungraded", flush=True)
        ctx = {d: v["judge_context"] for d, _, v in todo}
        with ThreadPoolExecutor(a.workers) as ex:
            res = dict(zip(calls, ex.map(lambda d: judge_one(c, a.model, d, True, a.rubric, ctx[d]), calls), strict=True)) if calls else {}
        for d, meta, v in todo:
            j = res.get(d) or (json.loads(out(d).read_text()) if out(d).exists() and (a.no_judge_calls or d not in res) else None)
            if j is None or "error" in j:
                why = (j or {}).get("error") or ("judge-cache miss (--no-judge-calls)" if a.no_judge_calls else "judge not run (--max-judge-calls)")
                rows[d] = grade_record(a.grader, meta, v, "ungradable", None, note=why)
            else:
                rows[d] = grade_record(a.grader, meta, v, "judge", bool(j["success"]), j,
                                       note=None if j.get("sim_context") else "judge verdict made without the sim facts (cache)")
    for d, r in rows.items():
        (d / f"grade_{a.grader}.json").write_text(json.dumps(r, indent=1))
    return [dict(r, dir=str(d.relative_to(a.folder))) for d, r in rows.items()]


def summarize_grades(rows: list[dict]) -> dict:
    """SR over decided rollouts (verdict not None) with Wilson CI; ungradable / skipped are counted, never failures."""
    dec = [r for r in rows if r["verdict"] is not None]
    succ = sum(r["verdict"] for r in dec)
    s = {"n": len(rows), "n_decided": len(dec), "n_success": succ,
         "sources": {k: sum(r["source"] == k for r in rows) for k in SOURCES}}
    if dec:
        s.update({"success_rate": round(succ / len(dec), 4), "success_ci95": wilson(succ, len(dec))})
    return s


def main_grade(a):
    dirs = sorted(f.parent for f in a.folder.rglob("rollout.json"))
    if a.limit:
        dirs = dirs[: a.limit]
    if not dirs:
        sys.exit(f"no rollouts under {a.folder}")
    print(f"grading {len(dirs)} rollouts under {a.folder} with grader={a.grader}", flush=True)
    rows = grade_all(a, dirs)
    cats = json.loads((Path(__file__).resolve().parent / "task_categories.json").read_text())
    for r in rows:
        r["category"] = cats.get(r["scene_id"])
    tasks, by_cat = {}, {}
    for r in rows:
        tasks.setdefault(r["scene_id"], []).append(r)
        by_cat.setdefault(r["category"], []).append(r)
    per_task = {sc: {**summarize_grades(rs), "category": rs[0]["category"], "seeds": sorted(rs, key=lambda r: r["seed"])}
                for sc, rs in sorted(tasks.items())}
    dec = [t["success_rate"] for t in per_task.values() if "success_rate" in t]
    out = {"folder": str(a.folder), "grader": a.grader, "model": a.model if a.grader == "hybrid" else None,
           "rubric": a.rubric if a.grader == "hybrid" else None, "spec_version": "pour-v1",
           "scored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "overall": summarize_grades(rows),
           "task_mean": {"success_rate": round(sum(dec) / len(dec), 4) if dec else None, "n_tasks": len(dec)},
           "per_category": {k: summarize_grades(rs) for k, rs in sorted(by_cat.items(), key=lambda kv: str(kv[0]))},
           "per_task": per_task}
    jp = a.folder / f"scores_{a.grader}.json"
    jp.write_text(json.dumps(out, indent=1))
    cols = ["n", "n_decided", "n_success", "success_rate"]
    with open(a.folder / f"scores_{a.grader}.tsv", "w") as f:
        f.write("scene_id\tcategory\t" + "\t".join(cols) + "\t" + "\t".join(SOURCES) + "\n")
        for sc, t in per_task.items():
            f.write(f"{sc}\t{t['category']}\t" + "\t".join(str(t.get(k, "")) for k in cols) + "\t"
                    + "\t".join(str(t["sources"][k]) for k in SOURCES) + "\n")
    o = out["overall"]
    print(f"\nwrote {jp}\noverall: n={o['n']} decided={o['n_decided']} SR={o.get('success_rate')} CI={o.get('success_ci95')} "
          f"sources={o['sources']}")
    for k, t in out["per_category"].items():
        print(f"  {str(k):26s} n={t['n']:4d} decided={t['n_decided']:4d} SR={t.get('success_rate')} {t['sources']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--rubric", default="r3", choices=["r1", "r2", "r3", "r4"],
                   help="r3 (default): r2 plus the rigid-body-only note; verdict cache is per rubric (judge_<model>_<rubric>.json)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="score only the first N rollouts (pilot / cost check)")
    p.add_argument("--force", action="store_true", help="re-judge rollouts that already have a verdict")
    p.add_argument("--grader", default="judge", choices=["judge", "verifier", "hybrid"],
                   help="judge: VLM judge for every rollout (default); verifier: sim state checks only, no API; "
                        "hybrid: verifier for sim-only scenes and as a gate, judge (with sim facts) for the rest")
    p.add_argument("--no-judge-calls", action="store_true", help="hybrid: reuse existing judge files, never call the API")
    p.add_argument("--max-judge-calls", type=int, default=None, help="hybrid: cap on real API calls")
    a = p.parse_args()
    if a.grader != "judge":
        return main_grade(a)

    frames_dir = "keyframes" if a.rubric == "r1" else "judge_frames"
    dirs = sorted(f.parent for f in a.folder.rglob("rollout.json") if (f.parent / frames_dir).is_dir())
    if a.limit:
        dirs = dirs[: a.limit]
    if not dirs:
        sys.exit(f"no rollouts with {frames_dir}/ under {a.folder}" + (" (run postprocess.py --frames-only)" if a.rubric != "r1" else ""))
    c = client()
    print(f"judging {len(dirs)} rollouts under {a.folder} with {a.model}", flush=True)

    rows = []
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (d, v) in enumerate(zip(dirs, ex.map(lambda d: judge_one(c, a.model, d, a.force, a.rubric), dirs), strict=True)):
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
    out = {"folder": str(a.folder), "model": a.model, "rubric_version": a.rubric, "scored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "rubric": "GWP-0.5 graded: 0.25 x (reach, grasp, move, complete) from the judge; success = judge binary",
           "tokens": {"input": sum(u.get("input", 0) for u in usage), "output": sum(u.get("output", 0) for u in usage)},
           "overall": summarize(rows),
           "task_mean": {"success_rate": round(sum(t.get("success_rate", 0) for t in per_task.values()) / len(per_task), 4),
                         "graded": round(sum(t.get("graded", 0) for t in per_task.values()) / len(per_task), 4)},
           "per_task": per_task}
    suffix = a.model if a.rubric == "r1" else f"{a.model}_{a.rubric}"
    jp = a.folder / f"scores_{suffix}.json"
    jp.write_text(json.dumps(out, indent=1))
    cols = ["n", "n_scored", "success_rate", "graded", "judge_reach", "judge_grasp", "judge_move", "judge_complete",
            "sim_reach", "sim_grasp", "sim_move"]
    with open(a.folder / f"scores_{suffix}.tsv", "w") as f:
        f.write("scene_id\ttask\t" + "\t".join(cols) + "\n")
        for sc, t in per_task.items():
            f.write(f"{sc}\t{t['task']}\t" + "\t".join(str(t.get(k, "")) for k in cols) + "\n")
    o = out["overall"]
    print(f"\nwrote {jp}\noverall: n={o['n']} scored={o['n_scored']} errors={o['n_errors']} "
          f"SR={o.get('success_rate')} CI={o.get('success_ci95')} graded={o.get('graded')} tokens={out['tokens']}")


if __name__ == "__main__":
    main()
