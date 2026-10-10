"""VLM judge for Sharpa sim rollouts: Claude Opus 5.5 watches 16 frames (ego | chest views) of video.mp4.

GWP-0.5 graded rubric (report Sec. 4): 0.25 each for reach, grasp, move-to-target, place/complete; plus binary
success ("task completed at any point", RoboTwin semantics). Sim signals (signals.json) decide reach/grasp/move
when they fire; the judge decides place/complete and success. Output: judge.json next to the video.

Credentials: ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / `ant auth login` profile.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import imageio.v3 as iio
from PIL import Image

MODEL = "claude-opus-5-5"
N_FRAMES = 16

RUBRIC = """You grade robot manipulation rollouts from a physics simulator. A bimanual robot (two arms with
five-fingered dexterous hands) is given a natural-language task. You see frames sampled uniformly in time from one
rollout. Each frame shows two cameras side by side: left half = head camera looking down at the table, right
half = chest camera looking forward across the table. Frame captions give the time in seconds.

Score four stages, each true or false, judging only what is visible:
1. reach: a hand gets to (touches or is within a few cm of) the object the task is about.
2. grasp: the hand(s) take hold of or firmly engage that object (grasp, pinch, push on a lid/handle as the task
   requires). Incidental brushing is not a grasp.
3. move: the object is moved purposefully toward the task goal (carried, rotated, opened/closed partially, poured...).
4. complete: the task's goal state is reached (e.g. object placed in/on the target, lid closed, cap on). For
   tasks without a placement (shake, stir, toss, hit), complete means the described action was clearly performed.

success = the full task was accomplished at any time in the rollout (even if disturbed later).
Be strict: if the evidence is ambiguous, answer false. Knocking objects off the table, or the goal object ending
in an implausible state, is not success. Later stages normally imply earlier ones; flag contradictions in notes."""

SCHEMA = {
    "type": "object",
    "properties": {
        "reach": {"type": "boolean"}, "grasp": {"type": "boolean"}, "move": {"type": "boolean"},
        "complete": {"type": "boolean"}, "success": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "rationale": {"type": "string"},
    },
    "required": ["reach", "grasp", "move", "complete", "success", "confidence", "rationale"],
    "additionalProperties": False,
}


def sample_frames(video: Path, n: int = N_FRAMES, fps: float = 10.0):
    frames = iio.imread(video, plugin="pyav")
    idx = [round(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
    out = []
    for i in idx:
        buf = io.BytesIO()
        Image.fromarray(frames[i]).save(buf, format="JPEG", quality=85)
        out.append((i / fps, base64.standard_b64encode(buf.getvalue()).decode()))
    return out


def keyframes(rollout_dir: Path):
    """Pre-rendered keyframes (postprocess --keyframes) if present, else sampled from video.mp4."""
    kd = rollout_dir / "keyframes"
    if kd.is_dir() and any(kd.glob("t*.jpg")):
        return [(float(f.stem[1:]), base64.standard_b64encode(f.read_bytes()).decode()) for f in sorted(kd.glob("t*.jpg"))]
    return sample_frames(rollout_dir / "video.mp4")


def judge_one(client: anthropic.Anthropic, rollout_dir: Path, force: bool = False) -> dict:
    out = rollout_dir / "judge.json"
    if out.exists() and not force:
        return json.loads(out.read_text())
    meta = json.loads((rollout_dir / "rollout.json").read_text())
    content = [{"type": "text", "text": f"Task: {meta['task']}\nRollout length: {meta['steps'] / 20:.1f} s."}]
    for t, b64 in keyframes(rollout_dir):
        content += [{"type": "text", "text": f"t = {t:.1f} s"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}]
    content.append({"type": "text", "text": "Grade this rollout."})
    resp = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=[{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral"}}],
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": content}],
    )
    if resp.stop_reason == "refusal":
        verdict = {"error": "refusal", "stop_details": str(resp.stop_details)}
    else:
        text = next(b.text for b in resp.content if b.type == "text")
        verdict = json.loads(text)
    verdict.update({"model": resp.model, "request_id": resp._request_id, "frames": N_FRAMES,
                    "usage": {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens,
                              "cache_read": resp.usage.cache_read_input_tokens}})
    out.write_text(json.dumps(verdict, indent=1))
    return verdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    client = anthropic.Anthropic()
    dirs = [d for d in a.dirs if (d / "video.mp4").exists() or (d / "keyframes").is_dir()]
    with ThreadPoolExecutor(a.workers) as ex:
        for d, v in zip(dirs, ex.map(lambda d: judge_one(client, d, a.force), dirs)):
            print(d, {k: v.get(k) for k in ("reach", "grasp", "move", "complete", "success", "confidence", "error")}, flush=True)


if __name__ == "__main__":
    main()
