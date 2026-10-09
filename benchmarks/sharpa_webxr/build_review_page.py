"""Video review page for sim-eval rollouts: one section per scene, one video card per checkpoint tag, side by side.

usage: build_review_page.py <run> --tags step005000_ema step005000_bf16 [--seeds 0] --out <dir>
Writes <dir>/index.html and <dir>/v/<tag>/<scene5>_s<k>.mp4 (re-encoded small for the web), plus <dir>/files.json
(the published paths, for publishing in batches under the 64 MB per-publish limit). Rollouts without a rendered
video.mp4 yet are shown as "video pending" and picked up on the next build.
"""
import argparse
import glob
import html
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TASKS = json.load(open(HERE / "tasks.json"))
STAGES = ("reach", "grasp", "move")


def find(run_dir, tag, scene, seed):
    hits = glob.glob(str(run_dir / tag / "**" / scene / f"seed_{seed}"), recursive=True)
    return Path(hits[0]) if hits else None


def full(s):
    return all(s["stage"].get(k) for k in STAGES)


def encode(src, dst):
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264",
                    "-crf", "30", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)], check=True)


def chip(ok, name):
    return f'<span class="chip {"ok" if ok else "no"}">{name} {"yes" if ok else "no"}</span>'


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default=None)
    a = p.parse_args()
    run_dir = ROOT / ".sharpa_sim_eval" / a.run
    a.out.mkdir(parents=True, exist_ok=True)

    sigs = {t: {} for t in a.tags}  # tag -> scene -> [signals]
    for t in a.tags:
        for f in glob.glob(str(run_dir / t / "**" / "seed_*" / "signals.json"), recursive=True):
            s = json.load(open(f))
            sigs[t].setdefault(s["scene_id"], []).append(s)

    jobs, files = [], []
    secs = []
    order = sorted(TASKS, key=lambda sc: -sum(sum(map(full, sigs[t].get(sc, []))) / max(len(sigs[t].get(sc, [])), 1) for t in a.tags))
    for sc in order:
        pre = sc.split("_")[0]
        cards = []
        for t in a.tags:
            for k in a.seeds:
                d = find(run_dir, t, sc, k)
                s = json.load(open(d / "signals.json")) if d and (d / "signals.json").exists() else None
                rate = sigs[t].get(sc, [])
                rate_s = f"{sum(map(full, rate))}/{len(rate)} seeds all three" if rate else "not run yet"
                if s is None:
                    cards.append(f'<figure class="card"><div class="pending">not run yet</div><figcaption><div class="top"><span class="label">{t} · seed {k}</span></div></figcaption></figure>')
                    continue
                r = json.load(open(d / "rollout.json"))
                st = s["stage"]
                chips = "".join(chip(st.get(x), x) for x in STAGES)
                if s.get("fallen_any"):
                    chips += '<span class="chip no">object fell</span>'
                if s.get("verifier"):
                    chips += chip(s["verifier"].get("success_any"), "verifier")
                rel = f"v/{t}/{pre}_s{k}.mp4"
                if (d / "video.mp4").exists():
                    jobs.append((d / "video.mp4", a.out / rel))
                    files.append(rel)
                    vid = f'<video src="{rel}" controls muted loop playsinline preload="metadata"></video>'
                else:
                    vid = '<div class="pending">video rendering, check back later</div>'
                cards.append(f'<figure class="card">{vid}<figcaption><div class="top"><span class="label">{html.escape(t)} · seed {k}</span>'
                             f'<span class="nums">{r["steps"] / 20:.1f} s · IK {s.get("ik_converged_frac", 0) * 100:.0f}%</span></div>'
                             f'<div class="chips">{chips}</div><div class="nums">{rate_s}</div></figcaption></figure>')
        star = "★ " if TASKS[sc]["n_eps"].get("gpt_fleet_depth") else ""
        secs.append(f'<section class="task" id="s{pre}"><div class="head"><h2>{star}{html.escape(TASKS[sc]["task"])}</h2>'
                    f'<span class="sid">{html.escape(sc)}</span></div><div class="row">{"".join(cards)}</div></section>')

    with ThreadPoolExecutor(32) as ex:
        list(ex.map(lambda j: encode(*j), jobs))

    rows = []
    for t in a.tags:
        ss = [s for v in sigs[t].values() for s in v]
        n = len(ss)
        if not n:
            rows.append(f"<tr><th scope=row>{t}</th><td>0</td>" + "<td>–</td>" * 6 + "</tr>")
            continue
        m = lambda f: sum(map(f, ss)) / n  # noqa: E731
        rows.append(f"<tr><th scope=row>{t}</th><td>{n}</td>" + "".join(f"<td>{m(lambda s: bool(s['stage'].get(x))):.2f}</td>" for x in STAGES)
                    + f"<td>{m(full):.2f}</td><td>{m(lambda s: bool(s.get('fallen_any'))):.2f}</td><td>{m(lambda s: s.get('ik_converged_frac', 0)):.2f}</td></tr>")
    css = (HERE / "review_page.css").read_text()
    title = a.title or f"{a.run} Sim Videos"
    page = f"""<title>{html.escape(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{css}
.pending{{aspect-ratio:640/240;max-width:100%;display:grid;place-items:center;background:var(--na-bg);color:var(--na);font-family:var(--mono);font-size:12px}}
</style>
<div class="wrap">
<header><h1>{html.escape(title)}</h1>
<p class="lede">Closed-loop WebXR MuJoCo sim, {len(TASKS)} training scenes. Cards in each row compare checkpoints on the same seeded reset. Each video shows the ego view on the left and the chest view on the right. Scenes are sorted best to worst; ★ marks scenes with gpt_fleet_depth data.</p></header>
<div class="facts"><span>run <b>{html.escape(a.run)}</b></span><span>seeds shown <b>{", ".join(map(str, a.seeds))}</b></span><span>chunk <b>32 steps, executed fully</b></span><span>flow steps <b>10</b></span><span>control <b>20 Hz</b></span></div>
<section class="task"><h2>Stage rates over all evaluated seeds</h2>
<div class="tablewrap"><table><thead><tr><th>checkpoint</th><th>rollouts</th><th>reach</th><th>grasp</th><th>move</th><th>all three</th><th>object fell</th><th>IK converged</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<p class="note">These are sim signals only. Reach means a hand came within 3 cm of an object. Grasp means 0.5 s or more of contact while the object was lifted or carried. Move means an object moved 5 cm or more, or a joint turned 20° or more. Task completion waits on the VLM judge.</p></section>
{"".join(secs)}
</div>"""
    (a.out / "index.html").write_text(page)
    json.dump(files, open(a.out / "files.json", "w"))
    print(f"wrote {a.out / 'index.html'}: {len(files)} videos, {sum((a.out / f).stat().st_size for f in files) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
