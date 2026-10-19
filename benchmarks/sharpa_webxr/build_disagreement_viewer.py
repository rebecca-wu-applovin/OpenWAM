"""Static viewer of human-vs-verifier disagreements on data/webxr_episodes (re-runnable; skips already rendered videos).

FN = verifier prefix failed on a human positive (success/recover); FP = verifier passed on a human failure.
Tasks ending in a judge: FPs (state checks passed on a labelled failure) carry the judge's cached verdict. Writes data/webxr_episodes/_viewer/index.html.
Run: .venv_webxr/bin/python benchmarks/sharpa_webxr/build_disagreement_viewer.py [--cap 4] [--jobs 32] [--no-render]
"""
import argparse
import html
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EPS = ROOT / "data/webxr_episodes"
OUT = EPS / "_viewer"
HERE = Path(__file__).resolve().parent
CATS = json.loads((HERE / "task_categories.json").read_text())
TASKS = json.loads((HERE / "tasks.json").read_text())
# recordings index written by pull_episodes.py: original_filename / upload_prefix per episode_uuid
try:
    INDEX = json.loads((EPS / "recordings_index.json").read_text())["entries"]
except Exception:
    INDEX = {}
LIVE = "gs://foundational-research/webxr-teleop/live"


def scan():
    sel, total = defaultdict(lambda: {"FN": [], "FP": []}), defaultdict(lambda: {"FN": 0, "FP": 0})
    for sd in sorted(p for p in EPS.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for d in sorted(p for p in sd.iterdir() if p.is_dir()):
            try:
                r = json.loads((d / "rollout.json").read_text())
                v = json.loads((d / "verifier.json").read_text())
            except Exception:
                continue
            if not r.get("revision_match") or "error" in v or "skipped" in v or r.get("outcome") not in ("success", "recover", "failure"):
                continue
            pos = r["outcome"] != "failure"
            if bool(v.get("sim_prefix_ok")) == pos:
                continue
            kind = "FN" if pos else "FP"  # FPs of tasks ending in a judge are shown too: the judge decides next
            total[sd.name][kind] += 1
            sel[sd.name][kind].append((d, r, v))
    return sel, total




def render(dirs, jobs):
    todo = [str(d) for d in dirs if not (d / "video.mp4").exists()]
    if not todo:
        return {}
    env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "osmesa"))
    py = str(ROOT / ".venv_webxr/bin/python")
    cmd = ["xargs", "-P", str(jobs), "-n", "1", py, str(HERE / "render_video.py")]
    p = subprocess.run(cmd, input="\n".join(todo) + "\n", text=True, env=env, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)
    errs = {}
    for d in todo:
        if not (Path(d) / "video.mp4").exists():
            errs[d] = "no video produced"
    if errs:
        lines = [l for l in p.stderr.splitlines() if l.strip()]
        for d in errs:
            errs[d] = next((l for l in reversed(lines) if Path(d).name in l or "Error" in l), lines[-1] if lines else "unknown")[:300]
    return errs


def esc(x):
    return html.escape(str(x))


def card(d, r, v, kind, vid):
    pos = r["outcome"] != "failure"
    rows = []
    for s in v["stages"]:
        cls = "ok" if s["achieved"] else "bad"
        rows.append(f"<tr class={cls}><td>{esc(s['name'])}</td><td>{esc(s['check'])}</td><td>{s['achieved']}</td><td>{s['any_time']}</td>"
                    f"<td>{s['end_state']}</td><td>{esc(s['t_first_s'])}</td></tr>")
    failing = "".join(
        f"<div class=why><b>{esc(s['name'])} failed because:</b><ul>{''.join('<li>' + esc(w) + '</li>' for w in s.get('why', []))}</ul></div>"
        + f"<details class=m><summary>all conditions and raw metrics for {esc(s['name'])}</summary><ul>{''.join('<li>' + esc(c) + '</li>' for c in s.get('conditions', []))}</ul>"
        f"{esc(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in s['metrics'].items()}))}</details>"
        for s in v["stages"] if not s["achieved"] and s["check"] != "judge")
    fallen = [k for k, x in (v.get("fallen") or {}).items() if x]
    jtxt = ""
    if v.get("needs_judge") and v.get("sim_prefix_ok"):  # state checks passed: the vision judge decides the episode
        jf = d / "judge_gpt-6-luna_r3.json"
        j = json.loads(jf.read_text()) if jf.exists() else None
        if j is None:
            jtxt = "<div class=judge><b>judge:</b> not called yet</div>"
        elif "error" in j:
            jtxt = f"<div class=judge><b>judge:</b> error {esc(j['error'])[:200]}</div>"
        else:
            stale = j.get("sim_context") != v.get("judge_context")
            jtxt = (f"<div class=judge><b>judge (gpt-6-luna, r3): {'SUCCESS' if j.get('success') else 'FAIL'}</b>"
                    f"{' <i>(made with an older sim summary)</i>' if stale else ''} &mdash; {esc(j.get('rationale', ''))}</div>")
    media = (f"<video controls loop muted preload=metadata src='{esc(vid)}'></video>" if vid else "<div class=novid>not rendered</div>")
    e = INDEX.get(r.get("episode_uuid"), {})
    orig = e.get("original_filename") or f"{r.get('episode_uuid', '')} ({r.get('user_id', '')})"
    src = f"{LIVE}/{e['upload_prefix']}/" if e.get("upload_prefix") else ""
    key = f"{d.parent.name}/{d.name}"
    meta = {"key": key, "scene": d.parent.name, "episode_uuid": r.get("episode_uuid"), "original_filename": e.get("original_filename"),
            "human": r["outcome"], "verifier": "PASS" if v["sim_prefix_ok"] else "FAIL", "kind": kind}
    review = (f"<div class=review data-meta='{esc(json.dumps(meta))}'>"
              f"<label><input type=radio name='dec-{esc(key)}' value=verifier> verifier right (human label wrong)</label> "
              f"<label><input type=radio name='dec-{esc(key)}' value=human> human right (verifier wrong)</label> "
              f"<label><input type=radio name='dec-{esc(key)}' value=both_wrong> both wrong</label> "
              f"<label><input type=radio name='dec-{esc(key)}' value=unsure> unsure</label>"
              f"<textarea placeholder='reason / what actually happened (auto-saves)' rows=2></textarea>"
              f"<button class=clear>clear</button> <span class=status>auto-saves on change</span></div>")
    return (f"<div class='card {kind}' id='{esc(key)}'>{media}<div class=info><div class=tag><span class='b {kind}'>{kind}</span> human: <b>{esc(r['outcome'])}</b> ({'pos' if pos else 'neg'}) "
            f"| verifier: <b>{'PASS' if v['sim_prefix_ok'] else 'FAIL'}</b>{' (needs_judge)' if v.get('needs_judge') else ''}"
            f"{' | fallen: ' + esc(', '.join(fallen)) if fallen else ''}</div>"
            f"<div class=orig><b>{esc(orig)}</b> by {esc(r.get('user_id', '?'))}, uploaded {esc(r.get('upload_date', '?'))}, {esc(r.get('source_num_steps', '?'))} steps"
            f"{(' <span class=gs>' + esc(src) + '</span>') if src else ''}</div>"
            f"<table><tr><th>stage</th><th>check</th><th>achieved</th><th>any_time</th><th>end_state</th><th>t_first_s</th></tr>{''.join(rows)}</table>"
            f"{jtxt}{failing}{review}<div class=dn>local: {esc(d.name)}</div></div></div>")


JS = r"""
<script>
async function loadReviews(){
  let rv = {};
  try { rv = await (await fetch('reviews.json', {cache:'no-store'})).json(); } catch(e) { document.getElementById('srv').textContent = 'reviews: static mode (serve with serve_viewer.py to save)'; return; }
  document.getElementById('srv').textContent = 'reviews saved: ' + Object.keys(rv).length;
  for (const box of document.querySelectorAll('.review')) {
    const m = JSON.parse(box.dataset.meta); const r = rv[m.key]; if (!r) continue;
    const inp = box.querySelector(`input[value="${r.decision}"]`); if (inp) inp.checked = true;
    box.querySelector('textarea').value = r.reason || '';
    box.querySelector('.status').textContent = 'saved ' + (r.saved_at || ''); box.classList.add('done');
    box.closest('.card').dataset.dec = r.decision || '';
  }
  updateHidden();
}
function updateHidden(){
  const hide = document.getElementById('hidevr').checked;
  document.body.classList.toggle('hide-vr', hide);
  try { localStorage.setItem('hidevr', hide ? '1' : '0'); } catch(e) {}
  const n = document.querySelectorAll('.card[data-dec="verifier"]').length;
  document.getElementById('hidevr-n').textContent = '(' + n + ')';
}
document.getElementById('hidevr').addEventListener('change', updateHidden);
try { if (localStorage.getItem('hidevr') === '1') document.getElementById('hidevr').checked = true; } catch(e) {}
async function send(box, clear){
  const m = JSON.parse(box.dataset.meta);
  const sel = box.querySelector('input:checked');
  const rec = Object.assign({}, m, {decision: clear ? '' : (sel ? sel.value : ''), reason: clear ? '' : box.querySelector('textarea').value});
  if (!clear && !rec.decision) { box.querySelector('.status').textContent = 'pick a decision first'; return; }
  const st = box.querySelector('.status'); st.textContent = 'saving...';
  try {
    const res = await fetch('review', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(rec)});
    const j = await res.json();
    box.closest('.card').dataset.dec = clear ? '' : rec.decision; updateHidden();
    if (clear) { for (const i of box.querySelectorAll('input')) i.checked = false; box.querySelector('textarea').value = ''; box.classList.remove('done'); st.textContent = 'cleared'; }
    else { box.classList.add('done'); st.textContent = 'saved (' + j.n + ' total)'; }
    document.getElementById('srv').textContent = 'reviews saved: ' + j.n;
  } catch(e) { st.textContent = 'save failed: ' + e; }
}
document.addEventListener('click', ev => {
  const box = ev.target.closest('.review'); if (!box) return;
  if (ev.target.classList.contains('clear')) send(box, true);
});
// auto-save: immediately on a decision click, debounced while typing the reason, and on leaving the text box
document.addEventListener('change', ev => {
  const box = ev.target.closest('.review'); if (!box) return;
  if (ev.target.matches('input[type=radio]')) send(box, false);
});
const timers = new Map();
document.addEventListener('input', ev => {
  const box = ev.target.closest('.review'); if (!box || !ev.target.matches('textarea')) return;
  clearTimeout(timers.get(box)); timers.set(box, setTimeout(() => send(box, false), 800));
});
document.addEventListener('focusout', ev => {
  const box = ev.target.closest('.review'); if (!box || !ev.target.matches('textarea')) return;
  clearTimeout(timers.get(box)); send(box, false);
});
loadReviews();
</script>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    sel, total = scan()
    chosen = {s: {k: sel[s][k][: a.cap] for k in ("FN", "FP")} for s in sel}
    alld = [d for s in chosen.values() for k in s.values() for d, _, _ in k]
    errs = {} if a.no_render else render(alld, a.jobs)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(exist_ok=True)
    keep = set()
    bycat = defaultdict(list)
    for s in chosen:
        bycat[CATS.get(s, "uncategorized")].append(s)
    summary, body = [], []
    for cat in sorted(bycat):
        body.append(f"<h2>{esc(cat)}</h2>")
        for s in sorted(bycat[cat]):
            nr, cards = 0, []
            for kind in ("FN", "FP"):
                for d, r, v in chosen[s][kind]:
                    vid = None
                    if (d / "video.mp4").exists():
                        link = OUT / "videos" / s / f"{d.name}.mp4"
                        link.parent.mkdir(exist_ok=True)
                        if link.is_symlink() or link.exists():
                            link.unlink()
                        link.symlink_to(os.path.relpath(d / "video.mp4", link.parent))
                        keep.add(link)
                        vid = f"videos/{s}/{d.name}.mp4"
                        nr += 1
                    cards.append(card(d, r, v, kind, vid))
            task = TASKS.get(s, {}).get("task", "")
            summary.append((cat, s, total[s]["FN"], total[s]["FP"], nr))
            body.append(f"<h3 id='{esc(s)}'>{esc(s)}</h3><div class=task>{esc(task)}</div><div class=grid>{''.join(cards)}</div>")
    for l in (OUT / "videos").rglob("*.mp4"):
        if l not in keep and l.is_symlink():
            l.unlink()
    srows = "".join(f"<tr><td>{esc(c)}</td><td><a href='#{esc(s)}'>{esc(s)}</a></td><td>{fn}</td><td>{fp}</td><td>{nr}</td></tr>" for c, s, fn, fp, nr in summary)
    tf, tp, tr = (sum(x[i] for x in summary) for i in (2, 3, 4))
    css = ("body{font:14px system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:2px 6px;text-align:left}"
           ".grid{display:flex;flex-wrap:wrap;gap:12px}.card{background:#fff;border:2px solid #ccc;padding:8px;width:700px}.card.FN{border-color:#e0a030}.card.FP{border-color:#c04040}"
           "video{width:100%}.novid{background:#eee;padding:30px;text-align:center}tr.ok td{background:#e8f6e8}tr.bad td{background:#fbe9e9}.m{font:12px monospace;margin:2px 0}"
           ".dn{font:11px monospace;color:#666;margin-top:4px;word-break:break-all}.orig{font:12px monospace;margin:4px 0;word-break:break-all}.gs{color:#888}"
           ".why{margin:6px 0;padding:6px;background:#fff3f0;border-left:4px solid #c04040;font-size:13px}.why ul{margin:4px 0 0 18px;padding:0}.m summary{cursor:pointer;color:#666}.m ul{margin:2px 0 4px 18px;color:#555}"
           ".judge{margin:6px 0;padding:6px;background:#fff6e0;border:1px solid #e0c080;font-size:12px}.review{margin:6px 0;padding:6px;background:#f3f3f3;border:1px solid #ddd;font-size:12px}.review.done{background:#e6f0ff;border-color:#8ab}.review label{margin-right:8px;white-space:nowrap}"
           ".review textarea{display:block;width:98%;margin:4px 0;font:12px monospace}.review .status{color:#555;margin-left:6px}#srv{color:#357;font-weight:bold}.task{margin:0 0 6px;color:#444}.b{padding:1px 5px;color:#fff;border-radius:3px}.b.FN{background:#e0a030}.b.FP{background:#c04040}"
           "body.hide-vr .card[data-dec=verifier]{display:none}.opt{margin-left:12px;padding:2px 6px;background:#eef;border:1px solid #aac;border-radius:3px;cursor:pointer;white-space:nowrap}")
    page = (f"<!doctype html><meta charset=utf-8><title>Human vs verifier disagreements</title><style>{css}</style><h1>Human vs verifier disagreements</h1>"
            f"<p>FN = verifier FAIL on human success/recover; FP = verifier PASS on human failure. Capped at {a.cap} each per scene (first by dir name). "
            f"Totals (all disagreements): FN {tf}, FP {tp}; rendered shown {tr}. "
            f"Review: pick who is right per episode and write why; saved to _viewer/reviews.json by serve_viewer.py. <span id=srv></span> "
            f"<label class=opt><input type=checkbox id=hidevr> hide cards already decided \"verifier right\" <span id=hidevr-n></span></label></p>"
            f"<table><tr><th>category</th><th>scene</th><th>n_FN</th><th>n_FP</th><th>n rendered</th></tr>{srows}"
            f"<tr><th colspan=2>total</th><th>{tf}</th><th>{tp}</th><th>{tr}</th></tr></table>{''.join(body)}{JS}")
    (OUT / "index.html").write_text(page)
    (OUT / "render_errors.json").write_text(json.dumps(errs, indent=1))
    print(f"disagreements FN={tf} FP={tp}; selected={len(alld)}; rendered_present={tr}; render_errors={len(errs)}; wrote {OUT/'index.html'}")


if __name__ == "__main__":
    main()
