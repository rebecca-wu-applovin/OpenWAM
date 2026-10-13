"""Build notebooks/gwp_sim_eval.ipynb: closed-loop WebXR sim eval of GWP-0.5 Sharpa runs (v1 sweep + v2 watcher).

Sections: summary table, stage rates per checkpoint, all-three success vs training step (EMA vs raw, v1 reference),
outcome breakdown, gpt vs simteleop-only scenes, per-scene heatmap, IK / fallen / episode length, keyframe strips
for the best and worst scenes.

Run:
    .venv/bin/python notebooks/_build_gwp_sim_eval_nb.py
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.kernel_name=openwam --ExecutePreprocessor.timeout=1800 notebooks/gwp_sim_eval.ipynb
Add a run to RUNS to include it (same structure). The v2 watcher re-runs both commands after each checkpoint.
"""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).with_name("gwp_sim_eval.ipynb")
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# GWP-0.5 Sharpa: closed-loop WebXR sim eval

Closed loop, wam.cpp `ActionChunkExecutor` recipe:
- The policy predicts a 32-step chunk; the sim executes all 32 steps, then re-observes.
- 10 flow-matching steps, shift 5, no CFG.
- 58 training scenes with seeded resets (object xy ±5 cm, yaw ±180°, arm jitter 0.08), identical across checkpoints.

Stages come from sim signals:
- **reach:** a hand came within 3 cm of an object.
- **grasp:** at least 0.5 s of contact while the object was lifted 2 cm or more, or carried 3 cm or more.
- **move:** an object moved at least 5 cm, or a joint turned at least 20°.
- **all three:** reach, grasp and move in the same rollout.

Task completion (placement) waits for the VLM judge.
""")

code("""
import glob, json, math, os, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

REPO = Path(os.environ.get('OPENWAM_ROOT', '/root/OpenWAM')); os.chdir(REPO)
EVAL = REPO / '.sharpa_sim_eval'
RUNS = ['gwp05_full_v1', 'gwp05_full_v2']
TASKS = json.load(open(REPO / 'benchmarks/sharpa_webxr/tasks.json'))
STAGES = ['reach', 'grasp', 'move']

def parse_tag(tag):
    m = re.match(r'(best_val_)?step0*(\\d+)_(ema|bf16)$', tag)
    if m:
        return int(m.group(2)), m.group(3), bool(m.group(1))
    return None

rows = []
for run in RUNS:
    for tdir in sorted((EVAL / run).glob('*')):
        p = parse_tag(tdir.name)
        if p is None:
            continue
        step, weights, best = p
        for f in glob.glob(str(tdir / '**' / 'seed_*' / 'signals.json'), recursive=True):
            s = json.load(open(f))
            rj = Path(f).with_name('rollout.json')
            st = s['stage']
            rows.append(dict(run=run, tag=tdir.name, step=step, weights=weights, best_val=best,
                             scene=s['scene_id'], seed=s['seed'], task=TASKS[s['scene_id']]['task'],
                             gpt=bool(TASKS[s['scene_id']]['n_eps'].get('gpt_fleet_depth')),
                             reach=bool(st.get('reach')), grasp=bool(st.get('grasp')), move=bool(st.get('move')),
                             fallen=bool(s.get('fallen_any')), ik=s.get('ik_converged_frac', np.nan),
                             steps=json.load(open(rj))['steps'] if rj.exists() else np.nan,
                             dir=str(Path(f).parent)))
df = pd.DataFrame(rows)
df['all3'] = df.reach & df.grasp & df.move
df['label'] = df.run.str.replace('gwp05_full_', '') + ' ' + df.tag\nprint('rollouts per tag:', df.groupby('label').size().to_dict())
print(f'{len(df):,} rollouts;', df.groupby('run').tag.nunique().to_dict(), 'checkpoint tags per run')
""")

md("## Summary per checkpoint")
code("""
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return c - h, c + h

g = df.groupby(['run', 'tag', 'step', 'weights', 'best_val'])
summ = g.agg(n=('all3', 'size'), seeds=('seed', 'nunique'), reach=('reach', 'mean'), grasp=('grasp', 'mean'),
             move=('move', 'mean'), all3=('all3', 'mean'), fallen=('fallen', 'mean'), ik=('ik', 'mean')).reset_index()
ci = g.all3.agg(['sum', 'size']).apply(lambda r: wilson(r['sum'], r['size']), axis=1)
summ['all3_lo'] = [c[0] for c in ci]; summ['all3_hi'] = [c[1] for c in ci]
summ = summ.sort_values(['run', 'best_val', 'step', 'weights'])
display(summ.style.format({c: '{:.3f}' for c in ['reach', 'grasp', 'move', 'all3', 'all3_lo', 'all3_hi', 'fallen', 'ik']})
        .background_gradient(subset=['all3'], cmap='Greens', vmin=0, vmax=0.7).hide(axis='index'))
""")

md("## Stage rates per checkpoint")
code("""
fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(summ)), 4))
x = np.arange(len(summ)); w = 0.2
for i, (k, c) in enumerate(zip(STAGES + ['all3'], ['#9ecae1', '#4292c6', '#08519c', '#238b45'])):
    ax.bar(x + (i - 1.5) * w, summ[k], w, label=k, color=c)
ax.errorbar(x + 1.5 * w, summ.all3, yerr=[summ.all3 - summ.all3_lo, summ.all3_hi - summ.all3], fmt='none', ecolor='k', capsize=2, lw=1)
ax.set_xticks(x, (summ.run.str.replace('gwp05_full_', '') + '\\n' + summ.tag).tolist(), rotation=45, ha='right', fontsize=8)
ax.set_ylim(0, 1.05); ax.set_ylabel('fraction of rollouts'); ax.legend(ncol=4, fontsize=8); ax.grid(axis='y', alpha=.3)
ax.set_title('Stage rates (error bars: 95% Wilson CI on all three)')
plt.tight_layout(); plt.show()
""")

md("## All three vs training step\nv2 intermediate checkpoints use 5 seeds; dashed lines are v1 checkpoints (10 seeds).")
code("""
v1 = summ[(summ.run == 'gwp05_full_v1') & (summ.weights == 'bf16')]
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, metric in zip(axes, ['all3', 'grasp']):
    ax.axhspan(v1[metric].min(), v1[metric].max(), color='gray', alpha=.2,
               label=f'v1 bf16 steps {v1.step.min()}-{v1.step.max()} (batch 1024)')
    for (run, wts), sub in summ[~summ.best_val & (summ.run != 'gwp05_full_v1')].groupby(['run', 'weights']):
        sub = sub.sort_values('step')
        ax.plot(sub.step, sub[metric], 'o-', label=f'{run.replace("gwp05_full_", "")} {wts}')
        if metric == 'all3':
            ax.fill_between(sub.step, sub.all3_lo, sub.all3_hi, alpha=.15)
    ax.set_xlim(0, 51000); ax.set_xlabel('training step'); ax.set_ylabel(f'{metric} rate'); ax.set_ylim(0, 1)
    ax.grid(alpha=.3); ax.legend(fontsize=8, loc='upper left'); ax.set_title(f'{metric} rate vs training step')
plt.tight_layout(); plt.show()
""")

md("## Where rollouts stop\nFirst stage each rollout failed to reach.")
code("""
def outcome(r):
    if not r.reach: return 'no reach'
    if not r.grasp and not r.move: return 'reach only'
    if not r.grasp: return 'move w/o grasp (push/knock)'
    if not r.move: return 'grasp, no move'
    return 'all three'
order = ['no reach', 'reach only', 'move w/o grasp (push/knock)', 'grasp, no move', 'all three']
colors = ['#a50f15', '#fb6a4a', '#fdae6b', '#9ecae1', '#238b45']
df['outcome'] = df.apply(outcome, axis=1)
tab = pd.crosstab(df.label, df.outcome, normalize='index').reindex(columns=order, fill_value=0)
tab = tab.loc[summ.apply(lambda r: r.run.replace('gwp05_full_', '') + ' ' + r.tag, axis=1)]
ax = tab.plot.barh(stacked=True, color=colors, figsize=(10, 0.35 * len(tab) + 1.5), width=.8)
ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel('fraction of rollouts'); ax.legend(ncol=3, fontsize=8, loc='lower center', bbox_to_anchor=(.5, 1))
plt.tight_layout(); plt.show()
""")

md("## Scenes with gpt_fleet_depth data vs simteleop-only scenes")
code("""
sp = df.groupby(['label', 'gpt']).all3.mean().unstack().rename(columns={True: 'gpt scenes (20)', False: 'simteleop-only (38)'})
sp = sp.loc[tab.index]
ax = sp.plot.bar(figsize=(max(8, 0.8 * len(sp)), 3.5), color=['#bdbdbd', '#6a51a3'])
ax.set_ylabel('all-three rate'); ax.set_ylim(0, 1); ax.grid(axis='y', alpha=.3); ax.set_xlabel('')
plt.xticks(rotation=45, ha='right', fontsize=8); plt.tight_layout(); plt.show()
""")

md("## Per-scene all-three rate\nRows sorted by mean over checkpoints; ★ marks scenes with gpt_fleet_depth data.")
code("""
hm = df.pivot_table(index='scene', columns='label', values='all3', aggfunc='mean')[tab.index]
hm = hm.loc[hm.mean(axis=1).sort_values(ascending=False).index]
names = [('★ ' if TASKS[s]['n_eps'].get('gpt_fleet_depth') else '') + TASKS[s]['task'][:48] + f'  [{s[:5]}]' for s in hm.index]
fig, ax = plt.subplots(figsize=(1.0 * hm.shape[1] + 6, 0.24 * len(hm) + 1.5))
im = ax.imshow(hm.values, cmap='Greens', vmin=0, vmax=1, aspect='auto')
ax.set_yticks(range(len(hm)), names, fontsize=7); ax.set_xticks(range(hm.shape[1]), hm.columns, rotation=45, ha='right', fontsize=8)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        v = hm.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=6, color='white' if v > .6 else 'black')
plt.colorbar(im, ax=ax, fraction=.02, label='all-three rate'); plt.tight_layout(); plt.show()
""")

md("## IK convergence, fallen objects, episode length")
code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
labels = list(tab.index)
axes[0].boxplot([df[df.label == l].ik.dropna() for l in labels], showfliers=False)
axes[0].set_title('IK converged fraction per rollout')
axes[1].bar(range(len(labels)), [df[df.label == l].fallen.mean() for l in labels], color='#fb6a4a')
axes[1].set_title('rollouts with an object fallen off the table')
axes[2].boxplot([df[df.label == l].steps.dropna() / 20 for l in labels], showfliers=False)
axes[2].set_title('episode length (s)')
for ax in axes:
    ax.set_xticks(range(1 if ax is not axes[1] else 0, len(labels) + (1 if ax is not axes[1] else 0)), labels, rotation=45, ha='right', fontsize=7)
    ax.grid(axis='y', alpha=.3)
plt.tight_layout(); plt.show()
""")

md("## Keyframes: best and worst scenes\nThe fully evaluated tag with the highest all-three rate, seed 0. Each panel shows 8 of the 16 judge keyframes, with the ego view on the left and the chest view on the right.")
code("""
from PIL import Image
full = summ[(summ.n >= 0.95 * 58 * summ.seeds) & (summ.n >= 58)]  # tags whose seeds are complete
best_tag = full.sort_values('all3').iloc[-1]
sub = df[(df.run == best_tag.run) & (df.tag == best_tag.tag)]
rank = sub.groupby('scene').all3.mean().sort_values()
show = list(rank.index[-3:][::-1]) + list(rank.index[:3])
print(f'checkpoint: {best_tag.run} {best_tag.tag}')
for s in show:
    r = sub[(sub.scene == s)].sort_values('seed').iloc[0]
    kf = sorted(Path(r.dir, 'keyframes').glob('*.jpg'))[::2]
    if not kf:
        continue
    fig, axs = plt.subplots(2, 4, figsize=(18, 3.6))
    for ax, k in zip(axs.flat, kf):
        ax.imshow(Image.open(k)); ax.axis('off'); ax.set_title(f't={float(k.stem[1:]):.1f}s', fontsize=8)
    for ax in list(axs.flat)[len(kf):]:
        ax.axis('off')
    fig.suptitle(f'{TASKS[s]["task"]}  [{s[:5]}]  scene rate {rank[s]:.1f}; seed {r.seed}: {r.outcome}', fontsize=10, x=0.01, ha='left')
    plt.tight_layout()
    plt.show()
""")

md("## Videos: best and worst scenes\nSame checkpoint and scenes as above, seed 0, ego view on the left and chest view on the right. Videos are embedded, so they play anywhere the notebook opens. For every scene, see the review page or `<rollout>/video.mp4`.")
code("""
import base64, subprocess, tempfile
from IPython.display import HTML
import imageio_ffmpeg
FF = imageio_ffmpeg.get_ffmpeg_exe()
cards = []
for s in show:
    r = sub[(sub.scene == s)].sort_values('seed').iloc[0]
    v = Path(r.dir, 'video.mp4')
    if not v.exists():
        cards.append(f'<div style="width:480px"><b>{TASKS[s]["task"]}</b><br>video not rendered yet</div>'); continue
    with tempfile.NamedTemporaryFile(suffix='.mp4') as t:
        subprocess.run([FF, '-y', '-loglevel', 'error', '-i', str(v), '-c:v', 'libx264', '-crf', '32', '-pix_fmt', 'yuv420p', t.name], check=True)
        b64 = base64.b64encode(open(t.name, 'rb').read()).decode()
    cards.append(f'<div style="width:480px;font:12px sans-serif"><b>{TASKS[s]["task"]}</b> [{s[:5]}]<br>'
                 f'scene rate {rank[s]:.1f}; seed {r.seed}: {r.outcome}<br>'
                 f'<video width="480" controls loop muted src="data:video/mp4;base64,{b64}"></video></div>')
HTML('<div style="display:flex;flex-wrap:wrap;gap:12px">' + ''.join(cards) + '</div>')
""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "openwam", "display_name": "openwam", "language": "python"}
nbf.write(nb, OUT)
print("wrote", OUT)
