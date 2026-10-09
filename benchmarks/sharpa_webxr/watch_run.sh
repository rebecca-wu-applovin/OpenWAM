#!/bin/bash
# Eval watcher: closed-loop sim eval of every checkpoint of a training run as it lands.
# usage: [VIDEO=1] [VIDEO_SEEDS=all] [VIDEO_P=32] [PORT_BASE=11600] \
#        watch_run.sh <run> <gpu_ema> <gpu_bf16> [seeds_intermediate=0-4] [seeds_final=5-9]
#   VIDEO=0          skip video.mp4 rendering (signals + keyframes only)
#   VIDEO_SEEDS=0    render only these seeds (comma list, or "all"); seed 0 is always rendered first
#   VIDEO_P=32       parallel renderers (each holds ~2.3 GB RAM; CPU OSMesa, nice 19)
#   PORT_BASE=11600  EMA server on PORT_BASE, raw on PORT_BASE+1 (use a different base for a second watcher)
#   PREP_DIR=...     training prep dir passed to policy_server.py --prep-dir (default prep_eef_full_v1)
#
# - Copier (background): copies each new checkpoint_stepN/transformer_bf16.pt (rotated away by --keep-checkpoints)
#   to .gwp_runs/<run>_eval_ckpts/stepN_bf16.pt as soon as its meta.json exists (save is atomic).
# - Main loop, oldest step first: for each step with ema_stepN/meta.json (kept by training) and/or a copied raw
#   checkpoint, serve EMA on <gpu_ema>:PORT_BASE and raw on <gpu_bf16>:PORT_BASE+1, run run_suite.sh for both in parallel
#   (seeds_intermediate, 10 workers each), stop the servers, mark DONE.
# - Renderer (background, one queue for the whole run): renders video.mp4 for every finished rollout without one,
#   seed 0 first, at nice 19 with VIDEO_P processes, so a backlog never multiplies memory use.
# - After training exits: top up the last step with seeds_final, evaluate the final best_val (EMA + raw, all seeds),
#   wait for the low-priority videos, write summary.tsv, print WATCH_ALL_DONE.
# Resumable: DONE markers in .sharpa_sim_eval/<run>/<tag>/DONE_<seeds>; run_suite.sh skips finished seeds.
set -u
cd "$(dirname "$(readlink -f "$0")")/../.."  # repo root
RUN=$1; GA=$2; GB=$3; SI=${4:-0-4}; SF=${5:-5-9}
VIDEO=${VIDEO:-1}; VIDEO_SEEDS=${VIDEO_SEEDS:-all}; VIDEO_P=${VIDEO_P:-32}; PORT_BASE=${PORT_BASE:-11600}
PE=$PORT_BASE; PB=$((PORT_BASE + 1))
SRC=.gwp_runs/$RUN; C=.gwp_runs/${RUN}_eval_ckpts; E=.sharpa_sim_eval/$RUN; L=.sharpa_sim_eval/logs/$RUN
mkdir -p $C $E $L
export RUN
ts() { date -u +%FT%TZ; }
alive() { pgrep -f "run_giga_sharpa_train_job.py.*--out-dir [^ ]*/$RUN( |$)" >/dev/null; }

copy_raw() {
  for d in $SRC/checkpoint_step[0-9]*; do
    [[ $d == *.tmp ]] && continue
    n=${d##*checkpoint_step}
    [ -f $d/meta.json ] && [ -f $d/transformer_bf16.pt ] && [ ! -f $C/step${n}_bf16.pt ] || continue
    cp $d/transformer_bf16.pt $C/step${n}_bf16.pt.tmp && mv $C/step${n}_bf16.pt.tmp $C/step${n}_bf16.pt \
      && cp $d/meta.json $C/step${n}_meta.json && echo "[copier] step $n raw copied $(ts)"
  done
}
( while true; do copy_raw; alive || { copy_raw; break; }; sleep 120; done ) &
COPIER=$!

# serve <gpu> <port> <ckpt> <log>: start a policy server, wait until it accepts connections; returns its pid in SPID
serve() {
  CUDA_VISIBLE_DEVICES=$1 .venv/bin/python -u benchmarks/sharpa_webxr/policy_server.py --ckpt $3 --port $2 --num-steps 10 ${PREP_DIR:+--prep-dir $PREP_DIR} > $4 2>&1 &
  SPID=$!
  for _ in $(seq 120); do
    .venv_webxr/bin/python -c "import socket;socket.create_connection(('localhost',$2),2)" 2>/dev/null && return 0
    kill -0 $SPID 2>/dev/null || { echo "[FAIL server] $3 exited, see $4"; SPID=; return 1; }
    sleep 10
  done
  echo "[FAIL server] $3 not up after 20 min"; kill $SPID; SPID=; return 1
}

# evaluate <tag> <ckpt> <gpu> <port> <seeds>: one suite against one server (blocking)
evaluate() {
  local tag=$1 ck=$2 gpu=$3 port=$4 seeds=$5
  [ -f $E/$tag/DONE_$seeds ] && return 0
  echo "[eval] $tag seeds $seeds ckpt $ck gpu $gpu $(ts)"
  serve $gpu $port $ck $L/server_$tag.log || return 1
  local spid=$SPID
  benchmarks/sharpa_webxr/run_suite.sh $tag $port ${seeds%-*} ${seeds#*-} 10 1 > $L/suite_${tag}_$seeds.log 2>&1
  kill $spid; wait $spid 2>/dev/null
  local n; n=$(ls $E/$tag/*/seed_*/signals.json 2>/dev/null | wc -l)
  local fails; fails=$(cat $L/$tag/*.log 2>/dev/null | grep -c '^\[FAIL')
  touch $E/$tag/DONE_$seeds
  echo "[eval done] $tag seeds $seeds: $n rollouts with signals, $fails FAIL lines $(ts)"
}

# eval_pair <tag_prefix> <ema_ckpt|-> <bf16_ckpt|-> <seeds>: EMA and raw suites in parallel
eval_pair() {
  local p=$1 ema=$2 raw=$3 seeds=$4
  local pids=()
  if [ "$ema" != - ]; then evaluate ${p}_ema $ema $GA $PE $seeds & pids+=($!); fi
  if [ "$raw" != - ]; then evaluate ${p}_bf16 $raw $GB $PB $seeds & pids+=($!); fi
  wait "${pids[@]}"
  .venv/bin/python benchmarks/sharpa_webxr/aggregate.py $RUN --tsv $E/summary.tsv | sed 's/^/[summary] /'
}

steps() { { ls -d $SRC/ema_step[0-9]* 2>/dev/null | sed 's/.*ema_step//'; ls $C/step*_bf16.pt 2>/dev/null | sed 's/.*step\([0-9]*\)_bf16.pt/\1/'; } | sort -u; }
pending() {
  for n in $(steps); do
    ema=-; raw=-
    [ -f $SRC/ema_step$n/meta.json ] && [ ! -f $E/step${n}_ema/DONE_$SI ] && ema=$SRC/ema_step$n/transformer_ema.pt
    [ -f $C/step${n}_bf16.pt ] && [ ! -f $E/step${n}_bf16/DONE_$SI ] && raw=$C/step${n}_bf16.pt
    [ $ema != - ] || [ $raw != - ] && { echo "$n $ema $raw"; return; }
  done
}

# videos to render: finished rollouts (signals.json) without video.mp4, seed 0 first, filtered by VIDEO_SEEDS
todo_videos() {
  find $E -name signals.json -path '*/seed_*' 2>/dev/null | while read f; do
    d=${f%/signals.json}; [ -f $d/video.mp4 ] && continue
    k=${d##*seed_}; [ "$VIDEO_SEEDS" = all ] || [[ ",$VIDEO_SEEDS," == *",$k,"* ]] || continue
    echo "$k $d"
  done | sort -n -s -k1,1 | cut -d' ' -f2
}
render_loop() {
  while true; do
    list=$(todo_videos)
    if [ -z "$list" ]; then [ -f $E/.evals_done ] && break; sleep 120; continue; fi
    echo "$list" | head -n 400 | MUJOCO_GL=osmesa LP_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      nice -n 19 xargs -n 4 -P $VIDEO_P .venv_webxr/bin/python benchmarks/sharpa_webxr/render_video.py >> $L/videos.log 2>&1
    echo "[videos] $(find $E -name video.mp4 | wc -l) rendered $(ts)"
  done
}
rm -f $E/.evals_done
RENDER=
[ "$VIDEO" = 1 ] && { render_loop & RENDER=$!; }

echo "[watch $RUN] start $(ts): gpus ema=$GA bf16=$GB, ports $PE/$PB, seeds $SI intermediate, +$SF final, video=$VIDEO seeds=$VIDEO_SEEDS P=$VIDEO_P"
while true; do
  job=$(pending)
  if [ -n "$job" ]; then
    set -- $job
    eval_pair step$1 $2 $3 $SI
    continue
  fi
  if ! alive; then
    wait $COPIER 2>/dev/null; copy_raw
    [ -n "$(pending)" ] && continue
    break
  fi
  sleep 120
done

echo "[watch $RUN] training finished $(ts); final top-ups"
last=$(steps | tail -1)
ema=-; raw=-
[ -f $SRC/ema_step$last/meta.json ] && ema=$SRC/ema_step$last/transformer_ema.pt
[ -f $C/step${last}_bf16.pt ] && raw=$C/step${last}_bf16.pt
eval_pair step$last $ema $raw $SF
if [ -f $SRC/best_val/meta.json ]; then
  b=$(.venv/bin/python -c "import json;print(f\"{json.load(open('$SRC/best_val/meta.json'))['step']:06d}\")")
  mkdir -p $C/best_val_step$b
  for f in transformer_ema.pt transformer_bf16.pt meta.json; do [ -f $C/best_val_step$b/$f ] || cp $SRC/best_val/$f $C/best_val_step$b/$f; done
  echo "[best_val] step $b copied $(ts)"
  for s in $SI $SF; do eval_pair best_val_step$b $C/best_val_step$b/transformer_ema.pt $C/best_val_step$b/transformer_bf16.pt $s; done
fi
touch $E/.evals_done
if [ -n "$RENDER" ]; then wait $RENDER; echo "[videos] all rendered $(ts)"; fi
.venv/bin/python benchmarks/sharpa_webxr/aggregate.py $RUN --tsv $E/summary.tsv | sed 's/^/[final] /'
echo "WATCH_ALL_DONE $(ts)"
