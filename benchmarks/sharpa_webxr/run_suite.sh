#!/bin/bash
# Closed-loop sweep of one checkpoint: 58 training scenes x seeds, longest scenes first.
# usage: RUN=<run> [EXECUTE_STEPS=N] run_suite.sh <tag> <port> <seed_first> <seed_last> <workers> [seeds_per_job]
# Each job: rollout.py (closed loop vs the policy server on <port>) then postprocess.py (stage signals, rule
# verifier, 16 judge keyframes; no full video). Resumable: finished seeds (rollout.json / signals.json) are skipped.
set -u
cd "$(dirname "$(readlink -f "$0")")/../.."  # repo root
TAG=$1; PORT=$2; S0=$3; S1=$4; W=$5; PER=${6:-5}
RUN=${RUN:?set RUN=<run name> (results go to .sharpa_sim_eval/<run>/<tag>)}
OUT=.sharpa_sim_eval/$RUN/$TAG
LOG=.sharpa_sim_eval/logs/$RUN/$TAG; mkdir -p $OUT $LOG
until .venv_webxr/bin/python -c "import socket;socket.create_connection(('localhost',$PORT),2)" 2>/dev/null; do sleep 15; done
echo "[suite $TAG] server :$PORT up $(date -u +%FT%TZ)"
JOBS=$(.venv_webxr/bin/python - "$S0" "$S1" "$PER" <<'PY'
import json, sys
s0, s1, per = map(int, sys.argv[1:])
t = json.load(open("benchmarks/sharpa_webxr/tasks.json"))
for sc in sorted(t, key=lambda k: -min(t[k]["max_steps"], 1200)):
    for a in range(s0, s1 + 1, per):
        print(f"{sc} {a}-{min(a + per - 1, s1)}")
PY
)
EXECUTE_STEPS=${EXECUTE_STEPS:-}  # empty: the server's upstream-matched value
export OUT LOG PORT EXECUTE_STEPS MUJOCO_GL=osmesa LP_NUM_THREADS=4
# Each worker exits 1 if its rollout or postprocess failed; xargs then exits nonzero (123) and so does this script.
echo "$JOBS" | xargs -P "$W" -L 1 bash -c '
  sc=$0; seeds=$1; tag=${sc:0:5}_${seeds}; rc=0
  .venv_webxr/bin/python -u benchmarks/sharpa_webxr/rollout.py --port $PORT --scene $sc --seeds $seeds --out $OUT ${EXECUTE_STEPS:+--execute-steps $EXECUTE_STEPS} >> $LOG/$tag.log 2>&1 \
    || { echo "[FAIL rollout] $sc $seeds" | tee -a $LOG/$tag.log; rc=1; }
  a=${seeds%-*}; b=${seeds#*-}; dirs=$(for s in $(seq $a $b); do echo $OUT/$sc/seed_$s; done)
  .venv_webxr/bin/python -u benchmarks/sharpa_webxr/postprocess.py $dirs --no-video --keyframes 16 >> $LOG/$tag.log 2>&1 \
    || { echo "[FAIL post] $sc $seeds" | tee -a $LOG/$tag.log; rc=1; }
  echo "[job done] $sc $seeds rc=$rc $(date -u +%T)"
  exit $rc
'
rc=$?
if [ $rc -ne 0 ]; then
  echo "[suite $TAG] FAILED (xargs exit $rc): see [FAIL ...] lines above and $LOG/ $(date -u +%FT%TZ)"
  exit 1
fi
echo "[suite $TAG] ALL_DONE $(date -u +%FT%TZ)"
