#!/usr/bin/env bash
# History-length sweep on Modal: Wan2.2-TI2V-5B sparse-history arms with H = 1..5 history frames.
# One 8xH100 run per H, same recipe as run_wan22_ti2v_5b.sh (run 84) except the history length:
# the H most recent of the 1 fps rows t-80..t (H=1 -> [0], H=3 -> [-40,-20,0], H=5 -> run 84's layout),
# H proprio tokens (derived from history_offsets), future rows [4,8,...,32], bs 6/GPU (global 48).
#
#   bash scripts/modal/sparse_history/run_wan22_nhist_sweep.sh                       # H = 1 2 3 4 5
#   NHISTS="1 2" bash scripts/modal/sparse_history/run_wan22_nhist_sweep.sh          # only some arms
#   bash scripts/modal/sparse_history/run_wan22_nhist_sweep.sh training.max_steps=20000   # extra Hydra overrides go to every arm
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root
export BACKBONE=wan22_ti2v_5b ACTION_MODE=fingertips_rel
export GPU_TYPE="${GPU_TYPE:-H100}"   # H100 | H200 (read by run_modal.sh)
export WANDB_GROUP="${WANDB_GROUP:-1020_sparse_wan_nhist}"
NHISTS="${NHISTS:-1 2 3 4}"
# All history rows the sweep picks from, oldest first, 1 s apart at 20 Hz.
ALL_OFFSETS=(-80 -60 -40 -20 0)
LOG_DIR="${LOG_DIR:-outputs/launch_logs/${WANDB_GROUP}}"   # local client output of each arm
mkdir -p "${LOG_DIR}"

for H in ${NHISTS}; do
    if (( H < 1 || H > ${#ALL_OFFSETS[@]} )); then
        echo "[nhist_sweep] H must be 1..${#ALL_OFFSETS[@]}, got ${H}" >&2; exit 1
    fi
    # Keep the H most recent rows, e.g. H=2 -> "[-20,0]".
    OFFSETS="[$(IFS=,; echo "${ALL_OFFSETS[*]: -H}")]"
    echo "[nhist_sweep] H=${H}  history_offsets=${OFFSETS}"
    # Each arm is its own detached Modal run (run_modal.sh picks the next run index). run_modal.sh stays attached to
    # its run's log stream until training ends, so every arm goes to the background with its own log file.
    LOG="${LOG_DIR}/nhist${H}.log"
    echo "[nhist_sweep] H=${H} launching, client log: ${LOG}"
    WANDB_RUN_NAME="sparse_wan22_nhist${H}_b48" bash run_modal.sh \
      dataloader.height=384 dataloader.width=224 \
      "dataloader.history_offsets=${OFFSETS}" 'dataloader.future_offsets=[4,8,12,16,20,24,28,32]' \
      model.architecture.clean_attention=bidirectional \
      training.batch_size=6 training.use_gradient_checkpointing=false \
      "$@" > "${LOG}" 2>&1 &
    sleep 20   # let this arm take its run index and pass run_modal.sh's dir check before the next one starts
done
echo "[nhist_sweep] all arms submitted. The local clients only stream logs: Ctrl-C detaches, the Modal runs keep going."
wait
