#!/usr/bin/env bash
# History-length sweep on Modal: Wan2.2-TI2V-5B sparse-history arms with H = 1..6 history frames.
# One 8xH100 run per H, same recipe as run_wan22_ti2v_5b.sh (run 84) except the history length:
# the H most recent of the 1 fps rows t-100..t (H=1 -> [0], H=3 -> [-40,-20,0], H=5 -> run 84's layout),
# H proprio tokens (derived from history_offsets), future rows [4,8,...,32], bs BS/GPU (default 6, global 48;
# run name suffix b<8*BS>).
#
#   bash scripts/modal/sparse_history/run_wan22_nhist_sweep.sh                       # H = 1 2 3 4
#   NHISTS="1 2" bash scripts/modal/sparse_history/run_wan22_nhist_sweep.sh          # only some arms
#   bash scripts/modal/sparse_history/run_wan22_nhist_sweep.sh training.max_steps=20000   # extra Hydra overrides go to every arm
#   BS=12 NHISTS="1 2" GPU_TYPE=H200 bash scripts/modal/sparse_history/run_wan22_nhist_sweep.sh   # global 96, suffix _b96
#   EPOCHS=5 bash scripts/modal/sparse_history/run_wan22_nhist_sweep.sh              # epochs per arm (default: train.yaml's)
#   HIST_SAMPLING=random bash scripts/modal/sparse_history/run_wan22_nhist_sweep.sh # train history = H-1 random rows of the
#                         past HIST_WINDOW rows (default 100 = 5 s) + frame t; val / deploy keep the 1 fps offsets; suffix _rand
#   HIST_ENCODING=clip NHISTS=6 bash scripts/modal/sparse_history/run_wan22_nhist_sweep.sh   # the 5 past frames VAE-encoded
#                         as one causal 1+4 clip (2 latents) + frame t alone (1): 3 history latents, 6 proprio tokens;
#                         H must be 6 (rows t-100..t); suffix _clip
#   RUN_TAG=_simdata1 NHISTS=1 bash scripts/modal/sparse_history/run_wan22_nhist_sweep.sh dataloader.sources.gpt_rollout.weight=0 \
#       dataloader.sources.simteleop.weight=1   # RUN_TAG: appended to every arm's run name / log (e.g. a data-mix arm)
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root
export BACKBONE=wan22_ti2v_5b ACTION_MODE=fingertips_rel
export GPU_TYPE="${GPU_TYPE:-H100}"   # H100 | H200 (read by run_modal.sh)
export WANDB_GROUP="${WANDB_GROUP:-1020_sparse_wan_nhist}"
NHISTS="${NHISTS:-1 2 3 4}"
BS="${BS:-6}"   # per-GPU batch size
GLOBAL_BS=$(( 8 * BS ))
EPOCHS="${EPOCHS:-}"   # training.num_epochs of every arm; empty = train.yaml's (an env var: Hydra rejects a repeated key)
EPOCH_OVERRIDE=$([[ -n "${EPOCHS}" ]] && echo "training.num_epochs=${EPOCHS}" || echo "")
HIST_SAMPLING="${HIST_SAMPLING:-fixed}"     # fixed | random (dataloader.history_sampling)
HIST_WINDOW="${HIST_WINDOW:-100}"           # random mode: rows back the history may reach (20 Hz)
HIST_ENCODING="${HIST_ENCODING:-per_frame}" # per_frame | clip (model.architecture.history_encoding)
RUN_TAG="${RUN_TAG:-}"                      # free-form suffix of the run name / log, e.g. _simdata1
# Run name / log suffix of the non-default history arms.
HIST_TAG=""
if [[ "${HIST_ENCODING}" == clip ]]; then HIST_TAG="_clip"; fi
if [[ "${HIST_SAMPLING}" == random ]]; then HIST_TAG="${HIST_TAG}_rand"; fi
# All history rows the sweep picks from, oldest first, 1 s apart at 20 Hz.
ALL_OFFSETS=(-100 -80 -60 -40 -20 0)
LOG_DIR="${LOG_DIR:-outputs/launch_logs/${WANDB_GROUP}}"   # local client output of each arm
mkdir -p "${LOG_DIR}"

for H in ${NHISTS}; do
    if (( H < 1 || H > ${#ALL_OFFSETS[@]} )); then
        echo "[nhist_sweep] H must be 1..${#ALL_OFFSETS[@]}, got ${H}" >&2; exit 1
    fi
    if [[ "${HIST_ENCODING}" == clip ]] && (( H != 6 )); then
        echo "[nhist_sweep] HIST_ENCODING=clip needs H=6 (5 past rows = 1+4, + frame t), got ${H}" >&2; exit 1
    fi
    # Keep the H most recent rows, e.g. H=2 -> "[-20,0]".
    OFFSETS="[$(IFS=,; echo "${ALL_OFFSETS[*]: -H}")]"
    NAME="sparse_wan22_nhist${H}${HIST_TAG}_b${GLOBAL_BS}${RUN_TAG}"
    echo "[nhist_sweep] H=${H}  history_offsets=${OFFSETS}  run ${NAME}"
    # Each arm is its own detached Modal run (run_modal.sh picks the next run index). run_modal.sh stays attached to
    # its run's log stream until training ends, so every arm goes to the background with its own log file.
    LOG="${LOG_DIR}/nhist${H}${HIST_TAG}_b${GLOBAL_BS}${RUN_TAG}.log"
    echo "[nhist_sweep] H=${H} launching, client log: ${LOG}"
    WANDB_RUN_NAME="${NAME}" bash run_modal.sh \
      dataloader.height=384 dataloader.width=224 \
      "dataloader.history_offsets=${OFFSETS}" 'dataloader.future_offsets=[4,8,12,16,20,24,28,32]' \
      "dataloader.history_sampling=${HIST_SAMPLING}" "dataloader.history_window=${HIST_WINDOW}" \
      "model.architecture.history_encoding=${HIST_ENCODING}" \
      model.architecture.clean_attention=bidirectional \
      training.batch_size=${BS} ${EPOCH_OVERRIDE} training.use_gradient_checkpointing=false \
      "$@" > "${LOG}" 2>&1 &
    sleep 20   # let this arm take its run index and pass run_modal.sh's dir check before the next one starts
done
echo "[nhist_sweep] all arms submitted. The local clients only stream logs: Ctrl-C detaches, the Modal runs keep going."
wait
