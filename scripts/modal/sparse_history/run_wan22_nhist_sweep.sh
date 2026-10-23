#!/usr/bin/env bash
# History-length sweep on Modal: Wan2.2-TI2V-5B sparse-history arms with H = 1..6 history frames.
# One 8xH100 run per H, same recipe as run_wan22_ti2v_5b.sh (run 84) except the history length:
# the H most recent of the 1 fps rows t-100..t (H=1 -> [0], H=3 -> [-40,-20,0], H=5 -> run 84's layout),
# H proprio tokens, future rows [5,10,15,20,25], bs BS/GPU (default 6, global 48; run name suffix b<8*BS>).
#
#   bash scripts/modal/sparse_history/run_wan22_nhist_sweep.sh                       # H = 1 2 3 4 5
#   NHISTS="1 2" bash scripts/modal/sparse_history/run_wan22_nhist_sweep.sh          # only some arms
#   bash scripts/modal/sparse_history/run_wan22_nhist_sweep.sh training.max_steps=20000   # extra Hydra overrides go to every arm
#   BS=12 NHISTS="1 2" GPU_TYPE=H200 bash scripts/modal/sparse_history/run_wan22_nhist_sweep.sh   # data-amount arms (global 96)
#   EPOCHS=2 bash scripts/modal/sparse_history/run_wan22_nhist_sweep.sh             # epochs per arm (default 5)
#   NODES=2 NHISTS=1 bash scripts/modal/sparse_history/run_wan22_nhist_sweep.sh     # 2 x 8 GPUs per arm (global 16*BS), suffix _n2
#   HIST_SAMPLING=random bash scripts/modal/sparse_history/run_wan22_nhist_sweep.sh # train history = H-1 random rows of the
#                         past HIST_WINDOW rows (default 100 = 5 s) + frame t; val / deploy keep the 1 fps offsets; suffix _rand
#   RUN_TAG=_simdata1 NHISTS=1 bash scripts/modal/sparse_history/run_wan22_nhist_sweep.sh dataloader.sources.gpt_rollout.weight=0 \
#       dataloader.sources.simteleop.weight=1   # RUN_TAG: appended to every arm's run name / log (e.g. a data-mix arm)
#   HIST_ENCODING=clip NHISTS=6 HIST_SAMPLING=random bash scripts/modal/sparse_history/run_wan22_nhist_sweep.sh   # the 5 past
#                         frames VAE-encoded as one causal 1+4 clip (2 latents) + frame t alone (1): 3 history latents,
#                         6 proprio tokens; H must be 6 (rows t-100..t); suffix _clip
#   HIST_LAYOUT=panel NHISTS=5 bash scripts/modal/sparse_history/run_wan22_nhist_sweep.sh   # the 4 past frames tiled into one
#                         half-resolution 2x2 panel + frame t (2 history latents, 5 proprio tokens); H must be 5; suffix _panel
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root
export BACKBONE=wan22_ti2v_5b ACTION_MODE=fingertips_rel
export GPU_TYPE="${GPU_TYPE:-H100}"   # H100 | H200 (read by run_modal.sh)
export WANDB_GROUP="${WANDB_GROUP:-1022_wan_nhist_fix_t5}"
NHISTS="${NHISTS:-1 2 3 4}"
BS="${BS:-16}"   # per-GPU batch size
export NODES="${NODES:-1}"   # 8-GPU nodes per arm (read by run_modal.sh)
GLOBAL_BS=$(( 8 * NODES * BS ))
NODE_TAG=$([[ "${NODES}" == 1 ]] && echo "" || echo "_n${NODES}")   # run name / log suffix of a multi-node arm
HIST_SAMPLING="${HIST_SAMPLING:-fixed}"   # fixed | random (dataloader.history_sampling)
HIST_WINDOW="${HIST_WINDOW:-100}"         # random mode: rows back the history may reach (20 Hz)
HIST_TAG=$([[ "${HIST_SAMPLING}" == fixed ]] && echo "" || echo "_rand")   # run name / log suffix of a random-history arm
HIST_LAYOUT="${HIST_LAYOUT:-frames}"     # frames | panel (dataloader.history_layout)
HIST_ENCODING="${HIST_ENCODING:-frames}"   # frames | clip (model.architecture.history_encoding, Wan only)
if [[ "${HIST_ENCODING}" == clip ]]; then HIST_TAG="_clip${HIST_TAG}"; fi
RUN_TAG="${RUN_TAG:-}"                   # free-form suffix of the run name / log, e.g. _simdata1
if [[ "${HIST_LAYOUT}" == panel ]]; then HIST_TAG="_panel${HIST_TAG}"; fi
EPOCHS="${EPOCHS:-5}"   # training.num_epochs of every arm (train.yaml default: 2); an env var, Hydra rejects a repeated key
# All history rows the sweep picks from, oldest first, 1 s apart at 20 Hz.
ALL_OFFSETS=(-100 -80 -60 -40 -20 0)
LOG_DIR="${LOG_DIR:-outputs/launch_logs/${WANDB_GROUP}}"   # local client output of each arm
mkdir -p "${LOG_DIR}"

for H in ${NHISTS}; do
    if (( H < 1 || H > ${#ALL_OFFSETS[@]} )); then
        echo "[nhist_sweep] H must be 1..${#ALL_OFFSETS[@]}, got ${H}" >&2; exit 1
    fi
    # Keep the H most recent rows, e.g. H=2 -> "[-20,0]".
    OFFSETS="[$(IFS=,; echo "${ALL_OFFSETS[*]: -H}")]"
    MODEL_H=${H}   # history frames the model sees: one per row, or (panel, frame t)
    if [[ "${HIST_LAYOUT}" == panel ]]; then
        if (( H != 5 )); then echo "[nhist_sweep] HIST_LAYOUT=panel needs H=5 (4 past rows + frame t), got ${H}" >&2; exit 1; fi
        MODEL_H=2
    fi
    if [[ "${HIST_ENCODING}" == clip ]]; then
        if (( H != 6 )); then echo "[nhist_sweep] HIST_ENCODING=clip needs H=6 (5 past rows = 1+4 + frame t), got ${H}" >&2; exit 1; fi
        MODEL_H=3   # 1 + (5-1)/4 past-clip latents + 1 for frame t
    fi
    echo "[nhist_sweep] H=${H}  history_offsets=${OFFSETS}"
    # Each arm is its own detached Modal run (run_modal.sh picks the next run index). run_modal.sh stays attached to
    # its run's log stream until training ends, so every arm goes to the background with its own log file.
    LOG="${LOG_DIR}/nhist${H}${HIST_TAG}_b${GLOBAL_BS}${NODE_TAG}${RUN_TAG}.log"
    echo "[nhist_sweep] H=${H} launching, client log: ${LOG}"
    WANDB_RUN_NAME="sparse_wan22_nhist${H}${HIST_TAG}_b${GLOBAL_BS}${NODE_TAG}${RUN_TAG}" bash run_modal.sh \
      dataloader.height=384 dataloader.width=224 \
      "+dataloader.history_offsets=${OFFSETS}" '+dataloader.future_offsets=[5,10,15,20,25]' +dataloader.min_rows_ahead=45 \
      "+dataloader.history_sampling=${HIST_SAMPLING}" "+dataloader.history_window=${HIST_WINDOW}" \
      "+dataloader.history_layout=${HIST_LAYOUT}" "+model.architecture.history_encoding=${HIST_ENCODING}" \
      "+model.architecture.num_history_frames=${MODEL_H}" "+model.architecture.num_proprio_tokens=${H}" \
      model.architecture.video_attention_mask_mode=first_frame_causal \
      training.batch_size=${BS} training.num_epochs=${EPOCHS} training.use_gradient_checkpointing=false \
      "$@" > "${LOG}" 2>&1 &
    sleep 20   # let this arm take its run index and pass run_modal.sh's dir check before the next one starts
done
echo "[nhist_sweep] all arms submitted. The local clients only stream logs: Ctrl-C detaches, the Modal runs keep going."
wait
