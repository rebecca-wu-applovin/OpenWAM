#!/usr/bin/env bash
# Modal version of run_dual.sh: same training config, 8xH100 on Modal
# (workspace axon-foundational-research). Details: scripts/modal/README.md
#
#   bash run_modal.sh              # detached: keeps running after the terminal closes
#   bash run_modal.sh --debug      # tiny-model smoke test, 1 GPU, 5 steps
#   bash run_modal.sh training.max_steps=50   # extra Hydra overrides are appended
#   GPU_TYPE=B200 bash run_modal.sh           # 8xB200 instead of 8xH100
#   ACTION_MODE=unify_abs bash run_modal.sh   # action-repr ablation arm (dim set automatically)
set -euo pipefail
cd "$(dirname "$0")"

# wandb naming: one group per ablation sweep, run name = action representation.
WANDB_PROJECT="${WANDB_PROJECT:-1008_wam_training}"
WANDB_GROUP="${WANDB_GROUP:-1012_action_repr_ablation}"
ACTION_MODE="${ACTION_MODE:-fingertips_rel}"   # fingertips_abs | fingertips_rel (dim 48); unify_abs | unify_rel (dim 80)
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${ACTION_MODE}_b64}"   # env vars override all of these
# action/state dim follows the action mode.
case "${ACTION_MODE}" in
    fingertips_abs|fingertips_rel) ACTION_DIM=48 ;;
    unify_abs|unify_rel)           ACTION_DIM=80 ;;
    *) echo "[run_modal] unknown ACTION_MODE=${ACTION_MODE}" >&2; exit 1 ;;
esac
# One launch timestamp (MM_DD_HH_MM_SS) shared by the run dir, Hydra log dir and Modal run name.
LAUNCH_TS="$(date +%m_%d_%H_%M_%S)"
RUN_DIR_NAME="${WANDB_RUN_NAME}_${LAUNCH_TS}"
# Name shown for this run in the Modal UI / `modal app list`.
MODAL_RUN_NAME="${MODAL_RUN_NAME:-${RUN_DIR_NAME//_/-}}"

GPU_TYPE="${GPU_TYPE:-H100}"   # H100 | B200
# Video decode is CPU-bound, but Modal caps one container at 64 cores (request > 64 is rejected).
CPU="${CPU:-64}"
MEMORY_GB="${MEMORY_GB:-512}"
NUM_WORKERS="${NUM_WORKERS:-8}"   # per rank; 8 x 8 = 64 workers on 64 cores
MODAL="${MODAL:-.venv/bin/modal}"
"${MODAL}" profile activate axon >/dev/null   # make sure we launch in the team workspace

if [[ "${1:-}" == "--debug" ]]; then
    export OPENWAM_GPU_TYPE="${GPU_TYPE}" OPENWAM_GPU_COUNT=1
    exec "${MODAL}" run --name "debug-$(date +%m%d-%H%M)" scripts/modal/train.py --debug
fi

# Run layout on the Volume: openwam-runs:/yjw/outputs/<group>/<run_name>_<MM_DD_HH_MM_SS>/
#   checkpoints, vis/, config.yaml, normalization_stats.npy, hydra/<launch time>/ (.hydra + train.log)
RUN_DIR="outputs/${WANDB_GROUP}/${RUN_DIR_NAME}"
RUN_DIR_OVERRIDE="training.output_path=outputs/${WANDB_GROUP} training.run_dir_name=${RUN_DIR_NAME}"

# Resuming: the trainer reuses the given run dir, so its Hydra logs go there too.
for arg in "$@"; do
    if [[ "${arg}" == training.resume_ckpt_path=* ]]; then
        RUN_DIR="${arg#training.resume_ckpt_path=}"
        RUN_DIR_OVERRIDE=""
    fi
done

# Fail fast (before paying for 8 GPUs) if a new run's dir is already taken.
if [[ -n "${RUN_DIR_OVERRIDE}" ]] && "${MODAL}" volume ls openwam-runs "/yjw/${RUN_DIR}" 2>/dev/null | grep -q "/config.yaml$"; then
    echo "[run_modal] openwam-runs:/yjw/${RUN_DIR} already holds a run; relaunch for a new timestamp" >&2
    exit 1
fi

# save_full_states_for_resume: Modal calls cap at 24h, so keep resumable state
# (relaunch with training.resume_ckpt_path=outputs/<group>/<run_name>_<MM_DD_HH_MM_SS>).
OVERRIDES="dataloader=yam_sharpa \
  dataloader.action_mode=${ACTION_MODE} \
  model=dual_system model/video_backbone=wan22_ti2v_5b \
  model.architecture.variant=joint_self_attn model.architecture.attention_mask_mode=mutual \
  model.architecture.action_dim=${ACTION_DIM} model.architecture.state_dim=${ACTION_DIM} \
  project.wandb.project=${WANDB_PROJECT} \
  project.wandb.group=${WANDB_GROUP} \
  project.wandb.run_name=${WANDB_RUN_NAME} \
  training.save_full_states_for_resume=true \
  training.dataset_num_workers=${NUM_WORKERS} \
  ${RUN_DIR_OVERRIDE} \
  hydra.run.dir=${RUN_DIR}/hydra/\${now:%Y-%m-%d_%H-%M-%S} \
  $*"

echo "[run_modal] ${GPU_TYPE}:8, ${CPU} CPU, ${MEMORY_GB} GB, ${NUM_WORKERS} workers/rank  |  Modal run name: ${MODAL_RUN_NAME}  |  run dir: openwam-runs:/yjw/${RUN_DIR}"
# GPU spec goes in via env: Modal fixes it when train.py defines the function.
export OPENWAM_GPU_TYPE="${GPU_TYPE}" OPENWAM_GPU_COUNT=8 OPENWAM_CPU="${CPU}" OPENWAM_MEMORY_GB="${MEMORY_GB}"
exec "${MODAL}" run --detach --name "${MODAL_RUN_NAME}" scripts/modal/train.py --overrides "${OVERRIDES}"
