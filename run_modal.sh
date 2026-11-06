#!/usr/bin/env bash
# Modal version of run_dual.sh: same training config, 8xH100 on Modal
# (workspace axon-foundational-research). Details: scripts/modal/README.md
#
#   bash run_modal.sh              # detached: keeps running after the terminal closes
#   bash run_modal.sh --debug      # tiny-model smoke test, 1 GPU, 5 steps
#   bash run_modal.sh training.max_steps=50   # extra Hydra overrides are appended
#   GPU_TYPE=H200 bash run_modal.sh           # 8xH200 (or B200) instead of 8xH100
#   ACTION_MODE=unify_abs bash run_modal.sh   # action-repr ablation arm (dim set automatically)
#   BACKBONE=cosmos3_nano bash run_modal.sh   # video-backbone ablation arm: wan22_ti2v_5b | cosmos3_edge | cosmos3_nano
#   bash scripts/modal/sparse_history/run_<backbone>.sh   # 07 sparse-history arms (5 backbones)
#   SIMEVAL=0 bash run_modal.sh   # no sim-policy-eval watch (default: every new run gets one, scripts/modal/simeval_watch.py)
set -euo pipefail
cd "$(dirname "$0")"
source ./base.sh   # next_run_index

# wandb naming: one group per ablation sweep, run name = action representation.
WANDB_PROJECT="${WANDB_PROJECT:-1008_wam_training}"
WANDB_GROUP="${WANDB_GROUP:-1013_action_repr_ablation}"
ACTION_MODE="${ACTION_MODE:-fingertips_rel}"   # fingertips_abs | fingertips_rel (dim 48); unify_abs | unify_rel (dim 80)
BACKBONE="${BACKBONE:-wan22_ti2v_5b}"           # wan22_ti2v_5b | cosmos3_edge | cosmos3_nano | longlive2_robot_s | ge_base_fast | flux3_action_base
# Non-default backbones get a name prefix so arms of one group stay distinguishable.
if [[ "${BACKBONE}" == "wan22_ti2v_5b" ]]; then NAME_PREFIX=""; else NAME_PREFIX="${BACKBONE}_"; fi
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${NAME_PREFIX}${ACTION_MODE}_b64}"   # env vars override all of these
# action/state dim follows the action mode.
case "${ACTION_MODE}" in
    fingertips_abs|fingertips_rel) ACTION_DIM=48 ;;
    unify_abs|unify_rel)           ACTION_DIM=80 ;;
    *) echo "[run_modal] unknown ACTION_MODE=${ACTION_MODE}" >&2; exit 1 ;;
esac
# Backbone -> weights dir on the openwam-assets Volume (mounted at /assets) + per-backbone extras.
# cosmos3_nano: bs 8/GPU OOMs on H100 without gradient checkpointing (verified locally,
# 06_video_backbone_ablation); with it on, bs 8 fits. Math is unchanged, only speed.
BACKBONE_EXTRA=""
case "${BACKBONE}" in
    wan22_ti2v_5b) BACKBONE_DIR=Wan2.2-TI2V-5B ;;
    cosmos3_edge)  BACKBONE_DIR=Cosmos3-Edge ;;
    cosmos3_nano)  BACKBONE_DIR=Cosmos3-Nano; BACKBONE_EXTRA="training.use_gradient_checkpointing=true" ;;
    # 07 sparse-history arms (weights uploaded by scripts/modal/sparse_history/upload_weights.sh).
    # LongLive = the Wan2.2 bundle + Robot-S DiT weights; GE reads VAE / T5 / tokenizer from an LTX-Video dir.
    longlive2_robot_s) BACKBONE_DIR=Wan2.2-TI2V-5B
                       BACKBONE_EXTRA="assets.video_backbone.dit_weights=/assets/video_backbone_ckpt/LongLive2.0-Robot-S/model.pt" ;;
    ge_base_fast)      BACKBONE_DIR=Genie-Envisioner-Base-fast
                       BACKBONE_EXTRA="assets.video_backbone.ltx_path=/assets/video_backbone_ckpt/LTX-Video" ;;
    flux3_action_base) BACKBONE_DIR=flux-3-action-base ;;
    *) echo "[run_modal] unknown BACKBONE=${BACKBONE}" >&2; exit 1 ;;
esac
GPU_TYPE="${GPU_TYPE:-H100}"   # H100 | H200 | B200
# Video decode is CPU-bound, but Modal caps one container at 64 cores (request > 64 is rejected).
CPU="${CPU:-64}"
MEMORY_GB="${MEMORY_GB:-512}"
NUM_WORKERS="${NUM_WORKERS:-8}"   # per rank; 8 x 8 = 64 workers on 64 cores
MODAL="${MODAL:-.venv/bin/modal}"
SIMEVAL="${SIMEVAL:-1}"   # 1: also launch the sim-policy-eval watch for a new run (needs the deployed openwam-yjw-simeval-refactor app)
"${MODAL}" profile activate axon >/dev/null   # make sure we launch in the team workspace

if [[ "${1:-}" == "--debug" ]]; then
    export OPENWAM_GPU_TYPE="${GPU_TYPE}" OPENWAM_GPU_COUNT=1
    exec "${MODAL}" run --name "debug-$(date +%m%d-%H%M)" scripts/modal/train.py --debug
fi

# Run layout on the Volume: openwam-runs:/yjw/outputs/<group>/<index>_<run_name>/
#   checkpoints, vis/, config.yaml, normalization_stats.npy, hydra/<launch time>/ (.hydra + train.log)
# <index>_<run_name> (index from base.sh) is the wandb run name, the run dir name, the wandb run ID
# (set from the dir in train.py) and, with hyphens, the Modal run name.
RESUME_DIR=""
for arg in "$@"; do
    [[ "${arg}" == training.resume_ckpt_path=* ]] && RESUME_DIR="${arg#training.resume_ckpt_path=}"
done
if [[ -z "${RESUME_DIR}" ]]; then
    WANDB_RUN_NAME="$(next_run_index)_${WANDB_RUN_NAME}"
    RUN_DIR="outputs/${WANDB_GROUP}/${WANDB_RUN_NAME}"
    RUN_DIR_OVERRIDE="training.output_path=outputs/${WANDB_GROUP} training.run_dir_name=${WANDB_RUN_NAME}"
else
    # Resuming: the trainer reuses the given run dir (Hydra logs go there too); no new index,
    # the wandb name is that dir's name again.
    RUN_DIR="${RESUME_DIR%/}"
    WANDB_RUN_NAME="${RUN_DIR##*/}"
    RUN_DIR_OVERRIDE=""
fi
# Name shown for this run in the Modal UI / `modal app list`.
MODAL_RUN_NAME="${MODAL_RUN_NAME:-${WANDB_RUN_NAME//_/-}}"

# Fail fast (before paying for 8 GPUs) if a new run's dir is already taken (e.g. the index file was reset).
if [[ -n "${RUN_DIR_OVERRIDE}" ]] && "${MODAL}" volume ls openwam-runs "/yjw/${RUN_DIR}" 2>/dev/null | grep -q "/config.yaml$"; then
    echo "[run_modal] openwam-runs:/yjw/${RUN_DIR} already holds a run; relaunch for a new index" >&2
    exit 1
fi

# save_full_states_for_resume: Modal calls cap at 24h, so keep resumable state
# (relaunch with training.resume_ckpt_path=outputs/<group>/<index>_<run_name>).
OVERRIDES="dataloader=yam_sharpa \
  dataloader.action_mode=${ACTION_MODE} \
  model=dual_system model/video_backbone=${BACKBONE} \
  assets.video_backbone.model_path=/assets/video_backbone_ckpt/${BACKBONE_DIR} \
  model.architecture.variant=joint_self_attn model.architecture.attention_mask_mode=mutual \
  common.action_dim=${ACTION_DIM} common.state_dim=${ACTION_DIM} \
  project.wandb.project=${WANDB_PROJECT} \
  project.wandb.group=${WANDB_GROUP} \
  project.wandb.run_name=${WANDB_RUN_NAME} \
  training.save_full_states_for_resume=true \
  training.dataset_num_workers=${NUM_WORKERS} \
  ${BACKBONE_EXTRA} \
  ${RUN_DIR_OVERRIDE} \
  hydra.run.dir=${RUN_DIR}/hydra/\${now:%Y-%m-%d_%H-%M-%S} \
  $*"

echo "[run_modal] run ${WANDB_RUN_NAME}  |  backbone ${BACKBONE} (/assets/video_backbone_ckpt/${BACKBONE_DIR})  |  ${GPU_TYPE}:8, ${CPU} CPU, ${MEMORY_GB} GB, ${NUM_WORKERS} workers/rank  |  Modal run name: ${MODAL_RUN_NAME}  |  run dir: openwam-runs:/yjw/${RUN_DIR}"
# GPU spec goes in via env: Modal fixes it when train.py defines the function.
export OPENWAM_GPU_TYPE="${GPU_TYPE}" OPENWAM_GPU_COUNT=8 OPENWAM_CPU="${CPU}" OPENWAM_MEMORY_GB="${MEMORY_GB}"
# Sim eval of every checkpoint, started BEFORE the training call: `modal run --detach` keeps streaming the training
# logs until training ends, so nothing after it would run. The watch polls until the trainer has written config.yaml.
# Not on resume: the run's watch from its first launch is still going (it respawns itself until TRAINING_DONE).
if [[ "${SIMEVAL}" == "1" && -n "${RUN_DIR_OVERRIDE}" ]]; then
    echo "[run_modal] sim-policy-eval watch for ${RUN_DIR#outputs/}"
    # A failed eval launch must not look like a failed training launch: warn and print the command to retry.
    "${MODAL}" run scripts/modal/simeval_watch.py::launch --run "${RUN_DIR#outputs/}" \
        || echo "[run_modal] WARNING: sim eval not launched; retry: ${MODAL} run scripts/modal/simeval_watch.py::launch --run ${RUN_DIR#outputs/}" >&2
fi

exec "${MODAL}" run --detach --name "${MODAL_RUN_NAME}" scripts/modal/train.py --overrides "${OVERRIDES}"
