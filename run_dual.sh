#!/usr/bin/env bash
# Local 8-GPU dual_system training (Modal version: run_modal.sh, same config).
#
#   bash run_dual.sh                                  # defaults below
#   ACTION_MODE=unify_abs bash run_dual.sh            # action-repr ablation arm (dim set automatically)
#   BACKBONE=cosmos3_nano bash run_dual.sh            # video-backbone ablation arm: wan22_ti2v_5b | cosmos3_edge | cosmos3_nano
#   bash run_dual.sh training.save_full_states_for_resume=true   # extra Hydra overrides are appended
set -euo pipefail
cd "$(dirname "$0")"
source ./base.sh   # next_run_index

# wandb backend: public wandb.ai, same as run_modal.sh (Modal can't reach the company server
# cbf-wandb.alcfd.com). Overrides base_url in ~/.config/wandb/settings; the api.wandb.ai key
# comes from ~/.netrc (account jiangwei-yu-applovin, same entity as the Modal secret wandb-yjw).
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
# wandb naming: one group per ablation sweep, run name = arm. Env vars override all of these.
WANDB_PROJECT="${WANDB_PROJECT:-1008_wam_training}"
WANDB_GROUP="${WANDB_GROUP:-1008_action_repr_ablation}"
ACTION_MODE="${ACTION_MODE:-fingertips_rel}"   # fingertips_abs | fingertips_rel (dim 48); unify_abs | unify_rel (dim 80)
BACKBONE="${BACKBONE:-wan22_ti2v_5b}"           # wan22_ti2v_5b | cosmos3_edge | cosmos3_nano
# Non-default backbones get a name prefix so arms of one group stay distinguishable.
if [[ "${BACKBONE}" == "wan22_ti2v_5b" ]]; then NAME_PREFIX=""; else NAME_PREFIX="${BACKBONE}_"; fi
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${NAME_PREFIX}${ACTION_MODE}}"
# action/state dim follows the action mode.
case "${ACTION_MODE}" in
    fingertips_abs|fingertips_rel) ACTION_DIM=48 ;;
    unify_abs|unify_rel)           ACTION_DIM=80 ;;
    *) echo "[run_dual] unknown ACTION_MODE=${ACTION_MODE}" >&2; exit 1 ;;
esac
# cosmos3_nano: bs 8/GPU OOMs on 8xH100 without gradient checkpointing (train.yaml has it off);
# with it on, bs 8 fits. Math is unchanged, only speed. Weights path comes from the backbone yaml.
BACKBONE_EXTRA=()
case "${BACKBONE}" in
    wan22_ti2v_5b|cosmos3_edge) ;;
    cosmos3_nano) BACKBONE_EXTRA=(training.use_gradient_checkpointing=true) ;;
    *) echo "[run_dual] unknown BACKBONE=${BACKBONE}" >&2; exit 1 ;;
esac

# <index>_<run_name> (index from base.sh, taken once per launch) is both the wandb run name and the
# run dir outputs/openwam_checkpoints/<index>_<run_name>/. A resume (training.resume_ckpt_path=...)
# reuses its old dir; the trainer ignores run_dir_name then.
WANDB_RUN_NAME="$(next_run_index)_${WANDB_RUN_NAME}"
echo "[run_dual] run ${WANDB_RUN_NAME}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}" bash scripts/train.sh dataloader=yam_sharpa \
  dataloader.action_mode="${ACTION_MODE}" \
  model=dual_system model/video_backbone="${BACKBONE}" \
  model.architecture.variant=joint_self_attn model.architecture.attention_mask_mode=mutual \
  model.architecture.action_dim="${ACTION_DIM}" model.architecture.state_dim="${ACTION_DIM}" \
  project.wandb.project="${WANDB_PROJECT}" \
  project.wandb.group="${WANDB_GROUP}" \
  project.wandb.run_name="${WANDB_RUN_NAME}" \
  training.run_dir_name="${WANDB_RUN_NAME}" \
  "${BACKBONE_EXTRA[@]}" \
  "$@"
