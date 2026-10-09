# wandb naming: one group per ablation sweep, run name = action representation.
WANDB_PROJECT=1008_wam_training
WANDB_GROUP=1008_action_repr_ablation
WANDB_RUN_NAME=fingertips_rel
ACTION_MODE=fingertips_rel   # fingertips_abs | fingertips_rel (dim 48); unify_abs | unify_rel (dim 80)
ACTION_DIM=48

NPROC_PER_NODE=8 bash scripts/train.sh dataloader=yam_sharpa \
  dataloader.action_mode=${ACTION_MODE} \
  model=dual_system model/video_backbone=wan22_ti2v_5b \
  model.architecture.variant=joint_self_attn model.architecture.attention_mask_mode=mutual \
  model.architecture.action_dim=${ACTION_DIM} model.architecture.state_dim=${ACTION_DIM} \
  project.wandb.project=${WANDB_PROJECT} \
  project.wandb.group=${WANDB_GROUP} \
  project.wandb.run_name=${WANDB_RUN_NAME}
