#!/usr/bin/env bash
# 07 sparse-history arm on Modal: Cosmos3-Edge.
# 8xH100 by default (GPU_TYPE=H200 for 8xH200), bs 6/GPU (global 48), no gradient checkpointing. Layout as 07_sparse_history_backbones/smoke.sh:
# 5 history frames at 1 fps (rows t-80..t) + future rows [4,8,...,32] (8 frames), 5 proprio tokens, two 192x224 views stacked.
#
#   bash scripts/modal/sparse_history/run_cosmos3_edge.sh                    # extra Hydra overrides are appended
#   GPU_TYPE=H200 bash scripts/modal/sparse_history/run_cosmos3_edge.sh   # H100 (default) | H200
#   WANDB_RUN_NAME=my_name bash scripts/modal/sparse_history/run_cosmos3_edge.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root
export BACKBONE=cosmos3_edge ACTION_MODE=fingertips_rel
export GPU_TYPE="${GPU_TYPE:-H100}"   # H100 | H200 (read by run_modal.sh)
export WANDB_GROUP="${WANDB_GROUP:-1016_sparse_history_backbones}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-sparse_cosmos3_edge_b48}"
exec bash run_modal.sh \
  dataloader.height=384 dataloader.width=224 \
  'dataloader.history_offsets=[-80,-60,-40,-20,0]' 'dataloader.future_offsets=[4,8,12,16,20,24,28,32]' \
  model.architecture.clean_attention=bidirectional \
  training.batch_size=6 training.use_gradient_checkpointing=false \
  "$@"
