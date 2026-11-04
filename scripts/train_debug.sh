#!/usr/bin/env bash
# DEBUG ONLY: single-process launch of the tiny read-the-code config (configs/train_debug.yaml).
# Plain `python` instead of torchrun, so `breakpoint()` / pdb / an IDE debugger attach normally.
#   bash scripts/train_debug.sh
#   bash scripts/train_debug.sh training.max_steps=1 dataloader.action_mode=unify_rel \
#        common.action_dim=80 common.state_dim=80
set -euo pipefail
cd "$(dirname "$0")/.."

# Fake a 1-rank torchrun environment; accelerate/DeepSpeed read these to init the process group.
export RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 LOCAL_WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1 MASTER_PORT="${MASTER_PORT:-29531}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE=disabled

exec "${PYTHON:-.venv/bin/python}" scripts/train.py --config-name train_debug "$@"
