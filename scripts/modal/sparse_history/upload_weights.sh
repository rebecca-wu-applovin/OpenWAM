#!/usr/bin/env bash
# One-time upload of the 07 backbone weights the openwam-assets Volume does not have yet
# (Wan2.2-TI2V-5B, Cosmos3-Edge and Genie-Envisioner-Base-fast are already there).
#   LongLive2.0-Robot-S/model.pt   19 GB  (DiT weights; VAE / umT5 come from Wan2.2-TI2V-5B)
#   flux-3-action-base/            24 GB  (trunk + video VAE + Qwen3-VL-4B text encoder)
#   LTX-Video/                            (GE's VAE / T5 / tokenizer in the one-dir layout ge_base_fast.yaml expects)
#
#   bash scripts/modal/sparse_history/upload_weights.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root
MODAL="${MODAL:-.venv/bin/modal}"
"${MODAL}" profile activate axon >/dev/null
SRC=assets/video_backbone_ckpt
"${MODAL}" volume put openwam-assets "${SRC}/LongLive2.0-Robot-S/model.pt" /video_backbone_ckpt/LongLive2.0-Robot-S/model.pt
for d in flux-3-action-base LTX-Video; do
    "${MODAL}" volume put openwam-assets "${SRC}/${d}" "/video_backbone_ckpt/${d}"
done
"${MODAL}" volume ls openwam-assets /video_backbone_ckpt
