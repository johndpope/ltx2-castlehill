#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python scripts/vfm_vanilla_inference.py \
    --model-path /media/2TB/ltx-models/ltx2.3/ltx-2.3-22b-distilled.safetensors \
    --lora-path /media/2TB/omnitransfer/output/vfm_v4a_full/checkpoints/lora_weights_step_08000.safetensors \
    --adapter-path /media/2TB/omnitransfer/output/vfm_v4a_full/checkpoints/noise_adapter_step_08000.safetensors \
    --cached-embedding /media/12TB/ddit_ditto_data_23_overfit10/conditions_final/000000.pt \
    --width 768 --height 448 --num-frames 25 --fps 24 \
    --adapter-variant v1b \
    --output /media/2TB/omnitransfer/inference/vfm_v4a_full_step8000.mp4 \
    --device cuda:0 --vae-device cuda:1
