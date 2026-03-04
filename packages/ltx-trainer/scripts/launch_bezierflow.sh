#!/bin/bash
# Wait for GPU to be free, then launch BézierFlow training.
# Polls every 30s until RTX 5090 (nvidia-smi index 1) has < 1GB used.

set -e
cd /home/johndpope/Documents/GitHub/ltx2-omnitransfer/ltx-trainer

echo "Waiting for RTX 5090 to be free..."
while true; do
    USED=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [ "$USED" -lt 1000 ]; then
        echo "GPU free (${USED}MB used). Launching BézierFlow training..."
        break
    fi
    echo "  GPU busy (${USED}MB used). Waiting 30s..."
    sleep 30
done

python scripts/train_bezierflow.py \
    --checkpoint /media/2TB/ltx-models/ltx2/ltx-2-19b-distilled.safetensors \
    --lora-path /media/2TB/omnitransfer/output/scd_distilled_perframe/checkpoints/lora_weights_step_02000.safetensors \
    --data-root /media/2TB/omnitransfer/data/ditto_subset \
    --teacher-steps 30 \
    --student-steps 4 \
    --n-control-points 32 \
    --num-iterations 200 \
    --lr 1e-3 \
    --guidance-scale 4.0 \
    --max-samples 100 \
    --quantization int8-quanto \
    --output /media/2TB/omnitransfer/output/bezierflow/schedule.pt \
    --wandb-project bezierflow-scd
