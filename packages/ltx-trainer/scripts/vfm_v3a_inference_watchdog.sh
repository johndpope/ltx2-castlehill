#!/bin/bash
# =============================================================================
# VFM v3a Inference Watchdog
# Monitors v3a checkpoints, runs 1-step inference on GPU 0 (training VAE side)
# =============================================================================

CHECKPOINT_DIR="/media/2TB/omnitransfer/output/vfm_v3a_dmd_5k/checkpoints"
OUTPUT_DIR="/media/2TB/omnitransfer/output/vfm_v3a_dmd_5k/inference_samples"
MODEL_PATH="/media/2TB/ltx-models/ltx2.3/ltx-2.3-22b-distilled.safetensors"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/home/johndpope/Documents/GitHub/ltx2-castlehill/.venv/bin/python3"
SEED=42

EMBEDDINGS=(
    "/media/12TB/ddit_ditto_data_23/conditions_final/000000.pt"
    "/media/12TB/ddit_ditto_data_23/conditions_final/000042.pt"
    "/media/12TB/ddit_ditto_data_23/conditions_final/000100.pt"
)

mkdir -p "$OUTPUT_DIR"
PROCESSED_FILE="$OUTPUT_DIR/.processed"
touch "$PROCESSED_FILE"

echo "$(date) | v3a Inference Watchdog started"
echo "  Monitoring: $CHECKPOINT_DIR"

while true; do
    for adapter in "$CHECKPOINT_DIR"/noise_adapter_step_*.safetensors; do
        [ -f "$adapter" ] || continue
        step=$(echo "$adapter" | grep -oP 'step_\K\d+')
        grep -q "^${step}$" "$PROCESSED_FILE" 2>/dev/null && continue

        lora="$CHECKPOINT_DIR/lora_weights_step_${step}.safetensors"
        [ ! -f "$lora" ] && continue

        echo "$(date) | Step $step: Running inference"

        for i in "${!EMBEDDINGS[@]}"; do
            emb="${EMBEDDINGS[$i]}"
            emb_idx=$(printf "%03d" $i)
            outfile="$OUTPUT_DIR/step_${step}_emb_${emb_idx}.mp4"
            [ -f "$outfile" ] && continue

            echo "  $(date) | Embedding $emb_idx"

            # Run on GPU 0 (training has VAE here ~1GB, ~23GB free for inference)
            CUDA_VISIBLE_DEVICES=0 $PYTHON "$SCRIPT_DIR/vfm_vanilla_inference.py" \
                --model-path "$MODEL_PATH" \
                --lora-path "$lora" \
                --adapter-path "$adapter" \
                --cached-embedding "$emb" \
                --num-steps 1 \
                --num-frames 9 \
                --width 768 --height 448 \
                --seed $SEED \
                --device cuda:0 \
                --vae-device cuda:0 \
                --quantize int8-quanto \
                --adapter-variant v1b \
                --adapter-hidden-dim 512 \
                --adapter-num-heads 8 \
                --adapter-num-layers 4 \
                --adapter-pos-dim 256 \
                --output "$outfile" 2>&1 | tail -5

            [ -f "$outfile" ] && echo "  ✓ $(du -h "$outfile" | cut -f1)" || echo "  ✗ Failed"
        done

        # Upload to wandb
        echo "$(date) | Step $step: Uploading to W&B"
        CUDA_VISIBLE_DEVICES="" $PYTHON -c "
import wandb, glob, os
videos = sorted(glob.glob('$OUTPUT_DIR/step_${step}_emb_*.mp4'))
if not videos: exit(0)
run = wandb.init(project='vfm-v3a', name='v3a-inference-step-${step}', tags=['inference','v3a','step-${step}'])
for v in videos:
    run.log({os.path.basename(v).replace('.mp4',''): wandb.Video(v, fps=24, format='mp4')})
run.finish()
" 2>&1 | grep -v "^wandb:"

        echo "$step" >> "$PROCESSED_FILE"
        echo "$(date) | Step $step: Done"
    done
    sleep 120
done
