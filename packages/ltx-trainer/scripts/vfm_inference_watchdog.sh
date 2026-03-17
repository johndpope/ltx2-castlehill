#!/bin/bash
# =============================================================================
# VFM Inference Watchdog
# Monitors checkpoint dir, runs 1-step inference on new checkpoints (GPU 1)
# Uploads results to W&B
# =============================================================================

CHECKPOINT_DIR="/media/2TB/omnitransfer/output/vfm_v1f_anticollapse_23/checkpoints"
OUTPUT_DIR="/media/2TB/omnitransfer/output/vfm_v1f_anticollapse_23/inference_samples"
MODEL_PATH="/media/2TB/ltx-models/ltx2.3/ltx-2.3-22b-distilled.safetensors"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/home/johndpope/Documents/GitHub/ltx2-castlehill/.venv/bin/python3"
SEED=42

# Pick a few diverse cached embeddings for inference
EMBEDDINGS=(
    "/media/12TB/ddit_ditto_data_23/conditions_final/000000.pt"
    "/media/12TB/ddit_ditto_data_23/conditions_final/000042.pt"
    "/media/12TB/ddit_ditto_data_23/conditions_final/000100.pt"
)

mkdir -p "$OUTPUT_DIR"

# Track which checkpoints we've already processed
PROCESSED_FILE="$OUTPUT_DIR/.processed"
touch "$PROCESSED_FILE"

echo "$(date) | VFM Inference Watchdog started"
echo "  Monitoring: $CHECKPOINT_DIR"
echo "  Embeddings: ${#EMBEDDINGS[@]} prompts"

while true; do
    # Find all noise adapter checkpoints
    for adapter in "$CHECKPOINT_DIR"/noise_adapter_step_*.safetensors; do
        [ -f "$adapter" ] || continue

        # Extract step number
        step=$(echo "$adapter" | grep -oP 'step_\K\d+')

        # Skip if already processed
        if grep -q "^${step}$" "$PROCESSED_FILE" 2>/dev/null; then
            continue
        fi

        # Check matching lora exists
        lora="$CHECKPOINT_DIR/lora_weights_step_${step}.safetensors"
        if [ ! -f "$lora" ]; then
            echo "$(date) | Step $step: waiting for lora weights..."
            continue
        fi

        echo "$(date) | Step $step: Running inference"

        # Run inference for each embedding
        for i in "${!EMBEDDINGS[@]}"; do
            emb="${EMBEDDINGS[$i]}"
            emb_idx=$(printf "%03d" $i)
            outfile="$OUTPUT_DIR/step_${step}_emb_${emb_idx}.mp4"

            if [ -f "$outfile" ]; then
                continue
            fi

            echo "  $(date) | Embedding $emb_idx → $outfile"

            # Use vfm_vanilla_inference.py on GPU 0 (training transformer is only ~3GB there)
            # Training holds GPU 1 (22GB for transformer), GPU 0 has ~21GB free
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
                --output "$outfile" 2>&1 | tail -10

            if [ -f "$outfile" ]; then
                echo "  $(date) | ✓ Saved $outfile ($(du -h "$outfile" | cut -f1))"
            else
                echo "  $(date) | ✗ Failed"
            fi
        done

        # Upload batch to wandb
        echo "$(date) | Step $step: Uploading to W&B"
        CUDA_VISIBLE_DEVICES="" $PYTHON -c "
import wandb, glob, os
videos = sorted(glob.glob('$OUTPUT_DIR/step_${step}_emb_*.mp4'))
if not videos:
    print('No videos to upload')
    exit(0)
run = wandb.init(project='vfm-v1f', name='inference-step-${step}', tags=['inference', 'watchdog', 'step-${step}'])
for v in videos:
    name = os.path.basename(v).replace('.mp4','')
    run.log({name: wandb.Video(v, fps=24, format='mp4')})
    print(f'  Uploaded {name}')
run.finish()
print('Done')
" 2>&1 | grep -v "^wandb:"

        # Mark as processed
        echo "$step" >> "$PROCESSED_FILE"
        echo "$(date) | Step $step: Complete"
    done

    # Wait before checking again
    sleep 60
done
