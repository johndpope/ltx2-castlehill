#!/bin/bash
# Watch for a checkpoint at a specific step and run microbenchmark when it appears.
# Usage: ./scripts/bench_at_step.sh <output_dir> <step> [bench_args...]
# Example: ./scripts/bench_at_step.sh /media/2TB/omnitransfer/output/vfm_v1f_anticollapse_23 1000

OUTPUT_DIR="${1:?Usage: bench_at_step.sh <output_dir> <step>}"
STEP="${2:?Usage: bench_at_step.sh <output_dir> <step>}"
STEP_PADDED=$(printf "%05d" "$STEP")

ADAPTER="$OUTPUT_DIR/checkpoints/noise_adapter_step_${STEP_PADDED}.safetensors"
LORA="$OUTPUT_DIR/checkpoints/lora_weights_step_${STEP_PADDED}.safetensors"
BENCH_DIR="$OUTPUT_DIR/bench_step_${STEP_PADDED}"

echo "Watching for checkpoint at step $STEP..."
echo "  Adapter: $ADAPTER"
echo "  LoRA:    $LORA"
echo "  Bench:   $BENCH_DIR"

# Poll every 30s
while [ ! -f "$ADAPTER" ]; do
    sleep 30
    echo -n "."
done
echo ""
echo "Checkpoint found! Waiting 10s for writes to finish..."
sleep 10

echo "Running microbenchmark..."
mkdir -p "$BENCH_DIR"

cd "$(dirname "$0")/.."
uv run python scripts/vfm_microbench.py \
    --adapter-path "$ADAPTER" \
    --lora-path "$LORA" \
    --data-root /media/12TB/ddit_ditto_data_23 \
    --num-samples 5 \
    --device cuda:0 \
    --vae-device cuda:0 \
    --output-dir "$BENCH_DIR" \
    2>&1 | tee "$BENCH_DIR/bench.log"

echo ""
echo "Benchmark complete! Videos at $BENCH_DIR"
echo "Check manifest: $BENCH_DIR/manifest.json"
