#!/bin/bash
# Pipeline: Wait for trajectories → Distillation Training
# Run from packages/ltx-trainer/

set -e

DATA_ROOT="/media/12TB/ddit_ditto_data"
TRAJ_DIR="$DATA_ROOT/trajectories"

echo "=========================================="
echo "  VFM Distillation Pipeline (5K samples)"
echo "  Linear schedule trajectories"
echo "=========================================="

# ── Wait for trajectories to finish ──
echo ""
echo "[Step 1/2] Waiting for trajectories to complete..."
while true; do
    COUNT=$(ls "$TRAJ_DIR"/*.pt 2>/dev/null | wc -l)
    if [ "$COUNT" -ge 5000 ]; then
        echo "  Trajectories complete: $COUNT / 5000"
        break
    fi
    echo "  Progress: $COUNT / 5000 — waiting 60s..."
    sleep 60
done

# ── Symlink latents if needed ──
if [ ! -d "$DATA_ROOT/latents" ] && [ -d "$DATA_ROOT/latents_19b" ]; then
    echo ""
    echo "  Creating latents symlink..."
    ln -s "$DATA_ROOT/latents_19b" "$DATA_ROOT/latents"
fi

# ── Start distillation training ──
echo ""
echo "[Step 2/2] Starting VFM distillation training..."
uv run python scripts/train.py configs/ltx2_vfm_distill_5k.yaml
