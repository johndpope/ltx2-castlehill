#!/bin/bash
# Auto-switch training to LTX-2.3 once download completes
# Run: nohup bash scripts/auto_switch_to_23.sh > /tmp/auto_switch_23.log 2>&1 &

MODEL_FILE="/media/2TB/ltx-models/ltx2.3/ltx-2.3-22b-distilled.safetensors"
EXPECTED_SIZE=46150000000  # ~46.15 GB
CONFIG="configs/ltx2_vfm_v1f_pertoken_5k.yaml"
NEW_CONFIG="configs/ltx2_vfm_v1f_pertoken_5k_23.yaml"
LOG_FILE="/tmp/vfm_v1f_pertoken_5k_23.log"

echo "[$(date)] Waiting for LTX-2.3 download to complete..."
echo "[$(date)] Watching: $MODEL_FILE"

# Wait for file to exist and reach expected size
while true; do
    if [ -f "$MODEL_FILE" ]; then
        SIZE=$(stat -c%s "$MODEL_FILE" 2>/dev/null || echo 0)
        echo "[$(date)] File size: $(numfmt --to=iec $SIZE) / $(numfmt --to=iec $EXPECTED_SIZE)"
        if [ "$SIZE" -gt "$EXPECTED_SIZE" ]; then
            echo "[$(date)] Download complete!"
            break
        fi
    else
        echo "[$(date)] File not yet created..."
    fi
    sleep 60
done

# Wait an extra 30s to ensure file is fully flushed
sleep 30

# Kill current 19B training
echo "[$(date)] Killing current LTX-2 (19B) training..."
pkill -f "train.py.*ltx2_vfm_v1f_pertoken_5k.yaml" 2>/dev/null
sleep 10

# Verify GPUs are free
echo "[$(date)] GPU status:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# Start LTX-2.3 training
echo "[$(date)] Starting LTX-2.3 (22B) training..."
cd /home/johndpope/Documents/GitHub/ltx2-castlehill/packages/ltx-trainer

nohup uv run python scripts/train.py "$NEW_CONFIG" --disable-progress-bars > "$LOG_FILE" 2>&1 &
NEW_PID=$!
echo "[$(date)] Training launched with PID: $NEW_PID"
echo "[$(date)] Log: $LOG_FILE"

# Wait a bit and verify
sleep 120
echo "[$(date)] Training status:"
tail -5 "$LOG_FILE"
echo "[$(date)] Done!"
