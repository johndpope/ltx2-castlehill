#!/usr/bin/env bash
# Standalone SCD + X-Cache prolonged-generation sweep.
#
# Usage: bench_scd_xcache.sh <cached_embedding.pt> <output_dir>
#
# Produces baseline + 3 xcache configurations × 3 durations (10s/30s/60s).
# Edit DURATIONS / CONFIGS below to push further (120s, 300s) once stable.

set -euo pipefail

EMBED="${1:?cached embedding path required}"
OUT="${2:?output dir required}"
mkdir -p "$OUT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="python ${SCRIPT_DIR}/scd_xcache_inference.py --distilled --cached-embedding ${EMBED}"

DURATIONS=(10 30 60)

run_one() {
    local label="$1"; shift
    local secs="$1"; shift
    local out="${OUT}/${label}_${secs}s.mp4"
    local stats="${OUT}/${label}_${secs}s.stats.json"
    echo ""
    echo "=== ${label} @ ${secs}s -> ${out} ==="
    local t0=$(date +%s)
    $RUN --num-seconds "$secs" --output "$out" --xcache-stats-json "$stats" "$@" 2>&1 \
        | tee "${OUT}/${label}_${secs}s.log"
    local t1=$(date +%s)
    echo "elapsed_seconds=$((t1 - t0))" | tee -a "${OUT}/${label}_${secs}s.log"
}

for s in "${DURATIONS[@]}"; do
    run_one baseline           "$s"
    run_one xcache_default     "$s" --xcache
    run_one xcache_relaxed     "$s" --xcache --xcache-tau-floor 0.93 --xcache-max-staleness 4
    run_one xcache_aggressive  "$s" --xcache --xcache-tau-floor 0.90 --xcache-max-staleness 6 --xcache-front-anchor 0
done

echo ""
echo "=== Sweep complete. Outputs in: $OUT ==="
ls -la "$OUT"
