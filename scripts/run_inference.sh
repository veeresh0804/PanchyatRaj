#!/usr/bin/env bash
# run_inference.sh — Run full inference pipeline.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="${1:-config/config.yaml}"
RUN_ID="${2:-run_$(date +%Y%m%d_%H%M%S)}"
VILLAGE="${3:-}"

echo "=== Hydra-Map Inference ==="
echo "Config: $CONFIG"
echo "Run ID: $RUN_ID"
if [ -n "$VILLAGE" ]; then
    echo "Village: $VILLAGE"
fi

EXTRA_ARGS=""
if [ -n "$VILLAGE" ]; then
    EXTRA_ARGS="--village $VILLAGE"
fi

python src/inference/run_inference.py \
    --config "$CONFIG" \
    --run-id "$RUN_ID" \
    $EXTRA_ARGS

echo ""
echo "=== Inference Complete ==="
echo "Outputs: output/$RUN_ID/"
