#!/usr/bin/env bash
# run_preprocess.sh — Run variance filter and tiling pipeline.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="${1:-config/config.yaml}"

echo "=== Hydra-Map Preprocessing ==="
echo "Config: $CONFIG"

echo ""
echo "--- Step 1: Variance Filter ---"
python src/preprocess/variance_filter.py --config "$CONFIG"

echo ""
echo "--- Step 2: Tiling ---"
python src/preprocess/tiler.py --config "$CONFIG"

echo ""
echo "=== Preprocessing Complete ==="
