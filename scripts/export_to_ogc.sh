#!/usr/bin/env bash
# export_to_ogc.sh — Export inference results to GeoPackage and COG.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

RUN_ID="${1:-default}"
CONFIG="${2:-config/config.yaml}"

INPUT_DIR="output/$RUN_ID/tiles"
GPKG_OUT="output/$RUN_ID/final.gpkg"
COG_OUT="output/$RUN_ID/final_mask.tif"

echo "=== OGC Export ==="
echo "Run ID: $RUN_ID"
echo "Input: $INPUT_DIR"
echo "Output GPKG: $GPKG_OUT"
echo "Output COG: $COG_OUT"

python src/export/export_ogc.py \
    --input "$INPUT_DIR" \
    --out "$GPKG_OUT" \
    --cog "$COG_OUT" \
    --config "$CONFIG"

echo ""
echo "=== Export Complete ==="
echo "GeoPackage: $GPKG_OUT"
echo "COG: $COG_OUT"
