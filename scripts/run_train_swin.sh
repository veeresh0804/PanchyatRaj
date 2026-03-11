#!/usr/bin/env bash
# run_train_swin.sh — Train Swin-UNet segmentation model.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="${1:-config/config.yaml}"
FOLD="${2:-0}"

echo "=== Training Swin-UNet (fold $FOLD) ==="
echo "Config: $CONFIG"

python src/train/train_swin.py --config "$CONFIG" --fold "$FOLD"

echo "=== Swin Training Complete ==="
