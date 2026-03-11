#!/usr/bin/env bash
# run_train_fusion.sh — Generate fusion dataset and train the fusion model.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="${1:-config/config.yaml}"
DATASET_PATH="${2:-data/preprocessed/fusion_dataset.jsonl}"

echo "=== Training Fusion Model ==="
echo "Config: $CONFIG"

echo ""
echo "--- Step 1: Generate fusion training dataset ---"
python src/train/train_fusion.py --config "$CONFIG" --generate-dataset --dataset-path "$DATASET_PATH"

echo ""
echo "--- Step 2: Train fusion model ---"
python src/train/train_fusion.py --config "$CONFIG" --dataset-path "$DATASET_PATH"

echo "=== Fusion Training Complete ==="
