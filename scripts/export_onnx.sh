#!/usr/bin/env bash
# export_onnx.sh — Export models to ONNX and quantize for edge deployment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="${1:-config/config.yaml}"

echo "=== ONNX Export & Quantization ==="

# Export Swin-UNet to ONNX
python3 -c "
import torch, os, sys
sys.path.insert(0, '.')
from src.models.swin_unet import SwinUNet
from src.utils.io import load_config

config = load_config('$CONFIG')
swin_cfg = config['swin']

model = SwinUNet(
    encoder_name=swin_cfg['encoder'],
    pretrained=False,
    num_classes=swin_cfg['num_classes'],
    input_size=swin_cfg['input_size'],
)

# Load checkpoint if available
ckpt = os.path.join(config['data']['models_dir'], 'swin', 'swin_fold0_best.pth')
if os.path.isfile(ckpt):
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    print(f'Loaded checkpoint: {ckpt}')

os.makedirs(os.path.join(config['data']['models_dir'], 'swin'), exist_ok=True)
onnx_path = os.path.join(config['data']['models_dir'], 'swin', 'swin.onnx')
model.export_onnx(onnx_path, input_size=swin_cfg['input_size'])
print(f'ONNX exported: {onnx_path}')

# Quantize if enabled
if config.get('onnx', {}).get('quantize', True):
    try:
        int8_path = SwinUNet.quantize_onnx(onnx_path)
        print(f'Quantized: {int8_path}')
    except Exception as e:
        print(f'Quantization failed: {e}')
"

echo ""
echo "=== ONNX Export Complete ==="
echo "Check models/swin/ for ONNX files"
