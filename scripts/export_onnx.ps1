# export_onnx.ps1 — Export models to ONNX and quantize for edge deployment.
param(
    [string]$Config = "config\config.yaml"
)

$ErrorActionPreference = "Stop"
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir

Write-Host "=== ONNX Export & Quantization ===" -ForegroundColor Cyan

python -c @"
import torch, os, sys
sys.path.insert(0, '.')
from src.models.swin_unet import SwinUNet
from src.utils.io import load_config

config = load_config('$Config')
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

if config.get('onnx', {}).get('quantize', True):
    try:
        int8_path = SwinUNet.quantize_onnx(onnx_path)
        print(f'Quantized: {int8_path}')
    except Exception as e:
        print(f'Quantization failed: {e}')
"@

if ($LASTEXITCODE -ne 0) { throw "ONNX export failed" }

Write-Host ""
Write-Host "=== ONNX Export Complete ===" -ForegroundColor Green
Write-Host "Check models\swin\ for ONNX files"
