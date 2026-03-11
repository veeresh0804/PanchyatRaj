# run_train_swin.ps1 — Train Swin-UNet segmentation model.
param(
    [string]$Config = "config\config.yaml",
    [int]$Fold = 0
)

$ErrorActionPreference = "Stop"
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir

Write-Host "=== Training Swin-UNet (fold $Fold) ===" -ForegroundColor Cyan
Write-Host "Config: $Config"

python src\train\train_swin.py --config $Config --fold $Fold
if ($LASTEXITCODE -ne 0) { throw "Swin training failed" }

Write-Host "=== Swin Training Complete ===" -ForegroundColor Green
