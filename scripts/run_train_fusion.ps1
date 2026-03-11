# run_train_fusion.ps1 — Generate fusion dataset and train the fusion model.
param(
    [string]$Config      = "config\config.yaml",
    [string]$DatasetPath = "data\preprocessed\fusion_dataset.jsonl"
)

$ErrorActionPreference = "Stop"
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir

Write-Host "=== Training Fusion Model ===" -ForegroundColor Cyan
Write-Host "Config: $Config"

Write-Host ""
Write-Host "--- Step 1: Generate fusion training dataset ---" -ForegroundColor Yellow
python src\train\train_fusion.py --config $Config --generate-dataset --dataset-path $DatasetPath
if ($LASTEXITCODE -ne 0) { throw "Dataset generation failed" }

Write-Host ""
Write-Host "--- Step 2: Train fusion model ---" -ForegroundColor Yellow
python src\train\train_fusion.py --config $Config --dataset-path $DatasetPath
if ($LASTEXITCODE -ne 0) { throw "Fusion training failed" }

Write-Host "=== Fusion Training Complete ===" -ForegroundColor Green
