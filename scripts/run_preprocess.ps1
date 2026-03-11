# run_preprocess.ps1 — Run variance filter and tiling pipeline.
param(
    [string]$Config = "config\config.yaml"
)

$ErrorActionPreference = "Stop"
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir

Write-Host "=== Hydra-Map Preprocessing ===" -ForegroundColor Cyan
Write-Host "Config: $Config"

Write-Host ""
Write-Host "--- Step 1: Variance Filter ---" -ForegroundColor Yellow
python src\preprocess\variance_filter.py --config $Config
if ($LASTEXITCODE -ne 0) { throw "Variance filter failed" }

Write-Host ""
Write-Host "--- Step 2: Tiling ---" -ForegroundColor Yellow
python src\preprocess\tiler.py --config $Config
if ($LASTEXITCODE -ne 0) { throw "Tiling failed" }

Write-Host ""
Write-Host "=== Preprocessing Complete ===" -ForegroundColor Green
