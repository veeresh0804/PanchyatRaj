# run_inference.ps1 — Run full inference pipeline.
param(
    [string]$Config  = "config\config.yaml",
    [string]$RunId   = ("run_" + (Get-Date -Format "yyyyMMdd_HHmmss")),
    [string]$Village = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir

Write-Host "=== Hydra-Map Inference ===" -ForegroundColor Cyan
Write-Host "Config : $Config"
Write-Host "Run ID : $RunId"
if ($Village) { Write-Host "Village: $Village" }

$extraArgs = @()
if ($Village) { $extraArgs = @("--village", $Village) }

python src\inference\run_inference.py --config $Config --run-id $RunId @extraArgs
if ($LASTEXITCODE -ne 0) { throw "Inference failed" }

Write-Host ""
Write-Host "=== Inference Complete ===" -ForegroundColor Green
Write-Host "Outputs: output\$RunId\"
