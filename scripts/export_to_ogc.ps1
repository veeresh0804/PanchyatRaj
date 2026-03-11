# export_to_ogc.ps1 — Export inference results to GeoPackage and COG.
param(
    [string]$RunId  = "default",
    [string]$Config = "config\config.yaml"
)

$ErrorActionPreference = "Stop"
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir

$InputDir = "output\$RunId\tiles"
$GpkgOut  = "output\$RunId\final.gpkg"
$CogOut   = "output\$RunId\final_mask.tif"

Write-Host "=== OGC Export ===" -ForegroundColor Cyan
Write-Host "Run ID       : $RunId"
Write-Host "Input        : $InputDir"
Write-Host "Output GPKG  : $GpkgOut"
Write-Host "Output COG   : $CogOut"

python src\export\export_ogc.py `
    --input  $InputDir `
    --out    $GpkgOut  `
    --cog    $CogOut   `
    --config $Config

if ($LASTEXITCODE -ne 0) { throw "OGC export failed" }

Write-Host ""
Write-Host "=== Export Complete ===" -ForegroundColor Green
Write-Host "GeoPackage : $GpkgOut"
Write-Host "COG        : $CogOut"
