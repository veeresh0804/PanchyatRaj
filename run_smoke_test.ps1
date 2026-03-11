# run_smoke_test.ps1 — Run the Hydra-Map end-to-end smoke test.
#
# The smoke test uses fully synthetic data (no real GeoTIFFs needed)
# and stub ML models (no large checkpoint downloads needed).
#
# Usage:
#   .\run_smoke_test.ps1              # uses system Python
#   .\run_smoke_test.ps1 -UseVenv     # activates .\venv first

param(
    [switch]$UseVenv
)

$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot
Set-Location $ProjectDir

Write-Host "======================================" -ForegroundColor Cyan
Write-Host " Hydra-Map Smoke Test" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

if ($UseVenv) {
    Write-Host "Activating venv..." -ForegroundColor Yellow
    if (-not (Test-Path ".\venv\Scripts\Activate.ps1")) {
        throw "venv not found. Run .\setup_windows.ps1 first."
    }
    & ".\venv\Scripts\Activate.ps1"
}

Write-Host "Running: python src\tests\smoke_test.py" -ForegroundColor Yellow
Write-Host ""

python src\tests\smoke_test.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Green
    Write-Host " SMOKE TEST PASSED ✓" -ForegroundColor Green
    Write-Host "======================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Red
    Write-Host " SMOKE TEST FAILED ✗" -ForegroundColor Red
    Write-Host "======================================" -ForegroundColor Red
    exit 1
}
