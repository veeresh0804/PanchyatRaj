# setup_windows.ps1 — One-time setup for Hydra-Map on Windows.
# Run this ONCE from the project root to create the venv and install deps.
#
# Usage:
#   .\setup_windows.ps1
#   .\setup_windows.ps1 -SkipInstall   # only create dirs, skip pip install

param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot
Set-Location $ProjectDir

Write-Host "======================================" -ForegroundColor Cyan
Write-Host " Hydra-Map Windows Setup" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# ── 1. Create virtual environment ──────────────────────────────────────────
if (-not (Test-Path "venv")) {
    Write-Host "`n[1/4] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) { throw "python -m venv failed. Is Python 3.8+ installed and on PATH?" }
    Write-Host "      venv created." -ForegroundColor Green
} else {
    Write-Host "`n[1/4] Virtual environment already exists, skipping." -ForegroundColor Green
}

# ── 2. Upgrade pip ─────────────────────────────────────────────────────────
Write-Host "`n[2/4] Upgrading pip..." -ForegroundColor Yellow
& ".\venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
Write-Host "      pip upgraded." -ForegroundColor Green

# ── 3. Install dependencies ────────────────────────────────────────────────
if (-not $SkipInstall) {
    Write-Host "`n[3/4] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
    Write-Host "      (This may take 10-15 minutes on first run)" -ForegroundColor DarkGray
    & ".\venv\Scripts\pip.exe" install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "pip install failed. See error above." }
    Write-Host "      Dependencies installed." -ForegroundColor Green
} else {
    Write-Host "`n[3/4] -SkipInstall flag set — skipping pip install." -ForegroundColor Yellow
}

# ── 4. Create required data directories ────────────────────────────────────
Write-Host "`n[4/4] Creating required directories..." -ForegroundColor Yellow
$dirs = @(
    "data\raw", "data\pointcloud", "data\dem",
    "data\meta", "data\annotations\archive",
    "data\preprocessed", "models", "output"
)
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) {
        New-Item -ItemType Directory -Path $d | Out-Null
        Write-Host "      Created: $d" -ForegroundColor DarkGray
    }
}
Write-Host "      Directories ready." -ForegroundColor Green

# ── Done ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host " Setup complete!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  Activate venv  :  .\venv\Scripts\Activate.ps1"
Write-Host "  Run smoke test :  .\run_smoke_test.ps1"
Write-Host "  Create test data: .\scripts\create_sample_data.ps1"
Write-Host ""
