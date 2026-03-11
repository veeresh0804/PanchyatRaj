# create_sample_data.ps1 — Generate dummy sample data for smoke testing.
# Creates a one-tile synthetic GeoTIFF, dummy LAS, and ground-truth polygon.

$ErrorActionPreference = "Stop"
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir

Write-Host "=== Creating sample data for hydra-map smoke test ===" -ForegroundColor Cyan

# Ensure directories exist
$dirs = @("data\raw","data\pointcloud","data\annotations","data\meta","data\dem",
          "data\preprocessed","data\annotations\archive","models","output")
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
}

# Generate a synthetic GeoTIFF using Python
python -c @"
import numpy as np
import os

np.random.seed(42)
h, w = 1024, 1024
img = np.random.randint(20, 80, (3, h, w), dtype=np.uint8)

# Add synthetic building-like rectangles (higher variance)
for _ in range(5):
    x1, y1 = np.random.randint(50, w-200), np.random.randint(50, h-200)
    bw, bh = np.random.randint(30, 100), np.random.randint(30, 100)
    color = np.random.randint(150, 250, 3)
    for c in range(3):
        img[c, y1:y1+bh, x1:x1+bw] = color[c] + np.random.randint(-10, 10, (bh, bw))

# Add road-like lines
for i in range(3):
    y = np.random.randint(100, h-100)
    img[:, y:y+5, :] = 120

try:
    import rasterio
    from rasterio.transform import from_bounds
    transform = from_bounds(82.0, 18.0, 82.01, 18.01, w, h)
    profile = {'driver': 'GTiff', 'dtype': 'uint8', 'width': w, 'height': h,
               'count': 3, 'crs': 'EPSG:4326', 'transform': transform}
    with rasterio.open('data/raw/SAMPLE_VILLAGE_T001.tif', 'w', **profile) as dst:
        dst.write(img)
    print('Created: data/raw/SAMPLE_VILLAGE_T001.tif (GeoTIFF)')
except ImportError:
    np.save('data/raw/SAMPLE_VILLAGE_T001.npy', img.transpose(1, 2, 0))
    print('Created: data/raw/SAMPLE_VILLAGE_T001.npy (numpy fallback)')
"@

if ($LASTEXITCODE -ne 0) { throw "GeoTIFF generation failed" }

# Generate a dummy ground truth mask
python -c @"
import numpy as np
import cv2

mask = np.zeros((1024, 1024), dtype=np.uint8)
np.random.seed(42)
for _ in range(5):
    x1 = np.random.randint(50, 824)
    y1 = np.random.randint(50, 824)
    bw = np.random.randint(30, 100)
    bh = np.random.randint(30, 100)
    mask[y1:y1+bh, x1:x1+bw] = 1
for i in range(3):
    y = np.random.randint(100, 924)
    mask[y:y+5, :] = 2
cv2.imwrite('data/annotations/SAMPLE_VILLAGE_T001_mask.png', mask)
print('Created: data/annotations/SAMPLE_VILLAGE_T001_mask.png')
"@

if ($LASTEXITCODE -ne 0) { throw "Mask generation failed" }

# Generate dummy tile metadata
python -c @"
import json
meta = {
    'crs': 'EPSG:4326',
    'transform': [9.765625e-06, 0.0, 82.0, 0.0, -9.765625e-06, 18.01],
    'acquisition_date': '2025-01-15',
    'sensor': 'UAV-RGB'
}
with open('data/meta/SAMPLE_VILLAGE_T001.json', 'w') as f:
    json.dump(meta, f, indent=2)
print('Created: data/meta/SAMPLE_VILLAGE_T001.json')
"@

# Generate a tiny dummy LAS file (optional)
python -c @"
try:
    import laspy, numpy as np
    np.random.seed(42)
    n = 1000
    header = laspy.LasHeader(point_format=0)
    header.offsets = [82.0, 18.0, 0.0]
    header.scales = [0.0001, 0.0001, 0.01]
    las = laspy.LasData(header)
    las.x = 82.0 + np.random.rand(n) * 0.01
    las.y = 18.0 + np.random.rand(n) * 0.01
    las.z = np.random.rand(n) * 10
    las.write('data/pointcloud/SAMPLE_VILLAGE_T001.las')
    print('Created: data/pointcloud/SAMPLE_VILLAGE_T001.las')
except ImportError:
    print('laspy not installed — skipping LAS generation')
"@

Write-Host ""
Write-Host "=== Sample data creation complete ===" -ForegroundColor Green
Write-Host "Files in data\raw\:"
Get-ChildItem "data\raw\" -ErrorAction SilentlyContinue | Format-Table Name, Length
Write-Host "Files in data\annotations\:"
Get-ChildItem "data\annotations\" -ErrorAction SilentlyContinue | Format-Table Name, Length
