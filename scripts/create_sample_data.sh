#!/usr/bin/env bash
# create_sample_data.sh — Generate dummy sample data for smoke testing.
# Creates a one-tile synthetic GeoTIFF, dummy LAS, and ground-truth polygon.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=== Creating sample data for hydra-map smoke test ==="

# Ensure directories exist
mkdir -p data/raw data/pointcloud data/annotations data/meta data/dem

# Generate a synthetic GeoTIFF using Python
python3 -c "
import numpy as np
import struct
import os

# Create a synthetic 1024x1024 RGB image with some 'buildings'
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
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'width': w,
        'height': h,
        'count': 3,
        'crs': 'EPSG:4326',
        'transform': transform,
    }
    with rasterio.open('data/raw/SAMPLE_VILLAGE_T001.tif', 'w', **profile) as dst:
        dst.write(img)
    print('Created: data/raw/SAMPLE_VILLAGE_T001.tif (GeoTIFF)')
except ImportError:
    # Fallback: save as raw numpy for testing
    np.save('data/raw/SAMPLE_VILLAGE_T001.npy', img.transpose(1, 2, 0))
    print('Created: data/raw/SAMPLE_VILLAGE_T001.npy (numpy fallback)')
"

# Generate a dummy ground truth mask
python3 -c "
import numpy as np
import cv2, os

mask = np.zeros((1024, 1024), dtype=np.uint8)
# Building class (1) rectangles
np.random.seed(42)
for _ in range(5):
    x1 = np.random.randint(50, 824)
    y1 = np.random.randint(50, 824)
    bw = np.random.randint(30, 100)
    bh = np.random.randint(30, 100)
    mask[y1:y1+bh, x1:x1+bw] = 1

# Road class (2)
for i in range(3):
    y = np.random.randint(100, 924)
    mask[y:y+5, :] = 2

cv2.imwrite('data/annotations/SAMPLE_VILLAGE_T001_mask.png', mask)
print('Created: data/annotations/SAMPLE_VILLAGE_T001_mask.png')
"

# Generate dummy tile metadata
python3 -c "
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
"

# Generate a tiny dummy LAS file (optional — just for testing)
python3 -c "
try:
    import laspy
    import numpy as np

    np.random.seed(42)
    n = 1000
    header = laspy.LasHeader(point_format=0)
    header.offsets = [82.0, 18.0, 0.0]
    header.scales = [0.0001, 0.0001, 0.01]

    las = laspy.LasData(header)
    las.x = 82.0 + np.random.rand(n) * 0.01
    las.y = 18.0 + np.random.rand(n) * 0.01
    las.z = np.random.rand(n) * 10  # height 0-10m
    las.write('data/pointcloud/SAMPLE_VILLAGE_T001.las')
    print('Created: data/pointcloud/SAMPLE_VILLAGE_T001.las')
except ImportError:
    print('laspy not installed — skipping LAS generation')
"

echo ""
echo "=== Sample data creation complete ==="
echo "Files created:"
ls -la data/raw/ 2>/dev/null || true
ls -la data/annotations/ 2>/dev/null || true
ls -la data/meta/ 2>/dev/null || true
ls -la data/pointcloud/ 2>/dev/null || true
