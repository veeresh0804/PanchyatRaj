import os
import cv2
import numpy as np
import argparse
from typing import Dict
import matplotlib.pyplot as plt

def analyze_dataset(ann_dir: str) -> None:
    """Analyze pixel distribution of ground truth classes."""
    classes = {
        0: {"name": "Background", "count": 0},
        1: {"name": "Building", "count": 0},
        2: {"name": "Road", "count": 0},
        3: {"name": "Vegetation", "count": 0},
        4: {"name": "Water", "count": 0},
        5: {"name": "Other", "count": 0}
    }
    
    mask_files = [f for f in os.listdir(ann_dir) if f.endswith(".png")]
    print(f"Analyzing {len(mask_files)} masks in {ann_dir}...")
    
    total_pixels = 0
    
    for f in mask_files:
        mask = cv2.imread(os.path.join(ann_dir, f), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
            
        unique, counts = np.unique(mask, return_counts=True)
        for val, count in zip(unique, counts):
            if val in classes:
                classes[val]["count"] += count
                total_pixels += count
                
    print("\n=== Dataset Distribution ===")
    for k, v in classes.items():
        pct = (v['count'] / max(total_pixels, 1)) * 100
        print(f"Class {k} ({v['name']:<12}): {v['count']:>12} pixels ({pct:.2f}%)")
        
    # Calculate inverse frequency weights
    print("\n=== Suggested Loss Weights ===")
    counts = np.array([v["count"] for k, v in classes.items() if k > 0])
    if counts.sum() > 0:
        freq = counts / counts.sum()
        # Cap to avoid extreme weights for rare classes
        weights = 1.0 / (np.log1p(freq * 100) + 0.1)
        # Normalize so min weight is ~1.0
        weights = weights / weights.min()
        
        for k, w in zip(range(1, 6), weights):
            print(f"Class {k} ({classes[k]['name']:<12}): {w:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_dir", default="data/annotations", help="Annotations directory")
    args = parser.parse_args()
    analyze_dataset(args.ann_dir)
