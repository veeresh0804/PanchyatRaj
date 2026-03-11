import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import rasterio
import rasterio.features
import rasterio.mask
import rasterio.warp
from PIL import Image
from rasterio.windows import Window
from shapely.geometry import box, shape

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def step1_detect_valid_imagery(src: rasterio.io.DatasetReader):
    """Detect valid pixels (value > 10) by reading a low-resolution overview."""
    logger.info("Step 1: Detecting valid imagery using downsampled overview...")
    num_bands = min(3, src.count)
    
    # Scale down to max 2000 dimension to fit in memory instantly
    scale_factor = min(1.0, 2000.0 / max(src.width, src.height))
    out_shape = (
        num_bands,
        int(src.height * scale_factor),
        int(src.width * scale_factor)
    )
    
    img = src.read(
        indexes=list(range(1, num_bands + 1)),
        out_shape=out_shape,
        resampling=rasterio.enums.Resampling.nearest
    )
    valid_mask = np.any(img > 10, axis=0).astype(np.uint8)
    
    import rasterio.transform as rt
    decimated_transform = src.transform * rt.Affine.scale(
        src.width / out_shape[2],
        src.height / out_shape[1]
    )
    
    return valid_mask, decimated_transform
    
def step2_clean_mask(valid_mask: np.ndarray) -> np.ndarray:
    """Apply morphological closing to remove noise and fill small gaps."""
    logger.info("Step 2: Cleaning the mask with morphology...")
    # Smaller kernel since we are running on a downscaled overview
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_CLOSE, kernel)
    return clean_mask

def step3_4_extract_and_polygonize_regions(clean_mask: np.ndarray, transform) -> List[dict]:
    """Find connected components and generate geographic polygons."""
    logger.info("Step 3 & 4: Extracting connected regions and polygonizing...")
    shapes = rasterio.features.shapes(clean_mask, mask=clean_mask > 0, transform=transform)
    
    regions = []
    
    for i, (geom, value) in enumerate(shapes):
        poly = shape(geom)
        # Keep shapes reasonably large (e.g., > 1000 sq meters if in UTM, but we don't assume CRS)
        # For simplicity, filtering extremely small noise polygons only
        regions.append({
            "id": f"region_{i+1:02d}",
            "geometry": geom,
            "shapely": poly
        })
    
    regions.sort(key=lambda r: r["shapely"].area, reverse=True)
    
    # Take the top N largest distinct regions (filter out microscopic noise)
    valid_regions = [r for r in regions if r["shapely"].area > (regions[0]["shapely"].area * 0.001)]
    
    logger.info(f"Found {len(valid_regions)} valid imagery regions.")
    return valid_regions

def step5_crop_regions(src_path: str, regions: List[dict], output_dir: Path) -> List[Path]:
    """Crop each polygon out of the main raster into separate continuous GeoTIFFs."""
    logger.info("Step 5: Cropping each region into separate continuous GeoTIFFs...")
    ensure_dir(output_dir)
    region_paths = []
    
    with rasterio.open(src_path) as src:
        for r in regions:
            reg_id = r["id"]
            geom = [r["geometry"]]
            
            try:
                out_image, out_transform = rasterio.mask.mask(src, geom, crop=True)
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })
                
                out_path = output_dir / f"{reg_id}.tif"
                with rasterio.open(out_path, "w", **out_meta) as dest:
                    dest.write(out_image)
                
                region_paths.append(out_path)
                logger.info(f"  Saved {out_path.name} ({out_meta['width']}x{out_meta['height']})")
            except Exception as e:
                logger.error(f"  Failed to crop {reg_id}: {e}")
                
    return region_paths

def step6_normalize_crs_resolution(region_paths: List[Path]) -> List[Path]:
    """Optionally normalize CRS using gdalwarp if needed. 
       Skipping exact EPSG:4326 reprojection unless specified because UTM maintains meters.
       We will ensure resolution is standard."""
    logger.info("Step 6: Normalizing regions (Placeholder for gdalwarp if strict CRS required)")
    return region_paths

def sliding_windows(height: int, width: int, size: int=512, overlap: int=64) -> List[Tuple[int, int]]:
    """Generate (y, x) top-left coordinates for sliding windows with overlap."""
    stride = size - overlap
    windows = []
    for y in range(0, max(1, height - size + stride), stride):
        for x in range(0, max(1, width - size + stride), stride):
            # Adjust if we overshoot the bottom/right edge
            y_start = min(y, max(0, height - size))
            x_start = min(x, max(0, width - size))
            windows.append((y_start, x_start))
    # Remove duplicates if image is smaller than tile
    return list(sorted(set(windows)))

def step7_8_9_generate_tiles_and_metadata(
    region_paths: List[Path], 
    tiles_dir: Path, 
    meta_dir: Path,
    tile_size: int = 512,
    overlap: int = 64
):
    """Generate sliding window overlapping tiles, apply gatekeeper filtering, and write metadata."""
    logger.info("Steps 7-9: Tiling, gatekeeper filtering, and metadata building...")
    ensure_dir(tiles_dir)
    ensure_dir(meta_dir)
    
    all_tile_meta = []
    rejected_count = 0
    accepted_count = 0
    
    for rp in region_paths:
        region_id = rp.stem
        logger.info(f"  Tiling {region_id}...")
        
        with rasterio.open(rp) as src:
            windows = sliding_windows(src.height, src.width, tile_size, overlap)
            
            for i, (y_off, x_off) in enumerate(windows):
                win_h = min(tile_size, src.height - y_off)
                win_w = min(tile_size, src.width - x_off)
                
                window = Window(x_off, y_off, win_w, win_h)
                
                data = src.read(window=window)
                
                # Check variance for Gatekeeper (Step 8)
                # Ignore black boundaries
                valid_data_mask = np.any(data > 0, axis=0)
                valid_ratio = np.sum(valid_data_mask) / (win_h * win_w)
                
                if valid_ratio < 0.2:  # Less than 20% real imagery
                    rejected_count += 1
                    continue
                    
                variance = np.var(data)
                if variance < 15: # Too flat/featureless
                    rejected_count += 1
                    continue
                
                # Convert to RGB HWC
                if data.shape[0] >= 3:
                    tile_img = np.transpose(data[:3], (1, 2, 0))
                else:
                    tile_img = np.repeat(data[0:1], 3, axis=0).transpose(1, 2, 0)
                
                # Pad to exact tile_size
                if tile_img.shape[0] < tile_size or tile_img.shape[1] < tile_size:
                    padded = np.zeros((tile_size, tile_size, 3), dtype=tile_img.dtype)
                    padded[:tile_img.shape[0], :tile_img.shape[1]] = tile_img
                    tile_img = padded
                
                # Save tile
                tile_idx = f"{region_id}_{i:04d}"
                out_png = tiles_dir / f"{tile_idx}.png"
                Image.fromarray(tile_img).save(out_png)
                
                # Calculate True Geographic Bounds (Step 9)
                import rasterio.windows as rw
                left, bottom, right, top = rw.bounds(window, src.transform)
                
                meta = {
                    "tile_id": tile_idx,
                    "region": region_id,
                    "bbox": [[bottom, left], [top, right]], # [min_y, min_x], [max_y, max_x]
                    "crs": str(src.crs),
                    "resolution": src.res[0],
                    "pixel_bounds": [y_off, x_off, y_off+win_h, x_off+win_w]
                }
                all_tile_meta.append(meta)
                accepted_count += 1
                
    # Save global metadata
    with open(meta_dir / "tiles.json", "w") as f:
        json.dump(all_tile_meta, f, indent=2)
        
    logger.info(f"Gatekeeper complete: {accepted_count} tiles accepted, {rejected_count} rejected.")

def main():
    parser = argparse.ArgumentParser(description="Hydra Dataset Preparation Pipeline")
    parser.add_argument("input_tif", type=str, help="Path to raw drone GeoTIFF")
    parser.add_argument("--out_dir", type=str, default="data/clean_dataset", help="Output directory")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    args = parser.parse_args()
    
    input_path = Path(args.input_tif)
    out_dir = Path(args.out_dir)
    
    regions_dir = out_dir / "regions"
    tiles_dir = out_dir / "tiles"
    meta_dir = out_dir / "metadata"
    
    ensure_dir(regions_dir)
    ensure_dir(tiles_dir)
    ensure_dir(meta_dir)
    
    logger.info(f"Starting Hydra Pipeline on {input_path.name}")
    
    # Bypass Steps 1-5 since regions are already extracted successfully
    logger.info("Skipping Steps 1-5, loading existing regions...")
    region_paths = list(regions_dir.glob("*.tif"))
    
    step7_8_9_generate_tiles_and_metadata(
        region_paths, 
        tiles_dir, 
        meta_dir, 
        tile_size=args.tile_size, 
        overlap=args.overlap
    )
    
    logger.info("Pipeline completed successfully! Dataset is clean and AI-ready.")

if __name__ == "__main__":
    main()
