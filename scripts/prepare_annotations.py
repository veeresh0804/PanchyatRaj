"""
Prepare ground truth annotations for Hydra-Map training.

Reads SVAMITVA shapefiles (Built_Up_Area, Road, Water_Body, etc.)
and rasterizes them onto GeoTIFF grids to produce per-tile mask images.

Class mapping:
  0 = background
  1 = building (from Built_Up_Area_type.shp)
  2 = road     (from Road.shp + Road_Centre_Line.shp buffered)
  3 = vegetation (not directly available — will be background)
  4 = water    (from Water_Body.shp)
  5 = other    (from Utility_Poly.shp, Bridge.shp, Railway.shp)

Usage:
  python scripts/prepare_annotations.py --config config/config.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Class assignment from SHP layer names
SHP_CLASS_MAP = {
    "Built_Up_Area_type": 1,  # building
    "Road": 2,                # road
    "Road_Centre_Line": 2,    # road (line → buffer)
    "Water_Body": 4,          # water
    "Water_Body_Line": 4,     # water (line → buffer)
    "Waterbody_Point": 4,     # water (point → buffer)
    "Utility_Poly": 5,        # other
    "Utility": 5,             # other (point → buffer)
    "Bridge": 5,              # other
    "Railway": 5,             # other (line → buffer)
}

LINE_BUFFER_METERS = 3.0  # Buffer for line geometries
POINT_BUFFER_METERS = 5.0  # Buffer for point geometries


def find_shp_dir(raw_dir: str) -> str:
    """Find the directory containing SHP files."""
    for root, dirs, files in os.walk(raw_dir):
        if any(f.endswith(".shp") for f in files):
            return root
    return ""


def find_geotiffs(raw_dir: str):
    """Find all GeoTIFF files recursively."""
    tifs = []
    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            if f.lower().endswith((".tif", ".tiff")):
                tifs.append(os.path.join(root, f))
    return sorted(tifs)


def rasterize_shp_onto_tif(tif_path: str, shp_dir: str, output_path: str) -> bool:
    """Rasterize all SHP layers onto a GeoTIFF grid and save as mask PNG.

    Args:
        tif_path: Path to the GeoTIFF.
        shp_dir: Directory containing shapefile layers.
        output_path: Path to save the output mask image.

    Returns:
        True if mask was created successfully.
    """
    try:
        import rasterio
        from rasterio.transform import Affine
        import geopandas as gpd
        from rasterio import features as rio_features
        import cv2
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False

    # Read TIF metadata
    with rasterio.open(tif_path) as src:
        transform = src.transform
        width = src.width
        height = src.height
        crs = src.crs

    logger.info(f"  TIF: {width}x{height}, CRS={crs}")

    # Initialize empty mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Process each SHP layer
    shp_files = sorted(Path(shp_dir).glob("*.shp"))
    geometries_count = 0

    for shp_path in shp_files:
        layer_name = shp_path.stem
        class_id = SHP_CLASS_MAP.get(layer_name, 0)
        if class_id == 0:
            continue

        try:
            gdf = gpd.read_file(str(shp_path))
        except Exception as e:
            logger.warning(f"  Failed to read {layer_name}: {e}")
            continue

        if gdf.empty:
            continue

        # Reproject to match TIF CRS if needed
        if gdf.crs and crs and gdf.crs != crs:
            try:
                gdf = gdf.to_crs(crs)
            except Exception as e:
                logger.warning(f"  CRS transform failed for {layer_name}: {e}")
                continue

        # Buffer line and point geometries
        geom_type = gdf.geometry.geom_type.iloc[0] if len(gdf) > 0 else ""
        if "Line" in geom_type or "line" in layer_name.lower():
            # Estimate buffer in CRS units (approximate for geographic CRS)
            buffer_deg = LINE_BUFFER_METERS / 111000.0  # rough meters→degrees
            gdf["geometry"] = gdf.geometry.buffer(buffer_deg)
        elif "Point" in geom_type or "point" in layer_name.lower():
            buffer_deg = POINT_BUFFER_METERS / 111000.0
            gdf["geometry"] = gdf.geometry.buffer(buffer_deg)

        # Clip to TIF bounds
        from shapely.geometry import box
        tif_bounds = box(*rasterio.open(tif_path).bounds)
        gdf = gdf[gdf.geometry.intersects(tif_bounds)]

        if gdf.empty:
            logger.info(f"  {layer_name}: no geometries in TIF extent")
            continue

        # Rasterize
        shapes = [(geom, class_id) for geom in gdf.geometry if geom is not None and not geom.is_empty]
        if not shapes:
            continue

        layer_mask = rio_features.rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )

        # Merge: higher class_id wins in overlaps (building > road > water > other)
        # Actually use priority: building first, then others fill background
        mask = np.where((mask == 0) & (layer_mask > 0), layer_mask, mask)
        count = int((layer_mask > 0).sum())
        geometries_count += len(shapes)
        logger.info(f"  {layer_name} (class {class_id}): {len(shapes)} shapes, {count} pixels")

    if geometries_count == 0:
        logger.warning(f"  No ground truth for this tile")
        return False

    # Save mask
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, mask)
    unique, counts = np.unique(mask, return_counts=True)
    logger.info(f"  Mask saved: {output_path}")
    logger.info(f"  Classes: {dict(zip(unique.tolist(), counts.tolist()))}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare annotations for Hydra-Map")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--max-tiles", type=int, default=0,
                        help="Max tiles to process (0=all)")
    args = parser.parse_args()

    from src.utils.io import load_config
    config = load_config(args.config)

    raw_dir = config["data"]["raw_dir"]
    ann_dir = config["data"]["annotations_dir"]

    # Find SHP directory
    shp_dir = find_shp_dir(raw_dir)
    if not shp_dir:
        logger.error(f"No SHP files found in {raw_dir}")
        return

    logger.info(f"SHP directory: {shp_dir}")
    logger.info(f"Annotations output: {ann_dir}")

    # Find GeoTIFFs
    tifs = find_geotiffs(raw_dir)
    if not tifs:
        logger.error(f"No GeoTIFF files found in {raw_dir}")
        return

    if args.max_tiles > 0:
        tifs = tifs[:args.max_tiles]

    logger.info(f"Processing {len(tifs)} GeoTIFFs...")

    success = 0
    for tif_path in tifs:
        tile_id = os.path.splitext(os.path.basename(tif_path))[0]
        logger.info(f"\nProcessing: {tile_id}")

        mask_path = os.path.join(ann_dir, f"{tile_id}_mask.png")
        if rasterize_shp_onto_tif(tif_path, shp_dir, mask_path):
            success += 1

    logger.info(f"\nDone: {success}/{len(tifs)} masks created in {ann_dir}")


if __name__ == "__main__":
    main()
