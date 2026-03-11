"""
Windowed Preprocessing for large SVAMITVA GeoTIFFs.

Reads 512x512 tiles directly from massive GeoTIFFs using rasterio
windowed reading (no full-image loading). Applies variance filter
per-tile and saves accepted tiles as PNGs with metadata JSONs.

Also rasterizes SHP ground truth at the tile level for training.

Usage:
  python scripts/preprocess_windowed.py --config config/small_config.yaml
  python scripts/preprocess_windowed.py --config config/small_config.yaml --max-tiles 50
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.io import ensure_dir, save_json, load_config, list_files_recursive

logging.basicConfig(level=logging.INFO, format="%(asctime)s [PREPROCESS] %(message)s")
logger = logging.getLogger(__name__)


def compute_tile_variance(tile: np.ndarray, window_size: int = 50) -> Tuple[float, float]:
    """Compute global and local variance for a tile."""
    if tile.ndim == 3:
        gray = np.mean(tile.astype(np.float64), axis=2)
    else:
        gray = tile.astype(np.float64)
    global_var = float(np.var(gray))

    h, w = gray.shape
    variances = []
    for y in range(0, h - window_size + 1, window_size):
        for x in range(0, w - window_size + 1, window_size):
            variances.append(np.var(gray[y:y+window_size, x:x+window_size]))
    local_var = float(np.median(variances)) if variances else global_var
    return global_var, local_var


def find_shp_dir(raw_dir: str) -> Optional[str]:
    """Find directory containing SHP files."""
    for root, dirs, files in os.walk(raw_dir):
        if any(f.endswith(".shp") for f in files):
            return root
    return None


def load_shp_geometries(shp_dir: str, tif_crs):
    """Load all SHP geometries and assign class IDs.

    Returns list of (geometry, class_id) tuples.
    """
    try:
        import geopandas as gpd
    except ImportError:
        logger.warning("geopandas not installed — skipping SHP annotations")
        return []

    SHP_CLASS_MAP = {
        "Built_Up_Area_type": 1,
        "Road": 2,
        "Road_Centre_Line": 2,
        "Water_Body": 4,
        "Water_Body_Line": 4,
        "Waterbody_Point": 4,
        "Utility_Poly": 5,
        "Utility": 5,
        "Bridge": 5,
        "Railway": 5,
    }

    all_shapes = []
    for shp_path in sorted(Path(shp_dir).glob("*.shp")):
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

        # Reproject if needed
        if gdf.crs and tif_crs and gdf.crs != tif_crs:
            try:
                gdf = gdf.to_crs(tif_crs)
            except Exception:
                continue

        # Buffer lines/points
        from shapely.geometry import LineString, Point, MultiLineString, MultiPoint
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            gtype = geom.geom_type
            if "Line" in gtype or "line" in layer_name.lower():
                geom = geom.buffer(3.0 / 111000.0)
            elif "Point" in gtype or "point" in layer_name.lower():
                geom = geom.buffer(5.0 / 111000.0)
            all_shapes.append((geom, class_id))

    logger.info(f"  Loaded {len(all_shapes)} ground truth geometries from SHP files")
    return all_shapes


def rasterize_tile_mask(shapes, tile_transform, tile_size: int) -> np.ndarray:
    """Rasterize SHP geometries for a specific tile window."""
    from rasterio import features as rio_features
    from shapely.geometry import box

    mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
    if not shapes:
        return mask

    # Tile bounding box in geographic coords
    a, b, c, d, e, f = tile_transform[:6]
    x_min = c
    y_max = f
    x_max = c + a * tile_size
    y_min = f + e * tile_size
    tile_box = box(min(x_min, x_max), min(y_min, y_max),
                   max(x_min, x_max), max(y_min, y_max))

    # Filter shapes that intersect this tile
    tile_shapes = []
    for geom, cls_id in shapes:
        try:
            if geom.intersects(tile_box):
                tile_shapes.append((geom, cls_id))
        except Exception:
            continue

    if not tile_shapes:
        return mask

    from rasterio.transform import Affine
    affine = Affine(a, b, c, d, e, f)
    mask = rio_features.rasterize(
        tile_shapes,
        out_shape=(tile_size, tile_size),
        transform=affine,
        fill=0,
        dtype=np.uint8,
    )
    return mask


def extract_valid_islands(src, downsample_factor=32) -> list:
    """Extract Shapely polygons representing contiguous valid-data islands in the GeoTIFF."""
    import time
    from shapely.geometry import shape
    from shapely.affinity import scale
    import rasterio.features
    
    t0 = time.time()
    out_shape = (1, int(src.height / downsample_factor), int(src.width / downsample_factor))
    img = src.read(1, out_shape=out_shape)
    
    mask = (img > 0).astype(np.uint8)
    
    # Close small holes
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    shapes = list(rasterio.features.shapes(mask, mask=mask))
    
    islands_px = []
    for geom, val in shapes:
        poly = shape(geom)
        poly = scale(poly, xfact=downsample_factor, yfact=downsample_factor, origin=(0, 0))
        islands_px.append(poly)
    
    logger.info(f"  Extracted {len(islands_px)} valid dataset island(s) in {time.time() - t0:.2f}s")
    return islands_px


def process_geotiff_windowed(
    tif_path: str,
    config: Dict,
    gt_shapes: list,
    tile_output_dir: str,
    mask_output_dir: str,
    max_tiles: int = 0,
) -> Dict:
    """Process a single GeoTIFF using windowed reading.

    Args:
        tif_path: Path to GeoTIFF.
        config: Config dict.
        gt_shapes: List of (geometry, class_id) for ground truth.
        tile_output_dir: Dir to save tile PNGs.
        mask_output_dir: Dir to save mask PNGs.
        max_tiles: Max tiles to extract (0=all).

    Returns:
        Stats dict.
    """
    import rasterio
    from rasterio.windows import Window

    prep_cfg = config.get("preprocess", {})
    tile_size = prep_cfg.get("tile_sizes", [512])[0]
    stride = prep_cfg.get("stride", 512)
    v_global = prep_cfg.get("v_global_thresh", 10)
    v_local = prep_cfg.get("v_local_thresh", 5)
    window_size = prep_cfg.get("local_window_size", 50)

    base_name = os.path.splitext(os.path.basename(tif_path))[0]
    # Sanitize filename
    base_name = base_name.replace(" ", "_").replace("__", "_")

    ensure_dir(tile_output_dir)
    ensure_dir(mask_output_dir)

    stats = {"total_windows": 0, "accepted": 0, "rejected": 0, "with_gt": 0}

    with rasterio.open(tif_path) as src:
        height = src.height
        width = src.width
        transform = list(src.transform)[:6]
        crs = src.crs

        logger.info(f"  {base_name}: {width}x{height} pixels, CRS={crs}")

        # 0. Dataset Normalization: Find Valid Islands
        islands = extract_valid_islands(src)
        from shapely.geometry import box

        tile_idx = 0
        for row_off in range(0, height, stride):
            for col_off in range(0, width, stride):
                if max_tiles > 0 and stats["accepted"] >= max_tiles:
                    return stats

                # Read window
                win_h = min(tile_size, height - row_off)
                win_w = min(tile_size, width - col_off)
                if win_h < tile_size // 2 or win_w < tile_size // 2:
                    continue  # Skip tiny edge tiles

                # Spatial bounds check: Does this tile intersect a valid island?
                tile_box = box(col_off, row_off, col_off + win_w, row_off + win_h)
                if islands and not any(island.intersects(tile_box) for island in islands):
                    # Skip empty background tiles completely
                    continue

                window = Window(col_off, row_off, win_w, win_h)
                try:
                    data = src.read(window=window)  # (C, H, W)
                except Exception as e:
                    continue

                # Convert to (H, W, C) RGB
                if data.shape[0] >= 3:
                    tile = np.transpose(data[:3], (1, 2, 0))
                elif data.shape[0] == 1:
                    tile = np.repeat(data[0:1], 3, axis=0).transpose(1, 2, 0)
                else:
                    continue

                # Pad if needed
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    padded = np.zeros((tile_size, tile_size, 3), dtype=tile.dtype)
                    padded[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded

                stats["total_windows"] += 1

                # Variance filter
                gv, lv = compute_tile_variance(tile, window_size)
                if gv < v_global and lv < v_local:
                    stats["rejected"] += 1
                    continue

                stats["accepted"] += 1
                tile_id = f"{base_name}_t{tile_idx:05d}"
                # Compute tile transform correctly with rasterio window_transform
                tile_transform_obj = src.window_transform(window)
                tile_transform = list(tile_transform_obj)[:6]

                # Get geographic bounds of this tile
                import rasterio.windows
                left, bottom, right, top = rasterio.windows.bounds(window, src.transform)
                # Store bound as [[min_y, min_x], [max_y, max_x]] equivalent for Leaflet CRS.Simple (actually [bottom, left], [top, right] usually, or [row, col])
                # We will just pass [[bottom, left], [top, right]] which is the spatial extent.
                tile_bounds = [[float(bottom), float(left)], [float(top), float(right)]]

                # Save tile PNG
                tile_path_out = os.path.join(tile_output_dir, f"{tile_id}.png")
                cv2.imwrite(tile_path_out, cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))

                # Save tile metadata
                tile_meta = {
                    "tile_id": tile_id,
                    "source_file": tif_path,
                    "tile_index": tile_idx,
                    "pixel_offset": [col_off, row_off],
                    "tile_size": tile_size,
                    "stride": stride,
                    "crs": str(crs) if crs else None,
                    "transform": tile_transform,
                    "bounds": tile_bounds,
                    "global_variance": gv,
                    "local_variance": lv,
                }
                save_json(tile_meta, os.path.join(tile_output_dir, f"tile_{tile_id}.json"))

                # Rasterize ground truth mask for this tile
                if gt_shapes:
                    mask = rasterize_tile_mask(gt_shapes, tile_transform, tile_size)
                    if mask.sum() > 0:
                        stats["with_gt"] += 1
                    mask_path = os.path.join(mask_output_dir, f"{tile_id}_mask.png")
                    cv2.imwrite(mask_path, mask)

                tile_idx += 1

                if stats["accepted"] % 100 == 0:
                    logger.info(f"    ... {stats['accepted']} tiles accepted so far")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Windowed preprocessing for large GeoTIFFs")
    parser.add_argument("--config", type=str, default="config/small_config.yaml")
    parser.add_argument("--max-tiles", type=int, default=0,
                        help="Max tiles per GeoTIFF (0=all)")
    parser.add_argument("--max-tifs", type=int, default=0,
                        help="Max GeoTIFFs to process (0=all)")
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = config["data"]["raw_dir"]
    preprocessed_dir = config["data"]["preprocessed_dir"]
    ann_dir = config["data"]["annotations_dir"]

    tile_size = config.get("preprocess", {}).get("tile_sizes", [512])[0]
    tile_output_dir = os.path.join(preprocessed_dir, "tiles", str(tile_size))
    mask_output_dir = ann_dir

    # Find GeoTIFFs
    tifs = list_files_recursive(raw_dir, extensions=[".tif", ".tiff"])
    if not tifs:
        logger.error(f"No GeoTIFF files found in {raw_dir}")
        return

    if args.max_tifs > 0:
        tifs = tifs[:args.max_tifs]

    logger.info(f"Found {len(tifs)} GeoTIFFs")

    # Load ground truth SHP geometries
    shp_dir = find_shp_dir(raw_dir)
    gt_shapes = []
    if shp_dir:
        logger.info(f"Loading ground truth from: {shp_dir}")
        import rasterio
        with rasterio.open(tifs[0]) as src:
            tif_crs = src.crs
        gt_shapes = load_shp_geometries(shp_dir, tif_crs)
    else:
        logger.warning("No SHP files found — will proceed without ground truth masks")

    # Process each GeoTIFF
    total_stats = {"total_windows": 0, "accepted": 0, "rejected": 0, "with_gt": 0}
    for tif_path in tifs:
        logger.info(f"\nProcessing: {os.path.basename(tif_path)}")
        stats = process_geotiff_windowed(
            tif_path, config, gt_shapes,
            tile_output_dir, mask_output_dir,
            max_tiles=args.max_tiles,
        )
        for k in total_stats:
            total_stats[k] += stats[k]
        logger.info(f"  → {stats['accepted']} accepted, {stats['rejected']} rejected, {stats['with_gt']} with GT")

    # Save summary
    save_json(total_stats, os.path.join(preprocessed_dir, "preprocess_summary.json"))

    logger.info(f"\n{'='*60}")
    logger.info(f"PREPROCESSING COMPLETE")
    logger.info(f"  Total windows scanned: {total_stats['total_windows']}")
    logger.info(f"  Tiles accepted:        {total_stats['accepted']}")
    logger.info(f"  Tiles rejected:        {total_stats['rejected']}")
    logger.info(f"  Tiles with GT:         {total_stats['with_gt']}")
    logger.info(f"  Output dir:            {tile_output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
