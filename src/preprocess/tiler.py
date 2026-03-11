"""
Hydra-Map Tiler.

Generates overlapping tiles from raw GeoTIFFs with configurable
tile size and stride. Saves each tile as PNG/NPY plus a metadata JSON
with geotransform and CRS.

CLI: python src/preprocess/tiler.py --config config/config.yaml
"""

import argparse
import logging
import os
import sys
from typing import Dict, List

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.io import ensure_dir, list_files, load_config, load_geotiff, save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_tiles(
    image: np.ndarray,
    meta: dict,
    tile_size: int,
    stride: int,
    source_file: str,
    output_dir: str,
    output_format: str = "png",
) -> List[Dict]:
    """Generate overlapping tiles from an image.

    Args:
        image: Input image (H, W, C).
        meta: GeoTIFF metadata dict.
        tile_size: Width and height of each tile.
        stride: Step size between tiles.
        source_file: Original source file path.
        output_dir: Directory to write tiles.
        output_format: 'png' or 'npy'.

    Returns:
        List of tile info dicts with paths and metadata.
    """
    h, w = image.shape[:2]
    tiles = []
    tile_idx = 0
    base_name = os.path.splitext(os.path.basename(source_file))[0]

    transform = meta.get("transform", [1, 0, 0, 0, 1, 0])
    crs = meta.get("crs", None)

    ensure_dir(output_dir)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Extract tile with zero-padding at edges
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)

            tile = np.zeros((tile_size, tile_size, image.shape[2]), dtype=image.dtype)
            tile[: y_end - y, : x_end - x] = image[y:y_end, x:x_end]

            tile_id = f"{base_name}_t{tile_idx:05d}"

            # Compute tile-specific geotransform
            # Original affine: [a, b, c, d, e, f]
            # c = x_origin, f = y_origin, a = pixel_width, e = pixel_height
            a, b, c, d, e, f = transform[:6]
            tile_c = c + x * a + y * b
            tile_f = f + x * d + y * e
            tile_transform = [a, b, tile_c, d, e, tile_f]

            # Save tile image
            if output_format == "npy":
                tile_path = os.path.join(output_dir, f"{tile_id}.npy")
                np.save(tile_path, tile)
            else:
                tile_path = os.path.join(output_dir, f"{tile_id}.png")
                if tile.shape[2] >= 3:
                    cv2.imwrite(tile_path, cv2.cvtColor(tile[:, :, :3], cv2.COLOR_RGB2BGR))
                else:
                    cv2.imwrite(tile_path, tile[:, :, 0])

            # Save tile metadata
            tile_meta = {
                "tile_id": tile_id,
                "source_file": source_file,
                "tile_index": tile_idx,
                "pixel_offset": [x, y],
                "tile_size": tile_size,
                "stride": stride,
                "crs": crs,
                "transform": tile_transform,
                "original_transform": transform,
                "padded": (y_end - y < tile_size) or (x_end - x < tile_size),
            }
            meta_path = os.path.join(output_dir, f"tile_{tile_id}.json")
            save_json(tile_meta, meta_path)

            tiles.append(tile_meta)
            tile_idx += 1

    return tiles


def run_tiler(config: Dict) -> Dict[str, int]:
    """Run tiling on all accepted raw GeoTIFFs.

    Args:
        config: Loaded config dict.

    Returns:
        Dict with per-tile-size counts.
    """
    prep_cfg = config.get("preprocess", {})
    raw_dir = config["data"]["raw_dir"]
    preprocessed_dir = config["data"]["preprocessed_dir"]
    tile_sizes = prep_cfg.get("tile_sizes", [512, 640])
    stride = prep_cfg.get("stride", 256)
    output_format = prep_cfg.get("output_format", "png")

    # Check for accepted filter results
    filter_dir = os.path.join(preprocessed_dir, "filter_results")
    accepted_files = []

    if os.path.isdir(filter_dir):
        from src.utils.io import load_json
        for fname in sorted(os.listdir(filter_dir)):
            if fname.endswith("_accepted.json"):
                record = load_json(os.path.join(filter_dir, fname))
                if record.get("source_file"):
                    accepted_files.append(record["source_file"])

    # If no filter results, tile all raw files
    if not accepted_files:
        logger.info("No filter results found, tiling ALL raw GeoTIFFs")
        accepted_files = list_files(raw_dir, extensions=[".tif", ".tiff"])

    if not accepted_files:
        logger.warning(f"No GeoTIFF files to tile")
        return {}

    logger.info(f"Tiling {len(accepted_files)} files with sizes {tile_sizes}, stride {stride}")
    counts = {}

    for ts in tile_sizes:
        output_dir = os.path.join(preprocessed_dir, "tiles", str(ts))
        ensure_dir(output_dir)
        total = 0

        for fpath in accepted_files:
            if not os.path.isfile(fpath):
                logger.warning(f"File not found: {fpath}, skipping")
                continue

            try:
                img, meta = load_geotiff(fpath)
            except Exception as e:
                logger.error(f"Failed to load {fpath}: {e}")
                continue

            tiles = generate_tiles(
                image=img,
                meta=meta,
                tile_size=ts,
                stride=stride,
                source_file=fpath,
                output_dir=output_dir,
                output_format=output_format,
            )
            total += len(tiles)
            logger.info(f"  {os.path.basename(fpath)}: {len(tiles)} tiles @ {ts}px")

        counts[str(ts)] = total
        logger.info(f"Tile size {ts}: {total} tiles total")

    # Save tiling summary
    summary_path = os.path.join(preprocessed_dir, "tiling_summary.json")
    save_json({"tile_counts": counts, "stride": stride, "format": output_format}, summary_path)

    return counts


def main():
    parser = argparse.ArgumentParser(description="Hydra-Map Tiler")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--debug", action="store_true",
                        help="Run single-tile mode with verbose logs")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config(args.config)
    counts = run_tiler(config)

    print(f"\n=== Tiling Results ===")
    for ts, count in counts.items():
        print(f"  {ts}px: {count} tiles")


if __name__ == "__main__":
    main()
