"""
Hydra-Map Variance Filter.

Reads tile-sized crops from raw GeoTIFFs and discards tiles
with low variance (featureless areas like empty fields or water).

CLI: python src/preprocess/variance_filter.py --config config/config.yaml
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.io import ensure_dir, list_files, load_config, load_geotiff, save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_global_variance(tile: np.ndarray) -> float:
    """Compute global pixel variance of a grayscale tile.

    Args:
        tile: Image tile (H, W, C) or (H, W).

    Returns:
        Global variance as float.
    """
    if tile.ndim == 3:
        gray = np.mean(tile, axis=2)
    else:
        gray = tile.astype(np.float64)
    return float(np.var(gray))


def compute_local_variance(tile: np.ndarray, window_size: int = 50) -> float:
    """Compute median of sliding-window local variance.

    Uses a non-overlapping sliding window across the grayscale tile,
    computes variance within each window, and returns the median.

    Args:
        tile: Image tile (H, W, C) or (H, W).
        window_size: Size of the sliding window.

    Returns:
        Median local variance as float.
    """
    if tile.ndim == 3:
        gray = np.mean(tile, axis=2)
    else:
        gray = tile.astype(np.float64)

    h, w = gray.shape
    variances = []

    for y in range(0, h - window_size + 1, window_size):
        for x in range(0, w - window_size + 1, window_size):
            window = gray[y : y + window_size, x : x + window_size]
            variances.append(np.var(window))

    if not variances:
        return float(np.var(gray))

    return float(np.median(variances))


def filter_tile(
    tile: np.ndarray,
    tile_id: str,
    v_global_thresh: float = 12.0,
    v_local_thresh: float = 6.0,
    local_window_size: int = 50,
) -> Tuple[bool, str, Dict]:
    """Decide whether to keep or discard a tile based on variance.

    A tile is discarded only if BOTH global variance < v_global_thresh
    AND median local variance < v_local_thresh.

    Args:
        tile: Image tile array (H, W, C).
        tile_id: Identifier for the tile.
        v_global_thresh: Global variance threshold.
        v_local_thresh: Local variance threshold.
        local_window_size: Window size for local variance.

    Returns:
        Tuple of (keep: bool, reason: str, stats: dict).
    """
    global_var = compute_global_variance(tile)
    local_var = compute_local_variance(tile, window_size=local_window_size)

    stats = {
        "tile_id": tile_id,
        "global_variance": global_var,
        "local_variance_median": local_var,
        "v_global_thresh": v_global_thresh,
        "v_local_thresh": v_local_thresh,
    }

    if global_var < v_global_thresh and local_var < v_local_thresh:
        return False, "low_variance", stats

    return True, "accepted", stats


def run_variance_filter(config: Dict) -> Dict[str, List]:
    """Run variance filter on all raw GeoTIFFs.

    Iterates over raw tiles, applies the filter, and logs results.

    Args:
        config: Loaded config dict.

    Returns:
        Dict with 'accepted' and 'rejected' lists of tile info dicts.
    """
    prep_cfg = config.get("preprocess", {})
    raw_dir = config["data"]["raw_dir"]
    v_global = prep_cfg.get("v_global_thresh", 12)
    v_local = prep_cfg.get("v_local_thresh", 6)
    window_size = prep_cfg.get("local_window_size", 50)

    # Output directory for filter results
    filter_dir = ensure_dir(os.path.join(config["data"]["preprocessed_dir"], "filter_results"))

    tiff_files = list_files(raw_dir, extensions=[".tif", ".tiff"])
    if not tiff_files:
        logger.warning(f"No GeoTIFF files found in {raw_dir}")
        return {"accepted": [], "rejected": []}

    logger.info(f"Found {len(tiff_files)} raw GeoTIFFs in {raw_dir}")
    accepted = []
    rejected = []

    for fpath in tiff_files:
        tile_id = os.path.splitext(os.path.basename(fpath))[0]
        logger.info(f"Processing tile: {tile_id}")

        try:
            img, meta = load_geotiff(fpath)
        except Exception as e:
            reason_record = {
                "tile_id": tile_id,
                "reason": "load_error",
                "error": str(e),
            }
            save_json(reason_record, os.path.join(filter_dir, f"{tile_id}_rejected.json"))
            rejected.append(reason_record)
            logger.error(f"Failed to load tile {tile_id}: {e}")
            continue

        keep, reason, stats = filter_tile(
            img, tile_id,
            v_global_thresh=v_global,
            v_local_thresh=v_local,
            local_window_size=window_size,
        )

        record = {
            "tile_id": tile_id,
            "reason": reason,
            "keep": keep,
            "values": stats,
            "source_file": fpath,
        }

        if keep:
            accepted.append(record)
            save_json(record, os.path.join(filter_dir, f"{tile_id}_accepted.json"))
            logger.info(f"  ACCEPTED: global_var={stats['global_variance']:.2f}, local_var={stats['local_variance_median']:.2f}")
        else:
            rejected.append(record)
            save_json(record, os.path.join(filter_dir, f"{tile_id}_rejected.json"))
            logger.info(f"  REJECTED ({reason}): global_var={stats['global_variance']:.2f}, local_var={stats['local_variance_median']:.2f}")

    # Summary
    summary = {
        "total": len(tiff_files),
        "accepted": len(accepted),
        "rejected": len(rejected),
    }
    save_json(summary, os.path.join(filter_dir, "filter_summary.json"))
    logger.info(f"Filter complete: {summary['accepted']}/{summary['total']} tiles accepted")

    return {"accepted": accepted, "rejected": rejected}


def main():
    parser = argparse.ArgumentParser(description="Hydra-Map Variance Filter")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--debug", action="store_true",
                        help="Run single-tile mode with verbose logs")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config(args.config)
    results = run_variance_filter(config)

    print(f"\n=== Variance Filter Results ===")
    print(f"Accepted: {len(results['accepted'])}")
    print(f"Rejected: {len(results['rejected'])}")


if __name__ == "__main__":
    main()
