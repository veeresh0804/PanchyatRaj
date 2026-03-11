"""
Hydra-Map I/O Utilities.

Helpers for loading GeoTIFFs, reading/writing JSON, loading YAML configs,
and general file operations used across the pipeline.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_geotiff(path: str) -> Tuple[np.ndarray, dict]:
    """Load a GeoTIFF and return the image array and metadata.

    Args:
        path: Path to the GeoTIFF file.

    Returns:
        Tuple of (image_array [H, W, C], metadata dict with keys
        'crs', 'transform', 'width', 'height', 'count', 'dtype').
    """
    import rasterio

    with rasterio.open(path) as src:
        img = src.read()  # (C, H, W)
        meta = {
            "crs": str(src.crs) if src.crs else None,
            "transform": list(src.transform)[:6],
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": str(src.dtypes[0]),
            "bounds": list(src.bounds),
        }
    # Transpose to (H, W, C) for downstream processing
    img = np.transpose(img, (1, 2, 0))
    return img, meta


def save_geotiff(
    path: str,
    data: np.ndarray,
    crs: str,
    transform: list,
    dtype: str = "uint8",
    compress: str = "LZW",
    tiled: bool = True,
    blockxsize: int = 256,
    blockysize: int = 256,
) -> None:
    """Save a numpy array as a GeoTIFF.

    Args:
        path: Output file path.
        data: Image array (H, W) or (H, W, C).
        crs: Coordinate reference system string.
        transform: Affine transform as a list of 6 floats.
        dtype: Output data type.
        compress: Compression method.
        tiled: Whether to write tiled GeoTIFF.
        blockxsize: Tile width.
        blockysize: Tile height.
    """
    import rasterio
    from rasterio.transform import Affine

    if data.ndim == 2:
        data = data[np.newaxis, :, :]  # (1, H, W)
    elif data.ndim == 3:
        data = np.transpose(data, (2, 0, 1))  # (C, H, W)

    count, height, width = data.shape
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "width": width,
        "height": height,
        "count": count,
        "crs": crs,
        "transform": Affine(*transform[:6]),
        "compress": compress,
    }
    if tiled:
        profile.update({"tiled": True, "blockxsize": blockxsize, "blockysize": blockysize})

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Save data to a JSON file.

    Args:
        data: Data to serialize.
        path: Output path.
        indent: JSON indentation.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=_json_serializer)


def load_json(path: str) -> Any:
    """Load data from a JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed JSON data.
    """
    with open(path, "r") as f:
        return json.load(f)


def save_jsonl(records: List[Dict], path: str) -> None:
    """Append records to a JSONL (JSON Lines) file.

    Args:
        records: List of dicts to write.
        path: Output path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        for rec in records:
            f.write(json.dumps(rec, default=_json_serializer) + "\n")


def load_jsonl(path: str) -> List[Dict]:
    """Load all records from a JSONL file.

    Args:
        path: Path to JSONL file.

    Returns:
        List of dicts.
    """
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist and return the path.

    Args:
        path: Directory path.

    Returns:
        The same path.
    """
    os.makedirs(path, exist_ok=True)
    return path


def list_files(directory: str, extensions: Optional[List[str]] = None) -> List[str]:
    """List files in a directory, optionally filtered by extension.

    Args:
        directory: Directory to scan.
        extensions: List of extensions (e.g., ['.tif', '.tiff']).

    Returns:
        Sorted list of absolute file paths.
    """
    files = []
    if not os.path.isdir(directory):
        return files
    for fname in sorted(os.listdir(directory)):
        fpath = os.path.join(directory, fname)
        if os.path.isfile(fpath):
            if extensions is None or any(fname.lower().endswith(ext) for ext in extensions):
                files.append(fpath)
    return files


def list_files_recursive(directory: str, extensions: Optional[List[str]] = None) -> List[str]:
    """List files recursively in directory and subdirectories.

    Args:
        directory: Root directory to scan.
        extensions: List of extensions to filter by (e.g., ['.tif', '.tiff']).

    Returns:
        Sorted list of absolute file paths.
    """
    files = []
    if not os.path.isdir(directory):
        return files
    for root, dirs, fnames in os.walk(directory):
        for fname in fnames:
            fpath = os.path.join(root, fname)
            if extensions is None or any(fname.lower().endswith(ext) for ext in extensions):
                files.append(fpath)
    return sorted(files)


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
