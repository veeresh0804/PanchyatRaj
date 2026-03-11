"""
Hydra-Map Geospatial Utilities.

Functions for coordinate transforms, mask-to-polygon conversion,
polygon sanitization, topology checks, and spatial statistics.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def masks_to_polygons(
    mask: np.ndarray,
    transform: list,
    crs: Optional[str] = None,
    min_area: float = 10.0,
    simplify_tolerance: float = 0.5,
) -> List[Dict[str, Any]]:
    """Convert a binary mask to georeferenced polygons.

    Args:
        mask: Binary mask (H, W) with 0=background, 1=object.
        transform: Affine transform as list of 6 floats.
        crs: CRS string (e.g., 'EPSG:4326').
        min_area: Minimum polygon area to keep (in CRS units²).
        simplify_tolerance: Simplification tolerance.

    Returns:
        List of dicts with 'geometry' (shapely), 'area', 'perimeter', 'valid'.
    """
    try:
        import rasterio.features
        from rasterio.transform import Affine
        from shapely.geometry import shape
    except ImportError:
        logger.warning(
            "rasterio/shapely not installed — masks_to_polygons returning stub. "
            "Install rasterio and shapely for real polygon output."
        )
        if mask is not None and mask.sum() > 0:
            return [_stub_polygon_dict(min_area)]
        return []

    affine = Affine(*transform[:6])
    polygons = []

    for geom, value in rasterio.features.shapes(
        mask.astype(np.uint8), transform=affine
    ):
        if value == 0:
            continue
        poly = shape(geom)
        poly = sanitize_polygon(poly)
        if poly is None or poly.is_empty:
            continue
        if poly.area < min_area:
            continue
        if simplify_tolerance > 0:
            poly = poly.simplify(simplify_tolerance, preserve_topology=True)

        polygons.append(
            {
                "geometry": poly,
                "area": poly.area,
                "perimeter": poly.length,
                "valid": poly.is_valid,
            }
        )

    return polygons


def _stub_polygon_dict(area: float = 100.0) -> Dict[str, Any]:
    """Return a minimal polygon dict for use when shapely is unavailable."""

    class _StubGeom:
        """Minimal shapely-like geometry stub."""
        wkt = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"
        is_empty = False
        is_valid = True
        area = area
        length = 4.0

    return {
        "geometry": _StubGeom(),
        "area": area,
        "perimeter": 4.0,
        "valid": True,
    }



def sanitize_polygon(polygon: Any) -> Any:
    """Sanitize a shapely polygon: fix validity, handle MultiPolygon.

    If invalid, attempts buffer(0) and make_valid. If result is
    MultiPolygon, keeps the largest piece and discards fragments.

    Args:
        polygon: A shapely geometry.

    Returns:
        Sanitized Polygon, or None if unrecoverable.
    """
    try:
        from shapely.geometry import MultiPolygon, Polygon
        from shapely.validation import make_valid
    except ImportError:
        # shapely not installed — return the polygon unchanged (best effort)
        return polygon if polygon is not None else None

    if polygon is None or polygon.is_empty:
        return None

    # Fix invalid geometry
    if not polygon.is_valid:
        try:
            polygon = make_valid(polygon)
        except Exception:
            try:
                polygon = polygon.buffer(0)
            except Exception:
                logger.warning("Could not sanitize polygon, returning None")
                return None

    # If MultiPolygon, keep the largest piece
    if isinstance(polygon, MultiPolygon):
        if len(polygon.geoms) == 0:
            return None
        polygon = max(polygon.geoms, key=lambda g: g.area)
        logger.debug(
            f"MultiPolygon reduced to largest piece (area={polygon.area:.2f}), "
            f"fragments_discarded={len(polygon.geoms) - 1 if hasattr(polygon, 'geoms') else 0}"
        )

    return polygon


def compute_polygon_stats(polygon: Any) -> Dict[str, float]:
    """Compute summary statistics for a polygon.

    Args:
        polygon: A shapely Polygon.

    Returns:
        Dict with area, perimeter, num_vertices, compactness.
    """
    if polygon is None or polygon.is_empty:
        return {"area": 0, "perimeter": 0, "num_vertices": 0, "compactness": 0}

    area = polygon.area
    perimeter = polygon.length
    num_vertices = len(polygon.exterior.coords) if hasattr(polygon, "exterior") else 0
    # Compactness: 4π × area / perimeter² (circle = 1)
    compactness = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0

    return {
        "area": float(area),
        "perimeter": float(perimeter),
        "num_vertices": int(num_vertices),
        "compactness": float(compactness),
    }


def polygon_iou(poly_a: Any, poly_b: Any) -> float:
    """Compute IoU between two shapely polygons.

    Args:
        poly_a: First polygon.
        poly_b: Second polygon.

    Returns:
        IoU value in [0, 1].
    """
    if poly_a is None or poly_b is None:
        return 0.0
    if poly_a.is_empty or poly_b.is_empty:
        return 0.0
    try:
        intersection = poly_a.intersection(poly_b).area
        union = poly_a.union(poly_b).area
        return intersection / union if union > 0 else 0.0
    except Exception:
        return 0.0


def bbox_to_polygon(bbox: List[float]) -> Any:
    """Convert [x1, y1, x2, y2] bounding box to shapely Polygon.

    Args:
        bbox: Bounding box as [x_min, y_min, x_max, y_max].

    Returns:
        Shapely Polygon.
    """
    from shapely.geometry import box

    return box(bbox[0], bbox[1], bbox[2], bbox[3])


def pixel_to_geo(
    pixel_coords: np.ndarray, transform: list
) -> np.ndarray:
    """Convert pixel coordinates to geographic coordinates.

    Args:
        pixel_coords: Array of shape (N, 2) with (col, row).
        transform: Affine transform as list of 6 floats.

    Returns:
        Array of shape (N, 2) with (x, y) in CRS coordinates.
    """
    from rasterio.transform import Affine

    affine = Affine(*transform[:6])
    geo = np.zeros_like(pixel_coords, dtype=np.float64)
    for i, (col, row) in enumerate(pixel_coords):
        x, y = affine * (col, row)
        geo[i] = [x, y]
    return geo
