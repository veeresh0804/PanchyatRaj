"""
Region clustering logic for Hydra-Map.

Groups adjacent 512x512 predicted tiles that contain the same class into contiguous 
Context Regions for GeoSAM refinement. 
For example, if Tile (3,4) and Tile (3,5) both contain 'Building', they merge into one region.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RegionClusterer:
    """Groups predicted tiles into contiguous morphological regions.
    
    Operates on a grid coordinate system. Extracts tile X, Y from bounds or names
    and uses connected components to group tiles sharing edges or corners.
    """

    def __init__(self, tile_size: int = 512):
        self.tile_size = tile_size

    def _get_grid_coords(self, meta: Dict[str, Any]) -> Tuple[int, int]:
        """Convert pixel bounds or transform into discrete (column, row) grid coords."""
        # bounds is typically [[min_y, min_x], [max_y, max_x]]
        bounds = meta.get("bounds")
        if bounds:
            # We want positive column and row indices
            min_y, min_x = bounds[0]
            col = int(abs(min_x) // self.tile_size)
            row = int(abs(min_y) // self.tile_size)
            return col, row
            
        # Fallback to parsing tile ID (e.g. KARTARPUR_00005 -> index 5)
        # Using a very naive sequential index to row/col if bounds are missing
        # Assuming a default width of 10 for demo purposes if no bounds exist
        tid = meta.get("tile_id", "")
        try:
            # Extract number from end of tile ID string
            idx = int(tid.split("_")[-1])
            return idx % 10, idx // 10
        except ValueError:
            return 0, 0

    def cluster_tiles(self, tile_diagnostics: List[Dict[str, Any]], tile_metas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group tiles and their predictions into contiguous regions.
        
        Args:
            tile_diagnostics: List of inference diagnostic dicts (fusion_decision).
            tile_metas: List of metadata dicts corresponding to the tiles.
            
        Returns:
            List of region dicts containing:
                - region_id: str
                - class_id: int
                - tile_ids: List[str]
                - bounding_box: [min_y, min_x, max_y, max_x]
        """
        # 1. Map tiles by class and map ID to grid coords
        class_to_tiles = defaultdict(list)
        grid_map = {}
        meta_dict = {}

        for diag, meta in zip(tile_diagnostics, tile_metas):
            tid = diag.get("tile_id")
            meta_dict[tid] = meta
            
            # Find dominant class from fusion decision
            cls_id = diag.get("fusion_decision", {}).get("class_id", 0)
            
            # Or from _output.json style class distributions if available (we will use dominant for clustering)
            if "class_distribution" in diag and diag["class_distribution"]:
                cls_id = max(diag["class_distribution"], key=lambda x: x["confidence"])["class_id"]

            if cls_id == 0:
                continue # Skip background

            col, row = self._get_grid_coords(meta)
            grid_map[tid] = (col, row)
            class_to_tiles[cls_id].append(tid)

        regions = []
        region_counter = 1

        # 2. Extract Connected Components for each class
        for cls_id, tids in class_to_tiles.items():
            unvisited = set(tids)
            
            while unvisited:
                # Start a new component
                start_tid = unvisited.pop()
                component = [start_tid]
                queue = [start_tid]
                
                while queue:
                    curr_tid = queue.pop(0)
                    curr_col, curr_row = grid_map[curr_tid]
                    
                    # Find neighbors in 8-connectivity (edges & corners)
                    neighbors_to_add = []
                    for n_tid in unvisited:
                        n_col, n_row = grid_map[n_tid]
                        if abs(curr_col - n_col) <= 1 and abs(curr_row - n_row) <= 1:
                            neighbors_to_add.append(n_tid)
                            
                    for n_tid in neighbors_to_add:
                        if n_tid in unvisited:
                            unvisited.remove(n_tid)
                            queue.append(n_tid)
                            component.append(n_tid)
                            
                # Calculate the bounding box for this entire region
                min_y, min_x = float('inf'), float('inf')
                max_y, max_x = float('-inf'), float('-inf')
                
                for tid in component:
                    bounds = meta_dict[tid].get("bounds")
                    if bounds:
                        min_y = min(min_y, bounds[0][0])
                        min_x = min(min_x, bounds[0][1])
                        max_y = max(max_y, bounds[1][0])
                        max_x = max(max_x, bounds[1][1])
                
                # If bounds were missing, use dummy based on grid
                if min_y == float('inf'):
                    min_y, min_x, max_y, max_x = 0, 0, self.tile_size, self.tile_size
                    
                regions.append({
                    "region_id": f"R{region_counter:04d}",
                    "class_id": cls_id,
                    "tile_ids": component,
                    "bounding_box": [min_y, min_x, max_y, max_x]
                })
                region_counter += 1
                
        logger.info(f"Clustered {len(tile_diagnostics)} active tiles into {len(regions)} continuous regions.")
        return regions
