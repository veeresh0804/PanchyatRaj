import json
import logging
import os
import shutil
import sys
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features as rio_features
from shapely.geometry import box

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Class assignment from SHP layer names
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

def rasterize_region(tif_path: Path, shp_dir: Path, output_path: Path):
    """Rasterize all SHP layers onto a single large Region GeoTIFF."""
    logger.info(f"Rasterizing SHP layers for {tif_path.name}...")
    
    with rasterio.open(tif_path) as src:
        transform = src.transform
        width = src.width
        height = src.height
        crs = src.crs
        tif_bounds = box(*src.bounds)

    mask = np.zeros((height, width), dtype=np.uint8)
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
            continue

        if gdf.empty:
            continue

        if gdf.crs and crs and gdf.crs != crs:
            try:
                gdf = gdf.to_crs(crs)
            except Exception as e:
                continue

        # Buffer line geometries
        geom_type = gdf.geometry.geom_type.iloc[0] if len(gdf) > 0 else ""
        if "Line" in geom_type or "line" in layer_name.lower():
            # Approx 3m buffer in map units
            gdf["geometry"] = gdf.geometry.buffer(3.0 / 111000.0 if "4326" in str(crs) else 3.0)
        elif "Point" in geom_type or "point" in layer_name.lower():
            gdf["geometry"] = gdf.geometry.buffer(5.0 / 111000.0 if "4326" in str(crs) else 5.0)

        # Clip to TIF bounds
        gdf = gdf[gdf.geometry.intersects(tif_bounds)]
        if gdf.empty:
            continue

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

        mask = np.where((mask == 0) & (layer_mask > 0), layer_mask, mask)
        geometries_count += len(shapes)

    if geometries_count > 0:
        os.makedirs(output_path.parent, exist_ok=True)
        cv2.imwrite(str(output_path), mask)
        logger.info(f"  Saved {output_path.name}")
        return True
    else:
        logger.warning(f"  No ground truth shapes overlap {tif_path.name}")
        return False

def slice_masks(meta_path: Path, regions_masks_dir: Path, out_ann_dir: Path):
    """Slice the large region masks into 512x512 tiles matching the images."""
    logger.info("Slicing region masks into tiles...")
    os.makedirs(out_ann_dir, exist_ok=True)
    
    with open(meta_path, 'r') as f:
        tiles = json.load(f)
        
    loaded_masks = {}
    success_count = 0
    
    for t in tiles:
        region_id = t["region"]
        tile_id = t["tile_id"]
        y, x, y2, x2 = t["pixel_bounds"]
        
        mask_path = regions_masks_dir / f"{region_id}_mask.png"
        
        if not mask_path.exists():
            continue
            
        if region_id not in loaded_masks:
            loaded_masks[region_id] = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
        full_mask = loaded_masks[region_id]
        
        # Crop exactly like the image
        win_h = min(512, full_mask.shape[0] - y)
        win_w = min(512, full_mask.shape[1] - x)
        
        tile_mask = full_mask[y:y+win_h, x:x+win_w]
        
        if tile_mask.shape[0] < 512 or tile_mask.shape[1] < 512:
            padded = np.zeros((512, 512), dtype=tile_mask.dtype)
            padded[:tile_mask.shape[0], :tile_mask.shape[1]] = tile_mask
            tile_mask = padded
            
        out_path = out_ann_dir / f"{tile_id}_mask.png"
        cv2.imwrite(str(out_path), tile_mask)
        success_count += 1
        
    logger.info(f"Generated {success_count} annotation tile masks.")

def main():
    shp_dir = Path("data/raw/CG_shp-file/shp-file")
    clean_dir = Path("data/clean_dataset_2")
    
    regions_dir = clean_dir / "regions"
    meta_path = clean_dir / "metadata" / "tiles.json"
    
    regions_masks_dir = clean_dir / "regions_masks"
    out_ann_dir = clean_dir / "annotations"
    
    # 1. Rasterize Region Masks
    for tif_path in regions_dir.glob("*.tif"):
        out_mask_path = regions_masks_dir / f"{tif_path.stem}_mask.png"
        rasterize_region(tif_path, shp_dir, out_mask_path)
        
    # 2. Slice into 512x512 tiles
    slice_masks(meta_path, regions_masks_dir, out_ann_dir)

if __name__ == "__main__":
    main()
