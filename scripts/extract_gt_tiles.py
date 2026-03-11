import os
import json
import logging
import rasterio
import geopandas as gpd
import numpy as np
import cv2
from rasterio.windows import Window
from shapely.geometry import box
from rasterio import features as rio_features
from shapely.strtree import STRtree

logging.basicConfig(level=logging.INFO, format="%(asctime)s [GT_EXTRACT] %(message)s")
logger = logging.getLogger(__name__)

def extract_gt_tiles(tif_path, shp_dir, output_dir, tile_size=512, stride=1024, max_tiles=200):
    os.makedirs(output_dir, exist_ok=True)
    tile_out = os.path.join(output_dir, "tiles", str(tile_size))
    mask_out = os.path.join(output_dir, "masks")
    os.makedirs(tile_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    with rasterio.open(tif_path) as src:
        tif_crs = src.crs
        height, width = src.height, src.width
        logger.info(f"TIF: {width}x{height}, CRS: {tif_crs}")
        
        # Load SHPs
        SHP_CLASS_MAP = {"Built_Up_Area_type": 1, "Road": 2, "Water_Body": 4}
        all_shapes = []
        geoms = []
        for shp_file in os.listdir(shp_dir):
            if not shp_file.endswith(".shp"): continue
            base_name = os.path.splitext(shp_file)[0]
            class_id = SHP_CLASS_MAP.get(base_name, 0)
            if class_id == 0: continue
            
            gdf = gpd.read_file(os.path.join(shp_dir, shp_file))
            if gdf.crs != tif_crs:
                gdf = gdf.to_crs(tif_crs)
            for geom in gdf.geometry:
                if geom is not None and not geom.is_empty:
                    all_shapes.append((geom, class_id))
                    geoms.append(geom)
        
        if not all_shapes:
            logger.error("No shapes found for ground truth.")
            return

        # S-Index for fast overlap check
        tree = STRtree(geoms)
        logger.info(f"Loaded {len(all_shapes)} geometries and built S-Index.")

        count = 0
        # Grid scan
        for row in range(0, height, stride):
            if count >= max_tiles: break
            for col in range(0, width, stride):
                if count >= max_tiles: break
                
                win_w = min(tile_size, width - col)
                win_h = min(tile_size, height - row)
                if win_w < 400 or win_h < 400: continue
                
                # Fast check using S-Index
                win_bounds = rasterio.windows.bounds(Window(col, row, win_w, win_h), src.transform)
                win_box = box(*win_bounds)
                
                # query indices of geoms that intersect win_box
                indices = tree.query(win_box)
                if len(indices) == 0: continue
                
                # Check for actual intersection if needed (some may just touch bbox)
                intersecting = []
                for idx in indices:
                    geom, class_id = all_shapes[idx]
                    if geom.intersects(win_box):
                        intersecting.append((geom, class_id))
                
                if not intersecting: continue 
                
                # Read 
                data = src.read(window=Window(col, row, win_w, win_h))
                if data.shape[0] < 3: continue
                img = np.transpose(data[:3], (1, 2, 0))
                
                # Rasterize with specific class values
                mask = rio_features.rasterize(
                    intersecting,
                    out_shape=(win_h, win_w),
                    transform=src.window_transform(Window(col, row, win_w, win_h)),
                    fill=0,
                    dtype=np.uint8
                )
                
                if mask.sum() < 200: continue 
                
                tile_id = f"real_gt_{count:05d}"
                cv2.imwrite(os.path.join(tile_out, f"{tile_id}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(mask_out, f"{tile_id}_mask.png"), mask)
                
                # Meta
                meta = {"tile_id": tile_id, "pixel_offset": [col, row], "crs": str(tif_crs)}
                with open(os.path.join(tile_out, f"tile_{tile_id}.json"), "w") as f:
                    json.dump(meta, f)
                
                count += 1
                if count % 10 == 0:
                    logger.info(f"Extracted {count}/{max_tiles} high-value tiles...")

    logger.info(f"Extraction complete. Total: {count}")

if __name__ == "__main__":
    extract_gt_tiles(
        "data/combined_real/BADETUMNAR_450157_BANGAPAL_450155_CHHOTETUMAR_450149_MOFALNAR_450150_ORTHO.tif",
        "data/combined_real/shp",
        "data/preprocessed_ready_real_gt",
        max_tiles=100
    )
