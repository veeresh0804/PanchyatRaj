import rasterio
import rasterio.features
import numpy as np
import time
from shapely.geometry import shape, box
from shapely.affinity import scale

tif_path = "data/raw/PB_live_demo_3/live_demo_3/KARTARPUR_AMRITSAR_37842_ORTHO/KARTARPUR_AMRITSAR_37842_ORTHO.tif"

def get_valid_islands(src, downsample_factor=32):
    t0 = time.time()
    out_shape = (1, int(src.height / downsample_factor), int(src.width / downsample_factor))
    img = src.read(1, out_shape=out_shape)
    
    mask = (img > 0).astype(np.uint8)
    
    # Close small holes
    import cv2
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    shapes = list(rasterio.features.shapes(mask, mask=mask))
    
    islands_px = []
    for geom, val in shapes:
        poly = shape(geom)
        poly = scale(poly, xfact=downsample_factor, yfact=downsample_factor, origin=(0, 0))
        islands_px.append(poly)
    
    print(f"Extracted {len(islands_px)} islands in {time.time() - t0:.2f}s")
    return islands_px

with rasterio.open(tif_path) as src:
    islands = get_valid_islands(src)
    
    # Simulate tiling
    stride = 512
    tile_size = 512
    valid_tiles = 0
    skipped_tiles = 0
    
    for row_off in range(0, src.height, stride):
        for col_off in range(0, src.width, stride):
            tile_box = box(col_off, row_off, col_off + tile_size, row_off + tile_size)
            if any(island.intersects(tile_box) for island in islands):
                valid_tiles += 1
            else:
                skipped_tiles += 1
                
    print(f"Valid Tiles: {valid_tiles}, Skipped: {skipped_tiles}")
