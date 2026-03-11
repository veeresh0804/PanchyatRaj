"""
Hydra-Map Gatekeeper (Step 1).

Compute pixel intensity variance and discard empty/flat tiles 
to save compute downstream.
"""

import cv2
import numpy as np

class Gatekeeper:
    """Variance filter to reject empty/meaningless tiles."""
    
    def __init__(self, variance_threshold: float = 100.0):
        self.variance_threshold = variance_threshold
        
    def check_tile(self, image: np.ndarray) -> bool:
        """Check if tile has enough variance to be processed.
        
        Args:
            image: RGB or BGR image array.
            
        Returns:
            True if tile should be processed, False if it should be skipped.
        """
        if image is None or image.size == 0:
            return False
            
        # Convert to grayscale for variance calculation
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        variance = np.var(gray)
        return variance >= self.variance_threshold

    def filter_batch(self, images: list, tile_ids: list) -> tuple:
        """Filter a batch of images.
        
        Returns:
            (valid_images, valid_tile_ids, skipped_tile_ids)
        """
        valid_images = []
        valid_tile_ids = []
        skipped_tile_ids = []
        
        for img, tid in zip(images, tile_ids):
            if self.check_tile(img):
                valid_images.append(img)
                valid_tile_ids.append(tid)
            else:
                skipped_tile_ids.append(tid)
                
        return valid_images, valid_tile_ids, skipped_tile_ids
