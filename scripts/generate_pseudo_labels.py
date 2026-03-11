"""
Generate High-Confidence Pseudo-Labels for the Teacher-Student Pipeline.

Loads the pre-trained Teacher Swin-UNet map model and runs inference over 
the new unannotated dataset tiles. It filters objects by a confidence threshold (>0.85)
and discards completely unconfident tiles.
"""

import argparse
import glob
import logging
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.swin_unet import SwinUNet
from src.utils.io import load_config, ensure_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def generate_pseudo_labels(config: dict, tile_dir: Path, output_dir: Path, weight_path: str, conf_thresh: float = 0.85):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Generating pseudo-labels on {device}...")
    
    # 1. Load Model
    num_classes = config["swin"]["num_classes"]
    model = SwinUNet(num_classes=num_classes)
    
    logger.info(f"Loading weights from: {weight_path}")
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
    else:
        logger.error(f"Cannot find weights at {weight_path}")
        return
        
    model.to(device)
    model.eval()
    
    ensure_dir(str(output_dir))
    
    tiles = sorted(glob.glob(str(tile_dir / "*.png")))
    logger.info(f"Found {len(tiles)} tiles to pseudo-label.")
    
    accept_count = 0
    discard_count = 0
    
    with torch.no_grad():
        for i, tile_path in enumerate(tiles):
            base_name = os.path.basename(tile_path)
            
            # Load and preprocess
            img = cv2.imread(tile_path)
            if img is None:
                continue
            orig_h, orig_w = img.shape[:2]
            
            # Resize exactly as in training
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
            
            tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device)
            
            # Inference
            output = model(tensor)
            logits = output["logits"]
            
            # probabilities (B, C, H, W)
            probs = F.softmax(logits, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            
            max_probs = max_probs[0].cpu().numpy()
            preds = preds[0].cpu().numpy()
            
            # Resize back to original
            # Note: Nearest Neighbor crucial for categorical masks
            preds_orig = cv2.resize(preds.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            probs_orig = cv2.resize(max_probs, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            # Apply Confidence Filter
            # If prob < 0.85, set to background (0)
            pseudo_mask = np.where(probs_orig >= conf_thresh, preds_orig, 0)
            
            # Apply Consistency Check (Skip tiles that are purely background after thresholding)
            # Actually, background is 0. If there are NO buildings (1), roads (2), or water (4)
            # and it's 100% background, we might discard it so we don't train only on empty space.
            # But the village background is useful. Let's keep it if there's any valid structure,
            # or if it's a solid terrain. For strong Teacher-Student, let's discard pure background tiles.
            unique, counts = np.unique(pseudo_mask, return_counts=True)
            class_counts = dict(zip(unique, counts))
            
            non_bg_pixels = sum(count for cls, count in class_counts.items() if cls != 0)
            
            if non_bg_pixels < 50:  # less than ~50 pixels of anything interesting
                discard_count += 1
            else:
                out_path = output_dir / f"{os.path.splitext(base_name)[0]}_mask.png"
                cv2.imwrite(str(out_path), pseudo_mask)
                accept_count += 1
                
            if (i+1) % 500 == 0:
                logger.info(f"Processed {i+1}/{len(tiles)}... Accepted: {accept_count}, Discarded: {discard_count}")
                
    logger.info(f"Pseudo-labeling complete! Generated {accept_count} high-confidence masks. Discarded {discard_count} empty tiles.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--tiles_dir", type=str, default="data/clean_dataset_2/tiles")
    parser.add_argument("--out_dir", type=str, default="data/clean_dataset_2/pseudo_masks")
    parser.add_argument("--weights", type=str, default="models/swin_fold0_best.pth")
    parser.add_argument("--conf", type=float, default=0.85)
    args = parser.parse_args()
    
    config = load_config(args.config)
    generate_pseudo_labels(
        config=config,
        tile_dir=Path(args.tiles_dir),
        output_dir=Path(args.out_dir),
        weight_path=args.weights,
        conf_thresh=args.conf
    )
