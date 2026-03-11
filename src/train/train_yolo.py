"""
Hydra-Map Small Object Squad (Step 2).

YOLOv8 Nano training script for small objects detection.
Generates YOLO-format labels from annotation masks using connected component analysis,
creates 640x640 crops from orthophoto tiles, and trains YOLOv8n.

Classes in YOLO labels:
  0: Building, 1: Road, 2: Vegetation, 3: Water

CLI: python src/train/train_yolo.py --config config/config.yaml --epochs 50
"""

import os
import sys
import glob
import argparse
import logging
import shutil

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────── Label Extraction ────────────────────────────

def extract_yolo_labels_from_masks(tile_dir: str, mask_dir: str, output_dir: str, 
                                    crop_size: int = 640, min_area: int = 100):
    """Extract YOLO bounding box labels from annotation masks.
    
    For each tile, finds connected components in the mask for each non-background
    class and generates YOLO-format labels (class_id, cx, cy, w, h normalized).
    
    Args:
        tile_dir: Directory containing tile images.
        mask_dir: Directory containing mask PNGs.
        output_dir: Root output directory for YOLO dataset.
        crop_size: Crop size for training images.
        min_area: Minimum pixel area for a component to become a bounding box.
    """
    img_out = os.path.join(output_dir, "images", "train")
    lbl_out = os.path.join(output_dir, "labels", "train")
    val_img_out = os.path.join(output_dir, "images", "val")
    val_lbl_out = os.path.join(output_dir, "labels", "val")
    
    for d in [img_out, lbl_out, val_img_out, val_lbl_out]:
        os.makedirs(d, exist_ok=True)
    
    tile_files = sorted(glob.glob(os.path.join(tile_dir, "*.png")))
    
    total_boxes = 0
    class_counts = {}
    
    for idx, tile_path in enumerate(tile_files):
        basename = os.path.splitext(os.path.basename(tile_path))[0]
        mask_path = os.path.join(mask_dir, f"{basename}_mask.png")
        
        if not os.path.exists(mask_path):
            continue
        
        img = cv2.imread(tile_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            continue
        
        h, w = img.shape[:2]
        labels = []
        
        # Extract bounding boxes for each non-background class
        for class_id in range(1, 5):  # 1=Building, 2=Road, 3=Vegetation, 4=Water
            class_mask = (mask == class_id).astype(np.uint8)
            
            if class_mask.sum() == 0:
                continue
            
            # Connected component analysis
            num_labels, label_map, stats, centroids = cv2.connectedComponentsWithStats(
                class_mask, connectivity=8
            )
            
            for comp_id in range(1, num_labels):  # Skip background (0)
                area = stats[comp_id, cv2.CC_STAT_AREA]
                if area < min_area:
                    continue
                
                x = stats[comp_id, cv2.CC_STAT_LEFT]
                y = stats[comp_id, cv2.CC_STAT_TOP]
                bw = stats[comp_id, cv2.CC_STAT_WIDTH]
                bh = stats[comp_id, cv2.CC_STAT_HEIGHT]
                
                # YOLO format: class_id cx cy w h (all normalized 0-1)
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h
                
                # Map to YOLO classes (0-indexed): Building=0, Road=1, Vegetation=2, Water=3
                yolo_class = class_id - 1
                labels.append(f"{yolo_class} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                
                total_boxes += 1
                class_counts[yolo_class] = class_counts.get(yolo_class, 0) + 1
        
        # 80/20 split
        is_val = idx % 5 == 0
        dest_img = val_img_out if is_val else img_out
        dest_lbl = val_lbl_out if is_val else lbl_out
        
        # Resize image to crop_size for YOLO
        resized = cv2.resize(img, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(dest_img, f"{basename}.jpg"), resized)
        
        # Write label file
        label_file = os.path.join(dest_lbl, f"{basename}.txt")
        with open(label_file, 'w') as f:
            f.write('\n'.join(labels))
    
    logger.info(f"Extracted {total_boxes} bounding boxes from {len(tile_files)} tiles")
    logger.info(f"Class distribution: {class_counts}")
    
    return total_boxes


def create_data_yaml(output_dir: str, yaml_path: str):
    """Create the YOLO data.yaml configuration file."""
    content = f"""# Hydra-Map YOLO Small Object Dataset
# Auto-generated from village orthophoto masks

path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

nc: 4
names:
  0: Building
  1: Road
  2: Vegetation
  3: Water
"""
    with open(yaml_path, 'w') as f:
        f.write(content)
    logger.info(f"Created data.yaml at {yaml_path}")


# ──────────────────────────── 4K Cropping ────────────────────────────

def crop_4k_to_640(image_dir: str, output_dir: str, crop_size: int = 640, overlap: int = 120):
    """Crop 4K imagery into overlapping 640x640 patches without resizing."""
    os.makedirs(output_dir, exist_ok=True)
    images = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
    
    logger.info(f"Slicing {len(images)} images into {crop_size}x{crop_size} patches...")
    
    stride = crop_size - overlap
    count = 0
    
    for img_path in images:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y1, y2 = y, min(y + crop_size, h)
                x1, x2 = x, min(x + crop_size, w)
                
                crop = img[y1:y2, x1:x2]
                ch, cw = crop.shape[:2]
                
                if ch < crop_size or cw < crop_size:
                    padded = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                    padded[:ch, :cw] = crop
                    crop = padded
                    
                crop_name = f"{os.path.splitext(img_name)[0]}_{y}_{x}.jpg"
                cv2.imwrite(os.path.join(output_dir, crop_name), crop)
                count += 1
                
    logger.info(f"Created {count} crops.")


# ──────────────────────────── Training ────────────────────────────

def train_yolov8_nano(data_yaml: str, epochs: int = 50, batch: int = 8, 
                       output_dir: str = "output/yolo_training"):
    """Train YOLOv8n on the extracted dataset."""
    from ultralytics import YOLO
    
    logger.info("Loading YOLOv8 Nano architecture...")
    model = YOLO("yolov8n.pt")
    
    logger.info(f"Starting training on {data_yaml} for {epochs} epochs...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        device="0",
        project=output_dir,
        name="hydra_small_objs",
        patience=15,
        optimizer="auto",
        cache=True,
        exist_ok=True,
    )
    
    # Copy best weights to models directory
    best_weights = os.path.join(output_dir, "hydra_small_objs", "weights", "best.pt")
    if os.path.exists(best_weights):
        dest = os.path.join("models", "yolo", "yolov8n_village_best.pt")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(best_weights, dest)
        logger.info(f"Best weights saved to {dest}")
    
    return results


if __name__ == "__main__":
    from src.utils.io import load_config
    
    parser = argparse.ArgumentParser(description="Hydra-Map YOLO Small Object Training")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--img-dir", type=str, help="Optional 4K imagery dir to crop")
    parser.add_argument("--skip-extract", action="store_true", help="Skip label extraction")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Setup paths
    tile_size = config["swin"]["input_size"]
    tile_dir = os.path.join(config["data"]["preprocessed_dir"], "tiles", str(tile_size))
    mask_dir = config["data"]["annotations_dir"]
    yolo_dataset_dir = os.path.join("data", "yolo_dataset")
    data_yaml = os.path.join(yolo_dataset_dir, "data.yaml")
    
    # Step 1: Extract labels from masks
    if not args.skip_extract:
        logger.info("=== Step 1: Extracting YOLO labels from annotation masks ===")
        num_boxes = extract_yolo_labels_from_masks(tile_dir, mask_dir, yolo_dataset_dir)
        
        if num_boxes == 0:
            logger.error("No bounding boxes extracted! Check your annotations.")
            sys.exit(1)
        
        create_data_yaml(yolo_dataset_dir, data_yaml)
    
    # Step 2: Optional 4K cropping
    if args.img_dir:
        logger.info("=== Step 2: Cropping 4K imagery ===")
        crop_4k_to_640(args.img_dir, os.path.join(yolo_dataset_dir, "images", "train"))
    
    # Step 3: Train
    logger.info("=== Step 3: Training YOLOv8 Nano ===")
    train_yolov8_nano(data_yaml, epochs=args.epochs, batch=args.batch)
