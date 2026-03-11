"""
Hydra-Map Teacher-Student Training Pipeline.

Trains the Student Swin-UNet using the generated pseudo-labels from the Teacher model.
This pipeline uses heavy augmentations such as Albumentations, MixUp, and CutMix
to enforce robust generalization beyond the Teacher's initial predictions.
"""

import argparse
import logging
import os
import random
import sys
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.swin_unet import SwinUNet
from src.utils.io import ensure_dir, load_config
from src.utils.metrics import DiceCELoss, pixel_iou
from src.train.train_swin import cutmix_batch, mixup_batch, SegmentationDataset, get_fold_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def collect_student_data(config: Dict) -> Tuple[List[str], List[str]]:
    """Collect tile and pseudo-mask paths for student training."""
    
    # These can be configured or hard-coded to the clean_dataset_2 for the teacher-student loop
    tile_dir = Path("data/clean_dataset_2/tiles")
    pseudo_dir = Path("data/clean_dataset_2/pseudo_masks")
    
    tile_paths = []
    mask_paths = []
    
    # We only train on tiles that HAVE a generated pseudo mask (since empty ones were discarded)
    masks = sorted(pseudo_dir.glob("*_mask.png"))
    for mp in masks:
        base_name = mp.name.replace("_mask.png", ".png")
        tp = tile_dir / base_name
        
        if tp.exists():
            tile_paths.append(str(tp))
            mask_paths.append(str(mp))
            
    logger.info(f"Student Data Collection: Found {len(tile_paths)} tile-mask pairs out of {len(list(tile_dir.glob('*.png')))} total tiles.")
    return tile_paths, mask_paths

def train_student(config: Dict, epochs: int = 50, batch_size: int = 8, lr: float = 1e-4):
    """Main function to train the Student model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training Student Swin-UNet on {device}...")
    
    tile_paths, mask_paths = collect_student_data(config)
    
    if len(tile_paths) == 0:
        logger.error("No pseudo-labeled data found! Ensure `generate_pseudo_labels.py` completed successfully.")
        return
        
    # Standard 80/20 train/val split since we don't have multiple folds for the self-training logic yet
    train_tiles, train_masks, val_tiles, val_masks = get_fold_splits(tile_paths, mask_paths, fold_k=5, fold_idx=0)
    
    # Datasets
    input_size = config["swin"]["input_size"]
    train_ds = SegmentationDataset(train_tiles, train_masks, input_size=input_size, augment=True, use_heavy_augs=True)
    val_ds = SegmentationDataset(val_tiles, val_masks, input_size=input_size, augment=False, use_heavy_augs=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    
    # Model
    num_classes = config["swin"]["num_classes"]
    model = SwinUNet(num_classes=num_classes)
    
    # Optional: Load Teacher weights as initialization to speed up student learning
    teacher_weights = config.get("student_training", {}).get("init_weights", "models/swin/swin_fold0_best.pth")
    if os.path.exists(teacher_weights):
        logger.info(f"Initializing Student with Teacher weights: {teacher_weights}")
        model.load_state_dict(torch.load(teacher_weights, map_location=device))
    else:
        logger.info("Initializing Student from scratch (No teacher weights found).")
        
    model.to(device)
    
    # Loss & Optimizer
    criterion = DiceCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    output_dir = ensure_dir("models/swin_student")
    best_iou = 0.0
    
    aug_cfg = config.get("augmentation", {})
    use_cutmix = aug_cfg.get("use_cutmix", True)
    use_mixup = aug_cfg.get("use_mixup", True)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Apply CutMix/MixUp on the pseudo-labels to force student generalization
            if use_cutmix and random.random() < 0.5:
                images, masks = cutmix_batch(images, masks, aug_cfg.get("cutmix_alpha", 1.0))
            elif use_mixup and random.random() < 0.5:
                images, masks = mixup_batch(images, masks, aug_cfg.get("mixup_alpha", 0.4))
                
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output["logits"], masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                output = model(images)
                loss = criterion(output["logits"], masks)
                val_loss += loss.item()
                
                preds = output["logits"].argmax(dim=1).cpu().numpy()
                targets = masks.cpu().numpy()
                all_preds.append(preds)
                all_targets.append(targets)
                
        avg_val_loss = val_loss / max(len(val_loader), 1)
        
        ious = []
        preds_cat = np.concatenate(all_preds, axis=0)
        targets_cat = np.concatenate(all_targets, axis=0)
        for i in range(len(preds_cat)):
            ious.append(pixel_iou(preds_cat[i], targets_cat[i], num_classes)["mIoU"])
        avg_miou = np.mean(ious)
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val mIoU: {avg_miou:.4f}")
        
        if avg_miou > best_iou:
            best_iou = avg_miou
            save_path = os.path.join(output_dir, "swin_student_best.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"  → Saved new best Student model (mIoU: {best_iou:.4f})")
            
    logger.info(f"Student Training Complete. Best Val mIoU: {best_iou:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Teacher-Student Swin-UNet Training")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    train_student(cfg, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
