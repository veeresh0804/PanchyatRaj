"""
Hydra-Map Context Engine (Step 3).

Swin-UNet Transfer Learning Script.
Uses the `timm` library to load a pretrained Swin Transformer, freezes the encoder,
and trains only the decoder and classification head.

Trained on REAL village orthophoto data from:
  data/preprocessed/tiles/512/ (images)
  data/annotations/ (masks)

Classes: 0: Background, 1: Building, 2: Road, 3: Vegetation, 4: Water, 5: Other

CLI: python src/train/train_swin_transfer.py --config config/config.yaml --epochs 30
"""

import os
import sys
import argparse
import logging
import glob
import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import timm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────── Model ────────────────────────────

class SwinTransferSegmenter(nn.Module):
    """Segmentation model with frozen Swin Transformer backbone."""
    
    def __init__(self, num_classes: int = 6, pretrained_model: str = "swin_tiny_patch4_window7_224"):
        super().__init__()
        
        logger.info(f"Loading pretrained {pretrained_model} from timm...")
        self.backbone = timm.create_model(
            pretrained_model, 
            pretrained=True, 
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        
        # Freeze the Backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        logger.info("Backbone frozen successfully.")
        
        # Feature channels of Swin-Tiny (96, 192, 384, 768)
        ch_in = [96, 192, 384, 768]
        
        # Simple UNet-style Decoder
        self.decoder3 = self._decoder_block(ch_in[3], ch_in[2])
        self.decoder2 = self._decoder_block(ch_in[2] * 2, ch_in[1])
        self.decoder1 = self._decoder_block(ch_in[1] * 2, ch_in[0])
        self.decoder0 = self._decoder_block(ch_in[0] * 2, 64)
        
        # Classification Head
        self.head = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2.0)
        self.final_upsample = nn.UpsamplingBilinear2d(scale_factor=4.0)

    def _decoder_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skips = self.backbone(x)
        # Swin features from timm are (B, H, W, C), need (B, C, H, W)
        f0 = skips[0].permute(0, 3, 1, 2)
        f1 = skips[1].permute(0, 3, 1, 2)
        f2 = skips[2].permute(0, 3, 1, 2)
        f3 = skips[3].permute(0, 3, 1, 2)
        
        # Decode
        d3 = self.decoder3(f3)
        d3_up = self.upsample(d3)
        
        d2 = self.decoder2(torch.cat([d3_up, f2], dim=1))
        d2_up = self.upsample(d2)
        
        d1 = self.decoder1(torch.cat([d2_up, f1], dim=1))
        d1_up = self.upsample(d1)
        
        d0 = self.decoder0(torch.cat([d1_up, f0], dim=1))
        d0_up = self.final_upsample(d0)
        
        logits = self.head(d0_up)
        if logits.shape[2:] != x.shape[2:]:
            logits = nn.functional.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
            
        return {"logits": logits}


# ──────────────────────────── Dataset ────────────────────────────

class VillageSegmentationDataset(Dataset):
    """Dataset for SVAMITVA village orthophoto segmentation.
    
    Loads 512×512 tile images and matching annotation masks from
    the preprocessed tiles directory.
    """
    
    def __init__(self, tile_paths: List[str], mask_paths: List[str], 
                 input_size: int = 224, augment: bool = True, num_classes: int = 6):
        self.tile_paths = tile_paths
        self.mask_paths = mask_paths
        self.input_size = input_size
        self.augment = augment
        self.num_classes = num_classes
        
        try:
            import albumentations as A
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
                A.GaussNoise(var_limit=(5, 25), p=0.3),
            ]) if augment else None
        except ImportError:
            self.transform = None
            logger.warning("albumentations not installed, skipping augmentation")
    
    def __len__(self):
        return len(self.tile_paths)
    
    def __getitem__(self, idx):
        # Load image
        img = cv2.imread(self.tile_paths[idx])
        if img is None:
            raise RuntimeError(f"Failed to load image: {self.tile_paths[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to load mask: {self.mask_paths[idx]}")
        
        # Resize to input_size (Swin-Base expects 224×224)
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
        
        # Apply augmentations
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        # Convert to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()
        
        # Clamp mask values to valid range
        mask = torch.clamp(mask, 0, self.num_classes - 1)
        
        return img, mask


# ──────────────────────────── Data Loading ────────────────────────────

def find_tile_mask_pairs(config: dict) -> Tuple[List[str], List[str]]:
    """Find matching tile images and annotation masks.
    
    Returns:
        Tuple of (tile_paths, mask_paths) lists.
    """
    tile_size = config["swin"]["input_size"]
    tile_dir = os.path.join(config["data"]["preprocessed_dir"], "tiles", str(tile_size))
    mask_dir = config["data"]["annotations_dir"]
    
    # Get all tile images (not masks, not JSON)
    tile_files = sorted(glob.glob(os.path.join(tile_dir, "*.png")))
    
    tile_paths = []
    mask_paths = []
    
    for tf in tile_files:
        basename = os.path.splitext(os.path.basename(tf))[0]
        mask_file = os.path.join(mask_dir, f"{basename}_mask.png")
        
        if os.path.exists(mask_file):
            tile_paths.append(tf)
            mask_paths.append(mask_file)
    
    logger.info(f"Found {len(tile_paths)} tile-mask pairs from {tile_dir}")
    return tile_paths, mask_paths


def split_data(tile_paths, mask_paths, val_ratio=0.2, seed=42):
    """Split data into train and validation sets."""
    combined = list(zip(tile_paths, mask_paths))
    random.seed(seed)
    random.shuffle(combined)
    
    split_idx = int(len(combined) * (1 - val_ratio))
    train_pairs = combined[:split_idx]
    val_pairs = combined[split_idx:]
    
    train_tiles, train_masks = zip(*train_pairs) if train_pairs else ([], [])
    val_tiles, val_masks = zip(*val_pairs) if val_pairs else ([], [])
    
    return list(train_tiles), list(train_masks), list(val_tiles), list(val_masks)


# ──────────────────────────── Training ────────────────────────────

def train_transfer(config: dict, epochs: int = 30, batch_size: int = 4, lr: float = 1e-4):
    """Train the frozen-encoder segmenter on real village data."""
    from src.utils.io import ensure_dir
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")
    
    # Find real data
    tile_paths, mask_paths = find_tile_mask_pairs(config)
    
    if len(tile_paths) == 0:
        logger.error("No tile-mask pairs found! Check data/preprocessed/tiles and data/annotations/")
        return
    
    # Split
    train_tiles, train_masks, val_tiles, val_masks = split_data(tile_paths, mask_paths)
    logger.info(f"Train: {len(train_tiles)}, Val: {len(val_tiles)}")
    
    # Create datasets
    num_classes = config["swin"]["num_classes"]
    input_size = 224  # Swin-Base optimal input
    
    train_ds = VillageSegmentationDataset(train_tiles, train_masks, input_size=input_size, augment=True, num_classes=num_classes)
    val_ds = VillageSegmentationDataset(val_tiles, val_masks, input_size=input_size, augment=False, num_classes=num_classes)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Build model
    encoder_name = config["swin"].get("encoder", "swin_tiny_patch4_window7_224")
    
    model = SwinTransferSegmenter(num_classes=num_classes, pretrained_model=encoder_name)
    model.to(device)
    
    # Only decoder params will be updated
    decoder_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters: {sum(p.numel() for p in decoder_params):,}")
    
    optimizer = torch.optim.AdamW(decoder_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    # Training loop
    output_dir = ensure_dir(os.path.join(config["data"]["models_dir"], "swin_transfer"))
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            output = model(images)
            logits = output["logits"]
            
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_loss / max(num_batches, 1)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        correct_pixels = 0
        total_pixels = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                output = model(images)
                logits = output["logits"]
                loss = criterion(logits, masks)
                
                val_loss += loss.item()
                val_batches += 1
                
                preds = logits.argmax(dim=1)
                correct_pixels += (preds == masks).sum().item()
                total_pixels += masks.numel()
        
        avg_val_loss = val_loss / max(val_batches, 1)
        pixel_acc = correct_pixels / max(total_pixels, 1) * 100
        
        scheduler.step()
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Pixel Acc: {pixel_acc:.1f}%")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(output_dir, "swin_transfer_best.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"  → Saved best model (val_loss={avg_val_loss:.4f}) to {save_path}")
    
    logger.info(f"Transfer training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    from src.utils.io import load_config
    
    parser = argparse.ArgumentParser(description="Swin Transfer Learning on Village Data")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    config = load_config(args.config)
    train_transfer(config, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
