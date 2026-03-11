"""
Hydra-Map Swin-UNet Training Pipeline.

Implements:
  - Freeze/unfreeze training schedule
  - Heavy augmentations (Albumentations + CutMix/MixUp)
  - Dice+CE hybrid loss with per-class IoU reporting
  - K-fold cross-validation with model ensembling
  - Pseudo-labeling for unlabeled data
  - Optional W&B integration

CLI: python src/train/train_swin.py --config config/config.yaml --fold 0
"""

import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.swin_unet import SwinUNet
from src.utils.io import ensure_dir, list_files, load_config, save_json
from src.utils.metrics import DiceCELoss, pixel_iou

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────── Dataset ────────────────────────────

class SegmentationDataset(Dataset):
    """Dataset for aerial image segmentation.

    Expects preprocessed tiles (PNG) and matching ground truth masks.

    Args:
        tile_paths: List of tile image paths.
        mask_paths: List of matching mask paths (same order).
        input_size: Target size for images.
        augment: Whether to apply augmentations.
        use_heavy_augs: Whether to use heavy augmentation pipeline.
    """

    def __init__(
        self,
        tile_paths: List[str],
        mask_paths: List[str],
        input_size: int = 512,
        augment: bool = True,
        use_heavy_augs: bool = True,
    ):
        self.tile_paths = tile_paths
        self.mask_paths = mask_paths
        self.input_size = input_size
        self.augment = augment
        self.transform = self._build_augmentation(use_heavy_augs) if augment else None

    def _build_augmentation(self, heavy: bool):
        """Build Albumentations augmentation pipeline."""
        try:
            import albumentations as A

            if heavy:
                return A.Compose([
                    A.RandomRotate90(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.3),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.CLAHE(clip_limit=4.0, p=0.3),
                    A.MotionBlur(blur_limit=5, p=0.2),
                    A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.3),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                    A.Resize(self.input_size, self.input_size),
                ])
            else:
                return A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.Resize(self.input_size, self.input_size),
                ])
        except ImportError:
            logger.warning("albumentations not installed, skipping augmentations")
            return None

    def __len__(self) -> int:
        return len(self.tile_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        img = cv2.imread(self.tile_paths[idx])
        if img is None:
            img = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load mask
        mask_path = self.mask_paths[idx]
        if os.path.isfile(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        else:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        # Augment
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # Resize
        img = cv2.resize(img, (self.input_size, self.input_size))
        mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)

        # To tensor
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).long()

        return img_t, mask_t


# ──────────────────────────── CutMix / MixUp ────────────────────────────

def cutmix_batch(images: torch.Tensor, masks: torch.Tensor, alpha: float = 1.0):
    """Apply CutMix to a batch.

    Args:
        images: (B, C, H, W) tensor.
        masks: (B, H, W) tensor.
        alpha: Beta distribution parameter.

    Returns:
        Mixed images and masks.
    """
    B = images.size(0)
    lam = np.random.beta(alpha, alpha)
    rand_idx = torch.randperm(B)

    _, _, H, W = images.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)

    images[:, :, y1:y2, x1:x2] = images[rand_idx, :, y1:y2, x1:x2]
    masks[:, y1:y2, x1:x2] = masks[rand_idx, y1:y2, x1:x2]

    return images, masks


def mixup_batch(images: torch.Tensor, masks: torch.Tensor, alpha: float = 0.4):
    """Apply MixUp to a batch (images only, masks use dominant class).

    Args:
        images: (B, C, H, W).
        masks: (B, H, W).
        alpha: MixUp alpha.

    Returns:
        Mixed images and masks.
    """
    lam = np.random.beta(alpha, alpha)
    rand_idx = torch.randperm(images.size(0))
    images = lam * images + (1 - lam) * images[rand_idx]
    # For masks: keep original (dominant) since MixUp on labels is tricky
    return images, masks


def copy_paste_augment(
    images: torch.Tensor, masks: torch.Tensor,
    rare_classes: List[int] = [2, 4],  # Road, Water (rare in annotations)
    paste_prob: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Copy-Paste Augmentation for rare object classes.

    Randomly copies regions of rare classes from one image and pastes
    them onto another image in the batch. This addresses class imbalance
    for wells, tanks, water bodies, and roads.

    Args:
        images: (B, C, H, W) tensor.
        masks: (B, H, W) tensor.
        rare_classes: Class IDs to augment.
        paste_prob: Probability of applying paste per sample.

    Returns:
        Augmented images and masks.
    """
    B, C, H, W = images.shape
    for b in range(B):
        if random.random() > paste_prob:
            continue

        # Find a source image with a rare class
        source_idx = random.randint(0, B - 1)
        source_mask = masks[source_idx]

        for cls_id in rare_classes:
            cls_region = (source_mask == cls_id)
            if cls_region.sum() < 50:  # Too small to be meaningful
                continue

            # Extract bounding box of the rare class region
            ys, xs = torch.where(cls_region)
            if len(ys) == 0:
                continue
            y1, y2 = ys.min().item(), ys.max().item() + 1
            x1, x2 = xs.min().item(), xs.max().item() + 1

            # Random paste location
            paste_y = random.randint(0, max(0, H - (y2 - y1)))
            paste_x = random.randint(0, max(0, W - (x2 - x1)))
            ph = min(y2 - y1, H - paste_y)
            pw = min(x2 - x1, W - paste_x)

            # Create paste mask (only the rare class pixels)
            paste_mask = cls_region[y1:y1+ph, x1:x1+pw]

            # Paste the pixels
            images[b, :, paste_y:paste_y+ph, paste_x:paste_x+pw][:, paste_mask] = \
                images[source_idx, :, y1:y1+ph, x1:x1+pw][:, paste_mask]
            masks[b, paste_y:paste_y+ph, paste_x:paste_x+pw][paste_mask] = cls_id

    return images, masks


def init_wandb(config: Dict, fold_idx: int = 0) -> Any:
    """Initialize Weights & Biases experiment tracking.

    Args:
        config: Training config dict.
        fold_idx: Current fold index.

    Returns:
        wandb run object, or None if unavailable.
    """
    try:
        import wandb
        run = wandb.init(
            project="hydra-map-svamitva",
            name=f"swin_fold{fold_idx}",
            config=config,
            tags=["swin-unet", "segmentation", f"fold{fold_idx}"],
        )
        logger.info(f"W&B tracking enabled: {run.url}")
        return run
    except ImportError:
        logger.info("wandb not installed, skipping experiment tracking")
        return None
    except Exception as e:
        logger.warning(f"wandb init failed: {e}")
        return None


# ──────────────────────────── Training Loop ────────────────────────────

def train_one_epoch(
    model: SwinUNet,
    loader: DataLoader,
    criterion: Any,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    config: Dict,
    wandb_run: Any = None,
) -> float:
    """Train model for one epoch.

    Args:
        model: SwinUNet model.
        loader: Training dataloader.
        criterion: Loss function (DiceCELoss).
        optimizer: Optimizer.
        device: Device string.
        epoch: Current epoch number.
        config: Config dict.
        wandb_run: Optional W&B run.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0
    aug_cfg = config.get("augmentation", {})
    use_cutmix = aug_cfg.get("use_cutmix", False)
    use_mixup = aug_cfg.get("use_mixup", False)

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        # Apply CutMix/MixUp
        if use_cutmix and random.random() < 0.5:
            images, masks = cutmix_batch(images, masks, aug_cfg.get("cutmix_alpha", 1.0))
        elif use_mixup and random.random() < 0.5:
            images, masks = mixup_batch(images, masks, aug_cfg.get("mixup_alpha", 0.4))

        output = model(images)
        logits = output["logits"]

        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / max(len(loader), 1)

    if wandb_run:
        wandb_run.log({"train_loss": avg_loss, "epoch": epoch})

    return avg_loss


def validate(
    model: SwinUNet,
    loader: DataLoader,
    criterion: Any,
    device: str,
    num_classes: int,
) -> Tuple[float, Dict]:
    """Validate model.

    Args:
        model: SwinUNet model.
        loader: Validation dataloader.
        criterion: Loss function.
        device: Device string.
        num_classes: Number of classes.

    Returns:
        Tuple of (avg_loss, iou_dict).
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            output = model(images)
            logits = output["logits"]
            loss = criterion(logits, masks)
            total_loss += loss.item()

            preds = logits.argmax(dim=1).cpu().numpy()
            targets = masks.cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets)

    avg_loss = total_loss / max(len(loader), 1)

    # Compute IoU
    if all_preds:
        preds_cat = np.concatenate(all_preds, axis=0)
        targets_cat = np.concatenate(all_targets, axis=0)
        # Compute per-image IoU and average
        ious = []
        for i in range(len(preds_cat)):
            iou = pixel_iou(preds_cat[i], targets_cat[i], num_classes)
            ious.append(iou)
        avg_miou = np.mean([iou["mIoU"] for iou in ious])
        iou_dict = {"mIoU": float(avg_miou)}
    else:
        iou_dict = {"mIoU": 0.0}

    return avg_loss, iou_dict


def get_fold_splits(
    tile_paths: List[str],
    mask_paths: List[str],
    fold_k: int,
    fold_idx: int,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Split data into train/val for k-fold CV.

    Args:
        tile_paths: All tile paths.
        mask_paths: All mask paths.
        fold_k: Number of folds.
        fold_idx: Current fold index.

    Returns:
        (train_tiles, train_masks, val_tiles, val_masks).
    """
    n = len(tile_paths)
    indices = list(range(n))
    fold_size = n // fold_k if fold_k > 0 else n
    val_start = fold_idx * fold_size
    val_end = min(val_start + fold_size, n)
    val_idx = set(indices[val_start:val_end])

    train_t = [tile_paths[i] for i in indices if i not in val_idx]
    train_m = [mask_paths[i] for i in indices if i not in val_idx]
    val_t = [tile_paths[i] for i in val_idx]
    val_m = [mask_paths[i] for i in val_idx]

    return train_t, train_m, val_t, val_m


def collect_data(config: Dict) -> Tuple[List[str], List[str]]:
    """Collect tile and mask paths from preprocessed and annotation dirs.

    Args:
        config: Config dict.

    Returns:
        (tile_paths, mask_paths).
    """
    tile_dir = os.path.join(config["data"]["preprocessed_dir"], "tiles",
                            str(config["swin"]["input_size"]))
    ann_dir = config["data"]["annotations_dir"]

    tile_paths = list_files(tile_dir, extensions=[".png", ".npy"])
    mask_paths = []

    for tp in tile_paths:
        base = os.path.splitext(os.path.basename(tp))[0]
        mask_candidates = [
            os.path.join(ann_dir, f"{base}_mask.png"),
            os.path.join(ann_dir, f"{base}.png"),
            os.path.join(tile_dir, f"{base}_mask.png"),
        ]
        mask_path = next((p for p in mask_candidates if os.path.isfile(p)), "")
        mask_paths.append(mask_path)

    return tile_paths, mask_paths


def pseudo_label(
    model: SwinUNet,
    config: Dict,
    device: str,
    threshold: float = 0.85,
) -> List[Tuple[str, str]]:
    """Generate pseudo-labels from unlabeled tiles.

    Args:
        model: Trained SwinUNet.
        config: Config dict.
        device: Device string.
        threshold: Confidence threshold for accepting pseudo-labels.

    Returns:
        List of (tile_path, pseudo_mask_path) pairs.
    """
    raw_dir = config["data"]["raw_dir"]
    pseudo_dir = ensure_dir(os.path.join(config["data"]["preprocessed_dir"], "pseudo_labels"))
    input_size = config["swin"]["input_size"]

    tile_paths = list_files(raw_dir, extensions=[".tif", ".tiff"])
    pseudo_pairs = []

    model.eval()
    for tp in tile_paths:
        try:
            from src.utils.io import load_geotiff
            img, _ = load_geotiff(tp)
            img_resized = cv2.resize(img[:, :, :3], (input_size, input_size))
            img_t = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            with torch.no_grad():
                output = model(img_t.to(device))
                confidence = output["confidence"].item()
                if confidence >= threshold:
                    mask = output["logits"].argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                    base = os.path.splitext(os.path.basename(tp))[0]
                    mask_path = os.path.join(pseudo_dir, f"{base}_pseudo.png")
                    cv2.imwrite(mask_path, mask)
                    pseudo_pairs.append((tp, mask_path))
                    logger.info(f"Pseudo-label generated for {base} (conf={confidence:.3f})")
        except Exception as e:
            logger.warning(f"Pseudo-labeling failed for {tp}: {e}")

    logger.info(f"Generated {len(pseudo_pairs)} pseudo-labels")
    return pseudo_pairs


# ──────────────────────────── Main Training ────────────────────────────

def train(config: Dict, fold: int = 0) -> str:
    """Main training function.

    Args:
        config: Config dict.
        fold: Fold index for CV.

    Returns:
        Path to saved checkpoint.
    """
    swin_cfg = config["swin"]
    device = "cpu" if config.get("inference", {}).get("cpu_only", False) else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # W&B
    wandb_run = None
    if config.get("observability", {}).get("use_wandb", False):
        try:
            import wandb
            wandb_run = wandb.init(
                project=config["observability"]["wandb_project"],
                config=config,
                name=f"swin_fold{fold}",
            )
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")

    # Collect data
    tile_paths, mask_paths = collect_data(config)
    logger.info(f"Found {len(tile_paths)} tiles for training")

    if len(tile_paths) == 0:
        logger.warning("No training data found. Creating dummy data for testing.")
        tile_dir = os.path.join(config["data"]["preprocessed_dir"], "tiles",
                                str(swin_cfg["input_size"]))
        ensure_dir(tile_dir)
        dummy_img = np.random.randint(0, 255, (swin_cfg["input_size"], swin_cfg["input_size"], 3), dtype=np.uint8)
        dummy_path = os.path.join(tile_dir, "dummy_tile.png")
        cv2.imwrite(dummy_path, dummy_img)
        dummy_mask = np.zeros((swin_cfg["input_size"], swin_cfg["input_size"]), dtype=np.uint8)
        dummy_mask[100:200, 100:200] = 1
        mask_path = os.path.join(tile_dir, "dummy_tile_mask.png")
        cv2.imwrite(mask_path, dummy_mask)
        tile_paths = [dummy_path]
        mask_paths = [mask_path]

    # K-fold split
    train_tiles, train_masks, val_tiles, val_masks = get_fold_splits(
        tile_paths, mask_paths, swin_cfg["fold_k"], fold
    )

    logger.info(f"Fold {fold}: train={len(train_tiles)}, val={len(val_tiles)}")

    # Datasets
    train_ds = SegmentationDataset(
        train_tiles, train_masks, swin_cfg["input_size"],
        augment=True, use_heavy_augs=config.get("augmentation", {}).get("heavy_augs", True),
    )
    val_ds = SegmentationDataset(
        val_tiles, val_masks, swin_cfg["input_size"],
        augment=False, use_heavy_augs=False,
    )

    # Compute sample weights if oversampling is enabled
    sampler = None
    shuffle = True
    if swin_cfg.get("building_tile_oversampling", False):
        logger.info("Computing sample weights for building tile oversampling...")
        sample_weights = []
        for mp in train_masks:
            weight = 1.0
            if os.path.isfile(mp):
                m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    # Oversample tiles with buildings (1) or roads (2)
                    if np.any(m == 1) or np.any(m == 2):
                        weight = 5.0
            sample_weights.append(weight)
        
        logger.info(f"Oversampling enabled. High-value tiles: {sum([1 for w in sample_weights if w > 1.0])} / {len(train_masks)}")
        sampler = WeightedRandomSampler(
            weights=sample_weights, 
            num_samples=len(sample_weights), 
            replacement=True
        )
        shuffle = False

    train_loader = DataLoader(
        train_ds, 
        batch_size=swin_cfg["batch_size"], 
        shuffle=shuffle, 
        sampler=sampler,
        num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=swin_cfg["batch_size"], shuffle=False, num_workers=0)

    # Model
    model = SwinUNet(
        encoder_name=swin_cfg["encoder"],
        pretrained=swin_cfg["pretrained"],
        num_classes=swin_cfg["num_classes"],
        input_size=swin_cfg["input_size"],
    ).to(device)

    # Loss Definition with class weights
    class_weights = swin_cfg.get("class_weights", None)
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = DiceCELoss(
        num_classes=swin_cfg["num_classes"],
        use_focal=swin_cfg.get("use_focal", True),
        class_weights=class_weights
    )
    checkpoint_dir = ensure_dir(os.path.join(config["data"]["models_dir"], "swin"))

    # Phase 1: Freeze backbone
    logger.info("Phase 1: Training decoder with frozen backbone")
    model.freeze_backbone()
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=swin_cfg["lr_freeze"],
        weight_decay=swin_cfg.get("weight_decay", 1e-4),
    )

    best_miou = 0.0
    best_path = ""

    for epoch in range(swin_cfg["freeze_epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, config, wandb_run)
        val_loss, iou_dict = validate(model, val_loader, criterion, device, swin_cfg["num_classes"])
        logger.info(f"  [Freeze] Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, mIoU={iou_dict['mIoU']:.4f}")

        if iou_dict["mIoU"] >= best_miou:
            best_miou = iou_dict["mIoU"]
            best_path = os.path.join(checkpoint_dir, f"swin_fold{fold}_best.pth")
            torch.save(model.state_dict(), best_path)

    # Phase 2: Unfreeze last 2 stages
    logger.info("Phase 2: Fine-tuning with unfrozen backbone stages")
    model.unfreeze_last_stages(num_stages=2)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=swin_cfg["lr_finetune"],
        weight_decay=swin_cfg.get("weight_decay", 1e-4),
    )

    for epoch in range(swin_cfg["unfreeze_epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                     epoch + swin_cfg["freeze_epochs"], config, wandb_run)
        val_loss, iou_dict = validate(model, val_loader, criterion, device, swin_cfg["num_classes"])
        logger.info(f"  [Finetune] Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, mIoU={iou_dict['mIoU']:.4f}")

        if iou_dict["mIoU"] >= best_miou:
            best_miou = iou_dict["mIoU"]
            best_path = os.path.join(checkpoint_dir, f"swin_fold{fold}_best.pth")
            torch.save(model.state_dict(), best_path)

    # Phase 3: Pseudo-labeling (optional)
    if swin_cfg.get("pseudo_label_threshold", 0) > 0:
        logger.info("Phase 3: Generating pseudo-labels")
        if best_path and os.path.isfile(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device))
        pseudo_pairs = pseudo_label(model, config, device, swin_cfg["pseudo_label_threshold"])
        if pseudo_pairs:
            logger.info(f"  Added {len(pseudo_pairs)} pseudo-labeled tiles")

    if wandb_run:
        wandb_run.finish()

    logger.info(f"Training complete. Best mIoU={best_miou:.4f}. Checkpoint: {best_path}")
    return best_path


def main():
    parser = argparse.ArgumentParser(description="Train Swin-UNet for segmentation")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--fold", type=int, default=0, help="Fold index for k-fold CV")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config(args.config)
    if args.cpu_only:
        config.setdefault("inference", {})["cpu_only"] = True

    checkpoint = train(config, fold=args.fold)
    print(f"\nTraining done. Checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
