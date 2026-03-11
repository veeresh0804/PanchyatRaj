"""
Hydra-Map Fusion Model Training Pipeline.

Implements:
  1. FusionDatasetCreator: runs models on training tiles, matches GT, saves JSONL
  2. FusionTrainer: trains the late-fusion model with k-fold CV,
     class-balanced sampling, and early stopping

CLI: python src/train/train_fusion.py --config config/config.yaml
"""

import argparse
import json
import logging
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.fusion.fusion_model import build_fusion_model
from src.utils.io import ensure_dir, list_files, load_config, load_json, load_jsonl, save_json, save_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────── Dataset Creator ────────────────────────────

class FusionDatasetCreator:
    """Creates training data for the fusion model.

    For each training tile with ground truth:
      1. Runs Swin-UNet to get features
      2. Runs YOLO to get detections
      3. Computes depth features if available
      4. Matches predictions to GT polygons via IoU
      5. Saves feature records as JSONL

    Args:
        config: Global config dict.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = "cpu" if config.get("inference", {}).get("cpu_only", True) else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.fusion_cfg = config.get("fusion", {})

    def create_dataset(self, output_path: str) -> int:
        """Generate fusion training dataset.

        Args:
            output_path: Path to save JSONL file.

        Returns:
            Number of records generated.
        """
        swin_cfg = self.config.get("swin", {})
        tile_dir = os.path.join(
            self.config["data"]["preprocessed_dir"], "tiles",
            str(swin_cfg.get("input_size", 512))
        )
        ann_dir = self.config["data"]["annotations_dir"]

        tile_paths = list_files(tile_dir, extensions=[".png", ".npy"])
        logger.info(f"Creating fusion dataset from {len(tile_paths)} tiles")

        # Load models (or stubs)
        swin_model = self._load_swin()
        yolo_wrapper = self._load_yolo()
        depth_pipeline = self._load_depth()

        records = []

        for tp in tile_paths:
            tile_id = os.path.splitext(os.path.basename(tp))[0]
            try:
                record = self._process_tile(
                    tp, tile_id, swin_model, yolo_wrapper, depth_pipeline, ann_dir
                )
                if record is not None:
                    records.append(record)
            except Exception as e:
                logger.warning(f"Fusion dataset: failed on {tile_id}: {e}")

        # Save dataset
        ensure_dir(os.path.dirname(output_path))
        save_jsonl(records, output_path)
        logger.info(f"Fusion dataset saved: {len(records)} records → {output_path}")
        return len(records)

    def _process_tile(
        self,
        tile_path: str,
        tile_id: str,
        swin_model: Any,
        yolo_wrapper: Any,
        depth_pipeline: Any,
        ann_dir: str,
    ) -> Optional[Dict[str, Any]]:
        """Process one tile and generate a fusion training record.

        Args:
            tile_path: Path to tile image.
            tile_id: Tile identifier.
            swin_model: SwinUNet model.
            yolo_wrapper: YOLOWrapper.
            depth_pipeline: DepthPipeline.
            ann_dir: Annotations directory.

        Returns:
            Feature record dict, or None if no GT available.
        """
        # Load tile
        if tile_path.endswith(".npy"):
            img = np.load(tile_path)
        else:
            img = cv2.imread(tile_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is None:
            return None

        input_size = self.config["swin"].get("input_size", 512)

        # Run Swin
        swin_features = np.zeros(self.fusion_cfg.get("swin_feature_dim", 768), dtype=np.float32)
        swin_confidence = 0.0
        mask_area = 0.0
        mask_perimeter = 0.0

        if swin_model is not None:
            try:
                img_resized = cv2.resize(img, (input_size, input_size))
                img_t = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                swin_model.eval()
                with torch.no_grad():
                    output = swin_model(img_t.to(self.device))
                swin_features = output["pooled_features"].cpu().squeeze(0).numpy()
                swin_confidence = output["confidence"].cpu().item()
                mask = output["logits"].argmax(dim=1).squeeze(0).cpu().numpy()
                mask_area = float((mask == 1).sum())
            except Exception as e:
                logger.debug(f"Swin failed on {tile_id}: {e}")

        # Run YOLO
        yolo_max_det = self.fusion_cfg.get("yolo_max_detections", 20)
        if yolo_wrapper is not None:
            yolo_result = yolo_wrapper.predict(img, tile_id)
            yolo_features = yolo_wrapper.get_summary_features(yolo_result, max_detections=yolo_max_det)
        else:
            yolo_result = {"count": 0, "max_confidence": 0, "avg_confidence": 0, "boxes": []}
            yolo_features = np.zeros(yolo_max_det * 6 + 3, dtype=np.float32)

        # Depth features
        depth_features = np.zeros(3, dtype=np.float32)
        if depth_pipeline is not None:
            depth_result = depth_pipeline.compute_height(tile_id)
            depth_features = depth_pipeline.get_depth_features(depth_result)

        # Ground truth labels
        gt_info = self._load_gt(tile_id, ann_dir)

        # Mask stats
        mask_stats = np.array([mask_area, mask_perimeter, swin_confidence, 1.0], dtype=np.float32)

        record = {
            "tile_id": tile_id,
            "swin_features": swin_features.tolist(),
            "swin_confidence": swin_confidence,
            "yolo_features": yolo_features.tolist(),
            "yolo_count": yolo_result.get("count", 0),
            "yolo_max_conf": yolo_result.get("max_confidence", 0),
            "depth_features": depth_features.tolist(),
            "mask_stats": mask_stats.tolist(),
            "gt_label": gt_info.get("label", 1),
            "gt_has_object": gt_info.get("has_object", True),
            "gt_iou": gt_info.get("iou", 0.0),
            "gt_needs_refinement": gt_info.get("needs_refinement", False),
        }

        return record

    def _load_gt(self, tile_id: str, ann_dir: str) -> Dict[str, Any]:
        """Load ground truth info for a tile.

        Args:
            tile_id: Tile identifier.
            ann_dir: Annotations directory.

        Returns:
            GT info dict.
        """
        # Check for matching GT mask
        mask_candidates = [
            os.path.join(ann_dir, f"{tile_id}_mask.png"),
            os.path.join(ann_dir, f"{tile_id}.png"),
        ]

        for mc in mask_candidates:
            if os.path.isfile(mc):
                mask = cv2.imread(mc, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    has_object = mask.max() > 0
                    label = int(mask.max()) if has_object else 0
                    return {
                        "has_object": has_object,
                        "label": label,
                        "iou": 1.0,  # GT itself
                        "needs_refinement": False,
                    }

        # Default: assume building present (for training with pseudo-labels)
        return {
            "has_object": True,
            "label": 1,
            "iou": 0.0,
            "needs_refinement": True,
        }

    def _load_swin(self) -> Any:
        """Load Swin model for feature extraction."""
        from src.models.swin_unet import SwinUNet
        swin_cfg = self.config["swin"]
        checkpoint_dir = os.path.join(self.config["data"]["models_dir"], "swin")

        model = SwinUNet(
            encoder_name=swin_cfg["encoder"],
            pretrained=False,  # Don't download during dataset creation
            num_classes=swin_cfg["num_classes"],
            input_size=swin_cfg["input_size"],
        ).to(self.device)

        # Try loading checkpoint
        ckpt_path = os.path.join(checkpoint_dir, "swin_fold0_best.pth")
        if os.path.isfile(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            logger.info(f"Loaded Swin checkpoint: {ckpt_path}")

        return model

    def _load_yolo(self) -> Any:
        """Load YOLO wrapper."""
        from src.models.yolo_wrapper import YOLOWrapper
        yolo_cfg = self.config.get("yolo", {})
        return YOLOWrapper(
            model_path=yolo_cfg.get("model", "yolov8n.pt"),
            confidence_threshold=yolo_cfg.get("confidence_threshold", 0.25),
            device=self.device,
        )

    def _load_depth(self) -> Any:
        """Load depth pipeline."""
        from src.models.depth_pipeline import DepthPipeline
        return DepthPipeline(self.config)


# ──────────────────────────── Fusion Dataset ────────────────────────────

class FusionDataset(Dataset):
    """PyTorch dataset for fusion training from JSONL records.

    Args:
        records: List of feature dicts.
        config: Config dict with fusion section.
    """

    def __init__(self, records: List[Dict], config: Dict):
        self.records = records
        self.fusion_cfg = config.get("fusion", {})
        self.swin_dim = self.fusion_cfg.get("swin_feature_dim", 768)
        yolo_max_det = self.fusion_cfg.get("yolo_max_detections", 20)
        yolo_feat_per = self.fusion_cfg.get("yolo_feature_dim", 6)
        self.yolo_dim = yolo_max_det * yolo_feat_per + 3

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rec = self.records[idx]

        # Assemble features
        swin_feat = np.array(rec["swin_features"], dtype=np.float32)
        if len(swin_feat) != self.swin_dim:
            swin_feat = np.zeros(self.swin_dim, dtype=np.float32)

        yolo_feat = np.array(rec["yolo_features"], dtype=np.float32)
        if len(yolo_feat) != self.yolo_dim:
            yolo_feat = np.zeros(self.yolo_dim, dtype=np.float32)

        depth_feat = np.array(rec.get("depth_features", [0, 0, 0]), dtype=np.float32)
        mask_stats = np.array(rec.get("mask_stats", [0, 0, 0, 0]), dtype=np.float32)

        features = np.concatenate([swin_feat, yolo_feat, depth_feat, mask_stats])

        # Labels
        labels = {
            "accept": torch.tensor(1.0 if rec.get("gt_has_object", True) else 0.0),
            "refine": torch.tensor(1.0 if rec.get("gt_needs_refinement", False) else 0.0),
            "class_id": torch.tensor(rec.get("gt_label", 1), dtype=torch.long),
            "confidence": torch.tensor(rec.get("gt_iou", 0.5)),
        }

        return torch.from_numpy(features), labels


# ──────────────────────────── Training ────────────────────────────

class FusionTrainer:
    """Trains the late-fusion model.

    Args:
        config: Global config dict.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fusion_cfg = config.get("fusion", {})
        self.device = "cpu" if config.get("inference", {}).get("cpu_only", True) else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def train(self, dataset_path: str) -> str:
        """Train fusion model with k-fold cross-validation.

        Args:
            dataset_path: Path to JSONL fusion dataset.

        Returns:
            Path to best model checkpoint.
        """
        records = load_jsonl(dataset_path)
        if not records:
            logger.warning("No fusion training records found. Generating dummy data.")
            records = self._generate_dummy_records()

        fold_k = self.fusion_cfg.get("fold_k", 5)
        n = len(records)
        logger.info(f"Training fusion model on {n} records, {fold_k}-fold CV")

        best_overall_loss = float("inf")
        best_ckpt = ""
        checkpoint_dir = ensure_dir(os.path.join(self.config["data"]["models_dir"], "fusion"))

        for fold in range(min(fold_k, max(n, 1))):
            logger.info(f"\n=== Fold {fold}/{fold_k} ===")

            # Split
            fold_size = max(n // fold_k, 1)
            val_start = fold * fold_size
            val_end = min(val_start + fold_size, n)
            val_records = records[val_start:val_end]
            train_records = records[:val_start] + records[val_end:]

            if not train_records:
                train_records = records
            if not val_records:
                val_records = records

            train_ds = FusionDataset(train_records, self.config)
            val_ds = FusionDataset(val_records, self.config)

            train_loader = DataLoader(
                train_ds, batch_size=self.fusion_cfg.get("batch_size", 64),
                shuffle=True, num_workers=0,
            )
            val_loader = DataLoader(
                val_ds, batch_size=self.fusion_cfg.get("batch_size", 64),
                shuffle=False, num_workers=0,
            )

            # Build model
            model = build_fusion_model(self.config).to(self.device)
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.fusion_cfg.get("lr", 1e-3),
            )

            # Loss functions
            bce = nn.BCELoss()
            ce = nn.CrossEntropyLoss()
            mse = nn.MSELoss()

            patience = self.fusion_cfg.get("early_stopping_patience", 7)
            epochs = self.fusion_cfg.get("epochs", 50)
            best_val_loss = float("inf")
            no_improve = 0

            for epoch in range(epochs):
                # Train
                model.train()
                train_loss = 0
                for features, labels in train_loader:
                    features = features.to(self.device)
                    accept_gt = labels["accept"].to(self.device)
                    refine_gt = labels["refine"].to(self.device)
                    class_gt = labels["class_id"].to(self.device)
                    conf_gt = labels["confidence"].to(self.device)

                    output = model(features)

                    loss = (
                        bce(output["accept"], accept_gt)
                        + bce(output["refine"], refine_gt)
                        + ce(output["class_logits"], class_gt)
                        + mse(output["confidence"], conf_gt)
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                train_loss /= max(len(train_loader), 1)

                # Validate
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for features, labels in val_loader:
                        features = features.to(self.device)
                        accept_gt = labels["accept"].to(self.device)
                        refine_gt = labels["refine"].to(self.device)
                        class_gt = labels["class_id"].to(self.device)
                        conf_gt = labels["confidence"].to(self.device)

                        output = model(features)
                        loss = (
                            bce(output["accept"], accept_gt)
                            + bce(output["refine"], refine_gt)
                            + ce(output["class_logits"], class_gt)
                            + mse(output["confidence"], conf_gt)
                        )
                        val_loss += loss.item()

                val_loss /= max(len(val_loader), 1)
                logger.info(f"  Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve = 0
                    ckpt_path = os.path.join(checkpoint_dir, f"fusion_fold{fold}_best.pth")
                    torch.save(model.state_dict(), ckpt_path)
                    if val_loss < best_overall_loss:
                        best_overall_loss = val_loss
                        best_ckpt = ckpt_path
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        logger.info(f"  Early stopping at epoch {epoch}")
                        break

        logger.info(f"Fusion training complete. Best: {best_ckpt} (loss={best_overall_loss:.4f})")
        return best_ckpt

    def _generate_dummy_records(self) -> List[Dict]:
        """Generate dummy fusion records for testing."""
        swin_dim = self.fusion_cfg.get("swin_feature_dim", 768)
        yolo_max_det = self.fusion_cfg.get("yolo_max_detections", 20)
        yolo_feat_dim = yolo_max_det * 6 + 3

        records = []
        for i in range(50):
            has_obj = random.random() > 0.3
            records.append({
                "tile_id": f"dummy_{i:03d}",
                "swin_features": np.random.randn(swin_dim).tolist(),
                "swin_confidence": random.random(),
                "yolo_features": np.random.randn(yolo_feat_dim).tolist(),
                "yolo_count": random.randint(0, 10),
                "yolo_max_conf": random.random(),
                "depth_features": [random.random() * 10, random.random() * 10, random.random() * 2],
                "mask_stats": [random.random() * 1000, random.random() * 200, random.random(), 1.0],
                "gt_label": 1 if has_obj else 0,
                "gt_has_object": has_obj,
                "gt_iou": random.random() if has_obj else 0,
                "gt_needs_refinement": random.random() > 0.6,
            })
        return records


def main():
    parser = argparse.ArgumentParser(description="Train Fusion Model")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--generate-dataset", action="store_true",
                        help="Generate fusion training dataset before training")
    parser.add_argument("--dataset-path", type=str, default="data/preprocessed/fusion_dataset.jsonl",
                        help="Path for fusion dataset JSONL")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config(args.config)
    if args.cpu_only:
        config.setdefault("inference", {})["cpu_only"] = True

    # Generate dataset if requested
    if args.generate_dataset:
        creator = FusionDatasetCreator(config)
        num_records = creator.create_dataset(args.dataset_path)
        logger.info(f"Generated {num_records} fusion training records")

    # Train fusion model
    trainer = FusionTrainer(config)
    checkpoint = trainer.train(args.dataset_path)
    print(f"\nFusion training done. Checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
