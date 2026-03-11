"""
Hydra-Map Evaluation Metrics.

Pixel-wise IoU, per-object polygon IoU, precision/recall,
and QA CSV report generation.
"""

import csv
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def pixel_iou(pred: np.ndarray, target: np.ndarray, num_classes: int) -> Dict[str, float]:
    """Compute per-class pixel-wise IoU.

    Args:
        pred: Predicted mask (H, W) with integer class labels.
        target: Ground truth mask (H, W) with integer class labels.
        num_classes: Number of classes.

    Returns:
        Dict mapping class index to IoU, plus 'mIoU'.
    """
    ious = {}
    valid_classes = 0
    total_iou = 0.0

    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()

        if union == 0:
            iou = float("nan")
        else:
            iou = float(intersection) / float(union)
            valid_classes += 1
            total_iou += iou

        ious[f"class_{cls}_iou"] = iou

    ious["mIoU"] = total_iou / valid_classes if valid_classes > 0 else 0.0
    return ious


def per_object_iou(
    pred_polygons: List[Any],
    gt_polygons: List[Any],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute per-object IoU, precision, and recall using polygon matching.

    Args:
        pred_polygons: List of predicted shapely polygons.
        gt_polygons: List of ground truth shapely polygons.
        iou_threshold: IoU threshold for a match.

    Returns:
        Dict with 'precision', 'recall', 'f1', 'mean_iou', 'matched', 'total_pred', 'total_gt'.
    """
    from src.utils.geo import polygon_iou

    if len(pred_polygons) == 0 and len(gt_polygons) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "mean_iou": 1.0,
                "matched": 0, "total_pred": 0, "total_gt": 0}
    if len(pred_polygons) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mean_iou": 0.0,
                "matched": 0, "total_pred": 0, "total_gt": len(gt_polygons)}
    if len(gt_polygons) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mean_iou": 0.0,
                "matched": 0, "total_pred": len(pred_polygons), "total_gt": 0}

    # Build IoU matrix
    iou_matrix = np.zeros((len(pred_polygons), len(gt_polygons)))
    for i, pp in enumerate(pred_polygons):
        for j, gp in enumerate(gt_polygons):
            iou_matrix[i, j] = polygon_iou(pp, gp)

    # Greedy matching
    matched_pred = set()
    matched_gt = set()
    matched_ious = []

    while True:
        if iou_matrix.size == 0:
            break
        max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        max_iou = iou_matrix[max_idx]
        if max_iou < iou_threshold:
            break

        i, j = max_idx
        matched_pred.add(i)
        matched_gt.add(j)
        matched_ious.append(max_iou)

        iou_matrix[i, :] = 0
        iou_matrix[:, j] = 0

    num_matched = len(matched_ious)
    precision = num_matched / len(pred_polygons) if len(pred_polygons) > 0 else 0.0
    recall = num_matched / len(gt_polygons) if len(gt_polygons) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "matched": num_matched,
        "total_pred": len(pred_polygons),
        "total_gt": len(gt_polygons),
    }


def generate_qa_csv(
    tile_results: List[Dict[str, Any]],
    output_path: str,
) -> str:
    """Generate a QA CSV with per-tile metrics.

    Each entry in tile_results should have:
        tile_id, precision, recall, mIoU, num_bboxes, avg_confidence

    Args:
        tile_results: List of per-tile result dicts.
        output_path: Path to write CSV.

    Returns:
        Path to the written CSV.
    """
    fieldnames = [
        "tile_id", "precision", "recall", "f1",
        "mIoU", "num_bboxes", "avg_confidence",
    ]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in tile_results:
            writer.writerow(row)

    logger.info(f"QA CSV written to {output_path} ({len(tile_results)} tiles)")
    return output_path


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        import torch
        import torch.nn.functional as F
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        
        # Get true class probabilities (pt)
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class DiceCELoss:
    """Combined Dice + Cross-Entropy (or Focal) loss for segmentation training.

    Implements the hybrid loss used in Swin-UNet training:
        L = alpha * DiceLoss + (1 - alpha) * BaseLoss
        (BaseLoss is CE by default, but can be Focal)
    """

    def __init__(
        self, 
        num_classes: int = 6, 
        alpha: float = 0.5, 
        smooth: float = 1e-6,
        use_focal: bool = True,
        class_weights: Optional[torch.Tensor] = None
    ):
        """Initialize DiceCELoss.

        Args:
            num_classes: Number of classes.
            alpha: Weight for Dice loss (1-alpha for CE/Focal loss).
            smooth: Smoothing factor for Dice computation.
            use_focal: Whether to use Focal Loss instead of Cross Entropy.
            class_weights: Tensor of shape (num_classes,) containing weights.
        """
        import torch
        import torch.nn as nn

        self.num_classes = num_classes
        self.alpha = alpha
        self.smooth = smooth
        
        if use_focal:
            self.base_loss = FocalLoss(alpha=class_weights, gamma=2.0)
        else:
            self.base_loss = nn.CrossEntropyLoss(weight=class_weights)

    def __call__(self, logits, targets):
        """Compute combined loss.

        Args:
            logits: (B, C, H, W) predicted logits.
            targets: (B, H, W) integer class labels.

        Returns:
            Combined scalar loss.
        """
        import torch
        import torch.nn.functional as F

        base = self.base_loss(logits, targets)

        # Dice loss
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        cardinality = (probs + targets_one_hot).sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1 - dice.mean()

        return self.alpha * dice_loss + (1 - self.alpha) * base
