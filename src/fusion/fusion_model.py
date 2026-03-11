"""
Hydra-Map Late-Fusion Model.

A LEARNED multimodal model that mediates between Swin-UNet, YOLOv8,
GeoSAM, and Depth signals. NOT hardcoded rules — trained on feature dumps.

Architectures:
  - MLP (default): Multi-layer perceptron with residual connections
  - Transformer: Small cross-attention model (optional)

Output heads:
  - object_acceptance (sigmoid): should this candidate be accepted?
  - refinement_required (sigmoid): should GeoSAM refine this?
  - class_prediction (softmax): final object class
  - confidence (sigmoid): aggregated confidence
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FusionMLP(nn.Module):
    """Late-fusion MLP for multi-model decision making.

    Concatenates features from Swin, YOLO, depth, and mask stats,
    then predicts per-object decisions.

    Args:
        swin_dim: Dimensionality of Swin pooled features.
        yolo_dim: Total YOLO feature dimensionality (max_det * 6 + 3).
        depth_dim: Depth feature dimensionality.
        mask_stats_dim: Mask statistics dimensionality.
        hidden_dims: List of hidden layer sizes.
        num_classes: Number of object classes.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        swin_dim: int = 768,
        yolo_dim: int = 123,  # 20 * 6 + 3
        depth_dim: int = 3,
        mask_stats_dim: int = 4,
        hidden_dims: list = None,
        num_classes: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.num_classes = num_classes
        total_input = swin_dim + yolo_dim + depth_dim + mask_stats_dim

        # Build MLP layers
        layers = []
        in_dim = total_input
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.feature_dim = hidden_dims[-1]

        # Output heads
        self.head_accept = nn.Linear(self.feature_dim, 1)      # object_acceptance
        self.head_refine = nn.Linear(self.feature_dim, 1)      # refinement_required
        self.head_class = nn.Linear(self.feature_dim, num_classes)  # class
        self.head_confidence = nn.Linear(self.feature_dim, 1)  # confidence

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            features: Concatenated feature vector (B, total_input_dim).

        Returns:
            Dict with:
                'accept': (B,) sigmoid probability
                'refine': (B,) sigmoid probability
                'class_logits': (B, num_classes)
                'class_probs': (B, num_classes) softmax
                'confidence': (B,) sigmoid
        """
        h = self.backbone(features)

        accept = torch.sigmoid(self.head_accept(h)).squeeze(-1)
        refine = torch.sigmoid(self.head_refine(h)).squeeze(-1)
        class_logits = self.head_class(h)
        class_probs = F.softmax(class_logits, dim=-1)
        confidence = torch.sigmoid(self.head_confidence(h)).squeeze(-1)

        return {
            "accept": accept,
            "refine": refine,
            "class_logits": class_logits,
            "class_probs": class_probs,
            "confidence": confidence,
        }


class FusionTransformer(nn.Module):
    """Small Transformer for late fusion (optional architecture).

    Treats each model's features as a token and uses self-attention
    to combine them.

    Args:
        swin_dim: Swin feature dim.
        yolo_dim: YOLO feature dim.
        depth_dim: Depth feature dim.
        mask_stats_dim: Mask stats dim.
        d_model: Transformer hidden dim.
        nhead: Number of attention heads.
        num_layers: Number of transformer layers.
        num_classes: Number of classes.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        swin_dim: int = 768,
        yolo_dim: int = 123,
        depth_dim: int = 3,
        mask_stats_dim: int = 4,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.d_model = d_model

        # Project each modality into d_model
        self.proj_swin = nn.Linear(swin_dim, d_model)
        self.proj_yolo = nn.Linear(yolo_dim, d_model)
        self.proj_depth = nn.Linear(depth_dim, d_model)
        self.proj_mask = nn.Linear(mask_stats_dim, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Output heads
        self.head_accept = nn.Linear(d_model, 1)
        self.head_refine = nn.Linear(d_model, 1)
        self.head_class = nn.Linear(d_model, num_classes)
        self.head_confidence = nn.Linear(d_model, 1)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            features: Concatenated features (B, total_dim).

        Returns:
            Same output dict as FusionMLP.
        """
        B = features.shape[0]
        swin_dim = 768
        yolo_dim = features.shape[1] - swin_dim - 3 - 4
        # Split features
        swin_f = features[:, :swin_dim]
        yolo_f = features[:, swin_dim:swin_dim + yolo_dim]
        depth_f = features[:, swin_dim + yolo_dim:swin_dim + yolo_dim + 3]
        mask_f = features[:, swin_dim + yolo_dim + 3:]

        # Project to tokens
        t_swin = self.proj_swin(swin_f).unsqueeze(1)   # (B, 1, d_model)
        t_yolo = self.proj_yolo(yolo_f).unsqueeze(1)
        t_depth = self.proj_depth(depth_f).unsqueeze(1)
        t_mask = self.proj_mask(mask_f).unsqueeze(1)

        cls = self.cls_token.expand(B, -1, -1)

        tokens = torch.cat([cls, t_swin, t_yolo, t_depth, t_mask], dim=1)  # (B, 5, d_model)
        out = self.transformer(tokens)

        # Use CLS token for predictions
        cls_out = out[:, 0]

        accept = torch.sigmoid(self.head_accept(cls_out)).squeeze(-1)
        refine = torch.sigmoid(self.head_refine(cls_out)).squeeze(-1)
        class_logits = self.head_class(cls_out)
        class_probs = F.softmax(class_logits, dim=-1)
        confidence = torch.sigmoid(self.head_confidence(cls_out)).squeeze(-1)

        return {
            "accept": accept,
            "refine": refine,
            "class_logits": class_logits,
            "class_probs": class_probs,
            "confidence": confidence,
        }


def build_fusion_model(config: Dict) -> nn.Module:
    """Factory function to build the fusion model from config.

    Args:
        config: Config dict with 'fusion' section.

    Returns:
        FusionMLP or FusionTransformer instance.
    """
    fusion_cfg = config.get("fusion", {})
    arch = fusion_cfg.get("arch", "mlp")
    yolo_max_det = fusion_cfg.get("yolo_max_detections", 20)
    yolo_feat_per_det = fusion_cfg.get("yolo_feature_dim", 6)
    yolo_dim = yolo_max_det * yolo_feat_per_det + 3  # +3 for summary stats

    kwargs = dict(
        swin_dim=fusion_cfg.get("swin_feature_dim", 768),
        yolo_dim=yolo_dim,
        depth_dim=fusion_cfg.get("depth_feature_dim", 3),
        mask_stats_dim=fusion_cfg.get("mask_stats_dim", 4),
        num_classes=fusion_cfg.get("num_classes", 6),
        dropout=fusion_cfg.get("dropout", 0.3),
    )

    if arch == "transformer":
        return FusionTransformer(**kwargs)
    else:
        return FusionMLP(hidden_dims=fusion_cfg.get("hidden_dims", [256, 128, 64]), **kwargs)
