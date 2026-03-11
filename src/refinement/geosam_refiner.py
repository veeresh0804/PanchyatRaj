"""
Hydra-Map GeoSAM Wrapper.

Wraps Meta's Segment Anything Model (SAM) for mask refinement.
Accepts bounding box or point prompts and returns refined masks
with quality metrics.

Supports:
  - Out-of-box SAM inference
  - LoRA fine-tuning stub
  - Deterministic stub when checkpoint is missing (smoke testing)
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class GeoSAMWrapper:
    """Wrapper for SAM-based mask refinement.

    Args:
        checkpoint: Path to SAM checkpoint (e.g., 'models/sam_vit_h.pth').
        model_type: SAM model type ('vit_h', 'vit_l', 'vit_b').
        device: 'cpu' or 'cuda'.
        use_lora: Whether to apply LoRA adapter (for fine-tuning).
        lora_rank: LoRA rank for approximate fine-tuning.
    """

    def __init__(
        self,
        checkpoint: str = "models/sam_vit_h.pth",
        model_type: str = "vit_h",
        device: str = "cpu",
        use_lora: bool = False,
        lora_rank: int = 4,
    ):
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.device = device
        self.use_lora = use_lora
        self.lora_rank = lora_rank

        self.model = None
        self.predictor = None
        self._stub_mode = False

        self._load_model()

    def _load_model(self) -> None:
        """Load SAM model and set up predictor."""
        if not os.path.isfile(self.checkpoint):
            logger.warning(
                f"SAM checkpoint not found at '{self.checkpoint}'. "
                f"Using stub mode. Download from: "
                f"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            )
            self._stub_mode = True
            return

        try:
            from segment_anything import SamPredictor, sam_model_registry

            self.model = sam_model_registry[self.model_type](
                checkpoint=self.checkpoint
            )
            self.model.to(self.device)

            if self.use_lora:
                self._apply_lora()

            self.predictor = SamPredictor(self.model)
            logger.info(f"SAM model loaded: {self.model_type} from {self.checkpoint}")
        except Exception as e:
            logger.warning(f"Failed to load SAM: {e}. Using stub mode.")
            self._stub_mode = True

    def _apply_lora(self) -> None:
        """Apply LoRA adapters to SAM image encoder for fine-tuning.

        This is a stub that demonstrates where LoRA would be applied.
        Full implementation requires a LoRA library (e.g., loralib).
        """
        logger.info(
            f"LoRA fine-tuning stub: rank={self.lora_rank}. "
            f"For production, integrate loralib or PEFT to adapt "
            f"SAM's image encoder attention layers."
        )
        # TODO: Replace with actual LoRA injection:
        # import loralib as lora
        # for name, module in self.model.image_encoder.named_modules():
        #     if isinstance(module, nn.Linear) and 'attn' in name:
        #         lora.replace_module(module, rank=self.lora_rank)

    def set_image(self, image: np.ndarray) -> None:
        """Set the image for SAM predictor (encodes features).

        Args:
            image: RGB image (H, W, 3) as uint8.
        """
        if self._stub_mode:
            self._stub_image = image
            return

        self.predictor.set_image(image)

    def predict_box(
        self,
        bbox: List[float],
        tile_id: str = "",
    ) -> Dict[str, Any]:
        """Generate refined mask from a bounding box prompt.

        Args:
            bbox: Bounding box [x1, y1, x2, y2] in pixel coords.
            tile_id: Tile identifier.

        Returns:
            Dict with 'mask' (H, W bool), 'mask_area', 'iou_prediction',
            'stability_score', 'quality_metric'.
        """
        if self._stub_mode:
            return self._stub_predict_box(bbox, tile_id)

        try:
            box_np = np.array(bbox)
            masks, scores, logits = self.predictor.predict(
                box=box_np,
                multimask_output=True,
            )

            # Select best mask by score
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            score = float(scores[best_idx])

            mask_area = float(mask.sum())
            return {
                "tile_id": tile_id,
                "mask": mask,
                "mask_area": mask_area,
                "iou_prediction": score,
                "stability_score": score,  # Approximate
                "quality_metric": score,
                "bbox": bbox,
            }
        except Exception as e:
            logger.error(f"GeoSAM prediction failed for tile {tile_id}: {e}")
            return self._fallback_result(bbox, tile_id, error=str(e))

    def predict_point(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        tile_id: str = "",
    ) -> Dict[str, Any]:
        """Generate refined mask from point prompts.

        Args:
            points: (N, 2) array of [x, y] coordinates.
            labels: (N,) array of 1=foreground, 0=background.
            tile_id: Tile identifier.

        Returns:
            Dict with mask and quality metrics.
        """
        if self._stub_mode:
            h, w = getattr(self, "_stub_image", np.zeros((1024, 1024, 3))).shape[:2]
            mask = np.zeros((h, w), dtype=bool)
            mask[100:300, 100:300] = True
            return {
                "tile_id": tile_id,
                "mask": mask,
                "mask_area": float(mask.sum()),
                "iou_prediction": 0.8,
                "stability_score": 0.8,
                "quality_metric": 0.8,
            }

        try:
            masks, scores, _ = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True,
            )
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            return {
                "tile_id": tile_id,
                "mask": mask,
                "mask_area": float(mask.sum()),
                "iou_prediction": float(scores[best_idx]),
                "stability_score": float(scores[best_idx]),
                "quality_metric": float(scores[best_idx]),
            }
        except Exception as e:
            logger.error(f"GeoSAM point prediction failed: {e}")
            return {"tile_id": tile_id, "mask": None, "mask_area": 0,
                    "iou_prediction": 0, "quality_metric": 0, "error": str(e)}

    def _stub_predict_box(self, bbox: List[float], tile_id: str) -> Dict[str, Any]:
        """Stub prediction for testing without SAM checkpoint."""
        h, w = getattr(self, "_stub_image", np.zeros((1024, 1024, 3))).shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        mask = np.zeros((h, w), dtype=bool)
        if y2 > y1 and x2 > x1:
            mask[y1:y2, x1:x2] = True

        return {
            "tile_id": tile_id,
            "mask": mask,
            "mask_area": float(mask.sum()),
            "iou_prediction": 0.75,
            "stability_score": 0.75,
            "quality_metric": 0.75,
            "bbox": bbox,
            "stub": True,
        }

    def _fallback_result(
        self, bbox: List[float], tile_id: str, error: str = ""
    ) -> Dict[str, Any]:
        """Fallback result when SAM fails (OOM etc.)."""
        return {
            "tile_id": tile_id,
            "mask": None,
            "mask_area": 0,
            "iou_prediction": 0.0,
            "stability_score": 0.0,
            "quality_metric": 0.0,
            "bbox": bbox,
            "error": error,
            "fallback": True,
        }

    def get_quality_features(self, result: Dict[str, Any]) -> np.ndarray:
        """Extract quality features for fusion model.

        Args:
            result: Result dict from predict_box/predict_point.

        Returns:
            Array [mask_area, iou_prediction, stability_score].
        """
        return np.array([
            result.get("mask_area", 0.0),
            result.get("iou_prediction", 0.0),
            result.get("stability_score", 0.0),
        ], dtype=np.float32)
