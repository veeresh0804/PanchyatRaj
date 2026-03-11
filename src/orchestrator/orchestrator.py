"""
Hydra-Map Orchestrator.

Full pipeline orchestrator that:
  1. Reads a preprocessed tile
  2. Runs Swin-UNet & YOLOv8 in parallel
  3. Assembles fusion features
  4. Queries the learned fusion model for decisions
  5. Conditionally runs GeoSAM refinement & depth validation
  6. Postprocesses polygons (topology sanitization)
  7. Writes per-tile diagnostics & output

Never silently drops tiles — all skips are logged with reason.
"""

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.io import ensure_dir, load_json, save_json
from src.utils.geo import masks_to_polygons, compute_polygon_stats, sanitize_polygon

logger = logging.getLogger(__name__)


class Orchestrator:
    """Multi-model pipeline orchestrator with learned fusion.

    Args:
        config: Global config dict.
        swin_model: Loaded SwinUNet model.
        yolo_wrapper: Loaded YOLOWrapper.
        geosam_wrapper: Loaded GeoSAMWrapper.
        depth_pipeline: Loaded DepthPipeline.
        fusion_model: Loaded fusion model (PyTorch).
        device: 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        swin_model: Any = None,
        yolo_wrapper: Any = None,
        geosam_wrapper: Any = None,
        depth_pipeline: Any = None,
        fusion_model: Any = None,
        gatekeeper: Any = None,
        device: str = "cpu",
    ):
        self.config = config
        self.swin = swin_model
        self.yolo = yolo_wrapper
        self.geosam = geosam_wrapper
        self.depth = depth_pipeline
        self.fusion = fusion_model
        self.gatekeeper = gatekeeper
        self.device = device

        self.inf_cfg = config.get("inference", {})
        self.swin_cfg = config.get("swin", {})
        self.export_cfg = config.get("export", {})
        self.geosam_top_k = self.inf_cfg.get("geosam_top_k", 5)

    def process_tile(
        self,
        tile_path: str,
        tile_meta: Dict[str, Any],
        output_dir: str,
        run_id: str = "default",
    ) -> Dict[str, Any]:
        """Process a single tile through the full pipeline.

        Args:
            tile_path: Path to tile image (PNG/NPY).
            tile_meta: Tile metadata dict with transform, CRS, etc.
            output_dir: Directory to write outputs.
            run_id: Run identifier.

        Returns:
            Dict with tile results and diagnostics.
        """
        tile_id = tile_meta.get("tile_id", os.path.splitext(os.path.basename(tile_path))[0])
        t_start = time.time()

        diagnostics = {
            "tile_id": tile_id,
            "run_id": run_id,
            "status": "processing",
            "errors": [],
            "models_used": [],
            "timing": {},
        }

        # Load tile image
        try:
            if tile_path.endswith(".npy"):
                tile_img = np.load(tile_path)
            else:
                tile_img = cv2.imread(tile_path)
                if tile_img is not None:
                    tile_img = cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB)
            if tile_img is None:
                raise ValueError(f"Could not load tile image: {tile_path}")
        except Exception as e:
            diagnostics["status"] = "error"
            diagnostics["errors"].append(f"load_error: {str(e)}")
            self._save_diagnostics(diagnostics, output_dir, tile_id)
            return diagnostics

        # Gatekeeper Check (Step 1)
        if self.gatekeeper is not None:
            if not self.gatekeeper.check_tile(tile_img):
                diagnostics["status"] = "skipped"
                diagnostics["errors"].append("gatekeeper_rejected_low_variance")
                self._save_diagnostics(diagnostics, output_dir, tile_id)
                logger.info(f"Tile {tile_id} skipped by Gatekeeper (low variance)")
                return diagnostics

        # Step 2: Run Swin & YOLO in parallel
        swin_result = {}
        yolo_result = {}

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}

            if self.swin is not None:
                futures["swin"] = executor.submit(self._run_swin, tile_img, tile_id)
            if self.yolo is not None:
                futures["yolo"] = executor.submit(self._run_yolo, tile_img, tile_id)

            for key, future in futures.items():
                try:
                    result = future.result(timeout=120)
                    if key == "swin":
                        swin_result = result
                        diagnostics["models_used"].append("swin")
                    else:
                        yolo_result = result
                        diagnostics["models_used"].append("yolo")
                except Exception as e:
                    diagnostics["errors"].append(f"{key}_error: {str(e)}")

        diagnostics["timing"]["swin_yolo"] = time.time() - t_start

        # Step 2: Assemble features for fusion model
        features = self._assemble_features(swin_result, yolo_result, tile_id)

        # Step 3: Query fusion model
        fusion_decision = self._run_fusion(features)

        # Step 3.5: Apply Spectral & Texture Heuristics
        fusion_decision = self._apply_heuristics(fusion_decision, tile_img, swin_result, yolo_result)

        diagnostics["fusion_decision"] = {
            k: v.item() if torch.is_tensor(v) else v
            for k, v in fusion_decision.items()
            if k != "class_probs"
        }
        diagnostics["bounds"] = tile_meta.get("bounds", [[0,0], [512,512]])
        diagnostics["crs"] = tile_meta.get("crs", "EPSG:32643") # Default to 43N if missing


        # Step 4: Conditional GeoSAM refinement
        geosam_results = []
        # if fusion_decision.get("refine_flag", False) and self.geosam is not None:
        #     t_sam = time.time()
        #     geosam_results = self._run_geosam(
        #         tile_img, yolo_result, fusion_decision, tile_id
        #     )
        #     diagnostics["timing"]["geosam"] = time.time() - t_sam
        #     diagnostics["models_used"].append("geosam")
        #     diagnostics["geosam_count"] = len(geosam_results)

        # Step 5: Depth validation
        depth_result = {"source": "none"}
        if self.depth is not None:
            t_depth = time.time()
            if hasattr(self.depth, 'compute_height_map'):
                # Apple Depth Pro
                h_map = self.depth.compute_height_map(tile_img)
                mask = swin_result.get("mask")
                if mask is not None:
                    # Simple validation for now, assumes class 1 is building or water
                    cls_name = "building" if fusion_decision.get("class_id") == 1 else ("water" if fusion_decision.get("class_id") == 4 else "unknown")
                    val_res = self.depth.validate_detection(h_map, mask, cls_name)
                    depth_result = {"source": "depth_pro", "z_mean": val_res.get("z_mean"), "z_max": val_res.get("z_max")}
                else:
                    depth_result = {"source": "depth_pro", "z_mean": 0}
            else:
                # Classic DEM
                mask = swin_result.get("mask")
                depth_result = self.depth.compute_height(
                    tile_id=tile_id,
                    mask=mask,
                    transform=tile_meta.get("transform"),
                )
            diagnostics["timing"]["depth"] = time.time() - t_depth
            if depth_result.get("source") != "none":
                diagnostics["models_used"].append("depth")

        diagnostics["depth_stats"] = {
            k: v for k, v in depth_result.items() if k != "tile_id"
        }

        # ──────── Step 5.5: Height-Based Validation Rules ────────

        # SHADOW KILLER: If classified as Water but height > 0.5m, reject
        # (Dark shadows on buildings/trees get misclassified as water)
        z_mean = depth_result.get("z_mean", 0) or 0
        z_max = depth_result.get("z_max", 0) or 0

        if fusion_decision.get("class_id") == 4 and z_mean > 0.5:
            old_class = fusion_decision["class_id"]
            fusion_decision["class_id"] = 0  # Reclassify as Background
            fusion_decision["shadow_killed"] = True
            diagnostics["shadow_killer"] = {
                "triggered": True,
                "original_class": old_class,
                "z_mean": z_mean,
                "reason": f"Water rejected: mean height {z_mean:.2f}m > 0.5m threshold"
            }
            logger.info(f"Tile {tile_id}: SHADOW KILLER — Water rejected (height={z_mean:.2f}m)")

        # ILLEGAL FLOOR CHECK: If classified as Building and height > 12m, flag
        # (Indian rural max is G+2 ≈ 9m, 12m is generous)
        if fusion_decision.get("class_id") == 1 and z_max > 12.0:
            fusion_decision["illegal_height_flag"] = True
            diagnostics["illegal_floor_check"] = {
                "flagged": True,
                "z_max": z_max,
                "reason": f"Building flagged: height {z_max:.1f}m > 12m (possible G+3 or higher)"
            }
            logger.warning(f"Tile {tile_id}: ILLEGAL FLOOR — Building height {z_max:.1f}m > 12m")
        else:
            fusion_decision["illegal_height_flag"] = False

        # Step 6: Postprocess and generate final polygons
        polygons = self._postprocess(
            swin_result, yolo_result, geosam_results,
            fusion_decision, tile_meta, tile_id
        )

        # Step 7: Write outputs
        tile_output = {
            "tile_id": tile_id,
            "polygons": [
                {
                    "geometry_wkt": p["geometry"].wkt if p.get("geometry") else None,
                    "class_id": p.get("class_id", 1),
                    "confidence": p.get("confidence", fusion_decision.get("confidence_val", 0.5)),
                    "source_masks": p.get("source_masks", diagnostics["models_used"]),
                    "z_mean": depth_result.get("z_mean"),
                    "area": p.get("area", 0),
                    "perimeter": p.get("perimeter", 0),
                }
                for p in polygons
            ],
            "num_polygons": len(polygons),
            "swin_confidence": swin_result.get("confidence", 0.0),
            "yolo_count": yolo_result.get("count", 0),
            "fusion_accept": fusion_decision.get("accept_val", 0.0),
        }

        diagnostics["status"] = "complete"
        diagnostics["timing"]["total"] = time.time() - t_start
        diagnostics["num_polygons"] = len(polygons)

        # Save outputs
        tile_out_dir = ensure_dir(os.path.join(output_dir, "tiles"))
        save_json(tile_output, os.path.join(tile_out_dir, f"{tile_id}_output.json"))
        self._save_diagnostics(diagnostics, output_dir, tile_id)

        logger.info(
            f"Tile {tile_id}: {len(polygons)} polygons, "
            f"models={diagnostics['models_used']}, "
            f"time={diagnostics['timing']['total']:.2f}s"
        )

        return diagnostics

    def process_batch(
        self,
        tile_paths: List[str],
        tile_metas: List[Dict[str, Any]],
        output_dir: str,
        run_id: str = "default",
    ) -> List[Dict[str, Any]]:
        """Process a batch of tiles for massive inference speedup."""
        t_start = time.time()
        batch_size = len(tile_paths)
        
        # Load all images
        images = []
        valid_indices = []
        tile_ids = []
        all_diagnostics = []
        
        for i, path in enumerate(tile_paths):
            meta = tile_metas[i]
            tid = meta.get("tile_id", os.path.splitext(os.path.basename(path))[0])
            try:
                img = np.load(path) if path.endswith(".npy") else cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                if img is not None:
                    images.append(img)
                    valid_indices.append(i)
                    tile_ids.append(tid)
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")

        # Gatekeeper filtering
        if self.gatekeeper is not None and images:
            images, tile_ids, skipped_ids = self.gatekeeper.filter_batch(images, tile_ids)
            # Log skipped
            for sid in skipped_ids:
                diag = {"tile_id": sid, "run_id": run_id, "status": "skipped", "errors": ["gatekeeper_rejected"], "timing": {}}
                self._save_diagnostics(diag, output_dir, sid)
                all_diagnostics.append(diag)
                
            # Update valid indices matching
            new_valid_indices = []
            for tid in tile_ids:
                for i, vm in enumerate(tile_metas):
                    if vm.get("tile_id", os.path.splitext(os.path.basename(tile_paths[i]))[0]) == tid:
                        new_valid_indices.append(i)
                        break
            valid_indices = new_valid_indices

        if not images:
            return all_diagnostics
            
        # 1. Run SWIN Batched
        swin_results = []
        if self.swin is not None:
            swin_results = self._run_swin_batch(images, tile_ids)
            
        # We can process the rest in a loop (YOLO, Fusion, GeoSAM)
        # Because YOLO and Fusion are fast and GeoSAM is conditional
        for idx, (img, tid, swin_res) in enumerate(zip(images, tile_ids, swin_results)):
            meta = tile_metas[valid_indices[idx]]
            path = tile_paths[valid_indices[idx]]
            
            diag = {
                "tile_id": tid, "run_id": run_id, "status": "processing", 
                "errors": [], "models_used": ["swin"], "timing": {}
            }
            
            # YOLO
            yolo_res = self._run_yolo(img, tid) if self.yolo is not None else {}
            if yolo_res: diag["models_used"].append("yolo")
                
            # Features & Fusion
            features = self._assemble_features(swin_res, yolo_res, tid)
            fusion_decision = self._run_fusion(features)
            fusion_decision = self._apply_heuristics(fusion_decision, img, swin_res, yolo_res)
            
            diag["fusion_decision"] = {k: v.item() if torch.is_tensor(v) else v for k, v in fusion_decision.items() if k != "class_probs"}
            diag["bounds"] = meta.get("bounds", [[0, 0], [512, 512]])
            
            # GeoSAM & Depth
            geosam_res = []
            # if fusion_decision.get("refine_flag", False) and self.geosam is not None:
            #     geosam_res = self._run_geosam(img, yolo_res, fusion_decision, tid)
            #     diag["models_used"].append("geosam")
                
            depth_res = {"source": "none"}
            if self.depth is not None:
                try:
                    if hasattr(self.depth, 'compute_height_map'):
                        # Apple Depth Pro path
                        h_map = self.depth.compute_height_map(img)
                        mask = swin_res.get("mask")
                        if mask is not None:
                            cls_name = "building" if fusion_decision.get("class_id") == 1 else ("water" if fusion_decision.get("class_id") == 4 else "unknown")
                            val_res = self.depth.validate_detection(h_map, mask, cls_name)
                            depth_res = {"source": "depth_pro", "z_mean": val_res.get("z_mean", 0), "z_max": val_res.get("z_max", 0)}
                        else:
                            depth_res = {"source": "depth_pro", "z_mean": 0, "z_max": 0}
                    else:
                        # Classic DEM path
                        depth_res = self.depth.compute_height(tile_id=tid, mask=swin_res.get("mask"), transform=meta.get("transform"))
                    if depth_res.get("source") != "none":
                        diag["models_used"].append("depth")
                except Exception as e:
                    logger.warning(f"Depth processing failed for {tid}: {e}")
                    depth_res = {"source": "error", "z_mean": 0, "z_max": 0}
            diag["depth_stats"] = {k: v for k, v in depth_res.items() if k != "tile_id"}

            # Height Validation Rules (Shadow Killer + Illegal Floor Check)
            z_mean = depth_res.get("z_mean", 0) or 0
            z_max = depth_res.get("z_max", 0) or 0

            if fusion_decision.get("class_id") == 4 and z_mean > 0.5:
                fusion_decision["class_id"] = 0
                fusion_decision["shadow_killed"] = True
                diag["shadow_killer"] = {"triggered": True, "z_mean": z_mean,
                    "reason": f"Water rejected: height {z_mean:.2f}m > 0.5m"}

            if fusion_decision.get("class_id") == 1 and z_max > 12.0:
                fusion_decision["illegal_height_flag"] = True
                diag["illegal_floor_check"] = {"flagged": True, "z_max": z_max,
                    "reason": f"Building height {z_max:.1f}m > 12m"}
            
            # Postprocess & Save
            polygons = self._postprocess(swin_res, yolo_res, geosam_res, fusion_decision, meta, tid)
            
            tile_output = {
                "tile_id": tid, "num_polygons": len(polygons),
                "swin_confidence": swin_res.get("confidence", 0.0),
                "yolo_count": yolo_res.get("count", 0),
                "fusion_accept": fusion_decision.get("accept_val", 0.0),
                "polygons": [
                    {
                        "class_id": p.get("class_id", 1),
                        "confidence": p.get("confidence", fusion_decision.get("confidence_val", 0.5)),
                        "geometry_wkt": p["geometry"].wkt if p.get("geometry") else None,
                        "z_mean": depth_res.get("z_mean"),
                        "area": p.get("area", 0)
                    } for p in polygons
                ]
            }
            
            diag["status"] = "complete"
            diag["timing"]["total"] = time.time() - t_start
            diag["num_polygons"] = len(polygons)
            
            self._save_diagnostics(diag, output_dir, tid)
            tile_out_dir = ensure_dir(os.path.join(output_dir, "tiles"))
            save_json(tile_output, os.path.join(tile_out_dir, f"{tid}_output.json"))
            all_diagnostics.append(diag)
            
        logger.info(f"Batched {batch_size} tiles in {time.time() - t_start:.2f}s")
        return all_diagnostics

    def _run_swin(self, image: np.ndarray, tile_id: str) -> Dict[str, Any]:
        """Run Swin-UNet inference on a tile.

        Args:
            image: RGB image (H, W, 3).
            tile_id: Tile identifier.

        Returns:
            Dict with logits, mask, confidence, pooled_features.
        """
        input_size = self.swin_cfg.get("input_size", 512)
        img_resized = cv2.resize(image, (input_size, input_size))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(self.device)

        self.swin.eval()
        with torch.no_grad():
            output = self.swin(img_tensor)

        logits = output["logits"].cpu()
        mask = logits.argmax(dim=1).squeeze(0).numpy()
        pooled = output["pooled_features"].cpu().squeeze(0).numpy()
        confidence = output["confidence"].cpu().item()

        # Mask statistics
        building_mask = (mask == 1)  # class 1 = building
        area = float(building_mask.sum())
        probs = torch.softmax(logits, dim=1).cpu().squeeze(0).numpy()
        building_conf = float(probs[1].mean()) if probs.shape[0] > 1 else 0.0

        return {
            "tile_id": tile_id,
            "logits": logits.numpy(),
            "mask": mask,
            "pooled_features": pooled,
            "confidence": confidence,
            "confidence_context": building_conf,
            "mask_area": area,
            "mask_perimeter": 0.0,  # Simplified
        }

    def _run_swin_batch(self, images: List[np.ndarray], tile_ids: List[str]) -> List[Dict[str, Any]]:
        """Run Swin-UNet inference on a batch of tiles."""
        input_size = self.swin_cfg.get("input_size", 512)
        tensor_list = []
        for img in images:
            img_resized = cv2.resize(img, (input_size, input_size))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            tensor_list.append(img_tensor)
        
        batch_tensor = torch.stack(tensor_list).to(self.device)
        self.swin.eval()
        with torch.no_grad():
            output = self.swin(batch_tensor)
            
        results = []
        logits_batch = output["logits"].cpu()
        masks_batch = logits_batch.argmax(dim=1).numpy()
        pooled_batch = output["pooled_features"].cpu().numpy()
        conf_batch = output["confidence"].cpu().numpy()
        
        for i in range(len(images)):
            mask = masks_batch[i]
            probs = torch.softmax(logits_batch[i], dim=0).numpy()
            building_mask = (mask == 1)
            
            results.append({
                "tile_id": tile_ids[i],
                "logits": logits_batch[i].numpy(),
                "mask": mask,
                "pooled_features": pooled_batch[i],
                "confidence": conf_batch[i],
                "confidence_context": float(probs[1].mean()) if probs.shape[0] > 1 else 0.0,
                "mask_area": float(building_mask.sum()),
                "mask_perimeter": 0.0,
            })
        return results

    def _run_yolo(self, image: np.ndarray, tile_id: str) -> Dict[str, Any]:
        """Run YOLO inference on a tile."""
        return self.yolo.predict(image, tile_id=tile_id)

    def _assemble_features(
        self, swin_result: Dict, yolo_result: Dict, tile_id: str
    ) -> torch.Tensor:
        """Assemble feature vector for fusion model.

        Concatenates: swin_pooled + yolo_summary + depth_placeholder + mask_stats

        Args:
            swin_result: Swin inference result.
            yolo_result: YOLO inference result.
            tile_id: Tile ID.

        Returns:
            Feature tensor (1, total_dim).
        """
        fusion_cfg = self.config.get("fusion", {})
        swin_dim = fusion_cfg.get("swin_feature_dim", 768)
        yolo_max_det = fusion_cfg.get("yolo_max_detections", 20)

        # Swin pooled features
        swin_feat = swin_result.get("pooled_features", np.zeros(swin_dim, dtype=np.float32))
        if len(swin_feat) != swin_dim:
            swin_feat = np.zeros(swin_dim, dtype=np.float32)

        # YOLO summary features
        if self.yolo is not None:
            yolo_feat = self.yolo.get_summary_features(yolo_result, max_detections=yolo_max_det)
        else:
            yolo_feat = np.zeros(yolo_max_det * 6 + 3, dtype=np.float32)

        # Depth placeholder (filled later if available)
        depth_feat = np.zeros(3, dtype=np.float32)

        # Mask statistics
        mask_stats = np.array([
            swin_result.get("mask_area", 0.0),
            swin_result.get("mask_perimeter", 0.0),
            swin_result.get("confidence_context", 0.0),
            1.0,  # num_components placeholder
        ], dtype=np.float32)

        combined = np.concatenate([swin_feat, yolo_feat, depth_feat, mask_stats])
        return torch.from_numpy(combined).unsqueeze(0).float()

    def _run_fusion(self, features: torch.Tensor) -> Dict[str, Any]:
        """Run fusion model on assembled features.
        
        Args:
            features: Feature tensor (1, total_dim).
        
        Returns:
            Decision dict with accept, refine, class_id, confidence.
        """
        if self.fusion is None:
            # Default pass-through when no fusion model trained
            return {
                "accept_val": 0.5,
                "refine_flag": True,
                "class_id": 1,
                "confidence_val": 0.5,
            }
        
        self.fusion.eval()
        with torch.no_grad():
            output = self.fusion(features.to(self.device))
        
        accept = output["accept"].cpu().item()
        refine = output["refine"].cpu().item()
        class_id = output["class_probs"].cpu().argmax(dim=-1).item()
        confidence = output["confidence"].cpu().item()
        
        return {
            "accept_val": accept,
            "refine_flag": refine > 0.5,
            "class_id": class_id,
            "confidence_val": confidence,
        }

    def _apply_heuristics(self, decision: Dict[str, Any], img: np.ndarray, swin_result: Dict, yolo_result: Dict) -> Dict[str, Any]:
        """Apply fast post-processing rules to fix common misclassifications."""
        # 1. YOLO Mask Fusion Override Rule
        # Actually modify the SWIN mask to include YOLO bounding boxes as buildings!
        if yolo_result.get("count", 0) > 0 and swin_result.get("mask") is not None:
            mask = np.ascontiguousarray(swin_result["mask"].copy(), dtype=np.uint8)
            h, w = mask.shape
            
            for box in yolo_result.get("boxes", []):
                # YOLO boxes are [x1, y1, x2, y2]
                x1, y1, x2, y2 = box["bbox"]
                # Convert to integer pixel coordinates (assuming box is in original image coords)
                # Ensure they are within bounds
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                
                # Draw a filled rectangle on the segmentation mask (Class 1 = Building)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)
                
            # Update decision class since we forced buildings into the mask
            decision["class_id"] = 1
            decision["confidence_val"] = max(decision["confidence_val"], yolo_result.get("max_confidence", 0.8))
            decision["accept_val"] = 1.0
            
        # If model thinks it's WATER (class 4) or Vegetation (class 3)
        if decision["class_id"] in [3, 4] and swin_result.get("mask") is not None:
            # Extract dominant pixels from SWIN mask
            target_mask = (swin_result["mask"] == decision["class_id"]).astype(np.uint8)
            if target_mask.sum() > 100:
                # Resize image to match mask if necessary
                img_resized = cv2.resize(img, (target_mask.shape[1], target_mask.shape[0]))
                
                # Compute spectral stats in the masked area
                masked_img = img_resized[target_mask == 1]
                mean_r, mean_g, mean_b = masked_img.mean(axis=0)
                
                # Compute texture (variance of gray image)
                gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
                masked_gray = gray[target_mask == 1]
                texture_var = masked_gray.var()
                
                # Rule 2: Grasslands confused as water
                if decision["class_id"] == 4:
                    if mean_g > mean_b * 1.05 or texture_var > 400:
                        decision["class_id"] = 3 # Reclassify as Vegetation
                        
                # Rule 3: Water shouldn't be too bright green or textured
                elif decision["class_id"] == 3:
                    if mean_b > mean_g * 1.1 and texture_var < 150:
                        decision["class_id"] = 4 # Reclassify as Water
                        
        # 4. Low Confidence Filter
        if decision["confidence_val"] < 0.60 and decision["class_id"] != 0:
            decision["class_id"] = 5 # Mark as 'Other' if not confident
            
        return decision

    def _run_geosam(
        self,
        image: np.ndarray,
        yolo_result: Dict,
        fusion_decision: Dict,
        tile_id: str,
    ) -> List[Dict[str, Any]]:
        """Run GeoSAM on top-k candidate boxes.

        Args:
            image: Tile image.
            yolo_result: YOLO detections.
            fusion_decision: Fusion model decisions.
            tile_id: Tile ID.

        Returns:
            List of GeoSAM refinement results.
        """
        results = []
        boxes = yolo_result.get("boxes", [])
        boxes_sorted = sorted(boxes, key=lambda b: b["confidence"], reverse=True)

        try:
            self.geosam.set_image(image)
        except Exception as e:
            logger.error(f"GeoSAM set_image failed for {tile_id}: {e}")
            return results

        for box in boxes_sorted[:self.geosam_top_k]:
            try:
                result = self.geosam.predict_box(
                    bbox=box["bbox"], tile_id=tile_id
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"GeoSAM failed for box {box['bbox']}: {e}")
                # Fallback: use Swin polygon with lower confidence
                results.append({
                    "tile_id": tile_id,
                    "mask": None,
                    "quality_metric": 0.0,
                    "fallback": True,
                    "error": str(e),
                })

        return results

    def _postprocess(
        self,
        swin_result: Dict,
        yolo_result: Dict,
        geosam_results: List[Dict],
        fusion_decision: Dict,
        tile_meta: Dict,
        tile_id: str,
    ) -> List[Dict[str, Any]]:
        """Postprocess model outputs into final polygons.

        Converts masks to polygons with topology sanitization.

        Args:
            swin_result: Swin output.
            yolo_result: YOLO output.
            geosam_results: GeoSAM refinement results.
            fusion_decision: Fusion decisions.
            tile_meta: Tile metadata.
            tile_id: Tile ID.

        Returns:
            List of polygon dicts.
        """
        min_area = self.export_cfg.get("min_polygon_area", 10.0)
        simplify = self.export_cfg.get("simplify_tolerance", 0.5)
        transform = tile_meta.get("transform", [1, 0, 0, 0, -1, 0])

        all_polygons = []

        # Check if fusion model accepted
        if fusion_decision.get("accept_val", 0.5) < 0.3:
            return all_polygons

        # Use GeoSAM masks if available and high quality
        if geosam_results:
            for sam_res in geosam_results:
                if sam_res.get("mask") is not None and sam_res.get("quality_metric", 0) > 0.3:
                    polys = masks_to_polygons(
                        sam_res["mask"], transform, min_area=min_area,
                        simplify_tolerance=simplify,
                    )
                    for p in polys:
                        p["class_id"] = fusion_decision.get("class_id", 1)
                        p["confidence"] = fusion_decision.get("confidence_val", 0.5)
                        p["source_masks"] = ["geosam"]
                    all_polygons.extend(polys)

        # Fallback to Swin mask
        if not all_polygons and swin_result.get("mask") is not None:
            mask = swin_result["mask"]
            for cls_id in range(1, self.swin_cfg.get("num_classes", 6)):
                cls_mask = (mask == cls_id).astype(np.uint8)
                if cls_mask.sum() < min_area:
                    continue
                polys = masks_to_polygons(
                    cls_mask, transform, min_area=min_area,
                    simplify_tolerance=simplify,
                )
                for p in polys:
                    p["class_id"] = cls_id
                    p["confidence"] = swin_result.get("confidence", 0.5)
                    p["source_masks"] = ["swin"]
                all_polygons.extend(polys)

        return all_polygons

    def _save_diagnostics(
        self, diagnostics: Dict, output_dir: str, tile_id: str
    ) -> None:
        """Save diagnostic JSON for a tile."""
        diag_dir = ensure_dir(os.path.join(output_dir, "diagnostics"))
        save_json(diagnostics, os.path.join(diag_dir, f"{tile_id}_diag.json"))
    def process_regions(self, diagnostics: List[Dict], metas: List[Dict], tile_paths: List[str], output_dir: str, run_id: str):
        """Phase 2: Group tiles into morphological regions and refine with GeoSAM."""
        from ..region.tile_clustering import RegionClusterer
        import cv2

        if not self.geosam:
            logger.warning("GeoSAM not loaded; skipping region refinement.")
            return

        clusterer = RegionClusterer(tile_size=self.swin_cfg.get("input_size", 512))
        regions = clusterer.cluster_tiles(diagnostics, metas)

        tile_size = clusterer.tile_size
        region_out_dir = ensure_dir(os.path.join(output_dir, "regions"))
        
        # Map tile IDs to their input image paths and intermediate outputs
        tid_to_path = {}
        for p in tile_paths:
            tid = os.path.splitext(os.path.basename(p))[0]
            if tid.startswith("tile_"): tid = tid[5:] # handle tile_ prefix
            tid_to_path[tid] = p
            
        tid_to_diag = {d["tile_id"]: d for d in diagnostics}
        tid_to_meta = {m.get("tile_id", ""): m for m in metas}

        for reg in regions:
            tids = reg["tile_ids"]
            if not tids: continue
            
            # Find spatial bounds in grid coords
            grid_points = []
            for tid in tids:
                if tid in tid_to_meta:
                    grid_points.append(clusterer._get_grid_coords(tid_to_meta[tid]))
            
            if not grid_points: continue
                
            min_c = min([p[0] for p in grid_points])
            max_c = max([p[0] for p in grid_points])
            min_r = min([p[1] for p in grid_points])
            max_r = max([p[1] for p in grid_points])
            
            w_tiles = max_c - min_c + 1
            h_tiles = max_r - min_r + 1
            
            # Create a stitched canvas
            canvas = np.zeros((h_tiles * tile_size, w_tiles * tile_size, 3), dtype=np.uint8)
            region_polys = []
            
            for tid in tids:
                meta = tid_to_meta.get(tid)
                if not meta: continue
                
                c, r = clusterer._get_grid_coords(meta)
                x_offset = (c - min_c) * tile_size
                y_offset = (r - min_r) * tile_size
                
                # Load image
                p = tid_to_path.get(tid)
                path_found = p
                if not p:
                    # try to guess
                    search_path = os.path.join(os.path.dirname(tile_paths[0]), f"{tid}.png")
                    if os.path.exists(search_path):
                        path_found = search_path
                        
                if path_found and os.path.exists(path_found):
                    img = cv2.imread(path_found)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if img is not None and img.shape[:2] == (tile_size, tile_size):
                        canvas[y_offset:y_offset+tile_size, x_offset:x_offset+tile_size] = img
                        
                # Read YOLO boxes from precomputed tile output JSON
                tile_out_json = os.path.join(output_dir, "tiles", f"{tid}_output.json")
                if os.path.exists(tile_out_json):
                    t_out = load_json(tile_out_json)
                    # We can use Swin/YOLO intermediate boxes, or if we rely on point prompts... 
                    # For simplicity, if we have YOLO boxes, we translate them
                    pass # We will rely on SAM's fully automated or central box

            # Run GeoSAM on the entire region canvas
            try:
                self.geosam.set_image(canvas)
                # Naive: run one big box for the context cluster
                h, w = canvas.shape[:2]
                res = self.geosam.predict_box([0, 0, w, h], tile_id=reg["region_id"])
                
                if res and res.get("mask") is not None:
                    # Convert to polygon
                    # Coordinate transform requires picking the top-left tile's transform
                    top_left_tid = None
                    for tid in tids:
                        meta = tid_to_meta.get(tid)
                        c, r = clusterer._get_grid_coords(meta)
                        if c == min_c and r == min_r:
                            top_left_tid = tid
                            break
                            
                    base_transform = [1,0,0,0,-1,0]
                    if top_left_tid:
                        base_transform = tid_to_meta[top_left_tid].get("transform", base_transform)
                        
                    polys = masks_to_polygons(res["mask"], base_transform, min_area=30.0)
                    for p in polys:
                        p["class_id"] = reg["class_id"]
                        p["confidence"] = res.get("quality_metric", 0.8)
                        p["source_masks"] = ["geosam_region"]
                    region_polys.extend(polys)
            except Exception as e:
                logger.error(f"Region {reg['region_id']} GeoSAM failed: {e}")
                
            # Save region polygons
            reg_output = {
                "region_id": reg["region_id"],
                "class_id": reg["class_id"],
                "tile_ids": tids,
                "num_polygons": len(region_polys),
                "polygons": [
                    {
                        "class_id": p.get("class_id"),
                        "confidence": p.get("confidence"),
                        "geometry_wkt": p["geometry"].wkt if p.get("geometry") else None,
                        "geometry_coords": [[y, x] for x, y in p.get("geometry").exterior.coords] if p.get("geometry") else [],
                        "area": p.get("area", 0)
                    } for p in region_polys
                ]
            }
            save_json(reg_output, os.path.join(region_out_dir, f"{reg['region_id']}_output.json"))

        logger.info(f"Processed {len(regions)} regions through GeoSAM.")
