"""
Hydra-Map Smoke Test  (Windows / Python-3.14 compatible).

Runs a minimal end-to-end pipeline test with synthetic data.
Requires only: numpy, opencv-python-headless, pyyaml, rasterio, shapely.
Does NOT require: torch, timm, ultralytics, geopandas, laspy.

Steps tested
------------
  1. Create a synthetic GeoTIFF in a temp directory
  2. Variance filter (numpy-only logic)
  3. Tiler (generates tile PNG + JSON metadata)
  4. Inference   — uses pure-numpy stubs (no torch)
  5. GeoPackage export (JSON fallback when geopandas absent)
  6. Diagnostic JSON files written
  7. Assert all output files exist with size > 0

Usage:  python src/tests/smoke_test.py
        OR: .\\run_smoke_test.ps1
"""

import json
import logging
import os
import shutil
import sys
import tempfile
import traceback

import cv2
import numpy as np

# ── Project root on path ────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [SMOKE] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Pure-numpy stubs — replace torch-dependent model classes
# ============================================================================

class _StubSwinUNet:
    """No-torch stub for SwinUNet. Returns deterministic numpy outputs."""

    def __init__(self, num_classes=6, swin_dim=768, input_size=512):
        self.num_classes = num_classes
        self.swin_dim = swin_dim
        self.input_size = input_size

    def eval(self):
        return self

    def __call__(self, *a, **kw):
        return self.forward_np()

    def forward_np(self):
        h = w = self.input_size
        nc = self.num_classes
        logits = np.zeros((1, nc, h, w), dtype=np.float32)
        # Mark a small square as "building" (class 1)
        logits[0, 1, 100:200, 100:200] = 2.0
        pooled = np.random.rand(1, self.swin_dim).astype(np.float32)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        confidence = np.max(probs, axis=1).mean()
        return {
            "logits_np": logits,
            "pooled_np": pooled[0],
            "confidence": float(confidence),
            "mask": np.argmax(logits[0], axis=0).astype(np.uint8),
        }


class _StubYOLO:
    """No-torch stub for YOLOWrapper."""

    def predict(self, image, tile_id=""):
        h, w = image.shape[:2]
        boxes = [{"bbox": [w*0.3, h*0.3, w*0.7, h*0.7],
                  "confidence": 0.85, "class_id": 0, "class_name": "building"}]
        return {"tile_id": tile_id, "boxes": boxes, "count": 1,
                "max_confidence": 0.85, "avg_confidence": 0.85, "confidence_yolo": 0.85}

    def get_summary_features(self, result, max_detections=10):
        feat = np.zeros(max_detections * 6 + 3, dtype=np.float32)
        feat[-3] = result.get("count", 0)
        feat[-2] = result.get("max_confidence", 0)
        feat[-1] = result.get("avg_confidence", 0)
        return feat


class _StubGeoSAM:
    """No-torch stub for GeoSAMWrapper."""

    def set_image(self, image):
        self._h, self._w = image.shape[:2]

    def predict_box(self, bbox, tile_id=""):
        h = getattr(self, "_h", 512)
        w = getattr(self, "_w", 512)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        mask = np.zeros((h, w), dtype=bool)
        mask[max(0,y1):min(h,y2), max(0,x1):min(w,x2)] = True
        return {"tile_id": tile_id, "mask": mask, "mask_area": float(mask.sum()),
                "iou_prediction": 0.75, "stability_score": 0.75, "quality_metric": 0.75}


class _StubDepth:
    """No-laspy stub for DepthPipeline."""

    def compute_height(self, tile_id, mask=None, transform=None, bbox=None):
        return {"tile_id": tile_id, "z_mean": None, "z_median": None,
                "z_std": None, "z_min": None, "z_max": None, "source": "none"}


class _StubFusion:
    """No-torch stub for FusionMLP."""

    def eval(self):
        return self

    def __call__(self, features):
        return {
            "accept":       np.array([0.85]),
            "refine":       np.array([0.4]),
            "class_probs":  np.eye(6)[[1]],    # always class 1 = building
            "confidence":   np.array([0.8]),
        }


# ============================================================================
# Patched Orchestrator that works without torch
# ============================================================================

class _StubOrchestrator:
    """Thin orchestrator that replaces torch tensors with numpy arrays."""

    def __init__(self, config, swin, yolo, geosam, depth, fusion, device="cpu"):
        self.config = config
        self.swin   = swin
        self.yolo   = yolo
        self.geosam = geosam
        self.depth  = depth
        self.fusion = fusion
        self.inf_cfg    = config.get("inference", {})
        self.swin_cfg   = config.get("swin", {})
        self.export_cfg = config.get("export", {})
        self.geosam_top_k = self.inf_cfg.get("geosam_top_k", 2)

    def process_tile(self, tile_path, tile_meta, output_dir, run_id="default"):
        from src.utils.io import ensure_dir, save_json
        from src.utils.geo import masks_to_polygons

        tile_id = tile_meta.get("tile_id",
                                os.path.splitext(os.path.basename(tile_path))[0])
        diagnostics = {
            "tile_id": tile_id, "run_id": run_id, "status": "processing",
            "errors": [], "models_used": [], "timing": {},
        }

        # Load tile
        tile_img = cv2.imread(tile_path)
        if tile_img is None:
            diagnostics["status"] = "error"
            diagnostics["errors"].append("load_error: could not read tile")
            self._save_diag(diagnostics, output_dir, tile_id)
            return diagnostics
        tile_img = cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB)

        # Swin
        swin_out = self.swin.forward_np()
        diagnostics["models_used"].append("swin")

        # YOLO
        yolo_out = self.yolo.predict(tile_img, tile_id=tile_id)
        diagnostics["models_used"].append("yolo")

        # Fusion
        swin_feat = swin_out["pooled_np"]
        yolo_feat = self.yolo.get_summary_features(yolo_out, max_detections=10)
        depth_feat = np.zeros(3, dtype=np.float32)
        mask_stats = np.array([swin_out.get("mask_area", 0.0) if hasattr(swin_out, "get") 
                                else float((swin_out["mask"] == 1).sum()),
                               0.0, float(swin_out["confidence"]), 1.0], dtype=np.float32)
        combined = np.concatenate([swin_feat, yolo_feat, depth_feat, mask_stats])
        fusion_raw = self.fusion(combined)

        accept_val   = float(np.squeeze(fusion_raw["accept"]))
        refine_flag  = float(np.squeeze(fusion_raw["refine"])) > 0.5
        class_id     = int(np.argmax(np.squeeze(fusion_raw["class_probs"])))
        confidence   = float(np.squeeze(fusion_raw["confidence"]))
        fusion_decision = {"accept_val": accept_val, "refine_flag": refine_flag,
                           "class_id": class_id, "confidence_val": confidence}
        diagnostics["fusion_decision"] = {k: v for k, v in fusion_decision.items()
                                          if not isinstance(v, np.ndarray)}

        # GeoSAM (if refine)
        geosam_results = []
        if refine_flag and yolo_out.get("boxes"):
            self.geosam.set_image(tile_img)
            for box in yolo_out["boxes"][:self.geosam_top_k]:
                res = self.geosam.predict_box(box["bbox"], tile_id=tile_id)
                geosam_results.append(res)
            diagnostics["models_used"].append("geosam")

        # Depth
        depth_res = self.depth.compute_height(tile_id=tile_id,
                                              mask=swin_out["mask"],
                                              transform=tile_meta.get("transform"))
        diagnostics["depth_stats"] = {k: v for k, v in depth_res.items()
                                      if k != "tile_id"}

        # Postprocess — polygons
        transform = tile_meta.get("transform", [1, 0, 0, 0, -1, 0])
        min_area  = self.export_cfg.get("min_polygon_area", 5.0)
        simplify  = self.export_cfg.get("simplify_tolerance", 1.0)
        polygons  = []

        if accept_val >= 0.3:
            if geosam_results:
                for sam_res in geosam_results:
                    if sam_res.get("mask") is not None and sam_res.get("quality_metric", 0) > 0.3:
                        polys = masks_to_polygons(sam_res["mask"], transform,
                                                  min_area=min_area,
                                                  simplify_tolerance=simplify)
                        for p in polys:
                            p["class_id"]     = class_id
                            p["confidence"]   = confidence
                            p["source_masks"] = ["geosam"]
                        polygons.extend(polys)

            if not polygons and swin_out.get("mask") is not None:
                mask = swin_out["mask"]
                for cls in range(1, self.swin_cfg.get("num_classes", 6)):
                    cls_mask = (mask == cls).astype(np.uint8)
                    if cls_mask.sum() < min_area:
                        continue
                    polys = masks_to_polygons(cls_mask, transform,
                                             min_area=min_area,
                                             simplify_tolerance=simplify)
                    for p in polys:
                        p["class_id"]     = cls
                        p["confidence"]   = float(swin_out["confidence"])
                        p["source_masks"] = ["swin"]
                    polygons.extend(polys)

        # Write tile output
        tile_out_dir = ensure_dir(os.path.join(output_dir, "tiles"))
        geom_attr = "wkt"  # _StubGeom uses .wkt; real shapely uses .wkt too
        tile_output = {
            "tile_id": tile_id,
            "polygons": [
                {
                    "geometry_wkt": (getattr(p["geometry"], "wkt", None)
                                     if p.get("geometry") else None),
                    "class_id":   p.get("class_id", 1),
                    "confidence": p.get("confidence", confidence),
                    "source_masks": p.get("source_masks", diagnostics["models_used"]),
                    "z_mean": depth_res.get("z_mean"),
                    "area":      p.get("area", 0),
                    "perimeter": p.get("perimeter", 0),
                }
                for p in polygons
            ],
            "num_polygons": len(polygons),
            "swin_confidence": float(swin_out["confidence"]),
            "yolo_count": yolo_out.get("count", 0),
            "fusion_accept": accept_val,
        }
        save_json(tile_output,
                  os.path.join(tile_out_dir, f"{tile_id}_output.json"))

        diagnostics["status"]       = "complete"
        diagnostics["num_polygons"] = len(polygons)
        self._save_diag(diagnostics, output_dir, tile_id)
        logger.info(f"Tile {tile_id}: {len(polygons)} polygons, "
                    f"models={diagnostics['models_used']}")
        return diagnostics

    def _save_diag(self, diagnostics, output_dir, tile_id):
        from src.utils.io import ensure_dir, save_json
        diag_dir = ensure_dir(os.path.join(output_dir, "diagnostics"))
        save_json(diagnostics, os.path.join(diag_dir, f"{tile_id}_diag.json"))


# ============================================================================
# Test data creation
# ============================================================================

def create_test_data(test_dir: str) -> dict:
    """Create minimal synthetic test data — requires only numpy + rasterio(optional)."""
    raw_dir    = os.path.join(test_dir, "data", "raw")
    ann_dir    = os.path.join(test_dir, "data", "annotations")
    meta_dir   = os.path.join(test_dir, "data", "meta")
    preproc    = os.path.join(test_dir, "data", "preprocessed")
    models_dir = os.path.join(test_dir, "models")
    output_dir = os.path.join(test_dir, "output")
    pc_dir     = os.path.join(test_dir, "data", "pointcloud")
    dem_dir    = os.path.join(test_dir, "data", "dem")
    archive    = os.path.join(ann_dir, "archive")

    for d in [raw_dir, ann_dir, meta_dir, preproc, models_dir, output_dir,
              pc_dir, dem_dir, archive]:
        os.makedirs(d, exist_ok=True)

    # Synthetic GeoTIFF
    np.random.seed(42)
    h, w = 512, 512
    img = np.random.randint(20, 80, (h, w, 3), dtype=np.uint8)
    img[100:200, 100:200] = np.random.randint(150, 250, (100, 100, 3), dtype=np.uint8)
    img[300:380, 250:350] = np.random.randint(140, 240, (80, 100, 3), dtype=np.uint8)

    tif_path = os.path.join(raw_dir, "TEST_VILLAGE_001.tif")
    try:
        import rasterio
        from rasterio.transform import from_bounds
        transform = from_bounds(82.0, 18.0, 82.01, 18.01, w, h)
        with rasterio.open(tif_path, "w", driver="GTiff", dtype="uint8",
                           width=w, height=h, count=3,
                           crs="EPSG:4326", transform=transform) as dst:
            dst.write(img.transpose(2, 0, 1))
        logger.info(f"Created GeoTIFF: {tif_path}")
    except ImportError:
        cv2.imwrite(tif_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        logger.info(f"Created PNG fallback: {tif_path}")

    # Ground truth mask
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[100:200, 100:200] = 1
    mask[300:380, 250:350] = 1
    cv2.imwrite(os.path.join(ann_dir, "TEST_VILLAGE_001_mask.png"), mask)

    # Config
    config = {
        "project": {"name": "smoke-test"},
        "data": {
            "raw_dir": raw_dir, "pointcloud_dir": pc_dir,
            "meta_dir": meta_dir, "dem_dir": dem_dir,
            "preprocessed_dir": preproc, "annotations_dir": ann_dir,
            "annotations_archive": archive, "models_dir": models_dir,
            "output_dir": output_dir,
        },
        "preprocess": {
            "v_global_thresh": 5, "v_local_thresh": 3,
            "local_window_size": 50, "tile_sizes": [512],
            "stride": 256, "output_format": "png",
        },
        "swin": {
            "encoder": "swin_tiny_patch4_window7_224", "pretrained": False,
            "num_classes": 6,
            "class_names": ["background","building","road","vegetation","water","other"],
            "input_size": 512, "batch_size": 1,
            "freeze_epochs": 1, "unfreeze_epochs": 1,
            "lr_freeze": 1e-4, "lr_finetune": 1e-5,
            "weight_decay": 1e-4, "scheduler": "cosine",
            "loss": "dice_ce", "fold_k": 2, "pseudo_label_threshold": 0,
        },
        "yolo": {
            "model": "yolov8n.pt", "input_size": 640,
            "confidence_threshold": 0.1, "iou_threshold": 0.45,
            "max_detections": 50,
        },
        "geosam": {"checkpoint": "nonexistent.pth", "model_type": "vit_h", "use_lora": False},
        "depth": {"source": "none"},
        "fusion": {
            "arch": "mlp", "swin_feature_dim": 768,
            "yolo_max_detections": 10, "yolo_feature_dim": 6,
            "depth_feature_dim": 3, "mask_stats_dim": 4,
            "hidden_dims": [64, 32], "dropout": 0.1, "num_classes": 6,
            "batch_size": 8, "lr": 1e-3, "epochs": 3,
            "early_stopping_patience": 2, "fold_k": 2,
        },
        "inference": {"batch_size": 1, "debug": True, "cpu_only": True, "geosam_top_k": 2},
        "export": {
            "simplify_tolerance": 1.0, "cog_tile_size": 256,
            "cog_compression": "LZW", "min_polygon_area": 5.0,
        },
        "onnx": {"quantize": False, "opset_version": 17},
        "observability": {"use_wandb": False, "log_diagnostics": True},
        "augmentation": {
            "use_cutmix": False, "cutmix_alpha": 1.0,
            "use_mixup": False, "mixup_alpha": 0.4,
            "use_copy_paste": False, "heavy_augs": False,
        },
    }
    return config


# ============================================================================
# Main smoke-test runner
# ============================================================================

def run_smoke_test():
    logger.info("=" * 60)
    logger.info("HYDRA-MAP SMOKE TEST — START")
    logger.info("=" * 60)

    test_dir = os.path.join(PROJECT_ROOT, "test_output_smoke")
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    try:
        # ── Step 1: Synthetic test data ──────────────────────────────────────
        logger.info("Step 1: Creating synthetic test data...")
        config = create_test_data(test_dir)
        logger.info("  ✓ Test data created")

        # ── Step 2: Variance filter ──────────────────────────────────────────
        logger.info("Step 2: Running variance filter...")
        from src.preprocess.variance_filter import run_variance_filter
        filter_results = run_variance_filter(config)
        assert len(filter_results["accepted"]) > 0, "No tiles accepted by variance filter"
        logger.info(f"  ✓ Variance filter: {len(filter_results['accepted'])} accepted")

        # ── Step 3: Tiling ─────────────────────────────────────────────────
        logger.info("Step 3: Running tiler...")
        from src.preprocess.tiler import run_tiler
        tile_counts = run_tiler(config)
        total_tiles = sum(tile_counts.values())
        assert total_tiles > 0, "No tiles generated"
        logger.info(f"  ✓ Tiler: {total_tiles} tiles generated")

        # ── Step 4: Inference with numpy stubs ─────────────────────────────
        logger.info("Step 4: Running inference (numpy stubs — no torch required)...")

        swin   = _StubSwinUNet(num_classes=6, swin_dim=768, input_size=512)
        yolo   = _StubYOLO()
        geosam = _StubGeoSAM()
        depth  = _StubDepth()
        fusion = _StubFusion()

        orchestrator = _StubOrchestrator(
            config=config, swin=swin, yolo=yolo,
            geosam=geosam, depth=depth, fusion=fusion,
        )

        from src.utils.io import list_files, load_json
        tile_dir    = os.path.join(config["data"]["preprocessed_dir"], "tiles", "512")
        tile_images = list_files(tile_dir, extensions=[".png"])
        assert len(tile_images) > 0, "No tile images found in preprocessed directory"

        run_id     = "smoke_test"
        output_dir = os.path.join(config["data"]["output_dir"], run_id)

        for tp in tile_images[:1]:
            tile_id  = os.path.splitext(os.path.basename(tp))[0]
            meta_path = os.path.join(tile_dir, f"tile_{tile_id}.json")
            tile_meta = (load_json(meta_path)
                         if os.path.isfile(meta_path)
                         else {"tile_id": tile_id, "transform": [1, 0, 0, 0, -1, 0]})
            diag = orchestrator.process_tile(tp, tile_meta, output_dir, run_id)
            assert diag["status"] == "complete", f"Tile processing failed: {diag}"

        logger.info("  ✓ Inference complete")

        # ── Step 5: Export to GeoPackage ────────────────────────────────────
        logger.info("Step 5: Exporting outputs...")
        from src.export.export_ogc import collect_tile_outputs, export_geopackage

        tiles_dir    = os.path.join(output_dir, "tiles")
        tile_outputs = collect_tile_outputs(tiles_dir)
        gpkg_path    = os.path.join(output_dir, "final.gpkg")
        export_geopackage(tile_outputs, gpkg_path)

        assert os.path.isfile(gpkg_path), f"Output not created at {gpkg_path}"
        assert os.path.getsize(gpkg_path) > 0, "Output file is empty"
        logger.info(f"  ✓ Output exported: {gpkg_path} ({os.path.getsize(gpkg_path)} bytes)")

        # ── Step 6: Diagnostics ────────────────────────────────────────────
        logger.info("Step 6: Verifying diagnostics...")
        diag_dir   = os.path.join(output_dir, "diagnostics")
        diag_files = list_files(diag_dir, extensions=[".json"]) if os.path.isdir(diag_dir) else []
        assert len(diag_files) > 0, "No diagnostic files written"
        logger.info(f"  ✓ Diagnostics: {len(diag_files)} files")

        logger.info("")
        logger.info("=" * 60)
        logger.info("SMOKE TEST PASSED ✓")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"SMOKE TEST FAILED: {e}")
        traceback.print_exc()
        return False

    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
