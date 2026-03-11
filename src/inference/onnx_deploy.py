"""
Hydra-Map Production ONNX Deployment.

Exports trained Swin-UNet model to ONNX format with INT8 quantization
and creates an ONNX Runtime inference server for low-latency serving.

CLI:
  Export:  python src/inference/onnx_deploy.py --export --config config/config.yaml
  Serve:   python src/inference/onnx_deploy.py --serve --port 5000
"""

import os
import sys
import logging
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def export_to_onnx(config: dict, output_path: str = "models/swin_unet.onnx"):
    """Export Swin-UNet to ONNX format.
    
    Args:
        config: Config dict with model paths.
        output_path: Output ONNX model path.
    """
    import torch
    from src.models.swin_unet import SwinUNet
    
    device = "cpu"  # Export on CPU for compatibility
    num_classes = config["swin"]["num_classes"]
    input_size = config["swin"]["input_size"]
    encoder = config["swin"].get("encoder", "swin_tiny_patch4_window7_224")
    
    model = SwinUNet(
        num_classes=num_classes,
        encoder_name=encoder,
        input_size=input_size,
    )
    
    # Load trained weights
    ckpt_path = config["swin"]["checkpoint"]
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        logger.info(f"Loaded weights from {ckpt_path}")
    
    model.eval()
    
    # Create dummy input
    dummy = torch.randn(1, 3, input_size, input_size)
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["image"],
        output_names=["logits", "pooled_features", "confidence"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    
    logger.info(f"ONNX model exported to {output_path}")
    
    # Quantize to INT8
    try:
        import onnxruntime as ort
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        quantized_path = output_path.replace(".onnx", "_int8.onnx")
        quantize_dynamic(output_path, quantized_path, weight_type=QuantType.QInt8)
        logger.info(f"INT8 quantized model saved to {quantized_path}")
        
        # Size comparison
        orig_size = os.path.getsize(output_path) / 1024 / 1024
        quant_size = os.path.getsize(quantized_path) / 1024 / 1024
        logger.info(f"Size: {orig_size:.1f}MB → {quant_size:.1f}MB ({(1-quant_size/orig_size)*100:.0f}% smaller)")
    except ImportError:
        logger.warning("onnxruntime not available for INT8 quantization")
    
    return output_path


def create_onnx_inference_session(model_path: str) -> "ort.InferenceSession":
    """Create ONNX Runtime inference session with optimizations.
    
    Args:
        model_path: Path to ONNX model.
    
    Returns:
        ONNX Runtime InferenceSession.
    """
    import onnxruntime as ort
    
    providers = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = 4
    
    session = ort.InferenceSession(model_path, session_options, providers=providers)
    logger.info(f"ONNX session created with providers: {session.get_providers()}")
    
    return session


def onnx_predict(session, image: np.ndarray, input_size: int = 512) -> dict:
    """Run inference using ONNX Runtime.
    
    Args:
        session: ONNX InferenceSession.
        image: RGB image (H, W, 3).
        input_size: Model input size.
    
    Returns:
        Dict with logits, mask, confidence.
    """
    import cv2
    
    img_resized = cv2.resize(image, (input_size, input_size))
    img_tensor = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_tensor = np.expand_dims(img_tensor, axis=0)
    
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_tensor})
    
    logits = outputs[0]
    mask = logits.argmax(axis=1).squeeze(0)
    confidence = float(logits.max())
    
    return {
        "logits": logits,
        "mask": mask,
        "confidence": confidence,
    }


def serve_onnx(model_path: str, port: int = 5000, input_size: int = 512):
    """Start a simple HTTP inference server using ONNX Runtime.
    
    Serves predictions via POST /predict endpoint.
    
    Args:
        model_path: Path to ONNX model.
        port: Server port.
        input_size: Model input size.
    """
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json
    import cv2
    
    session = create_onnx_inference_session(model_path)
    
    class PredictionHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/predict":
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len)
                
                # Decode image from request body
                img_array = np.frombuffer(body, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img is None:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"Invalid image data")
                    return
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = onnx_predict(session, img_rgb, input_size)
                
                response = {
                    "mask_shape": list(result["mask"].shape),
                    "confidence": result["confidence"],
                    "class_distribution": {
                        int(cls): int((result["mask"] == cls).sum())
                        for cls in np.unique(result["mask"])
                    },
                }
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            else:
                self.send_response(404)
                self.end_headers()
        
        def log_message(self, format, *args):
            logger.info(format % args)
    
    server = HTTPServer(("0.0.0.0", port), PredictionHandler)
    logger.info(f"ONNX inference server running on http://0.0.0.0:{port}/predict")
    server.serve_forever()


if __name__ == "__main__":
    from src.utils.io import load_config
    
    parser = argparse.ArgumentParser(description="Hydra-Map ONNX Deployment")
    parser.add_argument("--export", action="store_true", help="Export model to ONNX")
    parser.add_argument("--serve", action="store_true", help="Start inference server")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model", type=str, default="models/swin_unet.onnx")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    
    if args.export:
        config = load_config(args.config)
        export_to_onnx(config, args.model)
    
    if args.serve:
        config = load_config(args.config)
        serve_onnx(args.model, port=args.port, input_size=config.get("swin", {}).get("input_size", 512))
"""
Hydra-Map Production ONNX Deployment.

Exports trained Swin-UNet model to ONNX format with INT8 quantization
and creates an ONNX Runtime inference server for low-latency serving.

CLI:
  Export:  python src/inference/onnx_deploy.py --export --config config/config.yaml
  Serve:   python src/inference/onnx_deploy.py --serve --port 5000
"""

import os
import sys
import logging
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def export_to_onnx(config: dict, output_path: str = "models/swin_unet.onnx"):
    """Export Swin-UNet to ONNX format."""
    import torch
    from src.models.swin_unet import SwinUNet
    
    device = "cpu"
    num_classes = config["swin"]["num_classes"]
    input_size = config["swin"]["input_size"]
    encoder = config["swin"].get("encoder", "swin_tiny_patch4_window7_224")
    
    model = SwinUNet(num_classes=num_classes, encoder_name=encoder, input_size=input_size)
    
    ckpt_path = config["swin"]["checkpoint"]
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        logger.info(f"Loaded weights from {ckpt_path}")
    
    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size)
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["image"],
        output_names=["logits", "pooled_features", "confidence"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
    )
    
    logger.info(f"ONNX model exported to {output_path}")
    
    try:
        import onnxruntime as ort
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        quantized_path = output_path.replace(".onnx", "_int8.onnx")
        quantize_dynamic(output_path, quantized_path, weight_type=QuantType.QInt8)
        logger.info(f"INT8 quantized model saved to {quantized_path}")
    except ImportError:
        logger.warning("onnxruntime not available for INT8 quantization")
    
    return output_path


if __name__ == "__main__":
    from src.utils.io import load_config
    
    parser = argparse.ArgumentParser(description="Hydra-Map ONNX Deployment")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model", type=str, default="models/swin_unet.onnx")
    args = parser.parse_args()
    
    if args.export:
        config = load_config(args.config)
        export_to_onnx(config, args.model)
