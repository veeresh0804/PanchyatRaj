"""
Hydra-Map Dashboard Server.

A zero-dependency HTTP server to serve the visualization frontend
and provide API endpoints to read pipeline outputs (JSONs / tiles).
"""

import http.server
import json
import os
import socketserver
import urllib.parse
from pathlib import Path

# Paths to data
ROOT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = ROOT_DIR / "output"
TILES_DIR = ROOT_DIR / "data" / "clean_dataset_2" / "tiles"

PORT = 8080

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler for API routes and static files."""
    def __init__(self, *args, **kwargs):
        # Serve static files from the visualize folder
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)

    def log_message(self, format, *args):
        # Mute logging to prevent terminal spam
        return

    def end_headers(self):
        # Prevent browser caching of static assets during development
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path

        # API: Get run details
        if path.startswith("/api/runs/"):
            run_id = path.split("/")[-1]
            run_dir = OUTPUT_DIR / run_id
            summary_path = run_dir / "run_summary.json"
            diag_dir = run_dir / "diagnostics"
            
            if run_dir.exists() and summary_path.exists():
                with open(summary_path, "r") as f:
                    data = json.load(f)
                
                # Aggregate individual tile predictions
                predictions = []
                if diag_dir.exists():
                    tiles_dir = run_dir / "tiles"
                    for p in diag_dir.glob("*_diag.json"):
                        try:
                            with open(p, "r") as f:
                                diag_data = json.load(f)
                            
                            tile_id = diag_data.get("tile_id")
                            
                            # Load corresponding output json to get all detected features
                            class_distribution = []
                            output_path = tiles_dir / f"{tile_id}_output.json"
                            if output_path.exists():
                                try:
                                    with open(output_path, "r") as f_out:
                                        out_data = json.load(f_out)
                                        class_map = {}
                                        for poly in out_data.get("polygons", []):
                                            c_id = poly.get("class_id")
                                            conf = poly.get("confidence", 0)
                                            # Keep max confidence per class
                                            if c_id not in class_map or conf > class_map[c_id]:
                                                class_map[c_id] = conf
                                        class_distribution = [{"class_id": k, "confidence": v} for k, v in class_map.items()]
                                except Exception as e:
                                    pass
                            
                            
                            # Geographic Coordinate Resolution
                            bounds = diag_data.get("bounds", [[0,0], [512,512]])
                            if bounds and bounds[0][0] > 1000:
                                import pyproj
                                # Dynamically project coordinate system
                                data_crs = diag_data.get("crs", "EPSG:32643")
                                transformer = pyproj.Transformer.from_crs(data_crs, "EPSG:4326", always_xy=True)

                                # bounds are [Y, X]? No, rasterio bounds are [minX, minY], [maxX, maxY]
                                # But we might have stored them as [[min_x, min_y], [max_x, max_y]]
                                # Notice the script previously printed [[2105304, 524977], ...] which is [Y, X]
                                y1, x1 = bounds[0][0], bounds[0][1]
                                y2, x2 = bounds[1][0], bounds[1][1]
                                # UTM format usually X is in 500k, Y is in millions.
                                # Let's correctly order them for Transformer (X, Y) -> (Lon, Lat)
                                # Assuming [[Y, X], [Y, X]] based on the millions value being first
                                lon1, lat1 = transformer.transform(x1, y1)
                                lon2, lat2 = transformer.transform(x2, y2)
                                bounds = [[lat1, lon1], [lat2, lon2]]

                            # Map keys to what the frontend expects
                            pred = {
                                "tile_id": tile_id,
                                "final_class": diag_data.get("fusion_decision", {}).get("class_id", 0),
                                "final_confidence": diag_data.get("fusion_decision", {}).get("confidence_val", 0),
                                "swin_confidence": diag_data.get("fusion_decision", {}).get("accept_val", 0),
                                "yolo_count": diag_data.get("num_polygons", 0),
                                "yolo_max_conf": max(0.2, diag_data.get("fusion_decision", {}).get("confidence_val", 0)),
                                "metrics": diag_data.get("depth_stats", {}),
                                "action": "refine" if diag_data.get("fusion_decision", {}).get("refine_flag") else "accept",
                                "class_distribution": class_distribution,
                                "bounds": bounds
                            }
                            predictions.append(pred)
                        except Exception:
                            pass
                            
                data["predictions"] = predictions
                
                # Load region clusters
                regions = []
                regions_dir = run_dir / "regions"
                if regions_dir.exists():
                    for p in regions_dir.glob("*_output.json"):
                        try:
                            with open(p, "r") as f:
                                reg_data = json.load(f)
                            regions.append(reg_data)
                        except Exception:
                            pass
                data["regions"] = regions
                
                # Add processed_tiles field for frontend
                data["processed_tiles"] = data.get("completed", 0)
                
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode("utf-8"))
            else:
                self.send_error(404, "Run not found")
            return

        # Serve tile images from the data directory
        if path.startswith("/static/tiles/"):
            # expected format: /static/tiles/512/BADETUMNAR...png
            rel_path = path[len("/static/tiles/"):]
            tile_path = TILES_DIR / rel_path
            
            if tile_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                with open(tile_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Tile not found")
            return

        # Default static file serving for HTML/CSS/JS
        return super().do_GET()

def run_server():
    os.chdir(str(Path(__file__).parent))  # Ensure CWD is visualize/
    with socketserver.ThreadingTCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"Server started at http://localhost:{PORT}")
        print(f"Reading pipeline outputs from: {OUTPUT_DIR}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")

if __name__ == "__main__":
    run_server()
