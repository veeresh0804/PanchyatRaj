import os
import subprocess
import threading
import time
import uuid
from datetime import datetime
from database import SessionLocal
import models
from sqlalchemy.sql import func

class PipelineState:
    def __init__(self):
        self.state = "IDLE"  # IDLE, RUNNING, FAILED, COMPLETED
        self.tiles_total = 0
        self.tiles_done = 0
        self.dataset = ""
        self.run_id = ""
        self.process = None

pipeline_state = PipelineState()

import sys

def run_pipeline_worker(dataset: str, run_id: str):
    global pipeline_state
    
    db = SessionLocal()
    try:
        # Normalize dataset for matching
        dataset_lower = dataset.lower()
        config_file = "config/samlur_config.yaml" if "samlur" in dataset_lower else "config/config.yaml"
        print(f"DEBUG: Selected config: {config_file}")
        
        # Create DB entry
        db_run = models.PipelineRun(
            id=run_id,
            dataset=dataset,
            status="RUNNING",
            tiles_total=200, # Actual tile count for samlur processing
            tiles_done=0,
            started_at=func.now()
        )
        db.add(db_run)
        db.commit()

        pipeline_state.state = "RUNNING"
        pipeline_state.dataset = dataset
        pipeline_state.run_id = run_id
        pipeline_state.tiles_total = 200
        pipeline_state.tiles_done = 0

        # EXECUTE REAL PIPELINE
        # Use absolute python executable from current environment
        cmd = [
            sys.executable, "src/inference/run_inference.py",
            "--config", config_file,
            "--run-id", run_id
        ]
        print(f"DEBUG: Launching: {' '.join(cmd)}")
        
        env = os.environ.copy()
        # Ensure PYTHONPATH includes both the project root and the hydra-api
        root = os.getcwd()
        env["PYTHONPATH"] = f"{root};{os.path.join(root, 'hydra-api')};{env.get('PYTHONPATH', '')}"

        # Capture output to a log file for this run
        os.makedirs(os.path.join("output", "logs"), exist_ok=True)
        log_path = os.path.join("output", "logs", f"{run_id}.log")
        
        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            pipeline_state.process = process
            print(f"DEBUG: Subprocess PID {process.pid}, logging to {log_path}")

            # Watch progress (polling run_summary.json while process runs)
            while process.poll() is None:
                time.sleep(1)
                # Check for progress updates from the model side
                output_base = "output/samlur" if "samlur" in dataset_lower else "output"
                summary_path = os.path.join(output_base, run_id, "run_summary.json")
                if os.path.exists(summary_path):
                    try:
                        with open(summary_path, 'r') as f:
                            sum_data = json.load(f)
                            pipeline_state.tiles_done = sum_data.get("completed", 0)
                            pipeline_state.tiles_total = sum_data.get("num_tiles", 200)
                            db_run.tiles_done = pipeline_state.tiles_done
                            db_run.tiles_total = pipeline_state.tiles_total
                            db.commit()
                    except:
                        pass

        if process.returncode == 0:
            pipeline_state.state = "COMPLETED"
            db_run.status = "COMPLETED"
            db_run.completed_at = func.now()
        else:
            pipeline_state.state = "FAILED"
            db_run.status = "FAILED"
            print(f"DEBUG: Pipeline process failed with code {process.returncode}")
        
        db.commit()
        
    except Exception as e:
        pipeline_state.state = "FAILED"
        if 'db_run' in locals():
            db_run.status = "FAILED"
            db.commit()
        print(f"CRITICAL: Pipeline crashed in worker thread: {e}")
    finally:
        db.close()

def start_pipeline(dataset: str):
    global pipeline_state
    if pipeline_state.state == "RUNNING":
        return None
    
    run_id = str(uuid.uuid4())
    thread = threading.Thread(target=run_pipeline_worker, args=(dataset, run_id))
    thread.daemon = True
    thread.start()
    return run_id

def get_pipeline_metrics():
    global pipeline_state
    return {
        "state": pipeline_state.state,
        "run_id": pipeline_state.run_id,
        "dataset": pipeline_state.dataset,
        "tiles_total": pipeline_state.tiles_total,
        "tiles_done": pipeline_state.tiles_done
    }
