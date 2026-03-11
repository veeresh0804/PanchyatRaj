import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services.pipeline_runner import start_pipeline, get_pipeline_metrics

router = APIRouter()

@router.post("/run")
def run_pipeline(dataset: str, depth_enabled: bool = True):
    run_id = start_pipeline(dataset)
    if run_id:
        return {"status": "started", "run_id": run_id, "detail": "Pipeline dispatched to worker pool."}
    else:
        return {"status": "running", "detail": "A pipeline is already active."}

@router.get("/status")
def get_status():
    return get_pipeline_metrics()

@router.websocket("/ws/status")
async def status_socket(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            metrics = get_pipeline_metrics()
            await ws.send_json(metrics)
            if metrics["state"] in ["COMPLETED", "FAILED"]:
                # Keep emitting final state or close depending on UI logic
                await asyncio.sleep(2)
            else:
                await asyncio.sleep(1)
    except WebSocketDisconnect:
        print("WebSocket client disconnected")

from services.exporter import ExportService

@router.post("/export")
def export_data(run_id: str, format: str = "GeoPackage"):
    return ExportService.export_data(run_id, format)
