from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from database import get_db
import models
import json

router = APIRouter(prefix="/layers", tags=["layers"])

@router.get("/")
def get_layers():
    return [
        {"id": "swin", "name": "Swin Segmentation", "enabled": True},
        {"id": "yolov8", "name": "YOLO Detections", "enabled": True},
        {"id": "fusion", "name": "Fusion Polygons", "enabled": True},
        {"id": "depth", "name": "Depth Anomalies", "enabled": False}
    ]

@router.get("/geojson")
def get_geojson(run_id: str = None, db: Session = Depends(get_db)):
    """
    Fetches detected polygons from PostGIS and returns them as GeoJSON.
    This demonstrates the Spatial Indexing Power of Hydra v2.
    """
    query = db.query(
        models.DetectedPolygon.tile_id,
        models.DetectedPolygon.class_id,
        models.DetectedPolygon.confidence,
        models.DetectedPolygon.z_mean,
        func.ST_AsGeoJSON(models.DetectedPolygon.geom).label('geometry')
    )
    
    if run_id:
        query = query.filter(models.DetectedPolygon.run_id == run_id)
        
    polygons = query.limit(1000).all()
    
    features = []
    for p in polygons:
        features.append({
            "type": "Feature",
            "geometry": json.loads(p.geometry),
            "properties": {
                "tile_id": p.tile_id,
                "class_id": p.class_id,
                "confidence": p.confidence,
                "height": p.z_mean
            }
        })
        
    return {
        "type": "FeatureCollection",
        "features": features
    }
