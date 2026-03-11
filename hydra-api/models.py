from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.sql import func
from geoalchemy2 import Geometry
from database import Base

class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    
    id = Column(String, primary_key=True, index=True) # UUID
    dataset = Column(String, index=True)
    status = Column(String)
    tiles_total = Column(Integer)
    tiles_done = Column(Integer)
    created_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime, nullable=True)

class DetectedPolygon(Base):
    __tablename__ = "detected_polygons"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    run_id = Column(String, ForeignKey("pipeline_runs.id"), index=True)
    tile_id = Column(String, index=True)
    class_id = Column(Integer, index=True)
    confidence = Column(Float)
    z_mean = Column(Float, nullable=True)  # From DepthPro
    z_max = Column(Float, nullable=True)
    area_sqm = Column(Float)
    geom = Column(Geometry(geometry_type='POLYGON', srid=4326), index=True)
    created_at = Column(DateTime, default=func.now())

class Feedback(Base):
    __tablename__ = "feedback_queue"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    tile_id = Column(String, index=True)
    original_class = Column(Integer)
    corrected_class = Column(Integer)
    notes = Column(String, nullable=True)
    processed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
