from fastapi import APIRouter, Depends
from pydantic import BaseModel
import logging
from database import get_db
from sqlalchemy.orm import Session
import models

router = APIRouter(prefix="/feedback", tags=["feedback"])
logger = logging.getLogger(__name__)

class PolygonFeedback(BaseModel):
    tile_id: str
    original_class: int
    corrected_class: int
    notes: str = ""

@router.post("/")
def submit_feedback(feedback: PolygonFeedback, db: Session = Depends(get_db)):
    logger.info(f"Received HITL correction for {feedback.tile_id}: {feedback.original_class} -> {feedback.corrected_class}")
    
    # Store feedback in PostGIS database
    db_feedback = models.Feedback(
        tile_id=feedback.tile_id,
        original_class=feedback.original_class,
        corrected_class=feedback.corrected_class,
        notes=feedback.notes
    )
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    
    return {"status": "success", "message": f"Feedback registered (ID: {db_feedback.id}) for Continuous Learning queue."}
