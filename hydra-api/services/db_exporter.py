import logging
from sqlalchemy.orm import Session
from database import SessionLocal
import models
from shapely.geometry import shape, mapping
from shapely.ops import transform
import pyproj

logger = logging.getLogger(__name__)

class PostGISExporter:
    """Exports detection results directly to PostGIS for live dashboard view."""
    
    @staticmethod
    def export_tile_results(run_id: str, tile_id: str, polygons: list, crs: str = "EPSG:32644"):
        """Inserts real detected polygons into the PostGIS database.
        
        Args:
            run_id: Current pipeline run uuid.
            tile_id: Source tile identifier.
            polygons: List of polygon dicts from Orchestrator.
            crs: Input Coordinate Reference System (usually UTM).
        """
        db = SessionLocal()
        try:
            # Prepare CRS transformer for Lat/Lng (EPSG:4326) and 3857 for web
            # Dashboard expects 4326 for Leaflet or 3857 for Deck.gl
            # We'll store in 4326 for generalized mapping.
            transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

            for poly_data in polygons:
                # poly_data['geometry'] should be a GeoJSON-like dict or a shapely object
                geom_raw = poly_data.get('geometry')
                if not geom_raw: continue
                
                geom = shape(geom_raw)
                
                # Project to 4326
                geom_4326 = transform(transformer.transform, geom)
                wkt_4326 = geom_4326.wkt
                
                db_poly = models.DetectedPolygon(
                    run_id=run_id,
                    tile_id=tile_id,
                    class_id=poly_data.get('class_id', 0),
                    confidence=poly_data.get('confidence', 0.0),
                    z_mean=poly_data.get('z_mean', 0.0),
                    area_sqm=poly_data.get('area_sqm', 0.0),
                    geom=f'SRID=4326;{wkt_4326}'
                )
                db.add(db_poly)
            
            db.commit()
            logger.debug(f"Exported {len(polygons)} polygons for tile {tile_id} to PostGIS.")
        except Exception as e:
            logger.error(f"PostGIS export failed for tile {tile_id}: {e}")
            db.rollback()
        finally:
            db.close()
