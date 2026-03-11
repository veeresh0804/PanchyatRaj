from fastapi import APIRouter, File, UploadFile
import json
import os
import shutil

router = APIRouter(prefix="/datasets", tags=["datasets"])

@router.get("/")
def list_datasets():
    datasets_dir = "datasets"
    if not os.path.exists(datasets_dir):
        return []
        
    datasets = []
    for d in os.listdir(datasets_dir):
        meta_path = os.path.join(datasets_dir, d, "dataset.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                datasets.append(json.load(f))
        else:
            datasets.append({"name": d})
    return datasets

@router.post("/upload")
def upload_dataset(file: UploadFile = File(...), dataset_name: str = "New_Village"):
    datasets_dir = "datasets"
    dataset_path = os.path.join(datasets_dir, dataset_name)
    os.makedirs(dataset_path, exist_ok=True)
    
    with open(os.path.join(dataset_path, file.filename), "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    meta = {
        "name": dataset_name,
        "tiles": 8089, # Mock metadata processing
        "crs": "EPSG:32643"
    }
    with open(os.path.join(dataset_path, "dataset.json"), "w") as f:
        json.dump(meta, f)
        
    return {"status": "success", "dataset": meta}
