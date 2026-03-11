from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from routes import pipeline, datasets, layers, feedback
import models
from database import engine
import sentry_sdk

sentry_sdk.init(
    dsn="https://mock@sentry.io/12345", # Ensure tracking is mocked or configured
    traces_sample_rate=1.0,
)

# Create PostGIS tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Hydra Map API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to http://localhost:3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static data (tiles) to the frontend
if os.path.exists("data"):
    app.mount("/data", StaticFiles(directory="data"), name="data")

app.include_router(pipeline.router)
app.include_router(datasets.router)
app.include_router(layers.router)
app.include_router(feedback.router)
