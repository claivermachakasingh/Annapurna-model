"""Annapurna ML Serving API (FastAPI + Uvicorn).

Production REST API endpoints for:
- POST /predict-demand (Demand Forecasting & Newsvendor Production Planning)
- POST /quality/check (Multimodal Food Safety Fusion & Conformal Shelf-Life Estimation)
- GET /health
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

# Setup environment paths so ml_pipeline modules and model registries resolve correctly
ROOT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = ROOT_DIR / "ml_pipeline"
REGISTRY_DIR = PIPELINE_DIR / "registry"

sys.path.insert(0, str(PIPELINE_DIR))
os.environ["ML_PIPELINE_ROOT"] = str(ROOT_DIR)
os.environ["ML_REGISTRY_DIR"] = str(REGISTRY_DIR)

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from ml_quality_inference import assess_safety
from ml_demand_model import predict_demand
from ml_utils import get_paths

app = FastAPI(
    title="Annapurna ML Serving API",
    version="1.0.0",
    description="Production REST API for institutional kitchen demand prediction and food safety quality fusion.",
)

# Cache demand model bundle
_demand_bundle = None


def get_demand_bundle():
    global _demand_bundle
    if _demand_bundle is None:
        paths = get_paths()
        model_path = paths.latest_model_dir("demand") / "model.joblib"
        _demand_bundle = joblib.load(model_path)
    return _demand_bundle


class DemandPayload(BaseModel):
    attendance: float = Field(..., ge=0, description="Expected patrons or attendance")
    meal_type: str = Field(..., description="One of: BREAKFAST, LUNCH, SNACK, DINNER")
    day_of_week: int = Field(..., ge=0, le=6, description="0=Monday .. 6=Sunday (ISO)")
    service_level: Optional[float] = Field(0.85, ge=0.5, le=0.99, description="Target service level floor")


class SafetyPayload(BaseModel):
    category: str = Field(
        ...,
        description="One of: rice_dal, cooked_veg, non_veg_curry, dairy_based, bread_roti, fried_snack, raw_salad, sweets",
    )
    hours_since_prepared: float = Field(..., ge=0, description="Hours elapsed since batch was cooked")
    temps: List[float] = Field(..., description="Temperature readings trace in degrees C")
    step_min: float = Field(30.0, gt=0, description="Sampling interval between readings in minutes")
    humidity_mean: float = Field(55.0, ge=0, le=100, description="Mean relative humidity percentage")
    cv_unsafe_prob: float = Field(0.05, ge=0.0, le=1.0, description="Spoilage probability from vision screening model")
    hours_to_expiry: Optional[float] = Field(None, description="Label expiry in hours (if labelled)")


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Annapurna ML Serving Engine",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict_demand": "POST /predict-demand",
            "check_quality": "POST /quality/check",
        },
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict-demand")
def api_predict_demand(payload: DemandPayload):
    try:
        bundle = get_demand_bundle()
        return predict_demand(bundle, payload.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/quality/check")
def api_check_quality(payload: SafetyPayload):
    try:
        return assess_safety(
            category=payload.category,
            hours_since_prepared=payload.hours_since_prepared,
            temps=payload.temps,
            step_min=payload.step_min,
            humidity_mean=payload.humidity_mean,
            cv_unsafe_prob=payload.cv_unsafe_prob,
            hours_to_expiry=payload.hours_to_expiry,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
