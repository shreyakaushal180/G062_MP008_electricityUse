"""FastAPI prediction service for MP008 (Household Electricity Use, Lucknow)."""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.logging_config import configure_logging
from src.mp008.cleaning import clean_inference_data

logger = configure_logging()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

model = None
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Electricity prediction model loaded.", extra={"event": "model_loaded"})
except Exception:
    logger.exception("Could not load the trained model.", extra={"event": "model_load_failed"})

app = FastAPI(
    title="Household Electricity Prediction API",
    description="Predict monthly household electricity consumption (kWh) for Lucknow.",
    version="1.0.0",
)


class PredictionInput(BaseModel):
    """Request body: exactly the eight fields the client allows."""

    model_config = {"extra": "forbid"}

    rooms: int = Field(gt=0)
    members: int = Field(gt=0)
    ac_hours_per_day: float = Field(ge=0, le=24, allow_inf_nan=False)
    avg_temp_c: float = Field(allow_inf_nan=False)
    # Client rule 2: below 150 sq ft is always a typing mistake.
    house_area_sqft: float = Field(ge=150, allow_inf_nan=False)
    # Client rule 1: above 9 kW is impossible.
    solar_kw: float = Field(ge=0, le=9, allow_inf_nan=False)
    connection: str
    feeder: str

    @field_validator("connection", "feeder")
    @classmethod
    def validate_text_fields(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("This field cannot be empty.")
        return value


@app.get("/")
def home():
    return {"message": "Household Electricity Prediction API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Prediction model is unavailable.")

    started = time.perf_counter()
    try:
        features = clean_inference_data(pd.DataFrame([data.model_dump()]))
        prediction = max(0.0, float(model.predict(features)[0]))
    except Exception:
        logger.exception("Prediction failed.", extra={"event": "prediction_failed", "status_code": 500})
        raise HTTPException(status_code=500, detail="An error occurred while generating the prediction.")

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    # One JSON line per prediction. No personal details: only feeder, result and timing.
    logger.info(
        "Prediction served.",
        extra={
            "event": "prediction",
            "feeder": data.feeder,
            "predicted_units_kwh": round(prediction, 2),
            "latency_ms": latency_ms,
            "status_code": 200,
        },
    )
    return {"predicted_units_kwh": round(prediction, 2), "unit": "kWh"}
