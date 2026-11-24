
import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .cleaning import clean_inference_data


# -----------------------------
# Logging configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)


# -----------------------------
# Load trained model
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

model = None

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Electricity prediction model loaded successfully.")
except Exception:
    logger.exception("Could not load the trained model.")


# -----------------------------
# FastAPI application
# -----------------------------
app = FastAPI(
    title="Household Electricity Prediction API",
    description="Predict household electricity consumption in kWh.",
    version="1.0.0"
)


# -----------------------------
# Input validation
# -----------------------------
class PredictionInput(BaseModel):
    rooms: int = Field(gt=0)
    members: int = Field(gt=0)
    ac_hours_per_day: float = Field(ge=0, le=24, allow_inf_nan=False)
    avg_temp_c: float = Field(allow_inf_nan=False)
    house_area_sqft: float = Field(gt=0, allow_inf_nan=False)
    solar_kw: float = Field(ge=0, allow_inf_nan=False)
    connection: str
    feeder: str

    @field_validator("connection", "feeder")
    @classmethod
    def validate_text_fields(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("This field cannot be empty.")
        return value

    model_config = {"extra": "forbid"}


# -----------------------------
# Home endpoint
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "Household Electricity Prediction API",
        "docs": "/docs"
    }


# -----------------------------
# Health endpoint
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "unavailable",
        "model_loaded": model is not None
    }


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(data: PredictionInput):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is unavailable."
        )

    try:
        input_df = pd.DataFrame([data.model_dump()])

        # Apply the project's inference cleaning
        cleaned_data = clean_inference_data(input_df)

        # Generate prediction
        prediction = float(model.predict(cleaned_data)[0])

        # Electricity consumption cannot be negative
        prediction = max(0.0, prediction)

        logger.info("Electricity prediction completed successfully.")

        return {
            "predicted_units_kwh": round(prediction, 2),
            "unit": "kWh"
        }

    except Exception:
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while generating the prediction."
        )
