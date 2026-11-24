"""FastAPI prediction service for MP008 Household Electricity Use Prediction."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .cleaning import clean_inference_data


# -----------------------------
# Structured JSON logging (stdout)
# -----------------------------
class StdoutStreamHandler(logging.StreamHandler):
    """StreamHandler that always resolves the current sys.stdout."""

    @property
    def stream(self) -> Any:
        return sys.stdout

    @stream.setter
    def stream(self, _value: Any) -> None:
        pass


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, timezone.utc).isoformat()
        payload: dict[str, Any] = {
            "timestamp": ts,
            "time": ts,
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
            "message": record.getMessage(),
        }
        extra_fields = getattr(record, "fields", None)
        if isinstance(extra_fields, dict):
            for key, value in extra_fields.items():
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_api_logger() -> logging.Logger:
    """Configure the API logger to emit JSON lines to stdout."""
    api_logger = logging.getLogger("mp008.api")
    api_logger.setLevel(logging.INFO)
    api_logger.handlers.clear()
    api_logger.propagate = False

    handler = StdoutStreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(JsonFormatter())
    api_logger.addHandler(handler)
    return api_logger


logger = configure_api_logger()


def log_event(level: int, event: str, message: str, **fields: Any) -> None:
    """Log a structured JSON event without leaking sensitive request payloads."""
    logger.log(level, message, extra={"event": event, "fields": fields})


# -----------------------------
# Model loading
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

model: Any = None


def load_prediction_model(model_path: Path = MODEL_PATH) -> Any:
    """Load the serialized champion model from disk."""
    global model
    try:
        loaded = joblib.load(model_path)
        model = loaded
        log_event(
            logging.INFO,
            "model_loaded",
            "Electricity prediction model loaded successfully.",
            model_file=model_path.name,
            model_type=type(loaded).__name__,
        )
        return loaded
    except Exception as exc:
        model = None
        log_event(
            logging.ERROR,
            "model_load_failed",
            "Could not load the trained prediction model.",
            model_file=model_path.name,
            error_type=type(exc).__name__,
        )
        return None


load_prediction_model(MODEL_PATH)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Manage FastAPI startup and shutdown lifecycle events."""
    if model is None and MODEL_PATH.exists():
        load_prediction_model(MODEL_PATH)
    log_event(
        logging.INFO,
        "service_startup",
        "FastAPI prediction service started.",
        model_loaded=model is not None,
    )
    try:
        yield
    finally:
        log_event(
            logging.INFO,
            "service_shutdown",
            "FastAPI prediction service shutting down.",
        )


# -----------------------------
# FastAPI application
# -----------------------------
app = FastAPI(
    title="Household Electricity Prediction API",
    description="Predict household electricity consumption in kWh for Lucknow (MP008).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next: Any):
    """Attach request ID and start timer to every incoming HTTP request."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:8]
    request.state.request_id = request_id
    request.state.start_time = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


def _sanitize_for_json(value: Any) -> Any:
    """Replace non-finite floats (NaN/Inf) with strings so JSONResponse never raises ValueError."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Log validation failures in structured JSON and return HTTP 422."""
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:8])
    start_time = getattr(request.state, "start_time", None)
    duration_ms = (
        round((time.perf_counter() - start_time) * 1000.0, 2)
        if start_time is not None
        else None
    )
    error_fields = [
        str(err.get("loc", ("body",))[-1]) for err in exc.errors()
    ]
    log_event(
        logging.WARNING,
        "validation_failed",
        "Request input validation failed.",
        request_id=request_id,
        path=request.url.path,
        status_code=422,
        error_count=len(exc.errors()),
        fields=error_fields,
        duration_ms=duration_ms,
    )
    return JSONResponse(
        status_code=422,
        content={"detail": _sanitize_for_json(jsonable_encoder(exc.errors()))},
    )


# -----------------------------
# Input validation schema
# -----------------------------
class PredictionInput(BaseModel):
    """Validated prediction request payload containing the 8 approved features."""

    model_config = ConfigDict(extra="forbid")

    rooms: int = Field(..., gt=0, description="Number of rooms in the house (> 0)")
    members: int = Field(..., gt=0, description="Number of people living in the house (> 0)")
    ac_hours_per_day: float = Field(
        ...,
        ge=0.0,
        le=24.0,
        allow_inf_nan=False,
        description="Average AC hours per day (0 to 24)",
    )
    avg_temp_c: float = Field(
        ...,
        allow_inf_nan=False,
        description="Average temperature of the month in degrees Celsius",
    )
    house_area_sqft: float = Field(
        ...,
        ge=150.0,
        allow_inf_nan=False,
        description="Built-up house area in square feet (>= 150)",
    )
    solar_kw: float = Field(
        ...,
        ge=0.0,
        le=9.0,
        allow_inf_nan=False,
        description="Rooftop solar capacity in kW (0 to 9)",
    )
    connection: str = Field(..., min_length=1, description="Connection type")
    feeder: str = Field(..., min_length=1, description="Power feeder identifier")

    @field_validator("rooms", "members", mode="before")
    @classmethod
    def validate_strict_integer(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Value must be an integer.")
        return value

    @field_validator(
        "ac_hours_per_day",
        "avg_temp_c",
        "house_area_sqft",
        "solar_kw",
        mode="before",
    )
    @classmethod
    def validate_strict_finite_number(cls, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Value must be a finite number.")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("Value must be finite.")
        return numeric

    @field_validator("connection", "feeder", mode="before")
    @classmethod
    def validate_text_fields(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("Value must be a non-empty string.")
        stripped = value.strip()
        if not stripped:
            raise ValueError("This field cannot be empty.")
        return stripped


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def home() -> dict[str, str]:
    return {
        "message": "Household Electricity Prediction API",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok"}


@app.post("/predict")
def predict(request: Request, data: PredictionInput) -> dict[str, Any]:
    """Predict monthly household electricity consumption (units_kwh)."""
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:8])
    start_time = getattr(request.state, "start_time", time.perf_counter())

    if model is None:
        duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        log_event(
            logging.ERROR,
            "model_unavailable",
            "Prediction rejected because model is unavailable.",
            request_id=request_id,
            status_code=503,
            duration_ms=duration_ms,
        )
        raise HTTPException(
            status_code=503,
            detail="Prediction model is unavailable.",
        )

    try:
        input_df = pd.DataFrame([data.model_dump()])
        cleaned_data = clean_inference_data(input_df)
        raw_prediction = float(model.predict(cleaned_data)[0])
        if not math.isfinite(raw_prediction):
            raise ValueError("Model returned a non-finite prediction.")

        prediction = round(max(0.0, raw_prediction), 4)
        duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        log_event(
            logging.INFO,
            "prediction_succeeded",
            "Electricity prediction completed successfully.",
            request_id=request_id,
            status_code=200,
            duration_ms=duration_ms,
            prediction=prediction,
        )

        return {
            "prediction": prediction,
            "predicted_units_kwh": prediction,
            "unit": "kWh",
        }

    except Exception as exc:
        duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        log_event(
            logging.ERROR,
            "prediction_failed",
            "Prediction inference failed.",
            request_id=request_id,
            status_code=500,
            duration_ms=duration_ms,
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="An error occurred while generating the prediction.",
        )
