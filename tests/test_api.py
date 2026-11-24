"""API tests (use FastAPI's TestClient). No CSV files needed: the model is stubbed."""
import io
import json
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.logging_config import JsonFormatter

VALID = {
    "rooms": 3,
    "members": 4,
    "ac_hours_per_day": 6.5,
    "avg_temp_c": 30.0,
    "house_area_sqft": 1500,
    "solar_kw": 3.0,
    "connection": "domestic",
    "feeder": "Feeder-1",
}


class StubModel:
    """Stands in for the real champion so tests do not depend on model files."""

    def predict(self, frame):
        assert list(frame.columns) == list(VALID.keys())
        return np.array([123.456])


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(main, "model", StubModel())
    return TestClient(main.app)


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_prediction(client):
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    assert response.json() == {"predicted_units_kwh": 123.46, "unit": "kWh"}


def test_solar_above_nine_is_rejected(client):
    assert client.post("/predict", json={**VALID, "solar_kw": 9.1}).status_code == 422
    assert client.post("/predict", json={**VALID, "solar_kw": 9}).status_code == 200


def test_tiny_house_area_is_rejected(client):
    assert client.post("/predict", json={**VALID, "house_area_sqft": 149}).status_code == 422
    assert client.post("/predict", json={**VALID, "house_area_sqft": 150}).status_code == 200


def test_missing_and_extra_fields_are_rejected(client):
    missing = {k: v for k, v in VALID.items() if k != "feeder"}
    assert client.post("/predict", json=missing).status_code == 422
    assert client.post("/predict", json={**VALID, "units_kwh": 5}).status_code == 422


def test_model_unavailable_gives_503(monkeypatch):
    monkeypatch.setattr(main, "model", None)
    assert TestClient(main.app).post("/predict", json=VALID).status_code == 503


def test_json_formatter_outputs_one_json_line():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("mp008.test_json")
    logger.propagate = False
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info("Prediction served.", extra={"event": "prediction", "feeder": "Feeder-2"})
    lines = stream.getvalue().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["message"] == "Prediction served."
    assert record["event"] == "prediction" and record["feeder"] == "Feeder-2"
    assert record["level"] == "INFO"
