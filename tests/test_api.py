"""Automated FastAPI TestClient tests for MP008 prediction service."""
from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient
import numpy as np
import pytest

from src.mp008 import api


VALID_PAYLOAD: dict[str, Any] = {
    "rooms": 4,
    "members": 5,
    "ac_hours_per_day": 6.5,
    "avg_temp_c": 31.2,
    "house_area_sqft": 1800.0,
    "solar_kw": 3.5,
    "connection": "domestic_plus",
    "feeder": "Feeder-2",
}


class _DummyPositiveModel:
    """Deterministic mock regressor returning a positive value."""

    def predict(self, X: Any) -> np.ndarray:
        return np.array([345.6789] * len(X))


class _DummyNegativeModel:
    """Deterministic mock regressor returning a negative value."""

    def predict(self, X: Any) -> np.ndarray:
        return np.array([-42.5] * len(X))


class _FailingModel:
    """Deterministic mock regressor raising an internal exception."""

    def predict(self, _X: Any) -> np.ndarray:
        raise RuntimeError("sensitive_internal_stack_token")


client = TestClient(api.app)


def test_health_returns_200_and_exact_payload() -> None:
    """Test 1 & 2: GET /health returns HTTP 200 and exact {'status': 'ok'} JSON."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_prediction_with_actual_model_returns_200_and_nonnegative() -> None:
    """Test 3, 4 & 11: Valid prediction using the real champion model returns numeric non-negative output without needing target/leakage columns."""
    assert api.model is not None, "Expected real champion model loaded from models/model.joblib"
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["prediction"], (int, float))
    assert isinstance(body["predicted_units_kwh"], (int, float))
    assert body["prediction"] == body["predicted_units_kwh"]
    assert body["prediction"] >= 0.0
    assert body["unit"] == "kWh"


def test_prediction_clips_negative_model_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify API prevents negative electricity predictions by clipping at 0.0."""
    monkeypatch.setattr(api, "model", _DummyNegativeModel())
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 0.0
    assert body["predicted_units_kwh"] == 0.0


@pytest.mark.parametrize("invalid_solar", [9.01, 10, 10.0, 15.5, -0.1, -5])
def test_solar_kw_out_of_bounds_returns_422(
    monkeypatch: pytest.MonkeyPatch, invalid_solar: float
) -> None:
    """Test 5: solar_kw > 9 (e.g. 10) or < 0 returns HTTP 422."""
    monkeypatch.setattr(api, "model", _DummyPositiveModel())
    payload = {**VALID_PAYLOAD, "solar_kw": invalid_solar}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_solar_kw_at_boundary_9_returns_200(monkeypatch: pytest.MonkeyPatch) -> None:
    """Boundary check: solar_kw = 9.0 is valid and returns HTTP 200."""
    monkeypatch.setattr(api, "model", _DummyPositiveModel())
    payload = {**VALID_PAYLOAD, "solar_kw": 9.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


@pytest.mark.parametrize("invalid_area", [149, 149.99, 0, -100])
def test_house_area_sqft_below_150_returns_422(
    monkeypatch: pytest.MonkeyPatch, invalid_area: float
) -> None:
    """Test 6: house_area_sqft < 150 (e.g. 149) returns HTTP 422."""
    monkeypatch.setattr(api, "model", _DummyPositiveModel())
    payload = {**VALID_PAYLOAD, "house_area_sqft": invalid_area}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_house_area_sqft_at_boundary_150_returns_200(monkeypatch: pytest.MonkeyPatch) -> None:
    """Boundary check: house_area_sqft = 150.0 is valid and returns HTTP 200."""
    monkeypatch.setattr(api, "model", _DummyPositiveModel())
    payload = {**VALID_PAYLOAD, "house_area_sqft": 150.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


@pytest.mark.parametrize(
    "field_name,invalid_value",
    [
        ("rooms", 0),
        ("rooms", -2),
        ("members", 0),
        ("members", -1),
        ("ac_hours_per_day", -0.5),
        ("ac_hours_per_day", 24.1),
    ],
)
def test_domain_numeric_bounds_return_422(
    monkeypatch: pytest.MonkeyPatch, field_name: str, invalid_value: Any
) -> None:
    """Verify rooms > 0, members > 0, and 0 <= ac_hours_per_day <= 24 constraints return 422."""
    monkeypatch.setattr(api, "model", _DummyPositiveModel())
    payload = {**VALID_PAYLOAD, field_name: invalid_value}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize("missing_field", list(VALID_PAYLOAD.keys()))
def test_missing_required_features_return_422(
    monkeypatch: pytest.MonkeyPatch, missing_field: str
) -> None:
    """Test 7: Omitting any of the 8 required features returns HTTP 422."""
    monkeypatch.setattr(api, "model", _DummyPositiveModel())
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != missing_field}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "field_name,bad_value",
    [
        ("rooms", "four"),
        ("rooms", "4"),
        ("rooms", 3.5),
        ("rooms", True),
        ("members", "five"),
        ("members", False),
        ("ac_hours_per_day", "six"),
        ("ac_hours_per_day", True),
        ("avg_temp_c", "hot"),
        ("avg_temp_c", None),
        ("house_area_sqft", "1800"),
        ("solar_kw", "3.5"),
        ("solar_kw", False),
    ],
)
def test_invalid_numerical_types_not_coerced_and_return_422(
    monkeypatch: pytest.MonkeyPatch, field_name: str, bad_value: Any
) -> None:
    """Test 8: Invalid numerical types (strings, booleans, floats for int, None) return HTTP 422."""
    monkeypatch.setattr(api, "model", _DummyPositiveModel())
    payload = {**VALID_PAYLOAD, field_name: bad_value}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize("non_finite", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_numbers_return_422(
    monkeypatch: pytest.MonkeyPatch, non_finite: str
) -> None:
    """Verify non-finite JSON numeric literals (NaN/Infinity) are rejected with HTTP 422."""
    monkeypatch.setattr(api, "model", _DummyPositiveModel())
    raw_body = (
        '{"rooms": 4, "members": 5, "ac_hours_per_day": 6.5, '
        f'"avg_temp_c": {non_finite}, "house_area_sqft": 1800.0, '
        '"solar_kw": 3.5, "connection": "domestic_plus", "feeder": "Feeder-2"}'
    )
    response = client.post(
        "/predict",
        content=raw_body,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


@pytest.mark.parametrize("field_name", ["connection", "feeder"])
@pytest.mark.parametrize("bad_text", ["", "   ", "\t\n", 123, None])
def test_empty_or_invalid_text_fields_return_422(
    monkeypatch: pytest.MonkeyPatch, field_name: str, bad_text: Any
) -> None:
    """Test 9: Empty/whitespace or non-string connection/feeder returns HTTP 422."""
    monkeypatch.setattr(api, "model", _DummyPositiveModel())
    payload = {**VALID_PAYLOAD, field_name: bad_text}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "extra_fields",
    [
        {"unexpected_field": "abc"},
        {"billed_units_after_reading": 400.0},
        {"units_kwh": 400.0},
        {"record_id": "R99999", "date": "2026-09-01"},
    ],
)
def test_unexpected_extra_and_leakage_fields_rejected_with_422(
    monkeypatch: pytest.MonkeyPatch, extra_fields: dict[str, Any]
) -> None:
    """Test 10 & 11: Unexpected extra fields, leakage column, and target column are rejected."""
    monkeypatch.setattr(api, "model", _DummyPositiveModel())
    payload = {**VALID_PAYLOAD, **extra_fields}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_model_unavailable_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 12: Unavailable model returns HTTP 503 safely."""
    monkeypatch.setattr(api, "model", None)
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 503
    assert response.json()["detail"] == "Prediction model is unavailable."


def test_prediction_internal_error_returns_500_without_leaking_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify runtime inference errors return HTTP 500 without exposing internal exception strings."""
    monkeypatch.setattr(api, "model", _FailingModel())
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "An error occurred while generating the prediction."
    assert "sensitive_internal_stack_token" not in response.text


def test_structured_json_logging_to_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify startup, shutdown, valid prediction, validation failure, and error emit valid JSON lines to stdout."""
    monkeypatch.setattr(api, "model", _DummyPositiveModel())
    capsys.readouterr()  # Clear prior stdout

    with TestClient(api.app) as lifecycle_client:
        res_ok = lifecycle_client.post(
            "/predict",
            json=VALID_PAYLOAD,
            headers={"X-Request-ID": "req-ok-1"},
        )
        assert res_ok.status_code == 200
        assert res_ok.headers["X-Request-ID"] == "req-ok-1"

        res_bad = lifecycle_client.post(
            "/predict",
            json={**VALID_PAYLOAD, "solar_kw": 10},
            headers={"X-Request-ID": "req-bad-2"},
        )
        assert res_bad.status_code == 422
        assert res_bad.headers["X-Request-ID"] == "req-bad-2"

    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert len(lines) >= 4

    parsed_events = [json.loads(line) for line in lines]
    for record in parsed_events:
        assert "timestamp" in record
        assert "level" in record
        assert "event" in record

    event_names = [record["event"] for record in parsed_events]
    assert "service_startup" in event_names
    assert "prediction_succeeded" in event_names
    assert "validation_failed" in event_names
    assert "service_shutdown" in event_names

    pred_log = next(r for r in parsed_events if r["event"] == "prediction_succeeded")
    assert pred_log["request_id"] == "req-ok-1"
    assert isinstance(pred_log["duration_ms"], (int, float))
    assert "house_area_sqft" not in pred_log  # Do not log full payload

    val_log = next(r for r in parsed_events if r["event"] == "validation_failed")
    assert val_log["request_id"] == "req-bad-2"
    assert val_log["status_code"] == 422
