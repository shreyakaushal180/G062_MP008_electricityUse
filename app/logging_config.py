"""JSON logging for the prediction service (one JSON object per line, to stdout)."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

# Extra fields that may be attached to a log record via `extra={...}`.
# Only non-personal values (no customer details) are ever logged.
EXTRA_FIELDS = ("event", "feeder", "predicted_units_kwh", "latency_ms", "status_code")


class JsonFormatter(logging.Formatter):
    """Format a log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Send JSON logs to stdout only (never to a file). Safe to call repeatedly."""
    logger = logging.getLogger("mp008")
    logger.setLevel(level)
    logger.propagate = False
    if not any(getattr(h, "_mp008_json", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        handler._mp008_json = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logger
