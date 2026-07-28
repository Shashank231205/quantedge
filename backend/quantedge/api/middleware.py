"""Request latency capture.

The p95 shown on the System Health screen is computed from these records.
Writing one database row per request would make the API slower than the thing
it measures, so latencies accumulate in an in-memory ring and flush
periodically.
"""

from __future__ import annotations

import time
from collections import deque
from datetime import UTC, datetime

import numpy as np
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from quantedge.db.models import ApiRequestLog
from quantedge.db.session import session_scope
from quantedge.logging_config import get_logger

log = get_logger(__name__)

_LATENCIES: deque[dict] = deque(maxlen=5_000)
_PENDING: list[dict] = []
_FLUSH_EVERY = 50


class LatencyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        record = {
            "endpoint": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "latency_ms": elapsed_ms,
            "created_at": datetime.now(UTC).replace(tzinfo=None),
        }
        _LATENCIES.append(record)
        _PENDING.append(record)

        if len(_PENDING) >= _FLUSH_EVERY:
            _flush()

        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response


def _flush() -> None:
    if not _PENDING:
        return
    batch, _PENDING[:] = list(_PENDING), []
    try:
        with session_scope() as s:
            s.bulk_insert_mappings(ApiRequestLog, batch)
    except Exception as exc:  # telemetry must never break the API
        log.warning("latency.flush_failed error=%s", exc)


def latency_stats(window: int = 1_000) -> dict:
    """Live latency percentiles from the in-memory ring."""
    recent = list(_LATENCIES)[-window:]
    if not recent:
        return {
            "n_requests": 0, "p50_ms": None, "p95_ms": None,
            "p99_ms": None, "mean_ms": None, "error_rate": 0.0,
        }

    values = np.array([r["latency_ms"] for r in recent])
    errors = sum(1 for r in recent if r["status_code"] >= 500)

    return {
        "n_requests": len(recent),
        "p50_ms": round(float(np.percentile(values, 50)), 2),
        "p95_ms": round(float(np.percentile(values, 95)), 2),
        "p99_ms": round(float(np.percentile(values, 99)), 2),
        "mean_ms": round(float(values.mean()), 2),
        "max_ms": round(float(values.max()), 2),
        "error_rate": round(errors / len(recent), 4),
    }


def flush_on_shutdown() -> None:
    _flush()
