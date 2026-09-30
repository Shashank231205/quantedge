"""Request telemetry: latency capture, Prometheus metrics and request IDs.

The p95 shown on the System Health screen is computed from these records.
Writing one database row per request would make the API slower than the thing
it measures, so latencies accumulate in an in-memory ring and flush
periodically.

The same measurement feeds Prometheus, labelled by route template. One timer
serves both, so the screen and the Grafana dashboard can never disagree about
what a request cost.
"""

from __future__ import annotations

import re
import time
from collections import deque
from datetime import UTC, datetime
from functools import lru_cache

import numpy as np
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.routing import compile_path

from quantedge.db.models import ApiRequestLog
from quantedge.db.session import session_scope
from quantedge.logging_config import get_logger
from quantedge.observability.context import (
    REQUEST_ID_HEADER,
    request_id_var,
    resolve_request_id,
)
from quantedge.observability.metrics import HTTP_IN_FLIGHT, observe_request

log = get_logger(__name__)

_LATENCIES: deque[dict] = deque(maxlen=5_000)
_PENDING: list[dict] = []
_FLUSH_EVERY = 50

#: Probes and scrapes arrive every few seconds from every load balancer and
#: Prometheus. Counted in metrics, but kept out of the latency log, where they
#: would outnumber real traffic and drag the reported p95 toward zero.
_UNLOGGED_PATHS = frozenset({"/metrics", "/health", "/health/live", "/health/ready"})


@lru_cache(maxsize=4)
def _route_patterns(app) -> tuple[tuple[re.Pattern, str], ...]:
    """Compiled (regex, template) pairs for every route the app serves.

    Built from the OpenAPI schema plus the app's own top-level routes, both
    public interfaces, rather than from the router's internal structures.
    Routers record their match on a copy of the ASGI scope that middleware
    never sees, so the match has to be repeated here.
    """
    templates = set(app.openapi().get("paths", {}))
    templates |= {r.path for r in app.routes if isinstance(getattr(r, "path", None), str)}
    # Fewest parameters first, so /runs/latest wins over /runs/{run_id}.
    ordered = sorted(templates, key=lambda t: (t.count("{"), -len(t)))
    return tuple((compile_path(t)[0], t) for t in ordered)


def _route_template(request: Request) -> str:
    """The matched route's path template, or a fixed label for no match."""
    path = request.url.path
    for pattern, template in _route_patterns(request.app):
        if pattern.match(path):
            return template
    return "unmatched"


class LatencyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = resolve_request_id(request.headers.get(REQUEST_ID_HEADER))
        token = request_id_var.set(request_id)
        HTTP_IN_FLIGHT.inc()
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            HTTP_IN_FLIGHT.dec()
            request_id_var.reset(token)
        elapsed = time.perf_counter() - start
        elapsed_ms = elapsed * 1000.0

        observe_request(_route_template(request), request.method, response.status_code, elapsed)

        if request.url.path not in _UNLOGGED_PATHS:
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
        response.headers[REQUEST_ID_HEADER] = request_id
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
