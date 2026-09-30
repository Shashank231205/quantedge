"""Prometheus metrics.

The System Health screen answers "how is this instance doing"; these answer
"how is the service doing" once there are several instances behind a load
balancer and the screen can only ever see the one that served it. Prometheus
scrapes every replica and Grafana aggregates, which is also what the alert
rules in ``ops/prometheus/alerts.yml`` evaluate against.

Two kinds of metric:

* **Instrumented** — counters and histograms updated in-process as requests
  and jobs run.
* **Collected at scrape time** — pipeline freshness read from ``job_runs``.
  Those rows are written by whichever machine ran the job, so reading them
  from the database is the only way every replica reports the same truth.

Label values are bounded on purpose: requests are labelled with the route
*template* (``/v1/microstructure/study/{symbol}``), never the raw path, so a
crawler probing random URLs cannot create unbounded series.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

REGISTRY = CollectorRegistry(auto_describe=True)

#: Buckets span the p95 target (200ms) densely and still resolve cold starts.
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)

HTTP_REQUESTS = Counter(
    "quantedge_http_requests_total",
    "HTTP requests handled, by route template, method and status class.",
    ["route", "method", "status"],
    registry=REGISTRY,
)
HTTP_LATENCY = Histogram(
    "quantedge_http_request_duration_seconds",
    "Request latency by route template.",
    ["route", "method"],
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)
HTTP_IN_FLIGHT = Gauge(
    "quantedge_http_requests_in_flight",
    "Requests currently being handled by this instance.",
    registry=REGISTRY,
)
JOB_RUNS = Counter(
    "quantedge_job_runs_total",
    "Pipeline job executions by outcome.",
    ["job", "status"],
    registry=REGISTRY,
)
JOB_DURATION = Histogram(
    "quantedge_job_duration_seconds",
    "Pipeline job wall time.",
    ["job"],
    buckets=(1, 5, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600),
    registry=REGISTRY,
)
SCHEDULER_LEADER = Gauge(
    "quantedge_scheduler_is_leader",
    "1 while this scheduler instance holds the leader lock, else 0.",
    registry=REGISTRY,
)
LEADER_TRANSITIONS = Counter(
    "quantedge_scheduler_leader_transitions_total",
    "Leadership gained or lost by this instance.",
    ["transition"],
    registry=REGISTRY,
)
BUILD_INFO = Gauge(
    "quantedge_build_info",
    "Constant 1, labelled with the running version.",
    ["version"],
    registry=REGISTRY,
)
BUILD_INFO.labels(version="0.1.0").set(1)


def status_class(code: int) -> str:
    return f"{code // 100}xx"


def observe_request(route: str, method: str, status_code: int, seconds: float) -> None:
    HTTP_REQUESTS.labels(route=route, method=method, status=status_class(status_code)).inc()
    HTTP_LATENCY.labels(route=route, method=method).observe(seconds)


def observe_job(job: str, status: str, seconds: float) -> None:
    JOB_RUNS.labels(job=job, status=status).inc()
    JOB_DURATION.labels(job=job).observe(seconds)


class PipelineCollector(Collector):
    """Freshness gauges read from the database when Prometheus scrapes.

    Cached briefly so that a scrape every 15s from each of several Prometheus
    replicas costs one query, not one per scrape. A database that cannot be
    reached is reported as ``quantedge_metrics_db_up 0`` rather than raised —
    a scrape that errors produces no data at all, which would hide exactly the
    outage it should be reporting.
    """

    def __init__(self, ttl_seconds: float = 30.0) -> None:
        self.ttl = ttl_seconds
        self._cached_at = 0.0
        self._cached: tuple[bool, dict[str, float], float | None] = (False, {}, None)

    def _query(self) -> tuple[bool, dict[str, float], float | None]:
        from sqlalchemy import func, select

        from quantedge.db.models import JobRun, OhlcvClean
        from quantedge.db.session import session_scope

        try:
            with session_scope() as s:
                rows = s.execute(
                    select(JobRun.job_name, func.max(JobRun.finished_at))
                    .where(JobRun.status == "SUCCESS")
                    .group_by(JobRun.job_name)
                ).all()
                latest = s.scalar(select(func.max(OhlcvClean.date)))
        except Exception:
            return False, {}, None

        last_success = {
            name: finished.replace(tzinfo=UTC).timestamp() for name, finished in rows if finished
        }
        latest_ts = (
            datetime(latest.year, latest.month, latest.day, tzinfo=UTC).timestamp()
            if latest
            else None
        )
        return True, last_success, latest_ts

    def collect(self):
        now = time.monotonic()
        if now - self._cached_at > self.ttl:
            self._cached = self._query()
            self._cached_at = now
        db_up, last_success, latest_ts = self._cached

        up = GaugeMetricFamily(
            "quantedge_metrics_db_up", "1 if the metrics collector could read the database."
        )
        up.add_metric([], 1.0 if db_up else 0.0)
        yield up

        jobs = GaugeMetricFamily(
            "quantedge_job_last_success_timestamp_seconds",
            "Unix time of each job's most recent successful run, from any instance.",
            labels=["job"],
        )
        for job, ts in sorted(last_success.items()):
            jobs.add_metric([job], ts)
        yield jobs

        if latest_ts is not None:
            bar = GaugeMetricFamily(
                "quantedge_latest_price_date_timestamp_seconds",
                "Midnight UTC of the most recent stored daily bar.",
            )
            bar.add_metric([], latest_ts)
            yield bar


_collector_registered = False


def register_pipeline_collector() -> None:
    """Idempotent, since tests build the app more than once per process."""
    global _collector_registered
    if not _collector_registered:
        REGISTRY.register(PipelineCollector())
        _collector_registered = True


def render_latest() -> bytes:
    return generate_latest(REGISTRY)
