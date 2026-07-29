"""System Health endpoints (screen 5).

Every figure here is measured, not decorated. The Figma for this screen showed
Binance crypto feeds, a FIX order-flow gateway, a sentiment NLP stream and 128
auto-scaling workers — none of which this system has. Those panels are bound
instead to the jobs that genuinely run: ingestion, factor computation,
universe refresh and backtests.

Concretely:
  * uptime  -> successes / completed runs in ``job_runs``
  * streams -> the actual scheduled jobs and their last outcome
  * volume  -> rows written and bytes processed by the last ingest
  * p95     -> real request latencies captured by the middleware
  * syslog  -> the application's own log tail
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select

from quantedge.api.deps import TTLCache, require_api_key
from quantedge.api.middleware import latency_stats
from quantedge.config import settings
from quantedge.db.models import ApiRequestLog, JobRun, OhlcvClean, Security
from quantedge.db.session import session_scope
from quantedge.ingestion.pipeline import coverage_stats
from quantedge.ingestion.telemetry import compute_uptime, recent_runs
from quantedge.logging_config import get_recent_logs

router = APIRouter(prefix="/system", tags=["system"])

#: The jobs this platform actually runs.
KNOWN_JOBS = ("ohlcv_ingest", "universe_refresh", "factor_compute", "backtest_nightly")

#: Coverage is two sequential scans of ohlcv_clean — COUNT(*) and
#: COUNT(DISTINCT ticker) over 837k rows, measured at 88ms and 66ms. That was
#: the whole cost of /status and /ingestion, which the Pipeline screen polls.
#: The numbers only move when ingest writes, so serving them from a short
#: cache costs nothing in freshness.
_coverage_cache = TTLCache(ttl_seconds=60.0)


def cached_coverage() -> dict:
    hit = _coverage_cache.get("coverage")
    if hit is None:
        hit = coverage_stats()
        _coverage_cache.set("coverage", hit)
    return hit


@router.get("/health")
def health() -> dict:
    """Unauthenticated liveness probe, used by Docker's healthcheck."""
    try:
        with session_scope() as s:
            s.execute(select(1))
        db_ok = True
    except Exception:
        db_ok = False

    return {
        "status": "healthy" if db_ok else "degraded",
        "database": "connected" if db_ok else "unreachable",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": "0.1.0",
    }


@router.get("/status", dependencies=[Depends(require_api_key)])
def system_status() -> dict:
    """Top KPI strip: integrity, uptime, worker count, last sync."""
    uptime = compute_uptime(days=30)
    coverage = cached_coverage()

    with session_scope() as s:
        last_success = s.scalars(
            select(JobRun)
            .where(JobRun.status == "SUCCESS")
            .order_by(JobRun.finished_at.desc())
            .limit(1)
        ).first()
        running = s.scalar(
            select(func.count()).select_from(JobRun).where(JobRun.status == "RUNNING")
        ) or 0

    last_sync_seconds = None
    if last_success and last_success.finished_at:
        delta = datetime.now(UTC).replace(tzinfo=None) - last_success.finished_at
        last_sync_seconds = round(delta.total_seconds(), 1)

    has_data = coverage["total_rows"] > 0
    return {
        # "PRODUCTION_READY" only when data is loaded and jobs are passing.
        "system_integrity": "PRODUCTION_READY"
        if has_data and (uptime["uptime_pct"] or 0) >= 99
        else "DEGRADED"
        if has_data
        else "NO_DATA",
        "uptime_pct": uptime["uptime_pct"],
        "uptime_window_days": uptime["window_days"],
        "total_job_runs": uptime["total_runs"],
        "failed_job_runs": uptime["failed_runs"],
        # Real scheduler concurrency, not a fabricated worker fleet.
        "active_jobs": running,
        "configured_jobs": len(KNOWN_JOBS),
        "last_sync_seconds_ago": last_sync_seconds,
        "last_sync_job": last_success.job_name if last_success else None,
        "coverage": coverage,
    }


@router.get("/jobs", dependencies=[Depends(require_api_key)])
def jobs(limit: int = Query(default=20, le=100)) -> dict:
    """Job run history — the stream status table."""
    runs = recent_runs(limit)

    with session_scope() as s:
        per_job = []
        for name in KNOWN_JOBS:
            last = s.scalars(
                select(JobRun)
                .where(JobRun.job_name == name)
                .order_by(JobRun.started_at.desc())
                .limit(1)
            ).first()
            stats = compute_uptime(days=30, job_name=name)
            per_job.append(
                {
                    "job_name": name,
                    "last_run": last.started_at.isoformat() if last else None,
                    "last_status": last.status if last else "NEVER_RUN",
                    "last_duration_ms": last.duration_ms if last else None,
                    "last_rows_written": last.rows_written if last else 0,
                    "uptime_pct": stats["uptime_pct"],
                    "total_runs": stats["total_runs"],
                }
            )

    return {"streams": per_job, "recent_runs": runs}


@router.get("/ingestion", dependencies=[Depends(require_api_key)])
def ingestion_stats() -> dict:
    """Data Ingestion Engine panel — real counts from the last run."""
    coverage = cached_coverage()

    with session_scope() as s:
        last = s.scalars(
            select(JobRun)
            .where(JobRun.job_name == "ohlcv_ingest")
            .order_by(JobRun.started_at.desc())
            .limit(1)
        ).first()
        n_sec = s.scalar(select(func.count()).select_from(Security)) or 0
        latest_date = s.scalar(select(func.max(OhlcvClean.date)))

    details = (last.details or {}) if last else {}
    cleaning = details.get("cleaning", {})

    return {
        "tickers_updated": last.tickers_processed if last else 0,
        "rows_written_last_run": last.rows_written if last else 0,
        "bytes_processed_last_run": last.bytes_processed if last else 0,
        "duration_ms": last.duration_ms if last else None,
        "fetch_success_rate": details.get("fetch_success_rate"),
        "source": details.get("source", settings.data_source),
        "n_securities": n_sec,
        "latest_data_date": str(latest_date) if latest_date else None,
        "coverage": coverage,
        "cleaning": {
            "rows_in": cleaning.get("rows_in"),
            "rows_out": cleaning.get("rows_out"),
            "retention_pct": cleaning.get("retention_pct"),
            "dropped_bad_ohlc": cleaning.get("dropped_bad_ohlc"),
            "flagged_extreme_return": cleaning.get("flagged_extreme_return"),
        },
        "failed_tickers": details.get("failed", 0),
        "note": (
            "Tickers that fail are almost entirely delisted or acquired names "
            "for which the provider no longer serves history."
        ),
    }


@router.get("/api-metrics", dependencies=[Depends(require_api_key)])
def api_metrics() -> dict:
    """API Mesh Health — real p95 from the latency middleware."""
    live = latency_stats()

    with session_scope() as s:
        total = s.scalar(select(func.count()).select_from(ApiRequestLog)) or 0
        by_endpoint = s.execute(
            select(
                ApiRequestLog.endpoint,
                func.count().label("n"),
                func.avg(ApiRequestLog.latency_ms).label("mean_ms"),
                func.max(ApiRequestLog.latency_ms).label("max_ms"),
            )
            .group_by(ApiRequestLog.endpoint)
            .order_by(func.count().desc())
            .limit(15)
        ).all()

    return {
        "live": live,
        "total_requests_logged": total,
        "target_p95_ms": 200,
        "meets_target": (live["p95_ms"] or 0) < 200 if live["n_requests"] else None,
        "by_endpoint": [
            {
                "endpoint": row.endpoint,
                "requests": row.n,
                "mean_ms": round(float(row.mean_ms), 2),
                "max_ms": round(float(row.max_ms), 2),
            }
            for row in by_endpoint
        ],
    }


@router.get("/logs", dependencies=[Depends(require_api_key)])
def system_logs(
    limit: int = Query(default=100, le=500),
    level: str = Query(default="ALL", pattern="^(ALL|ERRORS|WARNINGS|INFO|DEBUG)$"),
) -> dict:
    """Live syslog panel with ALL / ERRORS / WARNINGS filtering."""
    entries = get_recent_logs(limit=limit, level=level)
    return {
        "filter": level,
        "n_entries": len(entries),
        "logs": entries,
        "note": "Real application log records emitted by this process.",
    }


@router.get("/info", dependencies=[Depends(require_api_key)])
def system_info() -> dict:
    """Runtime and configuration detail."""
    import platform
    import sys

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "data_source": settings.data_source,
        "benchmark": settings.benchmark_ticker,
        "history_years": settings.history_years,
        "trading_days_per_year": settings.trading_days_per_year,
        "costs": {
            "commission_bps": settings.commission_bps,
            "slippage_bps": settings.slippage_bps,
        },
        "risk_limits": {
            "max_position_weight": settings.max_position_weight,
            "max_sector_weight": settings.max_sector_weight,
            "max_drawdown_limit": settings.max_drawdown_limit,
            "target_annual_vol": settings.target_annual_vol,
        },
    }
