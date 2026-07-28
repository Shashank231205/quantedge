"""Job-run telemetry.

Every scheduled job wraps itself in ``track_job``. The resulting ``job_runs``
rows are the *only* source of the pipeline-uptime figure reported in the UI
and the README — nothing is hardcoded. If a job fails, uptime drops, and the
System Health screen shows it.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select

from quantedge.db.models import JobRun
from quantedge.db.session import session_scope
from quantedge.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class JobContext:
    """Mutable handle a running job uses to report what it did."""

    job_id: int | None = None
    rows_written: int = 0
    bytes_processed: int = 0
    tickers_processed: int = 0
    details: dict[str, Any] = field(default_factory=dict)


@contextmanager
def track_job(job_name: str) -> Generator[JobContext, None, None]:
    """Record a job execution, success or failure.

    The RUNNING row is written before the work starts so a crashed process
    still leaves evidence rather than vanishing.
    """
    ctx = JobContext()
    started = datetime.now(UTC).replace(tzinfo=None)

    with session_scope() as s:
        run = JobRun(job_name=job_name, status="RUNNING", started_at=started)
        s.add(run)
        s.flush()
        ctx.job_id = run.id

    log.info("job.start name=%s id=%s", job_name, ctx.job_id)

    try:
        yield ctx
    except Exception as exc:
        finished = datetime.now(UTC).replace(tzinfo=None)
        with session_scope() as s:
            run = s.get(JobRun, ctx.job_id)
            if run is not None:
                run.status = "FAILED"
                run.finished_at = finished
                run.duration_ms = (finished - started).total_seconds() * 1000
                run.error = f"{type(exc).__name__}: {exc}"[:4000]
                run.rows_written = ctx.rows_written
                run.tickers_processed = ctx.tickers_processed
                run.details = ctx.details
        log.error("job.failed name=%s error=%s", job_name, exc)
        raise
    else:
        finished = datetime.now(UTC).replace(tzinfo=None)
        duration = (finished - started).total_seconds() * 1000
        with session_scope() as s:
            run = s.get(JobRun, ctx.job_id)
            if run is not None:
                run.status = "SUCCESS"
                run.finished_at = finished
                run.duration_ms = duration
                run.rows_written = ctx.rows_written
                run.bytes_processed = ctx.bytes_processed
                run.tickers_processed = ctx.tickers_processed
                run.details = ctx.details
        log.info(
            "job.success name=%s rows=%s tickers=%s duration_ms=%.0f",
            job_name,
            ctx.rows_written,
            ctx.tickers_processed,
            duration,
        )


def compute_uptime(days: int = 30, job_name: str | None = None) -> dict[str, Any]:
    """Real uptime = successful runs / completed runs over the window."""
    cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)

    with session_scope() as s:
        # Only completed runs count; a RUNNING job is neither success nor failure.
        total_stmt = select(func.count(JobRun.id)).where(
            JobRun.started_at >= cutoff, JobRun.status != "RUNNING"
        )
        ok_stmt = select(func.count(JobRun.id)).where(
            JobRun.started_at >= cutoff, JobRun.status == "SUCCESS"
        )
        if job_name:
            total_stmt = total_stmt.where(JobRun.job_name == job_name)
            ok_stmt = ok_stmt.where(JobRun.job_name == job_name)

        total = s.scalar(total_stmt)
        succeeded = s.scalar(ok_stmt)

    total = total or 0
    succeeded = succeeded or 0
    return {
        "window_days": days,
        "total_runs": total,
        "successful_runs": succeeded,
        "failed_runs": total - succeeded,
        # No runs yet is not 100% uptime; report None so the UI can say "n/a".
        "uptime_pct": round(100.0 * succeeded / total, 3) if total else None,
    }


def recent_runs(limit: int = 20) -> list[dict[str, Any]]:
    """Latest job runs for the System Health stream table."""
    with session_scope() as s:
        rows = s.scalars(
            select(JobRun).order_by(JobRun.started_at.desc()).limit(limit)
        ).all()
        return [
            {
                "id": r.id,
                "job_name": r.job_name,
                "status": r.status,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                "duration_ms": r.duration_ms,
                "rows_written": r.rows_written,
                "tickers_processed": r.tickers_processed,
                "error": r.error,
            }
            for r in rows
        ]
