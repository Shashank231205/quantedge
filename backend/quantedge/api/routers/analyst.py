"""Analyst mode endpoints (screen 6).

Generating a report costs a provider call, so results are cached per
(run, universe size). The cache is what keeps a public demo inside a free
tier's rate limit: repeat viewers of the same run read a stored report
rather than triggering a new completion.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select

from quantedge.analyst import providers
from quantedge.analyst.agent import generate_report, report_to_dict
from quantedge.api.deps import TTLCache, require_api_key
from quantedge.config import settings
from quantedge.db.models import BacktestRun
from quantedge.db.session import session_scope
from quantedge.logging_config import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/analyst", tags=["analyst"])

_cache = TTLCache(ttl_seconds=settings.analyst_cache_seconds)


@router.get("/status", dependencies=[Depends(require_api_key)])
def status() -> dict:
    """Which backend the next report would use, and what the UI should say.

    Exposed so the screen can tell the reader whether prose was written by a
    model or assembled from templates, rather than leaving them to guess.
    """
    active = providers.active_provider_name()
    return {
        "active_provider": active,
        "is_template": active == "template",
        "configured_chain": settings.analyst_providers,
        "available": [p.name for p in providers.resolve_chain() if p.available()],
        "cache_ttl_seconds": settings.analyst_cache_seconds,
        "note": (
            "Scores are computed by the platform's rubric and do not depend on "
            "the provider. The model writes the explanation only."
        ),
    }


@router.get("/report", dependencies=[Depends(require_api_key)])
def report(
    run_id: int | None = Query(default=None),
    universe_size: int = Query(default=50, ge=1, le=1000),
    refresh: bool = Query(default=False, description="Bypass the cached report"),
) -> dict:
    """Full assessment for a run: overall verdict plus per-metric detail."""
    # Read every field inside the session: the ORM instance is detached once
    # the scope closes, and touching an unloaded attribute after that raises.
    with session_scope() as s:
        run = (
            s.get(BacktestRun, run_id)
            if run_id is not None
            else s.scalars(
                select(BacktestRun)
                .where(BacktestRun.is_walk_forward.is_(True))
                .order_by(BacktestRun.created_at.desc())
                .limit(1)
            ).first()
        )
        if run is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No walk-forward backtest run found. Run `quantedge backtest` "
                    "to create one before requesting an assessment."
                ),
            )
        context = {
            "run_id": run.id,
            "run_name": run.name,
            "engine": run.engine_type,
            "is_walk_forward": run.is_walk_forward,
            "period": {"start": str(run.start_date), "end": str(run.end_date)},
        }
        metrics = run.metrics or {}

    if not metrics:
        raise HTTPException(
            status_code=409,
            detail=f"Run {context['run_id']} has no stored metrics to assess.",
        )

    key = f"report:{context['run_id']}:{universe_size}"
    if not refresh:
        hit = _cache.get(key)
        if hit is not None:
            return {**hit, "cached": True}

    result = report_to_dict(
        generate_report(metrics, context, universe_size=universe_size)
    )
    _cache.set(key, result)
    return {**result, "cached": False}
