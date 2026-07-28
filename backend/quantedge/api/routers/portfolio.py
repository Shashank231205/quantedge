"""Portfolio Dashboard endpoints (screen 1)."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select

from quantedge.api.deps import cache, require_api_key
from quantedge.backtest.persistence import latest_run
from quantedge.db.models import BacktestRun, PortfolioSnapshot
from quantedge.db.session import session_scope
from quantedge.ingestion.telemetry import compute_uptime
from quantedge.logging_config import get_recent_logs
from quantedge.strategy import current_signals

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


def _headline_run() -> BacktestRun:
    """The walk-forward run is the one whose numbers we quote."""
    run = latest_run(walk_forward_only=True) or latest_run()
    if run is None:
        raise HTTPException(404, "No backtest run found. Run `make backtest` first.")
    return run


@router.get("/summary", dependencies=[Depends(require_api_key)])
def portfolio_summary() -> dict:
    """KPI strip: the four headline numbers plus pipeline status."""
    run = _headline_run()
    metrics = run.metrics or {}
    risk_adj = metrics.get("risk_adjusted", {})
    risk = metrics.get("risk", {})
    returns = metrics.get("returns", {})
    uptime = compute_uptime(days=30)

    return {
        "run_id": run.id,
        "run_name": run.name,
        # Labelled explicitly: this is the out-of-sample figure, not in-sample.
        "is_out_of_sample": run.is_walk_forward,
        "sharpe_ratio": risk_adj.get("sharpe_ratio"),
        "sortino_ratio": risk_adj.get("sortino_ratio"),
        "calmar_ratio": risk_adj.get("calmar_ratio"),
        "annualized_return": returns.get("annualized_return"),
        "total_return": returns.get("total_return"),
        "volatility": risk.get("annualized_volatility"),
        "max_drawdown": risk.get("max_drawdown"),
        "period": {"start": str(run.start_date), "end": str(run.end_date)},
        "pipeline": {
            "uptime_pct": uptime["uptime_pct"],
            "total_runs": uptime["total_runs"],
            "failed_runs": uptime["failed_runs"],
            "status": "OPERATIONAL" if (uptime["uptime_pct"] or 0) >= 99 else "DEGRADED",
        },
    }


@router.get("/equity-curve", dependencies=[Depends(require_api_key)])
def equity_curve(
    run_id: int | None = None,
    max_points: int = Query(default=1500, le=5000),
) -> dict:
    """Equity curve with an aligned benchmark series."""
    run = _headline_run() if run_id is None else _require_run(run_id)

    with session_scope() as s:
        rows = s.scalars(
            select(PortfolioSnapshot)
            .where(PortfolioSnapshot.run_id == run.id)
            .order_by(PortfolioSnapshot.date)
        ).all()
        points = [
            {
                "date": str(r.date),
                "equity": round(r.equity, 2),
                "returns": r.returns,
                "drawdown": r.drawdown,
                "is_oos": r.is_oos,
            }
            for r in rows
        ]

    if len(points) > max_points:
        step = len(points) // max_points + 1
        points = points[::step]

    return {
        "run_id": run.id,
        "n_points": len(points),
        "is_out_of_sample": run.is_walk_forward,
        "series": points,
        "benchmark": _benchmark_series(
            [p["date"] for p in points]
        ),
    }


@router.get("/drawdown", dependencies=[Depends(require_api_key)])
def drawdown(run_id: int | None = None) -> dict:
    run = _headline_run() if run_id is None else _require_run(run_id)
    with session_scope() as s:
        rows = s.scalars(
            select(PortfolioSnapshot)
            .where(PortfolioSnapshot.run_id == run.id)
            .order_by(PortfolioSnapshot.date)
        ).all()
        series = [
            {"date": str(r.date), "drawdown": r.drawdown} for r in rows if r.drawdown is not None
        ]

    current = series[-1]["drawdown"] if series else 0.0
    worst = min((p["drawdown"] for p in series), default=0.0)
    return {
        "run_id": run.id,
        "current_drawdown": current,
        "max_drawdown": worst,
        "series": series,
    }


@router.get("/signals", dependencies=[Depends(require_api_key)])
def live_signals(top_n: int = Query(default=20, le=100)) -> dict:
    """Current factor ranking — the strategy's live view as of the last bar."""
    key = f"signals:{top_n}"
    cached = cache.get(key)
    if cached is not None:
        return cached

    df = current_signals(top_n=top_n)
    payload = {
        "as_of": None,
        "n_signals": len(df),
        "signals": df.to_dict("records") if not df.empty else [],
        "note": "Ranking as of the most recent bar; scores are lagged one bar.",
    }

    with session_scope() as s:
        from quantedge.db.models import OhlcvClean
        from sqlalchemy import func

        payload["as_of"] = str(s.scalar(select(func.max(OhlcvClean.date))))

    cache.set(key, payload)
    return payload


@router.get("/logs", dependencies=[Depends(require_api_key)])
def logs(limit: int = Query(default=50, le=500), level: str | None = None) -> dict:
    """Real application log tail — not a synthetic feed."""
    return {"logs": get_recent_logs(limit=limit, level=level), "filter": level or "ALL"}


# ---------------------------------------------------------------------------


def _require_run(run_id: int) -> BacktestRun:
    with session_scope() as s:
        run = s.get(BacktestRun, run_id)
    if run is None:
        raise HTTPException(404, f"Backtest run {run_id} not found")
    return run


def _benchmark_series(dates: list[str]) -> list[dict]:
    """Benchmark equity rebased to 1.0 over the same dates."""
    from quantedge.config import settings
    from quantedge.db.models import OhlcvClean

    cached = cache.get("benchmark_prices")
    if cached is None:
        with session_scope() as s:
            rows = s.scalars(
                select(OhlcvClean)
                .where(OhlcvClean.ticker == settings.benchmark_ticker)
                .order_by(OhlcvClean.date)
            ).all()
        cached = {str(r.date): r.close for r in rows}
        cache.set("benchmark_prices", cached)

    wanted = [d for d in dates if d in cached]
    if not wanted:
        return []

    base = cached[wanted[0]]
    return [
        {"date": d, "value": round(cached[d] / base, 6)} for d in wanted
    ]
