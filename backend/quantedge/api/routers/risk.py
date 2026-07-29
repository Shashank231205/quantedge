"""Risk Monitor endpoints (screen 4)."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select

from quantedge.api.deps import cache, require_api_key
from quantedge.backtest.persistence import latest_run
from quantedge.config import settings
from quantedge.db.models import PortfolioSnapshot, RiskEvent, Security
from quantedge.db.session import session_scope
from quantedge.logging_config import get_recent_logs
from quantedge.risk.drawdown_guard import circuit_breaker_status
from quantedge.risk.exposure import check_breaches, exposure_summary, sector_exposure
from quantedge.risk.position_sizing import position_sizing_report, realized_volatility
from quantedge.risk.var import var_report
from quantedge.strategy import DEFAULT_SPEC, run_full_sample

router = APIRouter(prefix="/risk", tags=["risk"])


def _live_portfolio():
    """Current weights, volatilities and return series.

    Recomputing a full-sample backtest per request cost ~15s and pushed the
    p95 far past the 200ms target, so the portfolio is built once at first use
    and cached. ``warm_cache()`` primes it during API startup so even the first
    caller sees a fast response.
    """
    cached = cache.get("live_portfolio")
    if cached is not None:
        return cached

    # Prefer the book the snapshot step already computed. Running a full-sample
    # backtest per request needs the whole price panel resident, which a small
    # instance cannot hold -- and the answer is the same either way.
    from quantedge.factors.snapshot import read_diagnostic

    book = read_diagnostic("live_book")
    if book and book.get("weights"):
        weights = pd.Series(book["weights"], dtype=float)
        vols = pd.Series(book.get("volatility") or {}, dtype=float)
        payload = (_StoredRun(), None, weights, vols, None)
        cache.set("live_portfolio", payload)
        return payload

    result, panel, benchmark, _ = run_full_sample(DEFAULT_SPEC)
    weights = result.weights.iloc[-1]
    vols = realized_volatility(panel, window=60).iloc[-1]

    payload = (result, panel, weights, vols, benchmark)
    cache.set("live_portfolio", payload)
    return payload


def _as_of(panel) -> str:
    """The date this view describes.

    On the stored path there is no panel to read it from, so fall back to the
    snapshot's own as-of. Both name the same bar; only the source differs.
    """
    if panel is not None and len(panel.index):
        return str(panel.index[-1].date())

    from quantedge.factors.snapshot import read_diagnostic

    book = read_diagnostic("live_book") or {}
    return book.get("as_of") or "unknown"


class _StoredRun:
    """Equity and returns read from the persisted run.

    The risk endpoints need a drawdown series and a return series, both of
    which the last walk-forward run already recorded. Re-deriving them by
    re-running the backtest per request is what made this screen unaffordable
    on a small instance.
    """

    def __init__(self) -> None:
        from quantedge.db.models import BacktestRun, PortfolioSnapshot

        with session_scope() as s:
            run = s.scalars(
                select(BacktestRun)
                .where(BacktestRun.is_walk_forward.is_(True))
                .order_by(BacktestRun.created_at.desc())
                .limit(1)
            ).first()
            rows = (
                s.execute(
                    select(PortfolioSnapshot.date, PortfolioSnapshot.equity)
                    .where(PortfolioSnapshot.run_id == run.id)
                    .order_by(PortfolioSnapshot.date)
                ).all()
                if run
                else []
            )

        idx = pd.to_datetime([r.date for r in rows])
        self.equity_curve = pd.Series([float(r.equity) for r in rows], index=idx)
        self.returns = self.equity_curve.pct_change().dropna()


def warm_cache() -> None:
    """Build the live portfolio ahead of the first request."""
    try:
        _live_portfolio()
        _sectors()
    except Exception as exc:  # a cold cache must not stop the API booting
        from quantedge.logging_config import get_logger

        get_logger(__name__).warning("risk.warm_cache_failed error=%s", exc)


def _sectors() -> pd.Series:
    cached = cache.get("sector_series")
    if cached is None:
        with session_scope() as s:
            rows = s.scalars(select(Security)).all()
        cached = pd.Series({r.ticker: r.sector for r in rows if r.sector})
        cache.set("sector_series", cached)
    return cached


@router.get("/summary", dependencies=[Depends(require_api_key)])
def risk_summary() -> dict:
    """Circuit-breaker gauge, exposure and VaR headline."""
    result, panel, weights, vols, _ = _live_portfolio()
    breaker = circuit_breaker_status(result.equity_curve, settings.max_drawdown_limit)

    return {
        "circuit_breaker": breaker,
        "exposure": exposure_summary(weights),
        "var": var_report(result.returns, portfolio_value=1_000_000.0),
        "portfolio_volatility": round(
            float(result.returns.iloc[-60:].std() * (252**0.5)), 4
        ),
        "as_of": _as_of(panel),
    }


@router.get("/exposure", dependencies=[Depends(require_api_key)])
def exposure() -> dict:
    """Sector concentration — the treemap on the Risk Monitor."""
    _, panel, weights, _, _ = _live_portfolio()
    sectors = _sectors()
    table = sector_exposure(weights, sectors)

    return {
        "as_of": _as_of(panel),
        "summary": exposure_summary(weights),
        "sectors": table.to_dict("records") if not table.empty else [],
        "limits": {
            "max_position_weight": settings.max_position_weight,
            "max_sector_weight": settings.max_sector_weight,
        },
    }


@router.get("/positions", dependencies=[Depends(require_api_key)])
def positions(limit: int = Query(default=50, le=300)) -> dict:
    """Position sizing matrix: current weight vs vol-adjusted target."""
    _, panel, weights, vols, _ = _live_portfolio()
    report = position_sizing_report(weights, vols, _sectors())

    if report.empty:
        return {"as_of": _as_of(panel), "positions": []}

    report = report.head(limit).copy()
    for col in ("current_weight", "vol_adj_target", "volatility", "drift"):
        report[col] = report[col].round(6)

    return {
        "as_of": _as_of(panel),
        "n_positions": len(report),
        "positions": report.to_dict("records"),
    }


@router.get("/breaches", dependencies=[Depends(require_api_key)])
def breaches() -> dict:
    """Live limit breaches — powers the alert banner."""
    _, panel, weights, _, _ = _live_portfolio()
    detected = check_breaches(
        weights,
        _sectors(),
        max_position=settings.max_position_weight,
        max_sector=settings.max_sector_weight,
    )

    with session_scope() as s:
        stored = s.scalars(
            select(RiskEvent).order_by(RiskEvent.created_at.desc()).limit(20)
        ).all()
        history = [
            {
                "id": e.id,
                "type": e.event_type,
                "severity": e.severity,
                "message": e.message,
                "resolved": e.resolved,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in stored
        ]

    return {
        "as_of": _as_of(panel),
        "active_breaches": detected,
        "n_active": len(detected),
        "history": history,
    }


@router.get("/var", dependencies=[Depends(require_api_key)])
def value_at_risk(portfolio_value: float = Query(default=1_000_000.0, gt=0)) -> dict:
    result, _, _, _, _ = _live_portfolio()
    return var_report(result.returns, portfolio_value=portfolio_value)


@router.get("/drawdown-history", dependencies=[Depends(require_api_key)])
def drawdown_history() -> dict:
    """Drawdown path of the validated run."""
    run = latest_run(walk_forward_only=True) or latest_run()
    if run is None:
        raise HTTPException(404, "No backtest run found")

    with session_scope() as s:
        rows = s.scalars(
            select(PortfolioSnapshot)
            .where(PortfolioSnapshot.run_id == run.id)
            .order_by(PortfolioSnapshot.date)
        ).all()

    return {
        "run_id": run.id,
        "series": [
            {"date": str(r.date), "drawdown": r.drawdown}
            for r in rows
            if r.drawdown is not None
        ],
        "limit": -abs(settings.max_drawdown_limit),
    }


@router.get("/logs", dependencies=[Depends(require_api_key)])
def risk_logs(limit: int = Query(default=30, le=200)) -> dict:
    return {"logs": get_recent_logs(limit=limit)}
