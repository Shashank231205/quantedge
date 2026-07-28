"""Backtest Analysis endpoints (screen 3)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from quantedge.api.deps import require_api_key
from quantedge.backtest.persistence import latest_run, list_runs
from quantedge.db.models import BacktestRun, PortfolioSnapshot, Trade, WalkForwardFold
from quantedge.db.session import session_scope
from quantedge.strategy import DEFAULT_SPEC

router = APIRouter(prefix="/backtest", tags=["backtest"])


class RunRequest(BaseModel):
    """Parameters a caller may vary when launching a run."""

    name: str = Field(default="ad-hoc run")
    rebalance_frequency: str = Field(default="ME")
    long_short: bool = False
    long_quantile: float = Field(default=0.05, gt=0, le=0.5)
    vol_target: float | None = Field(default=0.20, ge=0, le=1.0)
    walk_forward: bool = True


@router.get("/runs", dependencies=[Depends(require_api_key)])
def runs(limit: int = Query(default=20, le=100)) -> dict:
    return {"runs": list_runs(limit)}


@router.get("/strategy", dependencies=[Depends(require_api_key)])
def strategy_spec() -> dict:
    """The canonical production strategy definition."""
    return {
        "spec": DEFAULT_SPEC.as_dict(),
        "rationale": {
            "long_only": "The long/short variant scored 0.17 OOS vs 1.42 long-only; "
                         "the short leg was a persistent drag in this regime.",
            "monthly_rebalance": "IC strengthens with horizon (0.024 at 1d, 0.063 at "
                                 "63d), so daily rebalancing pays costs to trade noise.",
            "top_5_percent": "Concentration in the strongest names beat wider quantiles.",
            "vol_target_20": "Sized so realised drawdown stays inside the 20% mandate.",
        },
    }


def _resolve(run_id: int | None) -> BacktestRun:
    run = latest_run(walk_forward_only=True) if run_id is None else None
    if run_id is not None:
        with session_scope() as s:
            run = s.get(BacktestRun, run_id)
    if run is None:
        raise HTTPException(404, "Backtest run not found")
    return run


@router.get("/metrics", dependencies=[Depends(require_api_key)])
def metrics(run_id: int | None = None) -> dict:
    """Full metric payload, with in-sample and out-of-sample side by side."""
    run = _resolve(run_id)
    payload = run.metrics or {}

    return {
        "run_id": run.id,
        "name": run.name,
        "is_walk_forward": run.is_walk_forward,
        "engine": run.engine_type,
        "period": {"start": str(run.start_date), "end": str(run.end_date)},
        "runtime_ms": run.runtime_ms,
        "config": run.config,
        "metrics": payload,
        "headline": {
            "sharpe": run.sharpe,
            "max_drawdown": run.max_drawdown,
            "total_return": run.total_return,
            "win_rate": run.win_rate,
            "n_trades": run.n_trades,
        },
        # Never let a caller mistake an in-sample figure for a validated one.
        "validation_note": (
            "Out-of-sample, walk-forward validated."
            if run.is_walk_forward
            else "IN-SAMPLE ONLY — not a validated result."
        ),
    }


@router.get("/equity-curve", dependencies=[Depends(require_api_key)])
def equity_curve(run_id: int | None = None) -> dict:
    run = _resolve(run_id)
    with session_scope() as s:
        rows = s.scalars(
            select(PortfolioSnapshot)
            .where(PortfolioSnapshot.run_id == run.id)
            .order_by(PortfolioSnapshot.date)
        ).all()

    return {
        "run_id": run.id,
        "is_out_of_sample": run.is_walk_forward,
        "series": [
            {
                "date": str(r.date),
                "equity": round(r.equity, 2),
                "drawdown": r.drawdown,
                "is_oos": r.is_oos,
            }
            for r in rows
        ],
    }


@router.get("/folds", dependencies=[Depends(require_api_key)])
def folds(run_id: int | None = None) -> dict:
    """Per-fold walk-forward results — the honest consistency check."""
    run = _resolve(run_id)
    with session_scope() as s:
        rows = s.scalars(
            select(WalkForwardFold)
            .where(WalkForwardFold.run_id == run.id)
            .order_by(WalkForwardFold.fold_index)
        ).all()

    folds_out = [
        {
            "fold": r.fold_index,
            "train": {"start": str(r.train_start), "end": str(r.train_end)},
            "test": {"start": str(r.test_start), "end": str(r.test_end)},
            "sharpe_is": r.sharpe_is,
            "sharpe_oos": r.sharpe_oos,
            "return_oos": r.return_oos,
            "max_drawdown_oos": r.max_drawdown_oos,
            "selected_params": (r.metrics or {}).get("selected_params"),
        }
        for r in rows
    ]
    positive = sum(1 for f in folds_out if (f["sharpe_oos"] or 0) > 0)

    return {
        "run_id": run.id,
        "n_folds": len(folds_out),
        "positive_folds": positive,
        "folds": folds_out,
    }


@router.get("/trades", dependencies=[Depends(require_api_key)])
def trades(
    run_id: int | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, le=500),
    outcome: str = Query(default="ALL", pattern="^(ALL|WINS|LOSSES)$"),
    ticker: str | None = None,
) -> dict:
    """Paginated trade log with ALL / WINS / LOSSES filtering."""
    run = _resolve(run_id)

    with session_scope() as s:
        base = select(Trade).where(Trade.run_id == run.id)
        count_stmt = select(func.count()).select_from(Trade).where(Trade.run_id == run.id)

        if outcome == "WINS":
            base = base.where(Trade.pnl_pct > 0)
            count_stmt = count_stmt.where(Trade.pnl_pct > 0)
        elif outcome == "LOSSES":
            base = base.where(Trade.pnl_pct < 0)
            count_stmt = count_stmt.where(Trade.pnl_pct < 0)

        if ticker:
            base = base.where(Trade.ticker.ilike(f"%{ticker.upper()}%"))
            count_stmt = count_stmt.where(Trade.ticker.ilike(f"%{ticker.upper()}%"))

        total = s.scalar(count_stmt) or 0
        # Closed trades first: the 29 still-open positions all share the last
        # bar's date and carry no realised P&L, so leading with them makes the
        # log look empty.
        rows = s.scalars(
            base.order_by(Trade.exit_date.desc().nullslast(), Trade.entry_date.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        ).all()

        items = [
            {
                "id": t.id,
                "ticker": t.ticker,
                "side": t.side,
                "entry_date": str(t.entry_date),
                "exit_date": str(t.exit_date) if t.exit_date else None,
                "entry_price": round(t.entry_price, 4) if t.entry_price else None,
                "exit_price": round(t.exit_price, 4) if t.exit_price else None,
                "pnl_pct": round(t.pnl_pct, 6) if t.pnl_pct is not None else None,
                "pnl_abs": round(t.pnl_abs, 2) if t.pnl_abs is not None else None,
                "holding_days": t.holding_days,
                "status": t.status,
            }
            for t in rows
        ]

    return {
        "run_id": run.id,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": (total + page_size - 1) // page_size,
        "filter": outcome,
        "trades": items,
    }


@router.post("/run", dependencies=[Depends(require_api_key)])
def trigger_run(request: RunRequest) -> dict:
    """Launch a backtest.

    Deliberately synchronous and bounded: the vectorized engine completes a
    full-history run in well under a second, so a queue would add complexity
    without solving a real problem here.
    """
    import pandas as pd

    from quantedge.backtest.naive import BacktestResult
    from quantedge.backtest.persistence import save_backtest_run
    from quantedge.metrics.report import compare_is_oos, full_report
    from quantedge.strategy import (
        StrategySpec,
        run_full_sample,
        run_walk_forward,
    )

    spec = StrategySpec(
        name=request.name,
        rebalance_frequency=request.rebalance_frequency,
        long_short=request.long_short,
        long_quantile=request.long_quantile,
        vol_target=request.vol_target,
    )

    if request.walk_forward:
        wf, panel, bench = run_walk_forward(spec)
        if wf.oos_returns.empty:
            raise HTTPException(400, "Walk-forward produced no out-of-sample data")

        report = compare_is_oos(
            wf.is_returns, wf.oos_returns, bench, n_trials=wf.n_configurations_tested
        )
        equity = (1.0 + wf.oos_returns).cumprod() * 1_000_000
        result = BacktestResult(
            returns=wf.oos_returns,
            equity_curve=equity,
            weights=wf.oos_weights,
            turnover=wf.oos_turnover,
            costs=pd.Series(0.0, index=wf.oos_returns.index),
            gross_exposure=wf.oos_weights.abs().sum(axis=1),
            net_exposure=wf.oos_weights.sum(axis=1),
            engine="vectorized",
            metadata={
                "config": spec.as_dict(),
                "costs": spec.cost_model().describe(),
                "n_bars": len(wf.oos_returns),
                "n_tickers": panel.shape[1],
            },
        )
        metrics = dict(report["out_of_sample"])
        metrics["comparison"] = report["comparison"]
        metrics["in_sample"] = report["in_sample"]
        # The stitched OOS series has its own turnover; compute_is_oos does not
        # see it, so attach it explicitly.
        if not wf.oos_turnover.empty:
            from quantedge.metrics.trades import turnover_statistics

            metrics["turnover"] = turnover_statistics(wf.oos_turnover)
        run_id = save_backtest_run(
            f"{spec.name} (walk-forward OOS)", result, metrics,
            prices=panel, walk_forward=wf, is_walk_forward=True,
        )
    else:
        result, panel, bench, _ = run_full_sample(spec)
        metrics = full_report(
            result.returns, bench, turnover=result.turnover, n_trials=1,
            label="full_sample_in_sample",
        )
        run_id = save_backtest_run(
            f"{spec.name} (full-sample, IN-SAMPLE)", result, metrics,
            prices=panel, is_walk_forward=False,
        )

    return {"run_id": run_id, "status": "completed", "walk_forward": request.walk_forward}
