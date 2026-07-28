"""Persist backtest runs so the API serves stored results, not recomputations.

Every run writes its full config alongside its metrics. That pairing is what
makes a result reproducible months later — a Sharpe ratio with no record of
the costs, universe and rebalance schedule that produced it is not evidence
of anything.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import delete, select

from quantedge.backtest.naive import BacktestResult
from quantedge.backtest.walk_forward import WalkForwardResult
from quantedge.db.models import (
    BacktestRun,
    PortfolioSnapshot,
    Trade,
    WalkForwardFold,
)
from quantedge.db.session import session_scope
from quantedge.logging_config import get_logger
from quantedge.metrics.trades import extract_trades

log = get_logger(__name__)

SNAPSHOT_CHUNK = 2_000


def save_backtest_run(
    name: str,
    result: BacktestResult,
    metrics: dict,
    prices: pd.DataFrame | None = None,
    walk_forward: WalkForwardResult | None = None,
    is_walk_forward: bool = False,
    initial_capital: float = 1_000_000.0,
) -> int:
    """Write a run, its folds, trades and daily snapshots. Returns the run id."""
    returns = result.returns
    equity = result.equity_curve

    config = dict(result.metadata.get("config", {}))
    config.update(
        {
            "costs": result.metadata.get("costs", {}),
            "engine": result.engine,
            "n_bars": result.metadata.get("n_bars"),
            "n_tickers": result.metadata.get("n_tickers"),
        }
    )
    if walk_forward is not None:
        config["walk_forward"] = walk_forward.config
        config["n_configurations_tested"] = walk_forward.n_configurations_tested

    risk_adj = metrics.get("risk_adjusted", {})
    ret_block = metrics.get("returns", {})
    risk_block = metrics.get("risk", {})

    with session_scope() as s:
        run = BacktestRun(
            name=name,
            engine_type=result.engine or "vectorized",
            config=config,
            start_date=returns.index[0].date() if len(returns) else date.today(),
            end_date=returns.index[-1].date() if len(returns) else date.today(),
            is_walk_forward=is_walk_forward,
            metrics=metrics,
            sharpe=risk_adj.get("sharpe_ratio"),
            sharpe_oos=metrics.get("comparison", {}).get("sharpe_oos"),
            max_drawdown=risk_block.get("max_drawdown"),
            total_return=ret_block.get("total_return"),
            win_rate=metrics.get("trades", {}).get("win_rate"),
            runtime_ms=result.runtime_seconds * 1000,
            n_trades=metrics.get("trades", {}).get("n_trades"),
        )
        s.add(run)
        s.flush()
        run_id = run.id

        # --- walk-forward folds -------------------------------------------
        if walk_forward is not None:
            for fold in walk_forward.folds:
                s.add(
                    WalkForwardFold(
                        run_id=run_id,
                        fold_index=fold["fold"],
                        train_start=pd.Timestamp(fold["train_start"]).date(),
                        train_end=pd.Timestamp(fold["train_end"]).date(),
                        test_start=pd.Timestamp(fold["test_start"]).date(),
                        test_end=pd.Timestamp(fold["test_end"]).date(),
                        sharpe_is=fold.get("train_score"),
                        sharpe_oos=fold.get("oos_sharpe"),
                        return_oos=fold.get("oos_return"),
                        max_drawdown_oos=fold.get("oos_max_drawdown"),
                        metrics=fold,
                    )
                )

        # --- daily snapshots -----------------------------------------------
        gross = result.gross_exposure
        net = result.net_exposure
        dd = (equity / equity.cummax() - 1.0) if len(equity) else pd.Series(dtype=float)
        n_pos = (result.weights.abs() > 1e-12).sum(axis=1)

        rows = [
            {
                "run_id": run_id,
                "date": ts.date(),
                "equity": float(equity.loc[ts]),
                "returns": float(returns.loc[ts]),
                "drawdown": float(dd.loc[ts]) if ts in dd.index else None,
                "gross_exposure": float(gross.loc[ts]) if ts in gross.index else None,
                "net_exposure": float(net.loc[ts]) if ts in net.index else None,
                "n_positions": int(n_pos.loc[ts]) if ts in n_pos.index else None,
                "is_oos": is_walk_forward,
            }
            for ts in returns.index
        ]
        for i in range(0, len(rows), SNAPSHOT_CHUNK):
            s.bulk_insert_mappings(PortfolioSnapshot, rows[i : i + SNAPSHOT_CHUNK])

        # --- trades -----------------------------------------------------------
        if prices is not None and not result.weights.empty:
            trades = extract_trades(result.weights, prices, initial_capital)
            if not trades.empty:
                trade_rows = [
                    {
                        "run_id": run_id,
                        "ticker": t.ticker,
                        "side": t.side,
                        "entry_date": pd.Timestamp(t.entry_date).date(),
                        "exit_date": pd.Timestamp(t.exit_date).date()
                        if t.exit_date is not None and pd.notna(t.exit_date)
                        else None,
                        "entry_price": float(t.entry_price)
                        if pd.notna(t.entry_price) else 0.0,
                        "exit_price": float(t.exit_price)
                        if pd.notna(t.exit_price) else None,
                        "quantity": 0.0,
                        "weight": float(t.weight),
                        "pnl_pct": float(t.pnl_pct),
                        "pnl_abs": float(t.pnl_abs),
                        "holding_days": float(t.holding_days),
                        "status": t.status,
                    }
                    for t in trades.itertuples()
                ]
                for i in range(0, len(trade_rows), SNAPSHOT_CHUNK):
                    s.bulk_insert_mappings(Trade, trade_rows[i : i + SNAPSHOT_CHUNK])

    log.info("backtest.saved id=%s name=%s bars=%s", run_id, name, len(returns))
    return run_id


def latest_run(walk_forward_only: bool = False) -> BacktestRun | None:
    with session_scope() as s:
        stmt = select(BacktestRun).order_by(BacktestRun.created_at.desc())
        if walk_forward_only:
            stmt = stmt.where(BacktestRun.is_walk_forward.is_(True))
        return s.scalars(stmt.limit(1)).first()


def get_run(run_id: int) -> BacktestRun | None:
    with session_scope() as s:
        return s.get(BacktestRun, run_id)


def list_runs(limit: int = 20) -> list[dict]:
    with session_scope() as s:
        runs = s.scalars(
            select(BacktestRun).order_by(BacktestRun.created_at.desc()).limit(limit)
        ).all()
        return [
            {
                "id": r.id,
                "name": r.name,
                "engine_type": r.engine_type,
                "is_walk_forward": r.is_walk_forward,
                "start_date": str(r.start_date),
                "end_date": str(r.end_date),
                "sharpe": r.sharpe,
                "sharpe_oos": r.sharpe_oos,
                "max_drawdown": r.max_drawdown,
                "total_return": r.total_return,
                "n_trades": r.n_trades,
                "runtime_ms": r.runtime_ms,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in runs
        ]


def delete_run(run_id: int) -> None:
    with session_scope() as s:
        s.execute(delete(Trade).where(Trade.run_id == run_id))
        s.execute(delete(PortfolioSnapshot).where(PortfolioSnapshot.run_id == run_id))
        s.execute(delete(WalkForwardFold).where(WalkForwardFold.run_id == run_id))
        s.execute(delete(BacktestRun).where(BacktestRun.id == run_id))
