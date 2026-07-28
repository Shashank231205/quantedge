"""Scheduled jobs.

Every job runs inside ``track_job``, so the uptime figure reported by the API
and the README is the observed success rate of these executions — not a
constant. Failures are recorded rather than swallowed, which is the point:
a pipeline that claims 100% uptime while silently erroring is worse than one
that admits 92%.

Cadence follows the data. US daily OHLCV settles once after the close, so
ingestion runs on a weekday evening schedule; anything faster would re-fetch
identical bars.
"""

from __future__ import annotations

import signal
import time

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from quantedge.logging_config import get_logger

log = get_logger(__name__)


def job_daily_ingest(lookback_days: int = 7) -> None:
    """Top up recent bars.

    The window deliberately overlaps already-stored data so vendor
    restatements and late corrections are picked up on the next run.
    """
    from quantedge.ingestion.pipeline import ingest_incremental

    ingest_incremental(lookback_days=lookback_days)


def job_weekly_universe() -> None:
    """Refresh index membership, including names added or removed this week."""
    from quantedge.ingestion.pipeline import refresh_universe

    refresh_universe()


def job_nightly_factors() -> None:
    """Recompute factor diagnostics against the latest data."""
    from quantedge.factors.composite import CORE_FACTORS, get_factor
    from quantedge.factors.diagnostics import ic_decay
    from quantedge.ingestion.telemetry import track_job
    from quantedge.strategy import load_panel

    with track_job("factor_compute") as ctx:
        panel, _ = load_panel()
        for name in CORE_FACTORS:
            ic_decay(get_factor(name).signal(panel), panel, factor=name)
        ctx.rows_written = len(panel) * len(CORE_FACTORS)
        ctx.tickers_processed = panel.shape[1]


def job_nightly_backtest() -> None:
    """Re-validate the strategy and store the run."""
    import pandas as pd

    from quantedge.backtest.naive import BacktestResult
    from quantedge.backtest.persistence import save_backtest_run
    from quantedge.ingestion.telemetry import track_job
    from quantedge.metrics.report import compare_is_oos
    from quantedge.metrics.trades import extract_trades, trade_statistics, turnover_statistics
    from quantedge.strategy import DEFAULT_SPEC, run_walk_forward

    with track_job("backtest_nightly") as ctx:
        spec = DEFAULT_SPEC
        wf, panel, benchmark = run_walk_forward(spec)
        if wf.oos_returns.empty:
            raise RuntimeError("walk-forward produced no out-of-sample data")

        report = compare_is_oos(
            wf.is_returns, wf.oos_returns, benchmark, n_trials=wf.n_configurations_tested
        )
        equity = (1.0 + wf.oos_returns).cumprod() * 1_000_000
        result = BacktestResult(
            returns=wf.oos_returns, equity_curve=equity, weights=wf.oos_weights,
            turnover=wf.oos_turnover, costs=pd.Series(0.0, index=wf.oos_returns.index),
            gross_exposure=wf.oos_weights.abs().sum(axis=1),
            net_exposure=wf.oos_weights.sum(axis=1), engine="vectorized",
            metadata={
                "config": spec.as_dict(), "costs": spec.cost_model().describe(),
                "n_bars": len(wf.oos_returns), "n_tickers": panel.shape[1],
            },
        )
        metrics = dict(report["out_of_sample"])
        metrics["comparison"] = report["comparison"]
        metrics["in_sample"] = report["in_sample"]
        if not wf.oos_turnover.empty:
            metrics["turnover"] = turnover_statistics(wf.oos_turnover)
        trades = extract_trades(wf.oos_weights, panel)
        if not trades.empty:
            metrics["trades"] = trade_statistics(trades)

        run_id = save_backtest_run(
            f"{spec.name} (walk-forward OOS)", result, metrics,
            prices=panel, walk_forward=wf, is_walk_forward=True,
        )
        ctx.rows_written = len(wf.oos_returns)
        ctx.tickers_processed = panel.shape[1]
        ctx.details = {"run_id": run_id, "oos_sharpe": report["comparison"]["sharpe_oos"]}


def build_scheduler(ingest_hour: int = 22) -> BackgroundScheduler:
    """Wire the job schedule. Times are UTC."""
    scheduler = BackgroundScheduler(timezone="UTC")

    # Weekdays after the US close; markets are shut at weekends.
    scheduler.add_job(
        job_daily_ingest, CronTrigger(day_of_week="mon-fri", hour=ingest_hour, minute=0),
        id="ohlcv_ingest", max_instances=1, coalesce=True,
    )
    scheduler.add_job(
        job_nightly_factors,
        CronTrigger(day_of_week="mon-fri", hour=(ingest_hour + 1) % 24, minute=0),
        id="factor_compute", max_instances=1, coalesce=True,
    )
    scheduler.add_job(
        job_nightly_backtest,
        CronTrigger(day_of_week="mon-fri", hour=(ingest_hour + 2) % 24, minute=0),
        id="backtest_nightly", max_instances=1, coalesce=True,
    )
    # Index changes are announced infrequently; weekly is ample.
    scheduler.add_job(
        job_weekly_universe, CronTrigger(day_of_week="sat", hour=6, minute=0),
        id="universe_refresh", max_instances=1, coalesce=True,
    )
    return scheduler


def run_scheduler(ingest_hour: int = 22) -> None:
    """Start the scheduler and block until interrupted."""
    scheduler = build_scheduler(ingest_hour)
    scheduler.start()

    for job in scheduler.get_jobs():
        log.info("scheduler.job id=%s next_run=%s", job.id, job.next_run_time)

    stopping = False

    def shutdown(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    log.info("scheduler.started jobs=%s", len(scheduler.get_jobs()))
    try:
        while not stopping:
            time.sleep(1)
    finally:
        scheduler.shutdown(wait=False)
        log.info("scheduler.stopped")
