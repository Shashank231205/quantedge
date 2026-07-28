"""Full metrics report — the single entry point for scoring a backtest.

Assembles every metric the UI and README display. Kept in one place so that
in-sample, out-of-sample and per-fold results are always computed the same
way; divergent metric code is how a backtest ends up quoting an OOS Sharpe
that was never comparable to its IS counterpart.
"""

from __future__ import annotations

import pandas as pd

from quantedge.config import settings
from quantedge.metrics import drawdown as dd
from quantedge.metrics import performance as perf
from quantedge.metrics.attribution import return_decomposition
from quantedge.metrics.deflated_sharpe import deflated_sharpe_ratio
from quantedge.metrics.trades import trade_statistics, turnover_statistics
from quantedge.risk.var import var_report

TRADING_DAYS = settings.trading_days_per_year


def full_report(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    trades: pd.DataFrame | None = None,
    turnover: pd.Series | None = None,
    n_trials: int = 1,
    portfolio_value: float = 1_000_000.0,
    label: str = "",
) -> dict:
    """Compute every metric for one return series."""
    r = returns.replace([float("inf"), float("-inf")], pd.NA).dropna()

    report: dict = {
        "label": label,
        "n_observations": int(len(r)),
        "start_date": str(r.index[0].date()) if len(r) else None,
        "end_date": str(r.index[-1].date()) if len(r) else None,
        "years": round(len(r) / TRADING_DAYS, 2) if len(r) else 0.0,
    }

    if len(r) < 2:
        report["note"] = "insufficient observations"
        return report

    # --- returns --------------------------------------------------------
    report["returns"] = {
        "total_return": round(perf.total_return(r), 6),
        "cagr": round(perf.cagr(r), 6),
        "annualized_return": round(perf.annualized_return(r), 6),
        "best_day": round(perf.best_worst(r)[0], 6),
        "worst_day": round(perf.best_worst(r)[1], 6),
        "hit_rate_daily": round(perf.hit_rate(r), 4),
    }

    # --- risk-adjusted ----------------------------------------------------
    report["risk_adjusted"] = {
        "sharpe_ratio": round(perf.sharpe_ratio(r), 4),
        "sortino_ratio": round(perf.sortino_ratio(r), 4),
        "calmar_ratio": round(perf.calmar_ratio(r), 4),
        "omega_ratio": round(perf.omega_ratio(r), 4),
    }

    # --- risk ---------------------------------------------------------------
    report["risk"] = {
        "annualized_volatility": round(perf.annualized_volatility(r), 6),
        "downside_deviation": round(
            float(r[r < 0].std(ddof=1)) * (TRADING_DAYS**0.5), 6
        )
        if (r < 0).sum() > 1
        else 0.0,
        "max_drawdown": round(dd.max_drawdown(r), 6),
        "avg_drawdown": round(dd.average_drawdown(r), 6),
        "max_drawdown_duration_days": dd.max_drawdown_duration(r),
        "time_underwater_pct": round(dd.time_underwater(r) * 100, 2),
        "ulcer_index": round(dd.ulcer_index(r), 6),
        "skewness": round(perf.skewness(r), 4),
        "excess_kurtosis": round(perf.kurtosis(r), 4),
    }
    report["var"] = var_report(r, portfolio_value)

    # --- benchmark-relative --------------------------------------------------
    if benchmark is not None and len(benchmark.dropna()) > 30:
        report["attribution"] = return_decomposition(r, benchmark)
        report["benchmark_relative"] = {
            "information_ratio": round(perf.information_ratio(r, benchmark), 4),
            "tracking_error": round(perf.tracking_error(r, benchmark), 6),
            "benchmark_return": round(perf.total_return(benchmark.dropna()), 6),
            "benchmark_sharpe": round(perf.sharpe_ratio(benchmark.dropna()), 4),
            "excess_return": round(
                perf.total_return(r) - perf.total_return(benchmark.dropna()), 6
            ),
        }

    # --- trades / turnover ----------------------------------------------------
    if trades is not None and not trades.empty:
        report["trades"] = trade_statistics(trades)
    if turnover is not None and not turnover.empty:
        report["turnover"] = turnover_statistics(turnover)

    # --- multiple-testing honesty ----------------------------------------------
    report["deflated_sharpe"] = deflated_sharpe_ratio(r, n_trials=n_trials)

    # --- drawdown episodes ------------------------------------------------------
    episodes = dd.drawdown_details(r, top_n=5)
    if not episodes.empty:
        report["worst_drawdowns"] = [
            {
                "peak_date": str(pd.Timestamp(row.peak_date).date()),
                "trough_date": str(pd.Timestamp(row.trough_date).date()),
                "recovery_date": str(pd.Timestamp(row.recovery_date).date())
                if row.recovery_date is not None and pd.notna(row.recovery_date)
                else None,
                "max_drawdown": round(row.max_drawdown, 6),
                "duration_days": int(row.duration_days),
                "recovered": bool(row.recovered),
            }
            for row in episodes.itertuples()
        ]

    # --- periodic ------------------------------------------------------------------
    yearly = perf.yearly_returns(r)
    if not yearly.empty:
        report["yearly_returns"] = {
            str(k): round(float(v), 6) for k, v in yearly.items()
        }

    return report


def compare_is_oos(
    is_returns: pd.Series,
    oos_returns: pd.Series,
    benchmark: pd.Series | None = None,
    n_trials: int = 1,
) -> dict:
    """Side-by-side in-sample vs out-of-sample.

    Reported together, always. A large IS/OOS gap is the signature of
    overfitting, and hiding it by quoting only one number is the failure mode
    this whole project is built to avoid.
    """
    is_rep = full_report(is_returns, benchmark, n_trials=n_trials, label="in_sample")
    oos_rep = full_report(oos_returns, benchmark, n_trials=n_trials, label="out_of_sample")

    is_sharpe = is_rep.get("risk_adjusted", {}).get("sharpe_ratio", 0.0)
    oos_sharpe = oos_rep.get("risk_adjusted", {}).get("sharpe_ratio", 0.0)

    degradation = (
        round((1.0 - oos_sharpe / is_sharpe) * 100, 2) if abs(is_sharpe) > 1e-9 else None
    )

    return {
        "in_sample": is_rep,
        "out_of_sample": oos_rep,
        "comparison": {
            "sharpe_is": is_sharpe,
            "sharpe_oos": oos_sharpe,
            "sharpe_degradation_pct": degradation,
            "verdict": _verdict(is_sharpe, oos_sharpe),
        },
    }


def _verdict(is_sharpe: float, oos_sharpe: float) -> str:
    if abs(is_sharpe) < 1e-9:
        return "No in-sample edge to evaluate."
    ratio = oos_sharpe / is_sharpe
    if ratio >= 0.8:
        return "Out-of-sample performance holds up well; little evidence of overfitting."
    if ratio >= 0.5:
        return "Moderate out-of-sample decay — typical for equity factor strategies."
    if ratio > 0:
        return "Substantial out-of-sample decay; the in-sample result is largely optimistic."
    return "No out-of-sample edge. The in-sample result does not generalise."
