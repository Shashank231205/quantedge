"""Naive loop-based backtest engine — the optimisation baseline.

This is deliberately written the way a backtest is usually first written:
an explicit Python loop over dates, and an inner loop over tickers, with
per-cell ``.loc`` lookups. It is not strawman code — the logic is correct and
it produces exactly the same results as the vectorized engine (enforced by
``tests/test_engine_parity.py``). It is simply the obvious implementation
before anyone thinks about NumPy.

Its purpose is to be the reference point for the runtime comparison. A
speedup measured against an intentionally crippled baseline would be
meaningless; this one is honest, which is why the parity test matters as much
as the benchmark.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from quantedge.backtest.costs import DEFAULT_COST_MODEL, CostModel
from quantedge.backtest.portfolio import (
    PortfolioConfig,
    build_weights,
    rebalance_dates,
    volatility_scalar,
)
from quantedge.config import settings
from quantedge.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class BacktestResult:
    """Output of either engine. Both must produce identical values."""

    returns: pd.Series
    equity_curve: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    gross_exposure: pd.Series
    net_exposure: pd.Series
    runtime_seconds: float = 0.0
    engine: str = ""
    n_rebalances: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def total_return(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        return float(self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1.0)


class NaiveBacktestEngine:
    """Row-by-row simulation with per-ticker inner loops."""

    engine_name = "naive"

    def __init__(
        self,
        config: PortfolioConfig | None = None,
        cost_model: CostModel | None = None,
        trading_days: int = settings.trading_days_per_year,
    ) -> None:
        self.config = config or PortfolioConfig()
        self.costs = cost_model or DEFAULT_COST_MODEL
        self.trading_days = trading_days

    def run(
        self,
        prices: pd.DataFrame,
        scores: pd.DataFrame,
        sectors: pd.Series | None = None,
        initial_capital: float = 1_000_000.0,
    ) -> BacktestResult:
        start_time = time.perf_counter()

        dates = prices.index
        tickers = list(prices.columns)
        rebal = set(rebalance_dates(dates, self.config.rebalance_frequency))

        current_weights = pd.Series(0.0, index=tickers, dtype=float)
        equity = initial_capital

        hist_returns: list[float] = []
        out_returns: list[float] = []
        out_equity: list[float] = []
        out_turnover: list[float] = []
        out_costs: list[float] = []
        out_gross: list[float] = []
        out_net: list[float] = []
        weight_rows: list[pd.Series] = []
        n_rebalances = 0

        for i in range(len(dates)):
            dt = dates[i]

            # --- mark the existing book to market -------------------------
            day_return = 0.0
            if i > 0:
                prev = dates[i - 1]
                for ticker in tickers:  # deliberate per-ticker inner loop
                    w = current_weights[ticker]
                    if w == 0.0:
                        continue
                    p_now = prices.loc[dt, ticker]
                    p_prev = prices.loc[prev, ticker]
                    if pd.isna(p_now) or pd.isna(p_prev) or p_prev == 0:
                        continue
                    day_return += w * (p_now / p_prev - 1.0)

            # --- rebalance -------------------------------------------------
            cost = 0.0
            turnover = 0.0
            if dt in rebal and dt in scores.index:
                row = scores.loc[dt]
                target = build_weights(row, self.config, sectors)

                if self.config.vol_target and len(hist_returns) > 20:
                    scalar = volatility_scalar(
                        pd.Series(hist_returns),
                        self.config.vol_target,
                        self.config.vol_lookback,
                        self.trading_days,
                        self.config.max_leverage,
                    )
                    target = target * scalar

                target = target.reindex(tickers).fillna(0.0)

                for ticker in tickers:  # deliberate per-ticker inner loop
                    turnover += abs(target[ticker] - current_weights[ticker])

                cost = self.costs.cost_of_turnover(turnover)
                current_weights = target
                n_rebalances += 1

            net_return = day_return - cost
            equity *= 1.0 + net_return
            hist_returns.append(net_return)

            gross = 0.0
            net = 0.0
            for ticker in tickers:  # deliberate per-ticker inner loop
                w = current_weights[ticker]
                gross += abs(w)
                net += w

            out_returns.append(net_return)
            out_equity.append(equity)
            out_turnover.append(turnover)
            out_costs.append(cost)
            out_gross.append(gross)
            out_net.append(net)
            weight_rows.append(current_weights.copy())

        runtime = time.perf_counter() - start_time
        log.info(
            "naive.done bars=%s rebalances=%s runtime=%.2fs",
            len(dates), n_rebalances, runtime,
        )

        return BacktestResult(
            returns=pd.Series(out_returns, index=dates, name="returns"),
            equity_curve=pd.Series(out_equity, index=dates, name="equity"),
            weights=pd.DataFrame(weight_rows, index=dates),
            turnover=pd.Series(out_turnover, index=dates, name="turnover"),
            costs=pd.Series(out_costs, index=dates, name="costs"),
            gross_exposure=pd.Series(out_gross, index=dates, name="gross"),
            net_exposure=pd.Series(out_net, index=dates, name="net"),
            runtime_seconds=runtime,
            engine=self.engine_name,
            n_rebalances=n_rebalances,
            metadata={
                "initial_capital": initial_capital,
                "config": self.config.as_dict(),
                "costs": self.costs.describe(),
                "n_bars": len(dates),
                "n_tickers": len(tickers),
            },
        )
