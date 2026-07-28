"""Vectorized backtest engine.

Same simulation as :mod:`quantedge.backtest.naive`, restructured so the work
happens inside NumPy rather than the Python interpreter:

* Returns for every ticker on every date are computed once as a single matrix
  operation instead of ``n_dates × n_tickers`` scalar lookups.
* Portfolio returns become one row-wise multiply-and-sum over that matrix.
* Turnover, costs, gross and net exposure are whole-column operations.

The one part that stays sequential is volatility targeting, because the
leverage applied at date ``t`` depends on realised portfolio volatility up to
``t``, which depends on the leverage applied earlier. That recursion is
genuine and cannot be vectorized away without changing the strategy — so it
is handled with an explicit loop over *rebalance* dates only (typically ~70
for a weekly schedule over six years) rather than over every bar.

``tests/test_engine_parity.py`` asserts this engine and the naive one agree
to floating-point tolerance; without that, a speedup claim is meaningless.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from quantedge.backtest.costs import DEFAULT_COST_MODEL, CostModel
from quantedge.backtest.naive import BacktestResult
from quantedge.backtest.portfolio import (
    PortfolioConfig,
    build_weights,
    rebalance_dates,
    volatility_scalar,
)
from quantedge.config import settings
from quantedge.logging_config import get_logger

log = get_logger(__name__)


class VectorizedBacktestEngine:
    """Matrix-based simulation. Results identical to the naive engine."""

    engine_name = "vectorized"

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
        tickers = prices.columns
        n_dates, n_tickers = len(dates), len(tickers)

        # --- one-shot return matrix ---------------------------------------
        price_matrix = prices.to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            returns_matrix = np.vstack(
                [
                    np.zeros((1, n_tickers)),
                    price_matrix[1:] / price_matrix[:-1] - 1.0,
                ]
            )
        # Missing or zero prior prices produce inf/NaN; a missing bar means
        # no return contribution, not a catastrophic one.
        returns_matrix[~np.isfinite(returns_matrix)] = 0.0

        # --- target weights on rebalance dates ----------------------------
        rebal_index = rebalance_dates(dates, self.config.rebalance_frequency)
        rebal_set = set(rebal_index)
        date_pos = {dt: i for i, dt in enumerate(dates)}

        target_matrix = np.zeros((n_dates, n_tickers))
        rebal_rows: list[int] = []

        for dt in rebal_index:
            if dt not in date_pos or dt not in scores.index:
                continue
            w = build_weights(scores.loc[dt], self.config, sectors)
            target_matrix[date_pos[dt]] = w.reindex(tickers).fillna(0.0).to_numpy()
            rebal_rows.append(date_pos[dt])

        weights_matrix = np.zeros((n_dates, n_tickers))
        portfolio_returns = np.zeros(n_dates)
        cost_vector = np.zeros(n_dates)
        turnover_vector = np.zeros(n_dates)

        vol_target = self.config.vol_target
        lookback = self.config.vol_lookback

        # --- simulation ----------------------------------------------------
        if not vol_target:
            # No feedback loop: hold target weights forward and settle the
            # entire path with array operations.
            held = np.zeros((n_dates, n_tickers))
            current = np.zeros(n_tickers)
            rebal_lookup = set(rebal_rows)
            for i in range(n_dates):
                if i in rebal_lookup:
                    current = target_matrix[i]
                held[i] = current

            weights_matrix = held
            # Weight from the *previous* bar earns today's return.
            prev_weights = np.vstack([np.zeros((1, n_tickers)), held[:-1]])
            gross_returns = np.einsum("ij,ij->i", prev_weights, returns_matrix)

            deltas = np.diff(held, axis=0, prepend=np.zeros((1, n_tickers)))
            turnover_vector = np.abs(deltas).sum(axis=1)
            cost_vector = turnover_vector * self.costs.rate
            portfolio_returns = gross_returns - cost_vector
        else:
            # Volatility targeting introduces a genuine dependency: leverage
            # at t is a function of realised vol of the levered series up to
            # t. Loop over rebalance dates only, vectorizing each segment.
            current = np.zeros(n_tickers)
            realized: list[float] = []
            rebal_rows_sorted = sorted(rebal_rows)
            next_rebal = {r: True for r in rebal_rows_sorted}

            for i in range(n_dates):
                if i > 0:
                    gross_r = float(current @ returns_matrix[i])
                else:
                    gross_r = 0.0

                cost = 0.0
                turnover = 0.0
                if i in next_rebal:
                    target = target_matrix[i]
                    if len(realized) > 20:
                        # Reuse the shared helper rather than reimplementing
                        # it: the two must agree on the std convention
                        # (ddof) or the engines silently apply different
                        # leverage. Enforced by test_engine_parity.
                        scalar = volatility_scalar(
                            pd.Series(realized),
                            vol_target,
                            lookback,
                            self.trading_days,
                            self.config.max_leverage,
                        )
                        target = target * scalar
                    turnover = float(np.abs(target - current).sum())
                    cost = turnover * self.costs.rate
                    current = target

                net_r = gross_r - cost
                realized.append(net_r)

                weights_matrix[i] = current
                portfolio_returns[i] = net_r
                cost_vector[i] = cost
                turnover_vector[i] = turnover

        equity = initial_capital * np.cumprod(1.0 + portfolio_returns)
        gross_exposure = np.abs(weights_matrix).sum(axis=1)
        net_exposure = weights_matrix.sum(axis=1)

        runtime = time.perf_counter() - start_time
        log.info(
            "vectorized.done bars=%s rebalances=%s runtime=%.3fs",
            n_dates, len(rebal_rows), runtime,
        )

        return BacktestResult(
            returns=pd.Series(portfolio_returns, index=dates, name="returns"),
            equity_curve=pd.Series(equity, index=dates, name="equity"),
            weights=pd.DataFrame(weights_matrix, index=dates, columns=tickers),
            turnover=pd.Series(turnover_vector, index=dates, name="turnover"),
            costs=pd.Series(cost_vector, index=dates, name="costs"),
            gross_exposure=pd.Series(gross_exposure, index=dates, name="gross"),
            net_exposure=pd.Series(net_exposure, index=dates, name="net"),
            runtime_seconds=runtime,
            engine=self.engine_name,
            n_rebalances=len(rebal_rows),
            metadata={
                "initial_capital": initial_capital,
                "config": self.config.as_dict(),
                "costs": self.costs.describe(),
                "n_bars": n_dates,
                "n_tickers": n_tickers,
            },
        )
