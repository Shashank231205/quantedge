"""Volatility factors.

The low-volatility anomaly is one of the more durable findings in equity
research: low-risk stocks have historically delivered better risk-adjusted
returns than high-risk stocks, contradicting a naive CAPM reading. These
factors are therefore oriented so that *low* volatility is the preferred
(long) side.

Volatility also does double duty in this system — the risk layer reuses
:class:`RealizedVolatilityFactor` output for position sizing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantedge.factors.base import Factor


class RealizedVolatilityFactor(Factor):
    """Annualised standard deviation of daily returns."""

    name = "volatility"
    higher_is_better = False  # low-vol anomaly: prefer the calm names

    def __init__(self, window: int = 60, trading_days: int = 252) -> None:
        self.window = window
        self.trading_days = trading_days
        self.min_periods = window

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        returns = prices.pct_change()
        mp = max(5, self.window // 2)
        return returns.rolling(self.window, min_periods=mp).std() * np.sqrt(
            self.trading_days
        )


class ATRFactor(Factor):
    """Average True Range, normalised by price.

    True range accounts for overnight gaps, which a close-to-close standard
    deviation misses entirely. Requires high/low, so it takes the full OHLC
    panel rather than a close-only panel.
    """

    name = "atr"
    higher_is_better = False

    def __init__(self, window: int = 14) -> None:
        self.window = window
        self.min_periods = window + 1

    def compute(
        self,
        prices: pd.DataFrame,
        high: pd.DataFrame | None = None,
        low: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        if high is None or low is None:
            # Degrade to close-to-close range rather than failing; the caller
            # gets a usable, clearly-documented approximation.
            rng = prices.diff().abs()
        else:
            high = high.reindex_like(prices)
            low = low.reindex_like(prices)
            prev_close = prices.shift(1)
            rng = pd.concat(
                [
                    (high - low).stack(future_stack=True),
                    (high - prev_close).abs().stack(future_stack=True),
                    (low - prev_close).abs().stack(future_stack=True),
                ],
                axis=1,
            ).max(axis=1).unstack()

        atr = rng.ewm(alpha=1 / self.window, min_periods=self.window, adjust=False).mean()
        return atr / prices.replace(0.0, np.nan)


class DownsideVolatilityFactor(Factor):
    """Standard deviation of negative returns only.

    Investors care about downside, not symmetric dispersion; this is the
    denominator behind the Sortino ratio.
    """

    name = "downside_vol"
    higher_is_better = False

    def __init__(self, window: int = 60, trading_days: int = 252) -> None:
        self.window = window
        self.trading_days = trading_days
        self.min_periods = window

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        returns = prices.pct_change()
        downside = returns.where(returns < 0)
        mp = max(5, self.window // 4)
        return downside.rolling(self.window, min_periods=mp).std() * np.sqrt(
            self.trading_days
        )


class BetaFactor(Factor):
    """Rolling market beta against a benchmark series."""

    name = "beta"
    higher_is_better = False  # low-beta side of the same anomaly

    def __init__(self, window: int = 126) -> None:
        self.window = window
        self.min_periods = window

    def compute(
        self, prices: pd.DataFrame, benchmark: pd.Series | None = None, **kwargs
    ) -> pd.DataFrame:
        if benchmark is None:
            # Equal-weight universe return is a serviceable market proxy.
            benchmark = prices.pct_change().mean(axis=1)

        returns = prices.pct_change()
        bench = benchmark.reindex(returns.index)
        mp = max(20, self.window // 2)

        cov = returns.rolling(self.window, min_periods=mp).cov(bench)
        var = bench.rolling(self.window, min_periods=mp).var()
        return cov.div(var.replace(0.0, np.nan), axis=0)
