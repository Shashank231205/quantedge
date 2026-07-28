"""Cross-sectional price momentum.

The canonical academic construction (Jegadeesh & Titman 1993; Carhart 1997)
is 12-month return skipping the most recent month — "12-1". The skip matters:
the latest month carries short-term reversal, which is the opposite effect and
partially cancels momentum if left in.
"""

from __future__ import annotations

import pandas as pd

from quantedge.factors.base import Factor


class MomentumFactor(Factor):
    """12-1 momentum: trailing 12-month return, skipping the last month."""

    name = "momentum"
    higher_is_better = True

    def __init__(self, lookback: int = 252, skip: int = 21) -> None:
        self.lookback = lookback
        self.skip = skip
        self.min_periods = lookback + 1

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # Both endpoints are shifted, so the window is [t-252, t-21] and the
        # value at t uses no information from after t.
        start = prices.shift(self.lookback)
        end = prices.shift(self.skip)
        return (end / start) - 1.0


class ShortTermReversalFactor(Factor):
    """1-month reversal: recent winners tend to give back short-horizon gains."""

    name = "st_reversal"
    higher_is_better = False  # high recent return -> expect reversal -> short

    def __init__(self, lookback: int = 21) -> None:
        self.lookback = lookback
        self.min_periods = lookback + 1

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return prices.pct_change(self.lookback)


class RiskAdjustedMomentumFactor(Factor):
    """Momentum scaled by realised volatility ("frog-in-the-pan" style).

    Dividing by path volatility favours names that trended smoothly over those
    that arrived at the same return through violent swings, which historically
    improves the factor's risk-adjusted profile.
    """

    name = "momentum_risk_adj"
    higher_is_better = True

    def __init__(self, lookback: int = 252, skip: int = 21, vol_window: int = 126) -> None:
        self.lookback = lookback
        self.skip = skip
        self.vol_window = vol_window
        self.min_periods = lookback + 1

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raw = (prices.shift(self.skip) / prices.shift(self.lookback)) - 1.0
        daily = prices.pct_change()
        vol = daily.shift(self.skip).rolling(
            self.vol_window, min_periods=self.vol_window // 2
        ).std()
        return raw / vol.replace(0.0, pd.NA)
