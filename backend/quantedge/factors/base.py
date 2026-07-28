"""Factor abstraction with lookahead protection built into the contract.

The single most damaging bug in a backtest is using information that was not
available at decision time. It inflates every downstream number and is
invisible unless you test for it specifically.

The defence here is structural rather than advisory:

* A factor's ``compute`` returns values *aligned to the bar they are derived
  from* — the value at row ``t`` uses data up to and including ``t``.
* ``compute_tradeable`` then shifts by one bar. A signal derived from the
  close of day ``t`` cannot inform a trade before day ``t+1``. Everything
  downstream consumes the shifted series.
* Every concrete factor is exercised by ``tests/test_no_lookahead.py``, which
  corrupts future bars and asserts past signals do not move.

Keeping the shift in one place means an individual factor author cannot
forget it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Factor(ABC):
    """Base class for cross-sectional equity factors.

    Subclasses implement :meth:`compute` on a wide price panel
    (index=date, columns=ticker) and return a same-shaped frame of raw
    factor values.
    """

    name: str = "base"
    #: Bars of history required before the factor produces a value.
    min_periods: int = 1
    #: True when a *higher* raw value should map to a *long* position.
    higher_is_better: bool = True

    @abstractmethod
    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Raw factor values, aligned to the bar they are computed from."""

    # -- tradeable view -------------------------------------------------

    def compute_tradeable(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Factor values shifted so row ``t`` is knowable before trading ``t``.

        This is the *only* view the portfolio construction layer should use.
        """
        return self.compute(prices, **kwargs).shift(1)

    # -- cross-sectional normalisation ------------------------------------

    @staticmethod
    def zscore(values: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
        """Standardise across tickers within each date.

        Row-wise (cross-sectional) rather than time-series: on any given day
        we care how a name ranks against its peers, not against its own past.
        """
        mean = values.mean(axis=1)
        std = values.std(axis=1, ddof=0)
        z = values.sub(mean, axis=0).div(std.replace(0.0, np.nan), axis=0)
        return z.clip(-clip, clip)

    @staticmethod
    def rank_pct(values: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional percentile rank in [0, 1].

        Rank is robust to the fat tails and outliers that survive cleaning,
        which is why portfolio construction ranks rather than z-scores.
        """
        return values.rank(axis=1, pct=True, na_option="keep")

    def signal(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Tradeable, direction-adjusted percentile rank.

        Always oriented so that a higher value means "prefer long", which
        lets the composite blend factors without sign bookkeeping.
        """
        raw = self.compute_tradeable(prices, **kwargs)
        ranked = self.rank_pct(raw)
        return ranked if self.higher_is_better else 1.0 - ranked

    def describe(self) -> dict:
        return {
            "name": self.name,
            "min_periods": self.min_periods,
            "higher_is_better": self.higher_is_better,
            "doc": (self.__doc__ or "").strip().split("\n")[0],
        }


def winsorize(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Clip each row to its own cross-sectional quantiles."""
    lo = df.quantile(lower, axis=1)
    hi = df.quantile(upper, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


def neutralize(values: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Demean a factor within groups (typically GICS sector).

    Without this a momentum factor can quietly become a bet on whichever
    sector happened to run, rather than on the factor itself.
    """
    aligned = groups.reindex(values.columns)
    out = values.copy()
    for group in aligned.dropna().unique():
        cols = aligned[aligned == group].index
        cols = [c for c in cols if c in out.columns]
        if len(cols) > 1:
            block = out[cols]
            out[cols] = block.sub(block.mean(axis=1), axis=0)
    return out
