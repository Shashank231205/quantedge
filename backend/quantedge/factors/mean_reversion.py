"""Short-horizon mean reversion.

Two complementary constructions:

* **Z-score** of price against its own moving average — how stretched a name
  is relative to its recent trading range.
* **RSI**, the classic bounded oscillator, which saturates rather than growing
  without limit and so behaves differently in violent moves.

Both are inverted: a stretched-high name is a short candidate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantedge.factors.base import Factor


class MeanReversionFactor(Factor):
    """Z-score of price vs. its rolling mean."""

    name = "mean_reversion"
    higher_is_better = False  # stretched high -> expect pullback -> short

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self.min_periods = window

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        mp = max(2, self.window // 2)
        mean = prices.rolling(self.window, min_periods=mp).mean()
        std = prices.rolling(self.window, min_periods=mp).std()
        return (prices - mean) / std.replace(0.0, np.nan)


class RSIFactor(Factor):
    """Wilder's Relative Strength Index (0-100)."""

    name = "rsi"
    higher_is_better = False  # overbought -> short

    def __init__(self, window: int = 14) -> None:
        self.window = window
        self.min_periods = window + 1

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        delta = prices.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        # Wilder smoothing is an EMA with alpha = 1/window.
        avg_gain = gain.ewm(alpha=1 / self.window, min_periods=self.window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / self.window, min_periods=self.window, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        # All-gain windows have zero average loss -> RSI is 100 by definition.
        return rsi.where(avg_loss != 0, 100.0).where(avg_gain.notna())


class BollingerPositionFactor(Factor):
    """Position within Bollinger bands, scaled to [-1, 1] at the band edges."""

    name = "bollinger_pos"
    higher_is_better = False

    def __init__(self, window: int = 20, n_std: float = 2.0) -> None:
        self.window = window
        self.n_std = n_std
        self.min_periods = window

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        mp = max(2, self.window // 2)
        mean = prices.rolling(self.window, min_periods=mp).mean()
        std = prices.rolling(self.window, min_periods=mp).std()
        band = (self.n_std * std).replace(0.0, np.nan)
        return (prices - mean) / band
