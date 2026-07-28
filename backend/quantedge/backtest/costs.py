"""Transaction cost and slippage models.

Costs are what separate a backtest from a fantasy. A daily-rebalanced
long/short book turning over 100% a day at 6bps round-trip loses roughly 15%
a year to friction alone — enough to erase most factor alphas. Modelling them
explicitly is what makes the reported Sharpe defensible.

Both engines share these functions so the cost treatment cannot drift between
the naive and vectorized implementations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantedge.config import settings


@dataclass(frozen=True)
class CostModel:
    """Linear cost model applied to traded notional.

    ``commission_bps`` covers explicit fees; ``slippage_bps`` covers the
    spread and market impact of crossing. Both are per side, so a full
    round trip pays roughly twice.
    """

    commission_bps: float = settings.commission_bps
    slippage_bps: float = settings.slippage_bps

    @property
    def total_bps(self) -> float:
        return self.commission_bps + self.slippage_bps

    @property
    def rate(self) -> float:
        return self.total_bps / 10_000.0

    def cost_of_turnover(self, turnover: float) -> float:
        """Cost as a fraction of portfolio value for a given turnover."""
        return abs(turnover) * self.rate

    def apply(self, weights_new: pd.Series, weights_old: pd.Series) -> float:
        """Cost of moving from one weight vector to another."""
        aligned_old = weights_old.reindex(weights_new.index).fillna(0.0)
        turnover = float((weights_new - aligned_old).abs().sum())
        return self.cost_of_turnover(turnover)

    def apply_panel(self, weights: pd.DataFrame) -> pd.Series:
        """Per-date cost for a full weight panel (index=date, columns=ticker)."""
        turnover = weights.fillna(0.0).diff().abs().sum(axis=1)
        # The first rebalance builds the whole book from cash.
        if len(weights):
            turnover.iloc[0] = float(weights.iloc[0].abs().sum())
        return turnover * self.rate

    def describe(self) -> dict:
        return {
            "commission_bps": self.commission_bps,
            "slippage_bps": self.slippage_bps,
            "total_bps_per_side": self.total_bps,
        }


@dataclass(frozen=True)
class VolumeAwareSlippage:
    """Slippage that scales with participation in daily volume.

    Trading 20% of a name's ADV moves the price against you far more than
    trading 0.1%. Included to show the effect is understood; the headline
    backtest uses the simpler linear model, and the participation cap keeps
    position sizes in a range where the linear approximation is reasonable.
    """

    base_bps: float = settings.slippage_bps
    impact_coefficient: float = 0.1
    max_participation: float = 0.05

    def slippage_bps(self, notional: float, adv: float) -> float:
        if adv <= 0:
            return self.base_bps * 3  # unknown liquidity: assume the worst
        participation = min(notional / adv, self.max_participation)
        # Square-root market impact is the standard practitioner form.
        impact = self.impact_coefficient * np.sqrt(participation) * 10_000
        return self.base_bps + impact


DEFAULT_COST_MODEL = CostModel()
