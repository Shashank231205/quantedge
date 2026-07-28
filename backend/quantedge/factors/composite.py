"""Multi-factor blending and the factor registry.

Signals are combined on the *rank* scale rather than the raw scale. Raw factor
values live on incompatible units — a momentum return of 0.4, an RSI of 70 and
an annualised vol of 0.25 cannot be averaged meaningfully. Percentile ranks
are unitless, bounded and robust to the outliers that survive cleaning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quantedge.factors.base import Factor, neutralize
from quantedge.factors.mean_reversion import (
    BollingerPositionFactor,
    MeanReversionFactor,
    RSIFactor,
)
from quantedge.factors.momentum import (
    MomentumFactor,
    RiskAdjustedMomentumFactor,
    ShortTermReversalFactor,
)
from quantedge.factors.volatility import (
    ATRFactor,
    BetaFactor,
    DownsideVolatilityFactor,
    RealizedVolatilityFactor,
)
from quantedge.logging_config import get_logger

log = get_logger(__name__)

#: Everything the engine knows how to compute.
FACTOR_REGISTRY: dict[str, type[Factor]] = {
    "momentum": MomentumFactor,
    "momentum_risk_adj": RiskAdjustedMomentumFactor,
    "st_reversal": ShortTermReversalFactor,
    "mean_reversion": MeanReversionFactor,
    "rsi": RSIFactor,
    "bollinger_pos": BollingerPositionFactor,
    "volatility": RealizedVolatilityFactor,
    "atr": ATRFactor,
    "downside_vol": DownsideVolatilityFactor,
    "beta": BetaFactor,
}

#: The three headline factors named in the project brief.
CORE_FACTORS = ("momentum", "mean_reversion", "volatility")


def get_factor(name: str, **kwargs) -> Factor:
    if name not in FACTOR_REGISTRY:
        raise KeyError(f"unknown factor {name!r}; known: {sorted(FACTOR_REGISTRY)}")
    return FACTOR_REGISTRY[name](**kwargs)


@dataclass
class CompositeFactor:
    """Weighted blend of individual factor signals.

    Each component contributes a percentile rank already oriented so that
    higher means "prefer long", so the blend is a simple weighted average.
    """

    weights: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(CORE_FACTORS, 1 / 3)
    )
    name: str = "composite"
    sector_neutral: bool = False

    def __post_init__(self) -> None:
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("composite weights must sum to a positive number")
        # Normalise so weights are interpretable regardless of input scale.
        self.weights = {k: v / total for k, v in self.weights.items()}

    def component_signals(
        self,
        prices: pd.DataFrame,
        extra: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Tradeable, direction-adjusted rank signal per component factor."""
        extra = extra or {}
        out: dict[str, pd.DataFrame] = {}
        for fname in self.weights:
            factor = get_factor(fname)
            out[fname] = factor.signal(prices, **extra)
        return out

    def compute(
        self,
        prices: pd.DataFrame,
        extra: dict[str, pd.DataFrame] | None = None,
        sectors: pd.Series | None = None,
    ) -> pd.DataFrame:
        signals = self.component_signals(prices, extra)

        blended: pd.DataFrame | None = None
        for fname, sig in signals.items():
            w = self.weights[fname]
            contribution = sig * w
            blended = contribution if blended is None else blended.add(contribution, fill_value=0.0)

        assert blended is not None, "composite requires at least one factor"

        # Only score names with a full set of components; a partial blend
        # silently changes the factor's meaning.
        valid = None
        for sig in signals.values():
            mask = sig.notna()
            valid = mask if valid is None else (valid & mask)
        blended = blended.where(valid)

        if self.sector_neutral and sectors is not None:
            blended = neutralize(blended, sectors)

        return blended

    def rank(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Final cross-sectional percentile rank used for selection."""
        return self.compute(prices, **kwargs).rank(axis=1, pct=True, na_option="keep")

    def describe(self) -> dict:
        return {
            "name": self.name,
            "weights": self.weights,
            "sector_neutral": self.sector_neutral,
            "components": [get_factor(f).describe() for f in self.weights],
        }
