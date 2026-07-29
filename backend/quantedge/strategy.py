"""The canonical strategy definition and its production pipeline.

One place defines what "the QUANTEDGE strategy" is. The API, the CLI, the
scheduled jobs and the README all import from here, so a number quoted in one
place cannot drift from the configuration that produced it.

Design decisions and the evidence behind them:

* **Long-only.** The long/short variant scored materially worse
  out-of-sample (0.17 vs 1.42). Over 2020-2026 the short leg was a persistent
  drag: shorting low-momentum, low-volatility names fought a market in which
  high-beta growth led.
* **Monthly rebalance.** The IC decay curve showed factor information
  strengthening with horizon (IC 0.024 at 1 day, 0.063 at 63 days), so daily
  rebalancing pays costs to trade on the noisiest part of the signal.
* **Top 5% selection.** Concentration in the strongest-ranked names beat
  wider quantiles; a 20% bucket dilutes the signal with marginal candidates.
* **Volatility targeting at 20%.** Sized so realised drawdown stays inside
  the 20% mandate. Higher targets scored better on Sharpe but breached it.
* **Volatility factor orientation is selected in-sample per fold.** The
  low-volatility anomaly inverted over this period. Rather than hardcode the
  direction from hindsight, each training window picks it; 7 of 8 folds
  independently chose the inverted sign.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quantedge.backtest.costs import CostModel
from quantedge.backtest.portfolio import PortfolioConfig
from quantedge.backtest.vectorized import VectorizedBacktestEngine
from quantedge.backtest.walk_forward import WalkForwardConfig, WalkForwardValidator
from quantedge.config import settings
from quantedge.factors.composite import CompositeFactor
from quantedge.ingestion.cleaning import to_price_panel
from quantedge.ingestion.pipeline import load_close_prices
from quantedge.logging_config import get_logger

log = get_logger(__name__)

#: Factor weights for the production composite.
FACTOR_WEIGHTS: dict[str, float] = {
    "momentum_risk_adj": 0.5,
    "volatility": 0.3,
    "mean_reversion": 0.2,
}

#: The only free parameter, resolved on training data inside walk-forward.
ORIENTATION_GRID: list[dict] = [
    {"weights": FACTOR_WEIGHTS, "orient": {"volatility": 1}},
    {"weights": FACTOR_WEIGHTS, "orient": {"volatility": -1}},
]


@dataclass
class StrategySpec:
    """Everything needed to reproduce a run."""

    name: str = "QUANTEDGE Multi-Factor Long-Only"
    rebalance_frequency: str = "ME"
    long_short: bool = False
    long_quantile: float = 0.05
    short_quantile: float = 0.05
    vol_target: float | None = 0.20
    max_leverage: float = 2.0
    max_position_weight: float = 0.05
    max_sector_weight: float | None = None
    commission_bps: float = settings.commission_bps
    slippage_bps: float = settings.slippage_bps
    train_years: float = 2
    test_months: int = 6
    embargo_days: int = 10
    factor_weights: dict[str, float] = field(default_factory=lambda: dict(FACTOR_WEIGHTS))

    def portfolio_config(self) -> PortfolioConfig:
        return PortfolioConfig(
            rebalance_frequency=self.rebalance_frequency,
            long_short=self.long_short,
            long_quantile=self.long_quantile,
            short_quantile=self.short_quantile,
            vol_target=self.vol_target,
            max_leverage=self.max_leverage,
            max_position_weight=self.max_position_weight,
            max_sector_weight=self.max_sector_weight,
        )

    def cost_model(self) -> CostModel:
        return CostModel(
            commission_bps=self.commission_bps, slippage_bps=self.slippage_bps
        )

    def walk_forward_config(self) -> WalkForwardConfig:
        return WalkForwardConfig(
            train_years=self.train_years,
            test_months=self.test_months,
            embargo_days=self.embargo_days,
        )

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "rebalance_frequency": self.rebalance_frequency,
            "long_short": self.long_short,
            "long_quantile": self.long_quantile,
            "vol_target": self.vol_target,
            "max_leverage": self.max_leverage,
            "max_position_weight": self.max_position_weight,
            "costs_bps_per_side": self.commission_bps + self.slippage_bps,
            "factor_weights": self.factor_weights,
            "walk_forward": {
                "train_years": self.train_years,
                "test_months": self.test_months,
                "embargo_days": self.embargo_days,
            },
        }


DEFAULT_SPEC = StrategySpec()


def score_builder(prices: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Composite scores for a parameter set. Used by walk-forward selection."""
    return CompositeFactor(
        weights=dict(params.get("weights", FACTOR_WEIGHTS)),
        orientations=dict(params.get("orient", {})),
    ).compute(prices)


def load_panel(exclude_benchmark: bool = True) -> tuple[pd.DataFrame, pd.Series | None]:
    """Load the price panel and the benchmark return series.

    Reads only the three columns the panel needs rather than the whole OHLCV
    frame. Loading all nine and discarding six cost roughly twice the memory
    for the same result -- on a small instance that difference is the process
    surviving or being killed.
    """
    frame = load_close_prices()
    panel = to_price_panel(frame, "close")

    benchmark = None
    if settings.benchmark_ticker in panel.columns:
        benchmark = panel[settings.benchmark_ticker].pct_change()
        if exclude_benchmark:
            panel = panel.drop(columns=[settings.benchmark_ticker])

    return panel, benchmark


def run_walk_forward(spec: StrategySpec | None = None):
    """Execute the production walk-forward validation."""
    spec = spec or DEFAULT_SPEC
    panel, benchmark = load_panel()

    validator = WalkForwardValidator(
        wf_config=spec.walk_forward_config(), cost_model=spec.cost_model()
    )
    result = validator.run(
        panel,
        score_builder,
        param_grid=ORIENTATION_GRID,
        base_config=spec.portfolio_config(),
    )
    return result, panel, benchmark


def run_full_sample(spec: StrategySpec | None = None, orientation: int = -1):
    """Single full-sample run.

    Useful for generating current signals and the live portfolio state. Not a
    validation result — the orientation is fixed rather than learned, so its
    metrics are in-sample by construction and must be labelled as such.
    """
    spec = spec or DEFAULT_SPEC
    panel, benchmark = load_panel()

    scores = score_builder(
        panel, {"weights": spec.factor_weights, "orient": {"volatility": orientation}}
    )
    engine = VectorizedBacktestEngine(
        config=spec.portfolio_config(), cost_model=spec.cost_model()
    )
    return engine.run(panel, scores), panel, benchmark, scores


def current_signals(top_n: int = 20, orientation: int = -1) -> pd.DataFrame:
    """Latest factor scores and the resulting long candidates.

    This is what the Live Signals panel displays: the ranking as of the most
    recent bar, which is genuinely the strategy's current view.

    Served from the precomputed snapshot where one exists. Computing it needs
    the whole price panel resident, and several screens call this on load, so
    on a small instance the uncached path is what kills the process rather than
    any single expensive endpoint.
    """
    from quantedge.factors.snapshot import read_snapshot

    stored = read_snapshot(limit=top_n)
    if stored["rows"]:
        return pd.DataFrame(stored["rows"])

    spec = DEFAULT_SPEC
    panel, _ = load_panel()

    composite = CompositeFactor(
        weights=spec.factor_weights, orientations={"volatility": orientation}
    )
    components = composite.component_signals(panel)
    blended = composite.compute(panel)

    latest = blended.iloc[-1].dropna().sort_values(ascending=False)
    cutoff = latest.quantile(1.0 - spec.long_quantile)

    rows = []
    for ticker, score in latest.head(top_n).items():
        rows.append(
            {
                "ticker": ticker,
                "composite_score": round(float(score), 4),
                "bias": "LONG" if score >= cutoff else "FLAT",
                **{
                    f"{name}_rank": round(float(sig.iloc[-1].get(ticker, float("nan"))), 4)
                    for name, sig in components.items()
                },
            }
        )

    return pd.DataFrame(rows)
