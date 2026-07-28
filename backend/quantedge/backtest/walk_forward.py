"""Walk-forward validation.

A single backtest over the full history tells you how a strategy would have
done if you had known the right parameters in advance — which you did not.
Walk-forward answers the question that actually matters: fit on a training
window, trade the following out-of-sample window, roll forward, repeat.

Two details separate a real implementation from a decorative one:

* **The embargo.** Train and test windows must not touch. A 12-month momentum
  factor computed on the first test day uses prices from the training period;
  without a gap, information leaks across the boundary. We insert a
  configurable embargo (default 10 bars) between them.
* **Parameters are chosen using training data only.** The selection routine
  never sees the test window. Whatever it picks is then traded blind.

The OOS numbers this produces are the ones that get reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from quantedge.backtest.costs import DEFAULT_COST_MODEL, CostModel
from quantedge.backtest.portfolio import PortfolioConfig
from quantedge.backtest.vectorized import VectorizedBacktestEngine
from quantedge.config import settings
from quantedge.logging_config import get_logger
from quantedge.metrics import performance as perf
from quantedge.metrics.drawdown import max_drawdown

log = get_logger(__name__)


@dataclass
class Fold:
    """One train/test split."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def as_dict(self) -> dict:
        return {
            "fold": self.index,
            "train_start": str(self.train_start.date()),
            "train_end": str(self.train_end.date()),
            "test_start": str(self.test_start.date()),
            "test_end": str(self.test_end.date()),
            "train_days": None,
            "test_days": None,
        }


@dataclass
class WalkForwardConfig:
    train_years: float = settings.train_years
    test_months: int = settings.test_months
    embargo_days: int = settings.embargo_days
    #: Expanding keeps all history in the training window; rolling drops it.
    anchored: bool = False
    trading_days_per_year: int = settings.trading_days_per_year

    def as_dict(self) -> dict:
        return {
            "train_years": self.train_years,
            "test_months": self.test_months,
            "embargo_days": self.embargo_days,
            "anchored": self.anchored,
        }


def generate_folds(
    dates: pd.DatetimeIndex, config: WalkForwardConfig | None = None
) -> list[Fold]:
    """Build non-overlapping train/test splits with an embargo gap."""
    cfg = config or WalkForwardConfig()
    dates = pd.DatetimeIndex(sorted(dates))
    if len(dates) < 100:
        return []

    train_len = int(cfg.train_years * cfg.trading_days_per_year)
    test_len = int(cfg.test_months * cfg.trading_days_per_year / 12)
    embargo = int(cfg.embargo_days)

    folds: list[Fold] = []
    train_start_i = 0
    train_end_i = train_len
    idx = 0

    while True:
        test_start_i = train_end_i + embargo
        test_end_i = test_start_i + test_len

        if test_start_i >= len(dates):
            break
        test_end_i = min(test_end_i, len(dates) - 1)
        if test_end_i - test_start_i < 20:
            break  # a fold too short to score meaningfully

        folds.append(
            Fold(
                index=idx,
                train_start=dates[train_start_i],
                train_end=dates[train_end_i],
                test_start=dates[test_start_i],
                test_end=dates[test_end_i],
            )
        )

        idx += 1
        train_end_i = test_end_i
        if not cfg.anchored:
            train_start_i = max(0, train_end_i - train_len)

    return folds


@dataclass
class WalkForwardResult:
    folds: list[dict] = field(default_factory=list)
    oos_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    is_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    oos_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    oos_turnover: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    selected_params: list[dict] = field(default_factory=list)
    n_configurations_tested: int = 1
    config: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "n_folds": len(self.folds),
            "oos_days": len(self.oos_returns),
            "oos_sharpe": round(perf.sharpe_ratio(self.oos_returns), 4),
            "oos_max_drawdown": round(max_drawdown(self.oos_returns), 4),
            "is_sharpe": round(perf.sharpe_ratio(self.is_returns), 4),
            "n_configurations_tested": self.n_configurations_tested,
        }


class WalkForwardValidator:
    """Runs the fit-then-trade-forward loop."""

    def __init__(
        self,
        wf_config: WalkForwardConfig | None = None,
        cost_model: CostModel | None = None,
    ) -> None:
        self.wf_config = wf_config or WalkForwardConfig()
        self.cost_model = cost_model or DEFAULT_COST_MODEL

    def run(
        self,
        prices: pd.DataFrame,
        score_builder,
        param_grid: list[dict] | None = None,
        base_config: PortfolioConfig | None = None,
        sectors: pd.Series | None = None,
        selection_metric: str = "sharpe",
    ) -> WalkForwardResult:
        """Execute walk-forward validation.

        ``score_builder(prices, params) -> DataFrame`` produces the factor
        scores for a given parameter set. It is called separately for the
        training and test windows so that no test data influences selection.
        """
        base_config = base_config or PortfolioConfig()
        param_grid = param_grid or [{}]
        folds = generate_folds(prices.index, self.wf_config)

        if not folds:
            log.warning("walk_forward.no_folds n_dates=%s", len(prices.index))
            return WalkForwardResult(config=self.wf_config.as_dict())

        log.info(
            "walk_forward.start folds=%s configs=%s embargo=%s",
            len(folds), len(param_grid), self.wf_config.embargo_days,
        )

        oos_chunks: list[pd.Series] = []
        is_chunks: list[pd.Series] = []
        weight_chunks: list[pd.DataFrame] = []
        turnover_chunks: list[pd.Series] = []
        fold_records: list[dict] = []
        selected: list[dict] = []

        for fold in folds:
            # --- selection, using training data only ----------------------
            train_prices = prices.loc[fold.train_start : fold.train_end]
            best_params, best_score, train_returns = self._select(
                train_prices, score_builder, param_grid, base_config, sectors,
                selection_metric,
            )

            # --- trade the test window blind -------------------------------
            # Scores need history to warm up (momentum needs ~252 bars), so
            # the factor is computed on data up to the test end and then
            # sliced. The one-bar shift inside the factor layer keeps this
            # causal; the embargo keeps the *selection* clean.
            scores_full = score_builder(prices.loc[: fold.test_end], best_params)
            test_prices = prices.loc[fold.test_start : fold.test_end]
            test_scores = scores_full.reindex(test_prices.index)

            engine = VectorizedBacktestEngine(
                config=base_config, cost_model=self.cost_model
            )
            oos = engine.run(test_prices, test_scores, sectors)

            oos_chunks.append(oos.returns)
            weight_chunks.append(oos.weights)
            turnover_chunks.append(oos.turnover)
            if train_returns is not None:
                is_chunks.append(train_returns)

            record = fold.as_dict()
            record.update(
                {
                    "train_days": len(train_prices),
                    "test_days": len(test_prices),
                    "selected_params": best_params,
                    "train_score": round(best_score, 4),
                    "oos_sharpe": round(perf.sharpe_ratio(oos.returns), 4),
                    "oos_return": round(perf.total_return(oos.returns), 6),
                    "oos_max_drawdown": round(max_drawdown(oos.returns), 6),
                    "oos_volatility": round(perf.annualized_volatility(oos.returns), 6),
                }
            )
            fold_records.append(record)
            selected.append(best_params)

            log.info(
                "walk_forward.fold %s test=%s..%s train_%s=%.3f oos_sharpe=%.3f",
                fold.index, record["test_start"], record["test_end"],
                selection_metric, best_score, record["oos_sharpe"],
            )

        # Training windows overlap between folds, so concatenated in-sample
        # returns carry duplicate dates. Keep the last observation per date:
        # the IS series is a diagnostic for comparison against OOS, and
        # double-counting bars would distort it.
        is_series = pd.concat(is_chunks) if is_chunks else pd.Series(dtype=float)
        if not is_series.empty:
            is_series = is_series[~is_series.index.duplicated(keep="last")].sort_index()

        return WalkForwardResult(
            folds=fold_records,
            oos_returns=pd.concat(oos_chunks) if oos_chunks else pd.Series(dtype=float),
            is_returns=is_series,
            oos_weights=pd.concat(weight_chunks) if weight_chunks else pd.DataFrame(),
            oos_turnover=pd.concat(turnover_chunks) if turnover_chunks else pd.Series(dtype=float),
            selected_params=selected,
            # Honest count for the deflated-Sharpe correction.
            n_configurations_tested=len(param_grid) * len(folds),
            config=self.wf_config.as_dict(),
        )

    def _select(
        self, train_prices, score_builder, param_grid, base_config, sectors, metric
    ):
        """Pick the best parameter set on the training window."""
        best_params: dict = param_grid[0]
        best_score = -np.inf
        best_returns: pd.Series | None = None

        for params in param_grid:
            try:
                scores = score_builder(train_prices, params)
            except Exception as exc:
                log.warning("walk_forward.score_failed params=%s error=%s", params, exc)
                continue

            engine = VectorizedBacktestEngine(
                config=base_config, cost_model=self.cost_model
            )
            res = engine.run(train_prices, scores, sectors)

            score = (
                perf.sharpe_ratio(res.returns)
                if metric == "sharpe"
                else perf.calmar_ratio(res.returns)
                if metric == "calmar"
                else perf.sortino_ratio(res.returns)
            )
            if np.isfinite(score) and score > best_score:
                best_score, best_params, best_returns = score, params, res.returns

        return best_params, (best_score if np.isfinite(best_score) else 0.0), best_returns
