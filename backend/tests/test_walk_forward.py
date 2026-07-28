"""Walk-forward split correctness.

If train and test windows overlap — or sit adjacent with no embargo — the
out-of-sample Sharpe is contaminated and means nothing. These tests pin the
split algebra so that cannot happen silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantedge.backtest.walk_forward import (
    WalkForwardConfig,
    WalkForwardValidator,
    generate_folds,
)


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2018-01-01", periods=1500)


@pytest.fixture
def prices(dates) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    tickers = [f"T{i:02d}" for i in range(30)]
    steps = rng.normal(0.0004, 0.013, size=(len(dates), len(tickers)))
    return pd.DataFrame(
        100 * np.exp(np.cumsum(steps, axis=0)), index=dates, columns=tickers
    )


class TestFoldGeneration:
    def test_produces_multiple_folds(self, dates):
        folds = generate_folds(dates, WalkForwardConfig())
        assert len(folds) >= 2

    def test_train_always_precedes_test(self, dates):
        for f in generate_folds(dates, WalkForwardConfig()):
            assert f.train_start < f.train_end < f.test_start < f.test_end

    def test_embargo_gap_is_respected(self, dates):
        """The core protection: no training bar may touch the test window."""
        cfg = WalkForwardConfig(embargo_days=10)
        for f in generate_folds(dates, cfg):
            gap = dates.get_loc(f.test_start) - dates.get_loc(f.train_end)
            assert gap >= cfg.embargo_days, (
                f"fold {f.index} has only {gap} bars between train end and "
                "test start; information can leak across the boundary"
            )

    def test_zero_embargo_is_honoured_when_requested(self, dates):
        cfg = WalkForwardConfig(embargo_days=0)
        for f in generate_folds(dates, cfg):
            gap = dates.get_loc(f.test_start) - dates.get_loc(f.train_end)
            assert gap >= 0

    def test_test_windows_do_not_overlap(self, dates):
        """Each OOS bar must be counted exactly once."""
        folds = generate_folds(dates, WalkForwardConfig())
        for a, b in zip(folds, folds[1:], strict=False):
            assert a.test_end <= b.test_start

    def test_rolling_window_has_constant_train_length(self, dates):
        cfg = WalkForwardConfig(anchored=False, train_years=2)
        folds = generate_folds(dates, cfg)
        lengths = [
            dates.get_loc(f.train_end) - dates.get_loc(f.train_start) for f in folds
        ]
        assert max(lengths) - min(lengths) <= 1

    def test_anchored_window_grows(self, dates):
        cfg = WalkForwardConfig(anchored=True, train_years=2)
        folds = generate_folds(dates, cfg)
        if len(folds) > 1:
            assert all(f.train_start == folds[0].train_start for f in folds)
            lengths = [
                dates.get_loc(f.train_end) - dates.get_loc(f.train_start) for f in folds
            ]
            assert lengths == sorted(lengths)

    def test_insufficient_history_yields_no_folds(self):
        short = pd.bdate_range("2023-01-01", periods=50)
        assert generate_folds(short, WalkForwardConfig()) == []


class TestWalkForwardRun:
    @staticmethod
    def _builder(prices: pd.DataFrame, params: dict) -> pd.DataFrame:
        window = params.get("window", 20)
        return prices.pct_change(window).rank(axis=1, pct=True).shift(1)

    def test_produces_oos_returns(self, prices):
        result = WalkForwardValidator(
            WalkForwardConfig(train_years=1, test_months=6)
        ).run(prices, self._builder)
        assert len(result.oos_returns) > 0
        assert len(result.folds) >= 1

    def test_oos_index_is_unique_and_sorted(self, prices):
        """Concatenated OOS segments must not double-count any date."""
        result = WalkForwardValidator(
            WalkForwardConfig(train_years=1, test_months=6)
        ).run(prices, self._builder)
        idx = result.oos_returns.index
        assert idx.is_unique
        assert idx.is_monotonic_increasing

    def test_trial_count_reflects_grid_size(self, prices):
        """The deflated-Sharpe correction needs an honest trial count."""
        grid = [{"window": w} for w in (10, 20, 40)]
        result = WalkForwardValidator(
            WalkForwardConfig(train_years=1, test_months=6)
        ).run(prices, self._builder, param_grid=grid)
        assert result.n_configurations_tested == len(grid) * len(result.folds)

    def test_selection_records_chosen_params(self, prices):
        grid = [{"window": w} for w in (10, 40)]
        result = WalkForwardValidator(
            WalkForwardConfig(train_years=1, test_months=6)
        ).run(prices, self._builder, param_grid=grid)
        assert len(result.selected_params) == len(result.folds)
        for params in result.selected_params:
            assert params["window"] in (10, 40)

    def test_summary_is_populated(self, prices):
        result = WalkForwardValidator(
            WalkForwardConfig(train_years=1, test_months=6)
        ).run(prices, self._builder)
        summary = result.summary()
        assert summary["n_folds"] >= 1
        assert summary["oos_days"] > 0
        assert "oos_sharpe" in summary
