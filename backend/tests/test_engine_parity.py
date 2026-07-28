"""Engine parity: the vectorized engine must reproduce the naive one exactly.

The resume claims an ~80% runtime reduction from vectorization. That claim is
only meaningful if both engines simulate the *same strategy* — otherwise the
"speedup" could just be the fast engine skipping work or computing something
different.

These tests are therefore the foundation of the benchmark, not an accessory
to it. They compare returns, equity, weights, turnover and costs bar by bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantedge.backtest.costs import CostModel
from quantedge.backtest.naive import NaiveBacktestEngine
from quantedge.backtest.portfolio import PortfolioConfig
from quantedge.backtest.vectorized import VectorizedBacktestEngine

N_DAYS = 260
N_TICKERS = 40


@pytest.fixture
def market():
    """Deterministic price panel and cross-sectional scores."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2022-01-03", periods=N_DAYS)
    tickers = [f"T{i:02d}" for i in range(N_TICKERS)]

    steps = rng.normal(0.0003, 0.014, size=(N_DAYS, N_TICKERS))
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(steps, axis=0)), index=dates, columns=tickers
    )
    scores = pd.DataFrame(
        rng.uniform(0, 1, size=(N_DAYS, N_TICKERS)), index=dates, columns=tickers
    )
    return prices, scores


@pytest.fixture
def sectors():
    return pd.Series(
        {f"T{i:02d}": ["Tech", "Health", "Energy", "Financials"][i % 4]
         for i in range(N_TICKERS)}
    )


def _run_both(prices, scores, config, sectors=None):
    cost = CostModel(commission_bps=1.0, slippage_bps=5.0)
    naive = NaiveBacktestEngine(config=config, cost_model=cost).run(prices, scores, sectors)
    vec = VectorizedBacktestEngine(config=config, cost_model=cost).run(prices, scores, sectors)
    return naive, vec


def _assert_parity(naive, vec, tol=1e-9):
    pd.testing.assert_series_equal(
        naive.returns, vec.returns, check_names=False, rtol=tol, atol=tol
    )
    pd.testing.assert_series_equal(
        naive.equity_curve, vec.equity_curve, check_names=False, rtol=tol, atol=1e-6
    )
    pd.testing.assert_series_equal(
        naive.turnover, vec.turnover, check_names=False, rtol=tol, atol=tol
    )
    pd.testing.assert_series_equal(
        naive.costs, vec.costs, check_names=False, rtol=tol, atol=tol
    )
    pd.testing.assert_frame_equal(
        naive.weights.astype(float),
        vec.weights.astype(float),
        check_names=False, check_column_type=False, rtol=tol, atol=tol,
    )


class TestEngineParity:
    def test_no_vol_target_weekly(self, market):
        """The fully-vectorized path (no volatility feedback)."""
        prices, scores = market
        cfg = PortfolioConfig(vol_target=None, rebalance_frequency="W-FRI")
        _assert_parity(*_run_both(prices, scores, cfg))

    def test_with_vol_target_weekly(self, market):
        """The sequential path, where leverage depends on past returns."""
        prices, scores = market
        cfg = PortfolioConfig(vol_target=0.10, rebalance_frequency="W-FRI")
        _assert_parity(*_run_both(prices, scores, cfg))

    def test_daily_rebalance(self, market):
        prices, scores = market
        cfg = PortfolioConfig(vol_target=None, rebalance_frequency="D")
        _assert_parity(*_run_both(prices, scores, cfg))

    def test_monthly_rebalance(self, market):
        prices, scores = market
        cfg = PortfolioConfig(vol_target=None, rebalance_frequency="ME")
        _assert_parity(*_run_both(prices, scores, cfg))

    def test_long_only(self, market):
        prices, scores = market
        cfg = PortfolioConfig(long_short=False, vol_target=None)
        _assert_parity(*_run_both(prices, scores, cfg))

    def test_with_sector_caps(self, market, sectors):
        prices, scores = market
        cfg = PortfolioConfig(vol_target=None, max_sector_weight=0.25)
        _assert_parity(*_run_both(prices, scores, cfg, sectors))

    def test_score_weighted(self, market):
        prices, scores = market
        cfg = PortfolioConfig(equal_weight=False, vol_target=None)
        _assert_parity(*_run_both(prices, scores, cfg))

    def test_handles_missing_prices(self, market):
        """Delisted names leave NaN gaps; both engines must treat them alike."""
        prices, scores = market
        holed = prices.copy()
        holed.iloc[100:, 0] = np.nan  # a name that stops trading
        holed.iloc[:50, 1] = np.nan   # a name that lists late
        cfg = PortfolioConfig(vol_target=None)
        _assert_parity(*_run_both(holed, scores, cfg))


class TestEngineSemantics:
    """Properties that must hold regardless of implementation."""

    def test_first_bar_has_no_return(self, market):
        prices, scores = market
        cfg = PortfolioConfig(vol_target=None)
        _, vec = _run_both(prices, scores, cfg)
        # There is no prior bar to earn a return from.
        assert vec.returns.iloc[0] == pytest.approx(0.0, abs=1e-12)

    def test_dollar_neutral_book(self, market):
        prices, scores = market
        cfg = PortfolioConfig(long_short=True, vol_target=None, max_sector_weight=None)
        _, vec = _run_both(prices, scores, cfg)
        active = vec.net_exposure[vec.gross_exposure > 0]
        assert active.abs().max() < 1e-9, "long/short book should be dollar neutral"

    def test_costs_are_never_negative(self, market):
        prices, scores = market
        _, vec = _run_both(prices, scores, PortfolioConfig())
        assert (vec.costs >= 0).all()

    def test_costs_reduce_returns(self, market):
        """A higher cost model must produce a lower terminal equity."""
        prices, scores = market
        cfg = PortfolioConfig(vol_target=None)
        cheap = VectorizedBacktestEngine(
            config=cfg, cost_model=CostModel(0.0, 0.0)
        ).run(prices, scores)
        pricey = VectorizedBacktestEngine(
            config=cfg, cost_model=CostModel(10.0, 20.0)
        ).run(prices, scores)
        assert pricey.equity_curve.iloc[-1] < cheap.equity_curve.iloc[-1]

    def test_zero_costs_means_zero_cost_series(self, market):
        prices, scores = market
        res = VectorizedBacktestEngine(
            config=PortfolioConfig(vol_target=None), cost_model=CostModel(0.0, 0.0)
        ).run(prices, scores)
        assert res.costs.abs().max() == pytest.approx(0.0)

    def test_weights_respect_position_cap(self, market):
        """A stated risk limit must hold exactly, not approximately.

        Regression: renormalising after clipping used to push names straight
        back over the cap — a 3% limit produced 25% positions.
        """
        prices, scores = market
        cfg = PortfolioConfig(max_position_weight=0.03, vol_target=None,
                              max_sector_weight=None)
        naive, vec = _run_both(prices, scores, cfg)
        assert vec.weights.abs().to_numpy().max() <= 0.03 + 1e-9
        assert naive.weights.abs().to_numpy().max() <= 0.03 + 1e-9

    def test_rebalance_count_matches_frequency(self, market):
        prices, scores = market
        weekly = VectorizedBacktestEngine(
            config=PortfolioConfig(rebalance_frequency="W-FRI", vol_target=None)
        ).run(prices, scores)
        monthly = VectorizedBacktestEngine(
            config=PortfolioConfig(rebalance_frequency="ME", vol_target=None)
        ).run(prices, scores)
        assert weekly.n_rebalances > monthly.n_rebalances

    def test_higher_turnover_costs_more(self, market):
        """Daily rebalancing must cost more than monthly."""
        prices, scores = market
        daily = VectorizedBacktestEngine(
            config=PortfolioConfig(rebalance_frequency="D", vol_target=None)
        ).run(prices, scores)
        monthly = VectorizedBacktestEngine(
            config=PortfolioConfig(rebalance_frequency="ME", vol_target=None)
        ).run(prices, scores)
        assert daily.costs.sum() > monthly.costs.sum()
