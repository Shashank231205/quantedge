"""Metric correctness against closed-form and hand-checkable cases.

Every reported figure traces back to these functions, so an error here
propagates into the README and the resume. Where a metric has an analytic
answer, the test asserts that answer rather than a regression snapshot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantedge.metrics import drawdown as dd
from quantedge.metrics import performance as perf
from quantedge.metrics.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
)
from quantedge.metrics.trades import trade_statistics
from quantedge.risk.drawdown_guard import (
    DrawdownGuardConfig,
    apply_drawdown_guard,
    circuit_breaker_status,
)
from quantedge.risk.var import conditional_var, historical_var

TRADING_DAYS = 252


def series(values, start="2022-01-03"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


class TestReturnMetrics:
    def test_total_return_compounds(self):
        # (1.1 * 0.9) - 1 = -0.01
        assert perf.total_return(series([0.10, -0.10])) == pytest.approx(-0.01)

    def test_cagr_of_doubling_over_one_year(self):
        daily = 2 ** (1 / TRADING_DAYS) - 1
        r = series([daily] * TRADING_DAYS)
        assert perf.cagr(r) == pytest.approx(1.0, rel=1e-3)

    def test_zero_returns_give_zero_cagr(self):
        assert perf.cagr(series([0.0] * 100)) == pytest.approx(0.0)

    def test_hit_rate(self):
        assert perf.hit_rate(series([0.01, -0.01, 0.01, 0.0])) == pytest.approx(0.5)


class TestSharpe:
    def test_constant_returns_have_no_volatility(self):
        """Zero variance must not produce an infinite Sharpe."""
        assert perf.sharpe_ratio(series([0.001] * 100)) == 0.0

    def test_known_value(self):
        rng = np.random.default_rng(3)
        r = series(rng.normal(0.001, 0.01, 2000))
        expected = (r.mean() - 0.0) / r.std(ddof=1) * np.sqrt(TRADING_DAYS)
        assert perf.sharpe_ratio(r, risk_free_rate=0.0) == pytest.approx(
            expected, rel=1e-6
        )

    def test_risk_free_rate_lowers_sharpe(self):
        """Quoting Sharpe on raw returns overstates it when cash pays."""
        rng = np.random.default_rng(5)
        r = series(rng.normal(0.0005, 0.01, 1000))
        assert perf.sharpe_ratio(r, risk_free_rate=0.05) < perf.sharpe_ratio(
            r, risk_free_rate=0.0
        )

    def test_sortino_exceeds_sharpe_when_downside_is_mild(self):
        """Sortino ignores upside dispersion, so a series whose losses are
        small and tightly clustered scores better on Sortino than Sharpe."""
        rng = np.random.default_rng(21)
        # Large, variable gains; small, uniform losses.
        values = np.where(
            rng.random(600) > 0.4, rng.normal(0.02, 0.02, 600), -0.002
        )
        r = series(values)
        assert perf.sortino_ratio(r, risk_free_rate=0.0) > perf.sharpe_ratio(
            r, risk_free_rate=0.0
        )

    def test_empty_series_is_zero_not_nan(self):
        assert perf.sharpe_ratio(pd.Series(dtype=float)) == 0.0


class TestDrawdown:
    def test_monotonic_gains_have_no_drawdown(self):
        assert dd.max_drawdown(series([0.01] * 50)) == pytest.approx(0.0)

    def test_known_drawdown_depth(self):
        # +25% then -20% returns exactly to the starting level.
        r = series([0.25, -0.20])
        assert dd.max_drawdown(r) == pytest.approx(-0.20)

    def test_drawdown_is_never_positive(self):
        rng = np.random.default_rng(9)
        assert dd.max_drawdown(series(rng.normal(0, 0.02, 500))) <= 0.0

    def test_duration_counts_underwater_bars(self):
        r = series([-0.10, 0.02, 0.02, 0.10, 0.01])
        assert dd.max_drawdown_duration(r) >= 3

    def test_unrecovered_episode_is_reported(self):
        """An episode still underwater must not be silently dropped."""
        r = series([0.05, -0.20, -0.05, 0.01])
        details = dd.drawdown_details(r)
        assert not details.empty
        assert bool(details.iloc[0]["recovered"]) is False
        assert details.iloc[0]["recovery_date"] is None

    def test_time_underwater_fraction(self):
        r = series([-0.10, 0.01, 0.01])
        assert 0.0 < dd.time_underwater(r) <= 1.0


class TestVaR:
    def test_historical_var_is_a_quantile(self):
        rng = np.random.default_rng(4)
        r = series(rng.normal(0, 0.01, 1000))
        assert historical_var(r, 0.95) == pytest.approx(np.percentile(r, 5), rel=1e-9)

    def test_cvar_is_worse_than_var(self):
        """Expected shortfall must be at least as severe as the threshold."""
        rng = np.random.default_rng(6)
        r = series(rng.normal(0, 0.015, 2000))
        assert conditional_var(r, 0.95) <= historical_var(r, 0.95)

    def test_higher_confidence_means_larger_loss(self):
        rng = np.random.default_rng(8)
        r = series(rng.normal(0, 0.01, 2000))
        assert historical_var(r, 0.99) < historical_var(r, 0.95)


class TestDeflatedSharpe:
    def test_expected_max_grows_with_trials(self):
        """More trials means a higher bar for significance."""
        assert expected_max_sharpe(100) > expected_max_sharpe(10) > expected_max_sharpe(2)

    def test_single_trial_has_no_selection_penalty(self):
        assert expected_max_sharpe(1) == 0.0

    def test_many_trials_deflate_a_marginal_result(self):
        """The correction must actually bite."""
        rng = np.random.default_rng(12)
        r = series(rng.normal(0.0004, 0.01, 1000))
        one = deflated_sharpe_ratio(r, n_trials=1)["deflated_sharpe"]
        many = deflated_sharpe_ratio(r, n_trials=500)["deflated_sharpe"]
        assert many < one

    def test_short_series_is_flagged_not_scored(self):
        out = deflated_sharpe_ratio(series([0.01] * 10), n_trials=5)
        assert out["is_significant"] is False
        assert "note" in out


class TestDrawdownGuard:
    def test_guard_caps_losses(self):
        """A guarded series must not fall as far as the unguarded one."""
        r = series([-0.03] * 30)
        guarded, events = apply_drawdown_guard(
            r, DrawdownGuardConfig(max_drawdown=0.10, reduced_exposure=0.0)
        )
        assert dd.max_drawdown(guarded) > dd.max_drawdown(r)
        assert not events.empty

    def test_guard_is_inert_without_a_breach(self):
        r = series([0.001] * 50)
        guarded, events = apply_drawdown_guard(r, DrawdownGuardConfig(max_drawdown=0.20))
        pd.testing.assert_series_equal(guarded, r, check_names=False)
        assert events.empty

    def test_status_reports_remaining_budget(self):
        curve = pd.Series([100.0, 110.0, 104.5])  # 5% off the peak
        status = circuit_breaker_status(curve, max_drawdown_limit=0.10)
        assert status["current_drawdown"] == pytest.approx(-0.05, abs=1e-9)
        assert status["remaining_pct"] == pytest.approx(50.0, abs=0.5)
        assert status["status"] == "ACTIVE"

    def test_status_detects_breach(self):
        curve = pd.Series([100.0, 80.0])
        assert circuit_breaker_status(curve, 0.10)["status"] == "BREACHED"


class TestTradeStatistics:
    def test_win_rate_and_profit_factor(self):
        trades = pd.DataFrame(
            {
                "pnl_pct": [0.10, 0.05, -0.04, -0.02],
                "side": ["LONG"] * 4,
                "holding_days": [5, 3, 8, 2],
                "status": ["CLOSED"] * 4,
            }
        )
        stats = trade_statistics(trades)
        assert stats["win_rate"] == pytest.approx(0.5)
        # 0.15 won / 0.06 lost
        assert stats["profit_factor"] == pytest.approx(2.5)
        assert stats["n_trades"] == 4

    def test_empty_input_is_safe(self):
        assert trade_statistics(pd.DataFrame())["n_trades"] == 0

    def test_expectancy_sign_matches_edge(self):
        losing = pd.DataFrame(
            {
                "pnl_pct": [-0.05, -0.03, 0.01],
                "side": ["LONG"] * 3,
                "holding_days": [1, 1, 1],
                "status": ["CLOSED"] * 3,
            }
        )
        assert trade_statistics(losing)["expectancy"] < 0
