"""Lookahead-bias tests — the most important tests in this project.

A factor that peeks at future prices produces a backtest that cannot be
traded and a Sharpe ratio that means nothing. The bug is silent: results
simply look better than they should.

The technique here is *future corruption*. Compute a factor on a price panel,
then replace every price after some cutoff with garbage and recompute. Any
factor value at or before the cutoff that changes was, by definition, reading
the future.

This runs against every factor in the registry, so a newly added factor is
covered automatically rather than by remembering to write a test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantedge.factors.base import Factor
from quantedge.factors.composite import FACTOR_REGISTRY, CompositeFactor, get_factor

N_DAYS = 400
N_TICKERS = 12
CUTOFF = 300


@pytest.fixture
def prices() -> pd.DataFrame:
    """Deterministic geometric random walk."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2021-01-01", periods=N_DAYS)
    tickers = [f"T{i:02d}" for i in range(N_TICKERS)]
    steps = rng.normal(0.0004, 0.015, size=(N_DAYS, N_TICKERS))
    return pd.DataFrame(100.0 * np.exp(np.cumsum(steps, axis=0)), index=dates, columns=tickers)


@pytest.fixture
def ohlc(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {"high": prices * 1.01, "low": prices * 0.99}


def corrupt_future(prices: pd.DataFrame, cutoff: int = CUTOFF) -> pd.DataFrame:
    """Replace everything strictly after ``cutoff`` with wildly different data."""
    out = prices.copy()
    rng = np.random.default_rng(999)
    shape = (len(out) - cutoff - 1, out.shape[1])
    # Deliberately extreme: a factor reading ahead will move visibly.
    out.iloc[cutoff + 1 :] = rng.uniform(1.0, 10_000.0, size=shape)
    return out


def assert_past_unchanged(
    original: pd.DataFrame, corrupted: pd.DataFrame, cutoff: int, label: str
) -> None:
    a = original.iloc[: cutoff + 1]
    b = corrupted.iloc[: cutoff + 1]
    both_nan = a.isna() & b.isna()
    close = np.isclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                       rtol=1e-9, atol=1e-12, equal_nan=True)
    ok = close | both_nan.to_numpy()
    if not ok.all():
        bad = int((~ok).sum())
        first = np.argwhere(~ok)[0]
        raise AssertionError(
            f"{label}: {bad} value(s) changed when future prices were corrupted "
            f"— first at row {first[0]}, column {a.columns[first[1]]}. "
            "This factor reads data it could not have known at the time."
        )


class TestFactorsDoNotReadTheFuture:
    @pytest.mark.parametrize("factor_name", sorted(FACTOR_REGISTRY))
    def test_compute_is_causal(self, factor_name, prices, ohlc):
        factor = get_factor(factor_name)
        original = factor.compute(prices, **ohlc)
        corrupted = factor.compute(corrupt_future(prices), **ohlc)
        assert_past_unchanged(original, corrupted, CUTOFF, f"{factor_name}.compute")

    @pytest.mark.parametrize("factor_name", sorted(FACTOR_REGISTRY))
    def test_signal_is_causal(self, factor_name, prices, ohlc):
        """The tradeable signal must be causal too, not just the raw value."""
        factor = get_factor(factor_name)
        original = factor.signal(prices, **ohlc)
        corrupted = factor.signal(corrupt_future(prices), **ohlc)
        assert_past_unchanged(original, corrupted, CUTOFF, f"{factor_name}.signal")

    def test_composite_is_causal(self, prices, ohlc):
        comp = CompositeFactor()
        original = comp.compute(prices, extra=ohlc)
        corrupted = comp.compute(corrupt_future(prices), extra=ohlc)
        assert_past_unchanged(original, corrupted, CUTOFF, "composite")


class TestTradeableShift:
    """``compute_tradeable`` must lag ``compute`` by exactly one bar."""

    @pytest.mark.parametrize("factor_name", sorted(FACTOR_REGISTRY))
    def test_shifted_by_one_bar(self, factor_name, prices, ohlc):
        factor = get_factor(factor_name)
        raw = factor.compute(prices, **ohlc)
        tradeable = factor.compute_tradeable(prices, **ohlc)

        pd.testing.assert_frame_equal(
            tradeable.iloc[1:].reset_index(drop=True),
            raw.iloc[:-1].reset_index(drop=True),
            check_names=False,
        )

    @pytest.mark.parametrize("factor_name", sorted(FACTOR_REGISTRY))
    def test_first_row_is_never_tradeable(self, factor_name, prices, ohlc):
        """Nothing is knowable before the first bar exists."""
        factor = get_factor(factor_name)
        assert factor.compute_tradeable(prices, **ohlc).iloc[0].isna().all()


class TestCrossSectionalNormalisation:
    def test_zscore_is_row_wise(self):
        """Standardisation must be across tickers on a date, not through time.

        A time-series z-score would use each name's own future distribution.
        """
        df = pd.DataFrame(
            {"A": [1.0, 10.0], "B": [2.0, 20.0], "C": [3.0, 30.0]},
            index=pd.to_datetime(["2022-01-03", "2022-01-04"]),
        )
        z = Factor.zscore(df)
        # Each row is independently standardised -> identical shape per row.
        assert z.loc["2022-01-03"].tolist() == pytest.approx(
            z.loc["2022-01-04"].tolist()
        )
        assert z.mean(axis=1).abs().max() < 1e-12

    def test_rank_pct_is_row_wise(self):
        df = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]})
        r = Factor.rank_pct(df)
        assert r.iloc[0].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])

    def test_rank_ignores_missing_names(self):
        df = pd.DataFrame({"A": [1.0], "B": [np.nan], "C": [3.0]})
        r = Factor.rank_pct(df)
        assert pd.isna(r.iloc[0]["B"])
        assert r.iloc[0]["C"] == pytest.approx(1.0)


class TestFactorDirection:
    """Orientation errors silently invert a strategy."""

    def test_momentum_prefers_winners(self, prices):
        factor = get_factor("momentum")
        raw = factor.compute_tradeable(prices)
        sig = factor.signal(prices)
        row = raw.iloc[-1].dropna()
        if len(row) > 2:
            best, worst = row.idxmax(), row.idxmin()
            assert sig.iloc[-1][best] > sig.iloc[-1][worst]

    def test_volatility_prefers_calm_names(self, prices):
        """Low-vol anomaly: the calmer name must score higher."""
        factor = get_factor("volatility")
        raw = factor.compute_tradeable(prices)
        sig = factor.signal(prices)
        row = raw.iloc[-1].dropna()
        if len(row) > 2:
            calmest, wildest = row.idxmin(), row.idxmax()
            assert sig.iloc[-1][calmest] > sig.iloc[-1][wildest]

    def test_mean_reversion_prefers_stretched_low(self, prices):
        factor = get_factor("mean_reversion")
        raw = factor.compute_tradeable(prices)
        sig = factor.signal(prices)
        row = raw.iloc[-1].dropna()
        if len(row) > 2:
            assert sig.iloc[-1][row.idxmin()] > sig.iloc[-1][row.idxmax()]


class TestCompositeBlending:
    def test_weights_are_normalised(self):
        comp = CompositeFactor(weights={"momentum": 2.0, "volatility": 2.0})
        assert sum(comp.weights.values()) == pytest.approx(1.0)
        assert comp.weights["momentum"] == pytest.approx(0.5)

    def test_rejects_non_positive_weights(self):
        with pytest.raises(ValueError, match="positive"):
            CompositeFactor(weights={"momentum": 0.0})

    def test_requires_all_components_present(self, prices):
        """A name missing one component must not receive a partial score."""
        comp = CompositeFactor()
        blended = comp.compute(prices)
        # Early rows lack the 252-day momentum window entirely.
        assert blended.iloc[0].isna().all()

    def test_output_is_bounded_like_a_rank(self, prices):
        comp = CompositeFactor()
        vals = comp.compute(prices).stack(future_stack=True).dropna()
        assert vals.min() >= 0.0
        assert vals.max() <= 1.0
