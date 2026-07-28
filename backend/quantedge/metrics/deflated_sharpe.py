"""Multiple-testing correction for Sharpe ratios.

If you try 50 strategy configurations and report the best one, its Sharpe is
biased upward — the maximum of 50 noisy draws exceeds the true mean even when
every configuration is worthless. This is the single most common way honest
people produce dishonest backtests.

Implements Bailey & López de Prado's Deflated Sharpe Ratio (2014), which
answers: given N trials, the observed skew and kurtosis, and the sample
length, what is the probability the true Sharpe is above zero?

Reporting this alongside the raw figure is what lets a reader judge whether a
result is a discovery or a selection artefact. It is included here precisely
because it can only make our headline number look *worse*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from quantedge.config import settings
from quantedge.metrics.performance import kurtosis, sharpe_ratio, skewness

TRADING_DAYS = settings.trading_days_per_year


def expected_max_sharpe(n_trials: int, variance_of_sharpes: float = 1.0) -> float:
    """Expected maximum Sharpe from ``n_trials`` independent worthless trials.

    Uses the standard extreme-value approximation for the maximum of N normal
    draws. This is the benchmark a genuine strategy must beat.
    """
    if n_trials < 2:
        return 0.0

    gamma = 0.5772156649015329  # Euler-Mascheroni
    sd = np.sqrt(variance_of_sharpes)

    # E[max] ~ sd * [(1-g) * z(1 - 1/N) + g * z(1 - 1/(N*e))]
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sd * ((1.0 - gamma) * z1 + gamma * z2))


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """P(true Sharpe > benchmark), adjusting for non-normal returns.

    Negative skew and fat tails make a given Sharpe less trustworthy, which
    this captures — the usual case for a levered long/short book.
    """
    if n_obs < 2:
        return 0.0

    # kurt here is the *raw* fourth moment (3.0 for a normal distribution).
    denom = 1.0 - skew * observed_sharpe + ((kurt - 1.0) / 4.0) * observed_sharpe**2
    if denom <= 0:
        return 0.0

    z = (observed_sharpe - benchmark_sharpe) * np.sqrt(n_obs - 1) / np.sqrt(denom)
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    variance_of_sharpes: float = 1.0,
    trading_days: int = TRADING_DAYS,
) -> dict:
    """Full deflated-Sharpe report for a return series.

    ``n_trials`` must be the honest count of configurations evaluated while
    arriving at this strategy — understating it defeats the purpose.
    """
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    n_obs = len(r)

    if n_obs < 30:
        return {
            "observed_sharpe": 0.0,
            "deflated_sharpe": 0.0,
            "expected_max_sharpe": 0.0,
            "n_trials": n_trials,
            "n_obs": n_obs,
            "is_significant": False,
            "note": "insufficient observations",
        }

    observed = sharpe_ratio(r, trading_days=trading_days)
    skew = skewness(r)
    # performance.kurtosis returns *excess* kurtosis; convert to raw.
    raw_kurt = kurtosis(r) + 3.0

    # Both the threshold and the observed value are expressed per-period,
    # since the test operates on the per-observation distribution.
    sr_periodic = observed / np.sqrt(trading_days)
    threshold_periodic = expected_max_sharpe(n_trials, variance_of_sharpes) / np.sqrt(
        trading_days
    )

    dsr = probabilistic_sharpe_ratio(
        sr_periodic, threshold_periodic, n_obs, skew, raw_kurt
    )
    psr = probabilistic_sharpe_ratio(sr_periodic, 0.0, n_obs, skew, raw_kurt)

    return {
        "observed_sharpe": round(observed, 4),
        "probabilistic_sharpe": round(psr, 4),
        "deflated_sharpe": round(dsr, 4),
        "expected_max_sharpe": round(
            expected_max_sharpe(n_trials, variance_of_sharpes), 4
        ),
        "n_trials": n_trials,
        "n_obs": n_obs,
        "skewness": round(skew, 4),
        "excess_kurtosis": round(raw_kurt - 3.0, 4),
        # The conventional 95% bar: P(true Sharpe > selection threshold).
        "is_significant": bool(dsr > 0.95),
        "interpretation": _interpret(dsr, n_trials),
    }


def _interpret(dsr: float, n_trials: int) -> str:
    if dsr > 0.95:
        return (
            f"Survives correction for {n_trials} trials "
            "(>95% probability the edge is real)."
        )
    if dsr > 0.80:
        return (
            f"Suggestive but not conclusive after {n_trials} trials "
            "(80-95% probability)."
        )
    return (
        f"Does not survive correction for {n_trials} trials — the result is "
        "consistent with selection bias rather than a genuine edge."
    )


def minimum_track_record_length(
    observed_sharpe: float,
    benchmark_sharpe: float = 0.0,
    confidence: float = 0.95,
    skew: float = 0.0,
    kurt: float = 3.0,
    trading_days: int = TRADING_DAYS,
) -> float:
    """Observations needed to establish significance at ``confidence``.

    Answers the practical question "is my backtest long enough?" — a 1.0
    Sharpe typically needs several years of daily data to be distinguishable
    from zero.
    """
    sr = observed_sharpe / np.sqrt(trading_days)
    bench = benchmark_sharpe / np.sqrt(trading_days)
    if sr <= bench:
        return float("inf")

    z = stats.norm.ppf(confidence)
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2
    return float(1.0 + denom * (z / (sr - bench)) ** 2)
