"""Value at Risk and Expected Shortfall.

Three estimators, because they disagree in exactly the situations that
matter:

* **Historical** — empirical quantile. No distributional assumption, but
  cannot see a loss larger than the worst one already observed.
* **Parametric** — Gaussian. Convenient and consistently too optimistic for
  equity strategies, whose returns are fat-tailed.
* **Cornish-Fisher** — adjusts the Gaussian quantile for skew and kurtosis,
  usually landing between the two.

CVaR (Expected Shortfall) is reported alongside because VaR says nothing
about how bad the tail gets once the threshold is breached.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Empirical quantile loss. Returned as a negative number."""
    r = returns.dropna()
    if len(r) < 20:
        return 0.0
    return float(np.percentile(r, (1.0 - confidence) * 100))


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Gaussian VaR. Understates tail risk for real equity returns."""
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    return float(r.mean() + stats.norm.ppf(1.0 - confidence) * r.std(ddof=1))


def cornish_fisher_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Gaussian VaR corrected for skewness and excess kurtosis."""
    r = returns.dropna()
    if len(r) < 4:
        return 0.0

    z = stats.norm.ppf(1.0 - confidence)
    s = float(r.skew())
    k = float(r.kurtosis())  # excess

    z_cf = (
        z
        + (z**2 - 1) * s / 6.0
        + (z**3 - 3 * z) * k / 24.0
        - (2 * z**3 - 5 * z) * (s**2) / 36.0
    )
    return float(r.mean() + z_cf * r.std(ddof=1))


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected Shortfall: mean loss *given* the VaR threshold is breached."""
    r = returns.dropna()
    if len(r) < 20:
        return 0.0
    threshold = historical_var(r, confidence)
    tail = r[r <= threshold]
    return float(tail.mean()) if len(tail) else float(threshold)


def var_report(
    returns: pd.Series,
    portfolio_value: float = 1_000_000.0,
    confidences: tuple[float, ...] = (0.95, 0.99),
) -> dict:
    """VaR/CVaR at several confidence levels, in both percent and currency."""
    out: dict = {"portfolio_value": portfolio_value}

    for c in confidences:
        tag = f"{int(c * 100)}"
        hist = historical_var(returns, c)
        cvar = conditional_var(returns, c)
        out[f"var_{tag}"] = {
            "historical_pct": round(hist, 6),
            "parametric_pct": round(parametric_var(returns, c), 6),
            "cornish_fisher_pct": round(cornish_fisher_var(returns, c), 6),
            "cvar_pct": round(cvar, 6),
            # Daily currency figures are what appear on the Risk Monitor.
            "historical_usd": round(abs(hist) * portfolio_value, 2),
            "cvar_usd": round(abs(cvar) * portfolio_value, 2),
        }
    return out


def rolling_var(
    returns: pd.Series, window: int = 252, confidence: float = 0.95
) -> pd.Series:
    """Time series of historical VaR — shows risk regime shifts."""
    return returns.rolling(window, min_periods=window // 4).quantile(1.0 - confidence)


def stress_test(returns: pd.Series, scenarios: dict[str, float] | None = None) -> dict:
    """Apply hypothetical market shocks scaled by the strategy's beta.

    A simple linear approximation, which is the honest description: it says
    what a beta-scaled move implies, not what would actually happen to a
    long/short book in a crisis (correlations converge, shorts get squeezed).
    """
    scenarios = scenarios or {
        "market_down_5pct": -0.05,
        "market_down_10pct": -0.10,
        "market_down_20pct": -0.20,
        "vol_spike_2x": -0.08,
    }

    r = returns.dropna()
    daily_vol = float(r.std(ddof=1)) if len(r) > 1 else 0.0

    return {
        name: {
            "shock": shock,
            "estimated_pnl_pct": round(shock * 0.3, 6),  # assumed net beta ~0.3
            "n_sigma": round(abs(shock * 0.3 / daily_vol), 2) if daily_vol > 0 else 0.0,
        }
        for name, shock in scenarios.items()
    }
