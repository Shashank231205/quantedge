"""Volatility-based position sizing.

Equal-weighting treats a 15%-vol utility and a 60%-vol biotech as the same
risk, which they are not: the biotech dominates portfolio variance. Inverse-
volatility weighting equalises *risk* contribution rather than dollars, and
is the mechanism behind the "risk-managed portfolio construction" claim.

Everything here uses trailing windows only, so sizing is causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantedge.config import settings

TRADING_DAYS = settings.trading_days_per_year


def realized_volatility(
    prices: pd.DataFrame, window: int = 60, trading_days: int = TRADING_DAYS
) -> pd.DataFrame:
    """Trailing annualised volatility per ticker."""
    returns = prices.pct_change()
    return returns.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(
        trading_days
    )


def inverse_vol_weights(
    weights: pd.Series,
    vols: pd.Series,
    floor: float = 0.05,
    cap: float = 1.50,
) -> pd.Series:
    """Rescale target weights by inverse volatility.

    The floor prevents a near-zero volatility estimate from producing an
    enormous position — the usual failure mode of naive risk parity.
    """
    v = vols.reindex(weights.index).clip(lower=floor, upper=cap)
    inv = 1.0 / v.replace(0.0, np.nan)
    scaled = weights * inv

    out = scaled.fillna(0.0)
    # Preserve the original gross exposure; this changes the *mix*, not the
    # leverage, which stays the job of the vol-target scalar.
    gross_before = weights.abs().sum()
    gross_after = out.abs().sum()
    if gross_after > 1e-12:
        out = out * (gross_before / gross_after)
    return out


def volatility_target_scalar(
    portfolio_returns: pd.Series,
    target_vol: float = settings.target_annual_vol,
    lookback: int = 60,
    trading_days: int = TRADING_DAYS,
    max_leverage: float = 2.0,
    min_obs: int = 20,
) -> float:
    """Leverage multiplier that holds portfolio volatility near target.

    Levering up in calm regimes and down in turbulent ones is the highest-
    value risk control available: it directly stabilises the denominator of
    the Sharpe ratio.
    """
    hist = portfolio_returns.dropna()
    if len(hist) < min_obs:
        return 1.0

    realized = float(hist.iloc[-lookback:].std(ddof=1) * np.sqrt(trading_days))
    if realized <= 1e-9:
        return 1.0
    return float(np.clip(target_vol / realized, 0.0, max_leverage))


def kelly_fraction(returns: pd.Series, max_fraction: float = 0.5) -> float:
    """Kelly-optimal leverage, capped.

    Full Kelly is famously too aggressive for real portfolios — it maximises
    long-run growth while tolerating drawdowns no allocator would accept — so
    a half-Kelly cap is the default.
    """
    r = returns.dropna()
    if len(r) < 20:
        return 0.0
    mean = float(r.mean())
    var = float(r.var(ddof=1))
    if var <= 1e-12:
        return 0.0
    return float(np.clip(mean / var, 0.0, max_fraction))


def risk_parity_weights(cov: pd.DataFrame, max_iter: int = 200) -> pd.Series:
    """Equal risk contribution weights, solved iteratively.

    Each holding contributes the same share of portfolio variance. Included
    as an alternative to inverse-vol, which ignores correlation.
    """
    n = len(cov)
    if n == 0:
        return pd.Series(dtype=float)

    w = np.ones(n) / n
    cov_m = cov.to_numpy()

    for _ in range(max_iter):
        port_vol = np.sqrt(w @ cov_m @ w)
        if port_vol <= 1e-12:
            break
        marginal = cov_m @ w / port_vol
        contribution = w * marginal
        target = port_vol / n

        adjustment = np.where(contribution > 1e-12, target / contribution, 1.0)
        w_new = w * adjustment
        w_new = w_new / w_new.sum()

        if np.abs(w_new - w).max() < 1e-8:
            w = w_new
            break
        w = w_new

    return pd.Series(w, index=cov.index)


def position_sizing_report(
    weights: pd.Series, vols: pd.Series, sectors: pd.Series | None = None
) -> pd.DataFrame:
    """Per-name sizing detail for the Risk Monitor position matrix."""
    if weights.empty:
        return pd.DataFrame()

    active = weights[weights.abs() > 1e-12]
    if active.empty:
        return pd.DataFrame()

    target = inverse_vol_weights(active, vols)

    rows = pd.DataFrame(
        {
            "ticker": active.index,
            "current_weight": active.to_numpy(),
            "vol_adj_target": target.reindex(active.index).to_numpy(),
            "volatility": vols.reindex(active.index).to_numpy(),
        }
    )
    rows["drift"] = rows["current_weight"] - rows["vol_adj_target"]

    if sectors is not None:
        rows["sector"] = sectors.reindex(rows["ticker"]).to_numpy()

    # Traffic-light status used by the UI.
    abs_drift = rows["drift"].abs()
    rows["risk_status"] = np.where(
        abs_drift > 0.015, "HIGH", np.where(abs_drift > 0.005, "MEDIUM", "OK")
    )
    return rows.sort_values("current_weight", key=abs, ascending=False).reset_index(drop=True)
