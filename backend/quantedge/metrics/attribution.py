"""Performance attribution: where did the return actually come from?

The question an interviewer asks about any equity strategy is whether the
returns are market beta in disguise. Regressing strategy returns on the
benchmark decomposes them into a systematic component (explained by market
exposure, which an investor can buy for a few basis points) and an
idiosyncratic component (genuine alpha).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from quantedge.config import settings

TRADING_DAYS = settings.trading_days_per_year


def alpha_beta(
    returns: pd.Series,
    benchmark: pd.Series,
    risk_free_rate: float = settings.risk_free_rate,
    trading_days: int = TRADING_DAYS,
) -> dict:
    """CAPM regression of excess strategy returns on excess benchmark returns."""
    aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return {
            "alpha_annual": 0.0, "beta": 0.0, "r_squared": 0.0,
            "alpha_t_stat": 0.0, "alpha_p_value": 1.0, "n_obs": len(aligned),
        }

    rf_daily = risk_free_rate / trading_days
    y = aligned.iloc[:, 0] - rf_daily
    x = aligned.iloc[:, 1] - rf_daily

    result = stats.linregress(x, y)
    n = len(aligned)

    # Standard error of the intercept, for the alpha t-statistic.
    residuals = y - (result.intercept + result.slope * x)
    dof = n - 2
    se_resid = np.sqrt((residuals**2).sum() / dof) if dof > 0 else 0.0
    x_var = ((x - x.mean()) ** 2).sum()
    se_alpha = (
        se_resid * np.sqrt(1.0 / n + (x.mean() ** 2) / x_var) if x_var > 0 else 0.0
    )

    t_stat = float(result.intercept / se_alpha) if se_alpha > 1e-15 else 0.0
    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), dof))) if dof > 0 else 1.0

    return {
        "alpha_daily": round(float(result.intercept), 8),
        "alpha_annual": round(float(result.intercept) * trading_days, 6),
        "beta": round(float(result.slope), 4),
        "r_squared": round(float(result.rvalue**2), 4),
        "alpha_t_stat": round(t_stat, 3),
        "alpha_p_value": round(p_value, 5),
        "n_obs": n,
    }


def return_decomposition(
    returns: pd.Series, benchmark: pd.Series, trading_days: int = TRADING_DAYS
) -> dict:
    """Split total variance into systematic and idiosyncratic shares.

    This is what the Risk Decomposition panel displays.
    """
    ab = alpha_beta(returns, benchmark, trading_days=trading_days)
    aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()

    if len(aligned) < 30:
        return {"systematic_pct": 0.0, "idiosyncratic_pct": 0.0, **ab}

    total_var = float(aligned.iloc[:, 0].var(ddof=1))
    if total_var <= 1e-15:
        return {"systematic_pct": 0.0, "idiosyncratic_pct": 0.0, **ab}

    # Systematic variance = beta^2 * benchmark variance.
    systematic_var = (ab["beta"] ** 2) * float(aligned.iloc[:, 1].var(ddof=1))
    systematic_pct = float(np.clip(systematic_var / total_var, 0.0, 1.0))

    return {
        "systematic_pct": round(systematic_pct * 100, 2),
        "idiosyncratic_pct": round((1.0 - systematic_pct) * 100, 2),
        "total_volatility": round(float(np.sqrt(total_var * trading_days)), 6),
        **ab,
    }


def sector_attribution(
    weights: pd.DataFrame, returns_panel: pd.DataFrame, sectors: pd.Series
) -> pd.DataFrame:
    """Contribution to total return by sector."""
    if weights.empty or returns_panel.empty:
        return pd.DataFrame()

    common = weights.columns.intersection(returns_panel.columns)
    w = weights[common]
    r = returns_panel[common].reindex(w.index).fillna(0.0)

    # Yesterday's weight earns today's return.
    contribution = w.shift(1).fillna(0.0) * r
    sector_map = sectors.reindex(common).fillna("Unknown")

    rows = []
    for sector in sector_map.unique():
        cols = sector_map[sector_map == sector].index
        series = contribution[cols].sum(axis=1)
        rows.append(
            {
                "sector": sector,
                "total_contribution": round(float(series.sum()), 6),
                "avg_daily_contribution": round(float(series.mean()), 8),
                "volatility": round(float(series.std(ddof=1)), 6),
                "n_names": len(cols),
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values("total_contribution", ascending=False).reset_index(drop=True)


def factor_contribution(
    component_signals: dict[str, pd.DataFrame],
    returns_panel: pd.DataFrame,
    weights: pd.DataFrame,
) -> dict:
    """Correlation between each factor's tilt and realised P&L.

    A diagnostic rather than a formal decomposition: it indicates which
    factors were pulling the book's performance, without claiming an exact
    variance split across correlated signals.
    """
    if weights.empty:
        return {}

    common = weights.columns.intersection(returns_panel.columns)
    pnl = (weights[common].shift(1).fillna(0.0) * returns_panel[common].fillna(0.0)).sum(axis=1)

    out: dict[str, float] = {}
    for name, signal in component_signals.items():
        cols = common.intersection(signal.columns)
        if len(cols) == 0:
            continue
        # Cross-sectional tilt toward this factor, day by day.
        tilt = (weights[cols].fillna(0.0) * signal[cols].fillna(0.5)).sum(axis=1)
        aligned = pd.concat([tilt, pnl], axis=1).dropna()
        if len(aligned) > 30 and aligned.iloc[:, 0].std() > 1e-12:
            out[name] = round(float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])), 4)
    return out
