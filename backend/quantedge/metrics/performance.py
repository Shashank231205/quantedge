"""Risk-adjusted performance metrics.

Conventions applied consistently throughout, because inconsistency here is
how backtest numbers become indefensible:

* Returns are **simple** (not log) daily returns, already **net of costs**.
* Annualisation uses 252 trading days; volatility scales by ``sqrt(252)``.
* Sharpe is computed on **excess** returns over the risk-free rate. Quoting a
  Sharpe on raw returns during a 5% rate environment overstates it materially.
* Every function tolerates short or degenerate input by returning 0.0 rather
  than raising — a walk-forward fold with few observations should not abort
  the run, but it also should not report a flattering number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantedge.config import settings

TRADING_DAYS = settings.trading_days_per_year


def _clean(returns: pd.Series) -> pd.Series:
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def total_return(returns: pd.Series) -> float:
    r = _clean(returns)
    return float((1.0 + r).prod() - 1.0) if len(r) else 0.0


def cagr(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """Compound annual growth rate."""
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    years = len(r) / trading_days
    growth = float((1.0 + r).prod())
    if years <= 0 or growth <= 0:
        return 0.0
    return float(growth ** (1.0 / years) - 1.0)


def annualized_return(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    return float(r.mean() * trading_days) if len(r) else 0.0


def annualized_volatility(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    return float(r.std(ddof=1) * np.sqrt(trading_days)) if len(r) > 1 else 0.0


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = settings.risk_free_rate,
    trading_days: int = TRADING_DAYS,
) -> float:
    """Annualised Sharpe on excess returns.

    The risk-free deduction is not cosmetic: at a 4% cash rate it moves a
    0.9 Sharpe to roughly 0.7 for a 10%-vol strategy.
    """
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    excess = r - (risk_free_rate / trading_days)
    sd = excess.std(ddof=1)
    if sd <= 1e-12:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(trading_days))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = settings.risk_free_rate,
    trading_days: int = TRADING_DAYS,
) -> float:
    """Like Sharpe, but penalising only downside deviation."""
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    excess = r - (risk_free_rate / trading_days)
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    dd = downside.std(ddof=1)
    if dd <= 1e-12:
        return 0.0
    return float(excess.mean() / dd * np.sqrt(trading_days))


def calmar_ratio(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """CAGR divided by max drawdown — return per unit of worst-case pain."""
    from quantedge.metrics.drawdown import max_drawdown

    mdd = abs(max_drawdown(returns))
    if mdd <= 1e-12:
        return 0.0
    return float(cagr(returns, trading_days) / mdd)


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """Probability-weighted ratio of gains to losses above a threshold."""
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    excess = r - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()
    if losses <= 1e-12:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def information_ratio(
    returns: pd.Series, benchmark: pd.Series, trading_days: int = TRADING_DAYS
) -> float:
    """Active return divided by tracking error."""
    aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return 0.0
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te = active.std(ddof=1)
    if te <= 1e-12:
        return 0.0
    return float(active.mean() / te * np.sqrt(trading_days))


def tracking_error(
    returns: pd.Series, benchmark: pd.Series, trading_days: int = TRADING_DAYS
) -> float:
    aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return 0.0
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return float(active.std(ddof=1) * np.sqrt(trading_days))


def skewness(returns: pd.Series) -> float:
    r = _clean(returns)
    return float(r.skew()) if len(r) > 2 else 0.0


def kurtosis(returns: pd.Series) -> float:
    """Excess kurtosis; > 0 means fatter tails than a normal distribution."""
    r = _clean(returns)
    return float(r.kurtosis()) if len(r) > 3 else 0.0


def hit_rate(returns: pd.Series) -> float:
    """Fraction of periods with a positive return."""
    r = _clean(returns)
    return float((r > 0).mean()) if len(r) else 0.0


def best_worst(returns: pd.Series) -> tuple[float, float]:
    r = _clean(returns)
    if not len(r):
        return 0.0, 0.0
    return float(r.max()), float(r.min())


def rolling_sharpe(
    returns: pd.Series,
    window: int = TRADING_DAYS,
    risk_free_rate: float = settings.risk_free_rate,
    trading_days: int = TRADING_DAYS,
) -> pd.Series:
    """Rolling annualised Sharpe — reveals whether performance is persistent."""
    excess = _clean(returns) - (risk_free_rate / trading_days)
    mean = excess.rolling(window, min_periods=window // 2).mean()
    std = excess.rolling(window, min_periods=window // 2).std(ddof=1)
    return (mean / std.replace(0.0, np.nan)) * np.sqrt(trading_days)


def monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """Year x month grid of compounded returns."""
    r = _clean(returns)
    if r.empty:
        return pd.DataFrame()
    idx = pd.to_datetime(r.index)
    monthly = r.groupby([idx.year, idx.month]).apply(lambda x: (1 + x).prod() - 1)
    table = monthly.unstack(level=-1)
    table.index.name = "year"
    table.columns = [
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][c - 1]
        for c in table.columns
    ]
    return table


def yearly_returns(returns: pd.Series) -> pd.Series:
    r = _clean(returns)
    if r.empty:
        return pd.Series(dtype=float)
    return r.groupby(pd.to_datetime(r.index).year).apply(lambda x: (1 + x).prod() - 1)
