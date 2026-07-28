"""Drawdown analysis.

Max drawdown is the number a risk committee asks about first, because it is
what an investor actually experiences. Depth alone is incomplete though —
a 15% drawdown recovered in a month is a different proposition from a 15%
drawdown that takes two years, so duration and recovery are reported too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    r = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return initial * (1.0 + r).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Percentage below the running peak, at every point in time.

    The running peak is seeded at the *starting* capital rather than at the
    first post-return value. Otherwise a series that opens with a loss folds
    that loss into its own baseline and reports zero drawdown — flattering
    and wrong.
    """
    curve = equity_curve(returns)
    if curve.empty:
        return curve
    peak = curve.cummax().clip(lower=1.0)
    return (curve / peak) - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough decline, as a negative fraction."""
    dd = drawdown_series(returns)
    return float(dd.min()) if len(dd) else 0.0


def drawdown_details(returns: pd.Series, top_n: int = 5) -> pd.DataFrame:
    """The worst drawdown episodes, with peak, trough and recovery dates.

    An episode still underwater at the end of the sample is reported with a
    null recovery date rather than being silently dropped — pretending it
    recovered would be the flattering error.
    """
    dd = drawdown_series(returns)
    if dd.empty:
        return pd.DataFrame()

    underwater = dd < -1e-12
    episodes: list[dict] = []
    start: int | None = None

    for i in range(len(dd)):
        if underwater.iloc[i] and start is None:
            start = i
        elif not underwater.iloc[i] and start is not None:
            episodes.append(_episode(dd, start, i - 1, recovered=True))
            start = None

    if start is not None:
        episodes.append(_episode(dd, start, len(dd) - 1, recovered=False))

    if not episodes:
        return pd.DataFrame()

    df = pd.DataFrame(episodes).sort_values("max_drawdown")
    return df.head(top_n).reset_index(drop=True)


def _episode(dd: pd.Series, start: int, end: int, recovered: bool) -> dict:
    window = dd.iloc[start : end + 1]
    trough_pos = int(window.to_numpy().argmin())
    trough_idx = start + trough_pos

    peak_date = dd.index[start]
    trough_date = dd.index[trough_idx]
    recovery_date = dd.index[end] if recovered else None

    return {
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "max_drawdown": float(window.min()),
        "duration_days": int(end - start + 1),
        "days_to_trough": int(trough_idx - start + 1),
        "days_to_recover": int(end - trough_idx) if recovered else None,
        "recovered": recovered,
    }


def average_drawdown(returns: pd.Series) -> float:
    dd = drawdown_series(returns)
    active = dd[dd < -1e-12]
    return float(active.mean()) if len(active) else 0.0


def max_drawdown_duration(returns: pd.Series) -> int:
    """Longest unbroken stretch below a previous peak, in bars."""
    dd = drawdown_series(returns)
    if dd.empty:
        return 0

    longest = current = 0
    for value in dd:
        if value < -1e-12:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def time_underwater(returns: pd.Series) -> float:
    """Fraction of the sample spent below a previous high-water mark."""
    dd = drawdown_series(returns)
    return float((dd < -1e-12).mean()) if len(dd) else 0.0


def ulcer_index(returns: pd.Series) -> float:
    """RMS of the drawdown series — penalises deep, prolonged declines."""
    dd = drawdown_series(returns)
    return float(np.sqrt((dd**2).mean())) if len(dd) else 0.0
