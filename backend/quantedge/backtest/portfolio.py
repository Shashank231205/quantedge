"""Portfolio construction: turning a cross-sectional score into weights.

Kept separate from both engines so the naive and vectorized implementations
are guaranteed to construct identical books — which is what makes the runtime
comparison a fair like-for-like measurement rather than two different
strategies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantedge.config import settings


@dataclass
class PortfolioConfig:
    """Everything that determines how scores become positions."""

    long_quantile: float = settings.long_quantile
    short_quantile: float = settings.short_quantile
    long_short: bool = True
    max_position_weight: float = settings.max_position_weight
    max_sector_weight: float = settings.max_sector_weight
    #: Scale positions to hit a target portfolio volatility.
    vol_target: float | None = settings.target_annual_vol
    vol_lookback: int = 60
    #: Cap on leverage after vol scaling.
    max_leverage: float = 2.0
    rebalance_frequency: str = settings.rebalance_frequency
    min_names: int = 20
    equal_weight: bool = True

    def as_dict(self) -> dict:
        return {
            "long_quantile": self.long_quantile,
            "short_quantile": self.short_quantile,
            "long_short": self.long_short,
            "max_position_weight": self.max_position_weight,
            "max_sector_weight": self.max_sector_weight,
            "vol_target": self.vol_target,
            "vol_lookback": self.vol_lookback,
            "max_leverage": self.max_leverage,
            "rebalance_frequency": self.rebalance_frequency,
            "min_names": self.min_names,
            "equal_weight": self.equal_weight,
        }


def select_names(
    scores: pd.Series, cfg: PortfolioConfig
) -> tuple[pd.Index, pd.Index]:
    """Split a single date's scores into long and short baskets."""
    valid = scores.dropna()
    if len(valid) < cfg.min_names:
        return pd.Index([]), pd.Index([])

    long_cut = valid.quantile(1.0 - cfg.long_quantile)
    longs = valid[valid >= long_cut].index

    if not cfg.long_short:
        return longs, pd.Index([])

    short_cut = valid.quantile(cfg.short_quantile)
    shorts = valid[valid <= short_cut].index
    # A name cannot be both; ties at a degenerate cut go long.
    shorts = shorts.difference(longs)
    return longs, shorts


def build_weights(
    scores: pd.Series,
    cfg: PortfolioConfig,
    sectors: pd.Series | None = None,
) -> pd.Series:
    """Dollar-neutral weights for one rebalance date.

    Longs sum to +1 and shorts to -1 before any volatility scaling, so gross
    exposure is 2.0 and net exposure is 0.
    """
    longs, shorts = select_names(scores, cfg)
    weights = pd.Series(0.0, index=scores.index, dtype=float)

    if len(longs) == 0:
        return weights

    if cfg.equal_weight:
        weights.loc[longs] = 1.0 / len(longs)
        if len(shorts):
            weights.loc[shorts] = -1.0 / len(shorts)
    else:
        # Score-proportional within each leg.
        ls = scores.loc[longs]
        weights.loc[longs] = (ls / ls.sum()).to_numpy() if ls.sum() > 0 else 1.0 / len(longs)
        if len(shorts):
            ss = scores.loc[shorts]
            inv = (1.0 - ss)
            weights.loc[shorts] = -(inv / inv.sum()).to_numpy() if inv.sum() > 0 else -1.0 / len(shorts)

    if sectors is not None and cfg.max_sector_weight:
        weights = _apply_sector_caps(weights, sectors, cfg.max_sector_weight)

    return _cap_and_normalize(weights, cfg.max_position_weight)


def _cap_and_normalize(
    weights: pd.Series, cap: float | None, max_iter: int = 20
) -> pd.Series:
    """Enforce a per-name cap while keeping each leg summing to +/-1.

    Clipping and normalising fight each other: normalising after a clip can
    push names straight back over the cap. Iterating to a fixed point
    resolves it, and the cap wins outright when it is mathematically
    impossible to satisfy both (fewer than 1/cap names in a leg), because
    silently exceeding a stated risk limit is the worse failure.
    """
    index = weights.index
    values = weights.to_numpy(dtype=float, copy=True)

    def normalize(arr: np.ndarray) -> np.ndarray:
        longs = arr > 0
        long_sum = arr[longs].sum()
        if long_sum > 0:
            arr[longs] /= long_sum
        shorts = arr < 0
        short_sum = arr[shorts].sum()
        if short_sum < 0:
            arr[shorts] /= -short_sum
        return arr

    values = normalize(values)
    if not cap:
        return pd.Series(values, index=index)

    for _ in range(max_iter):
        if not (np.abs(values) > cap + 1e-12).any():
            break
        values = normalize(np.clip(values, -cap, cap))

    # Final clip: if a leg cannot be normalised within the cap, respect the
    # cap and accept a leg that sums to less than 1.
    return pd.Series(np.clip(values, -cap, cap), index=index)


def _renormalize(weights: pd.Series) -> pd.Series:
    """Restore +1 / -1 leg sums after clipping.

    Operates on the underlying NumPy array rather than via boolean-mask
    assignment on the Series. Profiling showed the pandas path invoking
    ``Series.__repr__``/``to_string`` internally — formatting code running
    inside the rebalance loop, which dominated small-universe runtimes.
    """
    values = weights.to_numpy(dtype=float, copy=True)

    longs = values > 0
    long_sum = values[longs].sum()
    if long_sum > 0:
        values[longs] /= long_sum

    shorts = values < 0
    short_sum = values[shorts].sum()
    if short_sum < 0:
        values[shorts] /= -short_sum

    return pd.Series(values, index=weights.index)


def _apply_sector_caps(
    weights: pd.Series, sectors: pd.Series, cap: float
) -> pd.Series:
    """Scale down any sector whose gross weight exceeds the cap.

    Without this a momentum book in 2020-2021 becomes an undiversified bet on
    technology, and the resulting Sharpe measures sector luck rather than the
    factor.
    """
    out = weights.copy()
    aligned = sectors.reindex(out.index)

    for sector in aligned.dropna().unique():
        members = aligned[aligned == sector].index
        gross = out.loc[members].abs().sum()
        if gross > cap and gross > 0:
            out.loc[members] *= cap / gross
    return out


def volatility_scalar(
    portfolio_returns: pd.Series,
    target_vol: float,
    lookback: int = 60,
    trading_days: int = 252,
    max_leverage: float = 2.0,
) -> float:
    """Leverage multiplier that targets a constant portfolio volatility.

    Uses only past returns, so it is causal. Scaling up in calm regimes and
    down in turbulent ones is the single most effective Sharpe improvement
    available without touching the signal.
    """
    hist = portfolio_returns.dropna()
    if len(hist) < max(20, lookback // 2):
        return 1.0

    realized = float(hist.iloc[-lookback:].std() * np.sqrt(trading_days))
    if realized <= 1e-9:
        return 1.0
    return float(np.clip(target_vol / realized, 0.0, max_leverage))


def rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    """Trading dates on which the book is rebuilt.

    ``frequency`` accepts pandas offset aliases (``W-FRI``, ``ME``, ``QE``) or
    ``D`` for every bar.
    """
    if frequency.upper() in ("D", "DAILY", ""):
        return index

    marks = pd.Series(index=index, data=index)
    resampled = marks.resample(frequency).last().dropna()
    return pd.DatetimeIndex(resampled.to_numpy())


def forward_fill_weights(
    weights_on_rebalance: dict[pd.Timestamp, pd.Series],
    all_dates: pd.DatetimeIndex,
    columns: pd.Index,
) -> pd.DataFrame:
    """Hold positions between rebalances.

    Note this holds *target* weights constant rather than letting them drift
    with prices — equivalent to rebalancing back to target daily at zero cost.
    The approximation is conservative: real drift would add tracking error but
    also save turnover.
    """
    if not weights_on_rebalance:
        return pd.DataFrame(0.0, index=all_dates, columns=columns)

    frame = pd.DataFrame(weights_on_rebalance).T.reindex(columns=columns)
    frame.index = pd.DatetimeIndex(frame.index)
    return frame.reindex(all_dates).ffill().fillna(0.0)
