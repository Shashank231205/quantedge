"""Factor research diagnostics.

Backtest performance alone does not tell you whether a factor has predictive
content — a single lucky period can carry an equity curve. These are the
standard tools used to interrogate a signal directly:

* **Information Coefficient** — cross-sectional correlation between the factor
  and subsequent returns. Spearman (rank) IC is the usual choice because it
  is insensitive to outliers. A mean IC of 0.02-0.05 is respectable for a
  daily equity factor; anything above ~0.15 in a simple setup should prompt
  suspicion of lookahead rather than celebration.
* **IC decay** — how quickly predictive power fades with horizon, which tells
  you the natural rebalancing frequency.
* **Quantile spreads** — sort into buckets and check monotonicity. A factor
  whose top bucket beats its bottom bucket only because of one extreme decile
  is fragile.
* **Autocorrelation** — how persistent the signal is, which drives turnover
  and therefore transaction costs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from quantedge.logging_config import get_logger

log = get_logger(__name__)


def forward_returns(prices: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Return over the next ``horizon`` bars, aligned to the decision date.

    Row ``t`` holds the return from ``t`` to ``t+horizon``. Pairing this with
    a signal that is already lagged by one bar is what makes the IC honest.
    """
    return prices.shift(-horizon) / prices - 1.0


def information_coefficient(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    horizon: int = 1,
    method: str = "spearman",
    min_names: int = 20,
) -> pd.Series:
    """Per-date cross-sectional correlation of signal with forward return."""
    fwd = forward_returns(prices, horizon)
    common = signal.index.intersection(fwd.index)

    out: dict[pd.Timestamp, float] = {}
    for dt in common:
        s = signal.loc[dt]
        f = fwd.loc[dt]
        mask = s.notna() & f.notna()
        if int(mask.sum()) < min_names:
            continue
        x, y = s[mask], f[mask]
        if x.nunique() < 2 or y.nunique() < 2:
            continue
        corr = (
            stats.spearmanr(x, y).statistic
            if method == "spearman"
            else stats.pearsonr(x, y).statistic
        )
        if not np.isnan(corr):
            out[dt] = float(corr)

    return pd.Series(out, name=f"ic_{horizon}d").sort_index()


@dataclass
class ICSummary:
    """Aggregate IC statistics for one factor at one horizon."""

    factor: str
    horizon: int
    mean_ic: float
    std_ic: float
    ic_ir: float
    hit_rate: float
    t_stat: float
    p_value: float
    n_periods: int

    def as_dict(self) -> dict:
        return {
            "factor": self.factor,
            "horizon_days": self.horizon,
            "mean_ic": round(self.mean_ic, 5),
            "std_ic": round(self.std_ic, 5),
            # IC information ratio: mean/std, the risk-adjusted view of IC.
            "ic_ir": round(self.ic_ir, 4),
            "hit_rate": round(self.hit_rate, 4),
            "t_stat": round(self.t_stat, 3),
            "p_value": round(self.p_value, 5),
            "n_periods": self.n_periods,
        }


def summarize_ic(ic: pd.Series, factor: str, horizon: int) -> ICSummary:
    clean = ic.dropna()
    n = len(clean)
    if n < 2:
        return ICSummary(factor, horizon, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, n)

    mean = float(clean.mean())
    std = float(clean.std(ddof=1))
    t_stat, p_value = stats.ttest_1samp(clean, 0.0)

    return ICSummary(
        factor=factor,
        horizon=horizon,
        mean_ic=mean,
        std_ic=std,
        ic_ir=mean / std if std > 0 else 0.0,
        hit_rate=float((clean > 0).mean()),
        t_stat=float(t_stat),
        p_value=float(p_value),
        n_periods=n,
    )


def ic_decay(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 21, 42, 63),
    factor: str = "factor",
) -> pd.DataFrame:
    """IC across increasing horizons — the decay curve plotted in the UI."""
    rows = [
        summarize_ic(information_coefficient(signal, prices, h), factor, h).as_dict()
        for h in horizons
    ]
    return pd.DataFrame(rows)


def quantile_returns(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    n_quantiles: int = 5,
    horizon: int = 1,
) -> pd.DataFrame:
    """Mean forward return per signal quantile, per date.

    A healthy factor shows a monotonic progression from Q1 to Qn.
    """
    fwd = forward_returns(prices, horizon)
    common = signal.index.intersection(fwd.index)

    records: list[dict] = []
    for dt in common:
        s = signal.loc[dt].dropna()
        f = fwd.loc[dt]
        if len(s) < n_quantiles * 4:
            continue
        try:
            buckets = pd.qcut(s, n_quantiles, labels=False, duplicates="drop")
        except ValueError:
            continue
        row: dict = {"date": dt}
        for q in range(n_quantiles):
            names = buckets[buckets == q].index
            vals = f.reindex(names).dropna()
            row[f"q{q + 1}"] = float(vals.mean()) if len(vals) else np.nan
        records.append(row)

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.set_index("date")
    qcols = [c for c in df.columns if c.startswith("q")]
    if len(qcols) >= 2:
        # The tradeable spread: long the top bucket, short the bottom.
        df["spread"] = df[qcols[-1]] - df[qcols[0]]
    return df


def quantile_summary(qr: pd.DataFrame, trading_days: int = 252) -> dict:
    """Annualised summary of the quantile fan, including monotonicity."""
    if qr.empty:
        return {}

    qcols = [c for c in qr.columns if c.startswith("q")]
    means = {c: float(qr[c].mean()) for c in qcols}
    ordered = [means[c] for c in qcols]

    # Spearman correlation of bucket index vs. mean return: 1.0 is perfectly
    # monotonic, which is the shape a genuine factor should produce.
    monotonicity = (
        float(stats.spearmanr(range(len(ordered)), ordered).statistic)
        if len(ordered) > 2
        else 0.0
    )

    out = {
        "quantile_mean_returns": {k: round(v, 6) for k, v in means.items()},
        "quantile_annualized": {
            k: round(v * trading_days, 4) for k, v in means.items()
        },
        "monotonicity": round(monotonicity, 4),
    }

    if "spread" in qr.columns:
        spread = qr["spread"].dropna()
        if len(spread) > 1 and spread.std() > 0:
            out["spread_mean_daily"] = round(float(spread.mean()), 6)
            out["spread_annualized"] = round(float(spread.mean()) * trading_days, 4)
            out["spread_sharpe"] = round(
                float(spread.mean() / spread.std() * np.sqrt(trading_days)), 4
            )
    return out


def factor_autocorrelation(signal: pd.DataFrame, lags: tuple[int, ...] = (1, 5, 21)) -> dict:
    """Persistence of the cross-sectional ranking.

    Low autocorrelation means the ranking churns daily, which translates
    directly into turnover and transaction costs.
    """
    out: dict[str, float] = {}
    for lag in lags:
        corrs = [
            signal.iloc[i].corr(signal.iloc[i - lag], method="spearman")
            for i in range(lag, len(signal))
            if signal.iloc[i].notna().sum() > 20
        ]
        vals = [c for c in corrs if c is not None and not np.isnan(c)]
        out[f"lag_{lag}d"] = round(float(np.mean(vals)), 4) if vals else 0.0
    return out


def cross_factor_correlation(signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Average pairwise cross-sectional correlation between factor signals.

    Powers the correlation matrix on the Factor Explorer screen. Highly
    correlated factors add little diversification to a composite.
    """
    names = sorted(signals)
    mat = pd.DataFrame(np.eye(len(names)), index=names, columns=names)

    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            sa, sb = signals[a], signals[b]
            common = sa.index.intersection(sb.index)
            per_date = []
            for dt in common:
                x, y = sa.loc[dt], sb.loc[dt]
                mask = x.notna() & y.notna()
                if int(mask.sum()) > 20:
                    c = stats.spearmanr(x[mask], y[mask]).statistic
                    if not np.isnan(c):
                        per_date.append(c)
            val = round(float(np.mean(per_date)), 4) if per_date else 0.0
            mat.loc[a, b] = mat.loc[b, a] = val

    return mat
