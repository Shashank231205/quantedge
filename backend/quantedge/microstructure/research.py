"""Does top-of-book flow predict the next few seconds?

Four questions, each answered separately because they are easy to conflate:

1. **Explanation vs prediction.** OFI is famous for explaining a large share
   of the *same* interval's price change. That is a statement about how prices
   are formed, not a trading signal — by the time the interval's OFI is known,
   the move has happened. :func:`contemporaneous_vs_predictive` reports both
   R² figures side by side so the first can never be quoted as the second.
2. **Predictive IC by horizon.** Rank correlation of each feature with the
   forward mid return, sampled at non-overlapping intervals. Overlapping
   labels (a 60s return sampled every second) share 59 of their 60 seconds,
   which makes naive t-statistics roughly sqrt(60) times too large.
3. **When it works.** The same IC split by volatility, spread and session,
   and tracked hour by hour, because a signal that averages well can still be
   concentrated in one regime or have stopped working half-way through.
4. **Out of sample.** A linear combination fitted on one day and scored on the
   next, with an embargo so no training label reaches into the test day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from quantedge.microstructure.features import FEATURES, HORIZONS

PREDICTIVE_FEATURES = FEATURES


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def _clean(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].replace([np.inf, -np.inf], np.nan).dropna()


def non_overlapping(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Every ``horizon``-th second, so consecutive labels share no seconds."""
    return df.iloc[::horizon]


def ic_by_horizon(
    df: pd.DataFrame,
    features: tuple[str, ...] = PREDICTIVE_FEATURES,
    horizons: tuple[int, ...] = HORIZONS,
) -> list[dict]:
    """Spearman IC of each feature against each forward return."""
    rows = []
    for h in horizons:
        sample = non_overlapping(df, h)
        for f in features:
            data = _clean(sample, [f, f"fwd_{h}"])
            x, y = data[f].to_numpy(), data[f"fwd_{h}"].to_numpy()
            ic = _spearman(x, y)
            n = len(data)
            # Large-sample standard error of a rank correlation.
            t = ic * np.sqrt(max(n - 2, 1)) / np.sqrt(max(1 - ic**2, 1e-12)) if n > 2 else 0.0

            # Direction accuracy only where both the signal and the move are
            # non-zero; a flat mid is neither a hit nor a miss.
            moved = (y != 0) & (x != 0)
            hit = float((np.sign(x[moved]) == np.sign(y[moved])).mean()) if moved.any() else None

            per_day = [
                _spearman(g[f].to_numpy(), g[f"fwd_{h}"].to_numpy())
                for _, g in data.join(sample["day"]).groupby("day")
            ]
            per_day = [v for v in per_day if not np.isnan(v)]
            rows.append(
                {
                    "feature": f,
                    "horizon_s": h,
                    "ic": round(ic, 5),
                    "t_stat": round(float(t), 2),
                    "n": n,
                    "hit_rate": round(hit, 4) if hit is not None else None,
                    "days_positive": int(sum(v > 0 for v in per_day)),
                    "days": len(per_day),
                }
            )
    return rows


def _r2(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0:
        return float("nan")
    r = np.corrcoef(x, y)[0, 1]
    return float(r * r)


def contemporaneous_vs_predictive(df: pd.DataFrame, intervals=(1, 10, 60)) -> list[dict]:
    """R² of mid change on OFI over the same interval and over the next one.

    Intervals are formed by summing OFI and mid changes over consecutive,
    non-overlapping blocks. Contemporaneous: block k's price change on block
    k's OFI. Predictive: block k+1's price change on block k's OFI.
    """
    rows = []
    ofi_norm = (df["ofi_raw"] / df["avg_depth"]).to_numpy()
    dp_ticks = df["mid_change_ticks"].to_numpy()
    for k in intervals:
        n = len(df) // k * k
        # A plain sum over each row of a (blocks, k) reshape: any NaN second
        # makes the whole block NaN, so a block is only used when complete.
        agg = pd.DataFrame(
            {
                "ofi": ofi_norm[:n].reshape(-1, k).sum(axis=1),
                "dp": dp_ticks[:n].reshape(-1, k).sum(axis=1),
            }
        )
        agg["dp_next"] = agg["dp"].shift(-1)
        both = agg.replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "interval_s": k,
                "r2_contemporaneous": round(_r2(both["ofi"].to_numpy(), both["dp"].to_numpy()), 4),
                "r2_predictive": round(_r2(both["ofi"].to_numpy(), both["dp_next"].to_numpy()), 5),
                "n_intervals": len(both),
            }
        )
    return rows


def ic_by_regime(
    df: pd.DataFrame,
    features: tuple[str, ...] = ("qi", "ofi_5", "tfi_5"),
    horizon: int = 10,
) -> dict[str, list[dict]]:
    """IC conditioned on volatility tercile, spread and session.

    Tercile cut points come from the full sample, which is fine for a
    descriptive split like this one — nothing is traded on it. The walk-forward
    model and the execution policy never see these thresholds.
    """
    sample = non_overlapping(df, horizon).copy()
    sample["vol_regime"] = pd.qcut(sample["rv_300"], 3, labels=["low", "mid", "high"])
    sample["spread_regime"] = np.where(sample["spread_ticks"] <= 1, "1 tick", "wider")

    out: dict[str, list[dict]] = {}
    for regime in ("vol_regime", "spread_regime", "session"):
        rows = []
        for value, g in sample.groupby(regime, observed=True):
            row: dict = {"regime": str(value), "n": int(len(g))}
            for f in features:
                data = _clean(g, [f, f"fwd_{horizon}"])
                row[f"ic_{f}"] = round(_spearman(data[f].to_numpy(), data[f"fwd_{horizon}"].to_numpy()), 4)
            rows.append(row)
        out[regime] = rows
    return out


def ic_stability(df: pd.DataFrame, feature: str = "qi", horizon: int = 10) -> dict:
    """Hour-by-hour IC: is the average hiding a signal that comes and goes?"""
    sample = non_overlapping(df, horizon)
    data = _clean(sample, [feature, f"fwd_{horizon}"]).join(sample["time"])
    hourly = []
    for hour, g in data.groupby(data["time"].dt.floor("h")):
        if len(g) < 30:
            continue
        hourly.append(
            {
                "hour": hour.isoformat(),
                "ic": round(_spearman(g[feature].to_numpy(), g[f"fwd_{horizon}"].to_numpy()), 4),
                "n": len(g),
            }
        )
    ics = np.array([h["ic"] for h in hourly if not np.isnan(h["ic"])])
    if len(ics) == 0:
        return {"feature": feature, "horizon_s": horizon, "hours": 0, "hourly": []}

    half = len(ics) // 2
    first, second = ics[:half], ics[half:]
    welch = stats.ttest_ind(first, second, equal_var=False) if half >= 2 else None
    return {
        "feature": feature,
        "horizon_s": horizon,
        "hours": len(ics),
        "mean_ic": round(float(ics.mean()), 4),
        "std_ic": round(float(ics.std(ddof=1)), 4) if len(ics) > 1 else None,
        "pct_hours_positive": round(float((ics > 0).mean()), 4),
        "worst_hour_ic": round(float(ics.min()), 4),
        "best_hour_ic": round(float(ics.max()), 4),
        # Did the signal weaken over the sample? A small p-value here means the
        # two halves differ by more than hour-to-hour noise explains.
        "first_half_mean_ic": round(float(first.mean()), 4) if half else None,
        "second_half_mean_ic": round(float(second.mean()), 4),
        "halves_p_value": round(float(welch.pvalue), 4) if welch is not None else None,
        "hourly": hourly,
    }


def fit_linear(train: pd.DataFrame, features: list[str], target: str) -> dict:
    """OLS on standardised features. Returns everything needed to predict."""
    data = _clean(train, [*features, target])
    X = data[features].to_numpy()
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    beta, *_ = np.linalg.lstsq(Z, data[target].to_numpy(), rcond=None)
    return {"features": features, "mu": mu, "sd": sd, "beta": beta, "n_train": len(data)}


def predict_linear(model: dict, df: pd.DataFrame) -> pd.Series:
    X = df[model["features"]].replace([np.inf, -np.inf], np.nan)
    Z = (X.to_numpy() - model["mu"]) / model["sd"]
    pred = model["beta"][0] + Z @ model["beta"][1:]
    return pd.Series(pred, index=df.index)


def walk_forward_days(df: pd.DataFrame, embargo: int) -> list[tuple[object, pd.DataFrame, pd.DataFrame]]:
    """(test_day, train, test) triples: train on day d-1, test on day d.

    The last ``embargo`` seconds of each training day are dropped, because
    their forward labels are measured with prices from the test day.
    """
    days = sorted(d for d in df["day"].dropna().unique())
    out = []
    for prev, cur in zip(days[:-1], days[1:], strict=False):
        train = df[df["day"] == prev]
        train = train.iloc[: max(len(train) - embargo, 0)]
        test = df[df["day"] == cur]
        out.append((cur, train, test))
    return out


def walk_forward_model(
    df: pd.DataFrame,
    features: tuple[str, ...] = PREDICTIVE_FEATURES,
    horizon: int = 10,
) -> dict:
    """Fit on each day, score on the next. Only out-of-sample numbers leave here."""
    target = f"fwd_{horizon}"
    folds = []
    preds, actual = [], []
    for day, train, test in walk_forward_days(df, embargo=horizon):
        model = fit_linear(train, list(features), target)
        scored = non_overlapping(test, horizon)
        pred = predict_linear(model, scored)
        both = pd.DataFrame({"pred": pred, "y": scored[target]}).dropna()
        if len(both) < 30:
            continue
        sse = float(((both["y"] - both["pred"]) ** 2).sum())
        # Benchmark is a zero forecast, the right null for short-horizon
        # returns; beating the training-set mean would be a lower bar.
        sst = float((both["y"] ** 2).sum())
        folds.append(
            {
                "test_day": str(day),
                "n_train": model["n_train"],
                "n_test": len(both),
                "oos_r2": round(1.0 - sse / sst, 5) if sst > 0 else None,
                "oos_ic": round(_spearman(both["pred"].to_numpy(), both["y"].to_numpy()), 4),
                "coefficients": {
                    f: round(float(b), 4) for f, b in zip(features, model["beta"][1:], strict=True)
                },
            }
        )
        preds.append(both["pred"])
        actual.append(both["y"])

    if not folds:
        return {"horizon_s": horizon, "folds": []}

    p, y = pd.concat(preds), pd.concat(actual)
    pooled_ic = _spearman(p.to_numpy(), y.to_numpy())
    pooled_r2 = 1.0 - float(((y - p) ** 2).sum()) / float((y**2).sum())
    return {
        "horizon_s": horizon,
        "features": list(features),
        "folds": folds,
        "pooled_oos_ic": round(pooled_ic, 4),
        "pooled_oos_r2": round(pooled_r2, 5),
        "folds_positive_ic": int(sum((f["oos_ic"] or 0) > 0 for f in folds)),
    }
