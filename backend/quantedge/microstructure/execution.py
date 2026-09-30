"""Market or limit? A walk-forward execution study.

The decision: an order to buy (or sell) a small quantity arrives at second
``t``. Either cross the spread now, or rest at the touch for up to ``horizon``
seconds and cross at whatever the price is then if nobody trades with us.

Every cost is measured against the arrival mid, in basis points, and includes
exchange fees — which, on a one-tick-spread futures contract, dwarf the half
spread and are the main thing a resting order saves.

Fill model. We cannot see queue position in L1 data, so two bounds are
reported:

* **conservative** (headline) — filled only if the entire price level we are
  resting on is consumed: a sell trade prints *below* our bid, or the best ask
  drops to our bid. Either means every order at that price, ours included,
  has traded. Guaranteed fills, understated fill rate.
* **optimistic** — filled if any sell trade prints *at* our bid. That assumes
  we were at the front of the queue, which a newly placed order never is.

The truth lies between them; a conclusion that flips between the two bounds is
not a conclusion.

Policies:

* ``market``  — always cross immediately
* ``limit``   — always rest for ``horizon`` seconds, then cross
* ``signal``  — cross immediately when the model predicts the price will run
  away from us over ``horizon`` by more than a threshold, otherwise rest. Model and
  threshold are fitted on day d-1 and applied, unchanged, to day d.

Decisions are sampled ``stride`` seconds apart, with ``stride`` longer than
``horizon`` plus the markout window, so no two decisions share any price path and
the paired t-statistics are not inflated by overlap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from quantedge.microstructure.research import fit_linear, predict_linear, walk_forward_days

MARKOUT_S = 10


@dataclass(frozen=True)
class FeeSchedule:
    """Binance USDⓈ-M futures, regular tier, as published during the sample."""

    maker_bps: float = 2.0
    taker_bps: float = 5.0

    def as_dict(self) -> dict:
        return {"maker_bps": self.maker_bps, "taker_bps": self.taker_bps}


REGULAR_FEES = FeeSchedule()


def _decision_points(df: pd.DataFrame, horizon: int, stride: int) -> np.ndarray:
    """Row positions at which a decision is simulated."""
    need = df[["bid_px", "ask_px", "mid"]].notna().all(axis=1).to_numpy()
    positions = np.arange(0, len(df) - horizon - MARKOUT_S - 1, stride)
    return positions[need[positions]]


def simulate_orders(
    df: pd.DataFrame,
    horizon: int,
    stride: int,
    fees: FeeSchedule = REGULAR_FEES,
    fill_model: str = "conservative",
) -> pd.DataFrame:
    """Outcome of both a market and a resting limit order at each decision.

    One row per (decision, side). Costs are in bps of arrival mid; lower is
    better, and a negative cost means we did better than mid.
    """
    if fill_model not in ("conservative", "optimistic"):
        raise ValueError(f"unknown fill model {fill_model!r}")

    bid, ask, mid = df["bid_px"].to_numpy(), df["ask_px"].to_numpy(), df["mid"].to_numpy()
    ask_min, bid_max = df["ask_min"].to_numpy(), df["bid_max"].to_numpy()
    sell_min = df["sell_min_px"].to_numpy()
    buy_max = df["buy_max_px"].to_numpy()

    rows = []
    for t in _decision_points(df, horizon, stride):
        window = slice(t + 1, t + horizon + 1)
        m0 = mid[t]
        for side in (1, -1):
            if side == 1:
                level = bid[t]
                # np.nan comparisons are False, so seconds without trades
                # simply never produce a fill.
                with np.errstate(invalid="ignore"):
                    through = sell_min[window] < level if fill_model == "conservative" else sell_min[window] <= level
                    hit = through | (ask_min[window] <= level)
                market_px, late_px = ask[t], ask[t + horizon]
            else:
                level = ask[t]
                with np.errstate(invalid="ignore"):
                    through = buy_max[window] > level if fill_model == "conservative" else buy_max[window] >= level
                    hit = through | (bid_max[window] >= level)
                market_px, late_px = bid[t], bid[t + horizon]

            if np.isnan(late_px) or np.isnan(market_px):
                continue

            # Signed so that a positive number is always a cost to us.
            market_cost = side * (market_px - m0) / m0 * 1e4
            filled = bool(hit.any())
            if filled:
                fill_at = t + 1 + int(np.argmax(hit))
                limit_gross = side * (level - m0) / m0 * 1e4
                limit_cost = limit_gross + fees.maker_bps
                after = mid[fill_at + MARKOUT_S]
                # Mid move after the fill, in our favour if positive. A resting
                # order that fills is disproportionately one the market was
                # about to run through: this is adverse selection, measured.
                markout = side * (after - level) / m0 * 1e4 if not np.isnan(after) else np.nan
                wait = fill_at - t
            else:
                limit_gross = side * (late_px - m0) / m0 * 1e4
                limit_cost = limit_gross + fees.taker_bps
                markout = np.nan
                wait = horizon

            rows.append(
                {
                    "pos": t,
                    "side": side,
                    "market_gross_bps": market_cost,
                    "market_cost_bps": market_cost + fees.taker_bps,
                    "limit_gross_bps": limit_gross,
                    "limit_cost_bps": limit_cost,
                    "limit_filled": filled,
                    "limit_wait_s": wait,
                    "limit_markout_bps": markout,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out.index = df.index[out["pos"].to_numpy()]
    return out


def _choose_threshold(urgency: np.ndarray, market: np.ndarray, limit: np.ndarray) -> float:
    """Urgency cut-off that minimised mean cost on the training day.

    Candidates span the urgency distribution, plus the two static policies
    (+inf: always rest, -inf: always cross), so the chosen rule can never do
    worse in-sample than the better of the two.
    """
    finite = urgency[np.isfinite(urgency)]
    candidates = [np.inf, -np.inf, *np.quantile(finite, np.linspace(0.05, 0.99, 40))] if len(finite) else [np.inf]
    best, best_cost = np.inf, np.inf
    for c in candidates:
        cost = np.where(urgency > c, market, limit).mean()
        if cost < best_cost:
            best, best_cost = c, cost
    return float(best)


def _describe_threshold(threshold: float) -> float | str:
    if np.isfinite(threshold):
        return round(threshold, 4)
    # An infinite cut-off means the training day preferred a static policy.
    return "always_limit" if threshold > 0 else "always_market"


def _policy_stats(costs: np.ndarray) -> dict:
    return {
        "mean_cost_bps": round(float(np.mean(costs)), 4),
        "median_cost_bps": round(float(np.median(costs)), 4),
        "std_cost_bps": round(float(np.std(costs, ddof=1)), 4) if len(costs) > 1 else None,
        "n": int(len(costs)),
    }


def _paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Mean of a - b with a paired t-test. Negative means ``a`` is cheaper."""
    diff = a - b
    if len(diff) < 3 or np.std(diff) == 0:
        return {"mean_diff_bps": round(float(np.mean(diff)), 4), "t_stat": None, "p_value": None}
    res = stats.ttest_1samp(diff, 0.0)
    return {
        "mean_diff_bps": round(float(diff.mean()), 4),
        "t_stat": round(float(res.statistic), 2),
        "p_value": round(float(res.pvalue), 5),
    }


def evaluate_execution(
    df: pd.DataFrame,
    features: tuple[str, ...],
    horizon: int = 10,
    stride: int = 60,
    fees: FeeSchedule = REGULAR_FEES,
    fill_model: str = "conservative",
) -> dict:
    """Walk-forward comparison of market, limit and signal-conditioned routing."""
    if stride <= horizon + MARKOUT_S:
        raise ValueError("stride must exceed horizon + markout so decisions do not overlap")

    target = f"fwd_{horizon}"
    if target not in df.columns:
        raise ValueError(f"no {target} label; horizon must be one of the feature horizons")

    test_frames = []
    thresholds = []
    for day, train, test in walk_forward_days(df, embargo=horizon):
        model = fit_linear(train, list(features), target)

        sim_train = simulate_orders(train, horizon, stride, fees, fill_model)
        sim_test = simulate_orders(test, horizon, stride, fees, fill_model)
        if sim_train.empty or sim_test.empty:
            continue

        for sim, frame in ((sim_train, train), (sim_test, test)):
            pred = predict_linear(model, frame.loc[sim.index])
            # Positive urgency: the model expects the price to move against a
            # resting order on this side, i.e. up for a buy, down for a sell.
            sim["urgency"] = sim["side"].to_numpy() * pred.to_numpy()

        valid = sim_train["urgency"].notna()
        threshold = _choose_threshold(
            sim_train.loc[valid, "urgency"].to_numpy(),
            sim_train.loc[valid, "market_cost_bps"].to_numpy(),
            sim_train.loc[valid, "limit_cost_bps"].to_numpy(),
        )
        thresholds.append({"test_day": str(day), "threshold_bps": _describe_threshold(threshold)})

        sim_test = sim_test[sim_test["urgency"].notna()].copy()
        sim_test["route_market"] = sim_test["urgency"] > threshold
        sim_test["signal_cost_bps"] = np.where(
            sim_test["route_market"], sim_test["market_cost_bps"], sim_test["limit_cost_bps"]
        )
        sim_test["test_day"] = str(day)
        test_frames.append(sim_test)

    if not test_frames:
        return {"wait_s": horizon, "fill_model": fill_model, "n_decisions": 0}

    sims = pd.concat(test_frames)
    market = sims["market_cost_bps"].to_numpy()
    limit = sims["limit_cost_bps"].to_numpy()
    signal = sims["signal_cost_bps"].to_numpy()
    # The comparison bar is whichever static policy turned out cheaper on the
    # test days themselves — hindsight the signal policy never had. Beating it
    # is a stricter test than beating the static policy chosen in advance.
    best_static_name = "market" if market.mean() <= limit.mean() else "limit"
    best_static = market if best_static_name == "market" else limit

    fills = sims[sims["limit_filled"]]
    unfilled = sims[~sims["limit_filled"]]

    per_day = [
        {
            "test_day": d,
            "n": int(len(g)),
            "market": round(float(g["market_cost_bps"].mean()), 4),
            "limit": round(float(g["limit_cost_bps"].mean()), 4),
            "signal": round(float(g["signal_cost_bps"].mean()), 4),
            "routed_market_pct": round(float(g["route_market"].mean()), 4),
        }
        for d, g in sims.groupby("test_day")
    ]

    return {
        "wait_s": horizon,
        "stride_s": stride,
        "fill_model": fill_model,
        "fees": fees.as_dict(),
        "n_decisions": int(len(sims)),
        "policies": {
            "market": _policy_stats(market),
            "limit": _policy_stats(limit),
            "signal": _policy_stats(signal),
        },
        "gross_of_fees": {
            "market_mean_bps": round(float(sims["market_gross_bps"].mean()), 4),
            "limit_mean_bps": round(float(sims["limit_gross_bps"].mean()), 4),
        },
        "limit_detail": {
            "fill_rate": round(float(sims["limit_filled"].mean()), 4),
            "mean_wait_to_fill_s": round(float(fills["limit_wait_s"].mean()), 2) if len(fills) else None,
            "filled_mean_cost_bps": round(float(fills["limit_cost_bps"].mean()), 4) if len(fills) else None,
            "unfilled_mean_cost_bps": round(float(unfilled["limit_cost_bps"].mean()), 4) if len(unfilled) else None,
            "fill_markout_bps": round(float(fills["limit_markout_bps"].mean()), 4) if len(fills) else None,
        },
        "signal_detail": {
            "routed_market_pct": round(float(sims["route_market"].mean()), 4),
            "vs_best_static": {"policy": best_static_name, **_paired(signal, best_static)},
            "vs_market": _paired(signal, market),
            "vs_limit": _paired(signal, limit),
            "days_signal_beats_best_static": int(
                sum(r["signal"] < r[best_static_name] for r in per_day)
            ),
            "thresholds": thresholds,
        },
        "per_day": per_day,
    }
