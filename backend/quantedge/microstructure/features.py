"""Features and labels on one-second bars.

Timing convention, which every test in ``test_microstructure`` leans on: a
feature at second ``t`` uses only bars up to and including ``t`` — the state at
the *end* of that second — and a label at ``t`` is the mid-price change from
the end of ``t`` to the end of ``t + h``. Nothing is ever shifted backwards.

Features:

* ``qi``          queue imbalance at the touch, (qb - qa) / (qb + qa). On an
                  instrument whose spread is almost always one tick the price
                  can only move when one side's queue is exhausted, so the
                  relative size of the two queues is the most direct read of
                  which way the next move goes.
* ``ofi_{w}``     order-flow imbalance summed over the last ``w`` seconds,
                  divided by average touch depth over the last five minutes.
                  The normalisation is Cont et al.'s: the same OFI moves price
                  less when the book is deep.
* ``tfi_{w}``     trade-flow imbalance, (buy - sell) / (buy + sell) volume by
                  aggressor over the last ``w`` seconds.
* ``past_ret_5``  mid return over the last 5 seconds, in bps — a control for
                  short-term reversal, so the other features are not credited
                  with it.

Regime variables (for conditioning, not prediction):

* ``rv_300``        realised volatility of 1-second mid returns over 5 minutes
* ``spread_ticks``  quoted spread in ticks
* ``session``       UTC trading session
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OFI_WINDOWS = (1, 5, 30)
TFI_WINDOWS = (5, 30)
HORIZONS = (1, 5, 10, 30, 60)
DEPTH_WINDOW = 300
VOL_WINDOW = 300

FEATURES = (
    "qi",
    *(f"ofi_{w}" for w in OFI_WINDOWS),
    *(f"tfi_{w}" for w in TFI_WINDOWS),
    "past_ret_5",
)

#: UTC session boundaries. Crypto trades around the clock, but liquidity and
#: participation follow the regional equity sessions.
SESSIONS = (
    ("asia", 0, 8),
    ("europe", 8, 13),
    ("us", 13, 21),
    ("late", 21, 24),
)


def infer_tick_size(prices: pd.Series) -> float:
    """Smallest price increment observed.

    Read from the data rather than configured, so a symbol whose tick size
    changed between archive days is caught rather than silently mis-scaled.
    """
    unique = np.unique(prices.dropna().to_numpy())
    if len(unique) < 2:
        raise ValueError("need at least two distinct prices to infer a tick size")
    diffs = np.diff(unique)
    tick = float(diffs[diffs > 0].min())
    # Prices arrive as decimal strings parsed to float; snap the difference
    # back to the decimal grid it came from.
    return float(np.round(tick, 10))


def session_of(hours: pd.Series | np.ndarray) -> np.ndarray:
    hours = np.asarray(hours)
    out = np.empty(len(hours), dtype=object)
    for name, lo, hi in SESSIONS:
        out[(hours >= lo) & (hours < hi)] = name
    return out


def to_full_grid(bars: pd.DataFrame) -> pd.DataFrame:
    """Reindex concatenated days onto one continuous second grid.

    Missing seconds — a day that failed to download, say — become NaN rows, so
    every rolling window and forward label that would span the gap is NaN too,
    instead of silently joining two non-adjacent days.
    """
    bars = bars.sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]
    grid = pd.RangeIndex(int(bars.index.min()), int(bars.index.max()) + 1, name="second")
    out = bars.reindex(grid)
    out["time"] = pd.to_datetime(out.index, unit="s", utc=True)
    return out


def build_features(bars: pd.DataFrame, tick: float | None = None) -> pd.DataFrame:
    """Compute features, regime variables and labels. One row per second."""
    bars = to_full_grid(bars)
    tick = tick or infer_tick_size(bars["bid_px"])

    out = pd.DataFrame(index=bars.index)
    out["time"] = bars["time"]
    mid = (bars["bid_px"] + bars["ask_px"]) / 2.0
    out["mid"] = mid
    out["bid_px"] = bars["bid_px"]
    out["ask_px"] = bars["ask_px"]
    out["spread_ticks"] = ((bars["ask_px"] - bars["bid_px"]) / tick).round()

    depth = bars["bid_qty"] + bars["ask_qty"]
    out["qi"] = (bars["bid_qty"] - bars["ask_qty"]) / depth.where(depth > 0)

    # Average touch depth per update over the window, not per second: quiet
    # seconds have no updates and should not dilute the estimate.
    n_upd = bars["n_updates"]
    avg_depth = (
        bars["depth_sum"].rolling(DEPTH_WINDOW, min_periods=DEPTH_WINDOW).sum()
        / n_upd.rolling(DEPTH_WINDOW, min_periods=DEPTH_WINDOW).sum().where(lambda s: s > 0)
    )
    for w in OFI_WINDOWS:
        ofi = bars["ofi"].rolling(w, min_periods=w).sum()
        out[f"ofi_{w}"] = ofi / avg_depth

    for w in TFI_WINDOWS:
        buy = bars["buy_qty"].rolling(w, min_periods=w).sum()
        sell = bars["sell_qty"].rolling(w, min_periods=w).sum()
        total = buy + sell
        # No trades in the window is balanced flow, not missing data.
        out[f"tfi_{w}"] = ((buy - sell) / total.where(total > 0)).fillna(0.0).where(
            buy.notna()
        )

    log_mid = np.log(mid)
    out["past_ret_5"] = (log_mid - log_mid.shift(5)) * 1e4
    one_sec = (log_mid - log_mid.shift(1)) * 1e4
    out["ret_1"] = one_sec
    out["rv_300"] = one_sec.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std()
    out["mid_change_ticks"] = (mid - mid.shift(1)) / tick
    out["ofi_raw"] = bars["ofi"]
    out["avg_depth"] = avg_depth
    out["session"] = session_of(bars["time"].dt.hour)
    out["day"] = bars["time"].dt.date

    for h in HORIZONS:
        out[f"fwd_{h}"] = (mid.shift(-h) / mid - 1.0) * 1e4

    # Execution-simulator inputs travel with the features so one frame serves
    # both halves of the research.
    for col in ("ask_min", "bid_max", "sell_min_px", "buy_max_px"):
        out[col] = bars[col]

    out.attrs["tick"] = tick
    return out
