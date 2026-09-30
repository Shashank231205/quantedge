"""Event streams to one-second bars.

Order-flow imbalance (Cont, Kukanov & Stoikov, 2014) is defined on each
consecutive pair of top-of-book states:

    e_n =  1{Pb_n >= Pb_n-1} qb_n  -  1{Pb_n <= Pb_n-1} qb_n-1
         - 1{Pa_n <= Pa_n-1} qa_n  +  1{Pa_n >= Pa_n-1} qa_n-1

A bid that improves adds its whole size as buying pressure; a bid that is
unchanged contributes only its change in size; a bid that falls away removes
what was there. The ask side mirrors it. Summing e_n inside a bar gives the
net pressure over that bar.

It has to be computed on the raw events, before any sampling: two updates that
cancel within a second still move OFI, and a per-second snapshot would see
neither. So the full stream is read here — in chunks, carrying the last state
across each boundary so the result does not depend on the chunk size.

Each bar also keeps the intra-second extremes the fill simulator needs: the
lowest ask and highest bid seen, and the most aggressive sell and buy trade
prices. Without those a limit order could only be checked against end-of-second
state, which misses fills that happen and reverse inside the second.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

BOOK_STATE = ["bid_px", "bid_qty", "ask_px", "ask_qty"]


def ofi_increments(book: pd.DataFrame, prev: pd.Series | None = None) -> np.ndarray:
    """Per-event OFI contribution, in base-asset units.

    ``prev`` is the book state immediately before ``book``'s first row. When it
    is absent (the first event of a stream) that event contributes zero rather
    than its full size, since there is nothing to difference against.
    """
    bp, bq = book["bid_px"].to_numpy(), book["bid_qty"].to_numpy()
    ap, aq = book["ask_px"].to_numpy(), book["ask_qty"].to_numpy()
    n = len(book)
    if n == 0:
        return np.empty(0)

    if prev is None:
        prev_bp, prev_bq, prev_ap, prev_aq = bp[0], bq[0], ap[0], aq[0]
    else:
        prev_bp, prev_bq = float(prev["bid_px"]), float(prev["bid_qty"])
        prev_ap, prev_aq = float(prev["ask_px"]), float(prev["ask_qty"])

    bp0 = np.concatenate(([prev_bp], bp[:-1]))
    bq0 = np.concatenate(([prev_bq], bq[:-1]))
    ap0 = np.concatenate(([prev_ap], ap[:-1]))
    aq0 = np.concatenate(([prev_aq], aq[:-1]))

    e = (
        (bp >= bp0) * bq
        - (bp <= bp0) * bq0
        - (ap <= ap0) * aq
        + (ap >= ap0) * aq0
    )
    if prev is None:
        e[0] = 0.0
    return e.astype(float)


@dataclass
class BookAggregate:
    """Per-second book aggregates plus the state needed to continue the stream."""

    bars: pd.DataFrame
    events: int


def aggregate_book(chunks: Iterable[pd.DataFrame]) -> BookAggregate:
    """Fold a chunked quote stream into per-second book bars."""
    partials: list[pd.DataFrame] = []
    prev: pd.Series | None = None
    events = 0

    for chunk in chunks:
        if chunk.empty:
            continue
        chunk = chunk.reset_index(drop=True)
        ofi = ofi_increments(chunk, prev)
        prev = chunk.iloc[-1][BOOK_STATE]
        events += len(chunk)

        frame = pd.DataFrame(
            {
                "second": chunk["ts_ms"].to_numpy() // 1000,
                "ofi": ofi,
                "depth": (chunk["bid_qty"].to_numpy() + chunk["ask_qty"].to_numpy()) / 2.0,
                "bid_px": chunk["bid_px"].to_numpy(),
                "bid_qty": chunk["bid_qty"].to_numpy(),
                "ask_px": chunk["ask_px"].to_numpy(),
                "ask_qty": chunk["ask_qty"].to_numpy(),
            }
        )
        g = frame.groupby("second", sort=True)
        partials.append(
            pd.DataFrame(
                {
                    "ofi": g["ofi"].sum(),
                    "depth_sum": g["depth"].sum(),
                    "n_updates": g["ofi"].size(),
                    "bid_px": g["bid_px"].last(),
                    "bid_qty": g["bid_qty"].last(),
                    "ask_px": g["ask_px"].last(),
                    "ask_qty": g["ask_qty"].last(),
                    "ask_min": g["ask_px"].min(),
                    "bid_max": g["bid_px"].max(),
                }
            )
        )

    if not partials:
        return BookAggregate(pd.DataFrame(), 0)

    # A second that straddles two chunks appears in both partials. Combining
    # them again with the same rules — sums add, extremes take the extreme,
    # state takes the later chunk — gives the single-pass answer exactly.
    stacked = pd.concat(partials)
    g = stacked.groupby(level=0, sort=True)
    bars = pd.DataFrame(
        {
            "ofi": g["ofi"].sum(),
            "depth_sum": g["depth_sum"].sum(),
            "n_updates": g["n_updates"].sum(),
            "bid_px": g["bid_px"].last(),
            "bid_qty": g["bid_qty"].last(),
            "ask_px": g["ask_px"].last(),
            "ask_qty": g["ask_qty"].last(),
            "ask_min": g["ask_min"].min(),
            "bid_max": g["bid_max"].max(),
        }
    )
    bars.index.name = "second"
    return BookAggregate(bars, events)


def aggregate_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Per-second trade flow, split by aggressor.

    ``is_buyer_maker`` true means the resting order was the buy, so the seller
    crossed the spread: a sell-initiated trade.
    """
    if trades.empty:
        return pd.DataFrame()
    sell = trades["is_buyer_maker"].to_numpy()
    frame = pd.DataFrame(
        {
            "second": trades["ts_ms"].to_numpy() // 1000,
            "buy_qty": np.where(sell, 0.0, trades["qty"].to_numpy()),
            "sell_qty": np.where(sell, trades["qty"].to_numpy(), 0.0),
            # Aggressive sells hit bids and aggressive buys lift offers, so these
            # are the prices a resting bid or offer would have been tested at.
            "sell_min_px": np.where(sell, trades["price"].to_numpy(), np.inf),
            "buy_max_px": np.where(sell, -np.inf, trades["price"].to_numpy()),
        }
    )
    g = frame.groupby("second", sort=True)
    out = pd.DataFrame(
        {
            "buy_qty": g["buy_qty"].sum(),
            "sell_qty": g["sell_qty"].sum(),
            "n_trades": g["buy_qty"].size(),
            "sell_min_px": g["sell_min_px"].min(),
            "buy_max_px": g["buy_max_px"].max(),
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan)
    out.index.name = "second"
    return out


def build_bars(book: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Join book and trade aggregates onto a gap-free one-second grid.

    Seconds with no quote update carry the previous state forward — the book
    genuinely did not change — while flow columns are zero, not missing.
    Rows before the first quote are dropped, since there is no book to report.
    """
    if book.empty:
        raise ValueError("no book updates to build bars from")

    start, end = int(book.index.min()), int(book.index.max())
    grid = pd.RangeIndex(start, end + 1, name="second")
    bars = book.reindex(grid)
    if not trades.empty:
        bars = bars.join(trades.reindex(grid))
    else:
        for col in ("buy_qty", "sell_qty", "n_trades", "sell_min_px", "buy_max_px"):
            bars[col] = np.nan

    state = ["bid_px", "bid_qty", "ask_px", "ask_qty"]
    bars[state] = bars[state].ffill()
    for col in ("ofi", "depth_sum", "n_updates", "buy_qty", "sell_qty", "n_trades"):
        bars[col] = bars[col].fillna(0.0)

    # The book at the start of a second is the previous second's closing state,
    # so it belongs in that second's extremes too. With no update at all the
    # extremes are just the unchanged state.
    bars["ask_min"] = np.fmin(bars["ask_min"], bars["ask_px"].shift(1)).fillna(bars["ask_px"])
    bars["bid_max"] = np.fmax(bars["bid_max"], bars["bid_px"].shift(1)).fillna(bars["bid_px"])

    bars["n_updates"] = bars["n_updates"].astype("int64")
    bars["n_trades"] = bars["n_trades"].astype("int64")
    bars.insert(0, "time", pd.to_datetime(bars.index, unit="s", utc=True))
    return bars
