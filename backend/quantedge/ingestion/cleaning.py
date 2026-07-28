"""Cleaning and corporate-action adjustment.

Raw vendor data is not tradeable as-is. Two things matter most here:

1. **Adjustment.** A 4:1 split shows up as a -75% one-day return in raw close
   prices. Using ``adj_close`` (which folds in splits and dividends) is what
   makes returns comparable through time. Every factor reads the adjusted
   series; the raw close is kept only for audit.

2. **Bad ticks.** Free data contains zero prices, negative volume, and
   occasional absurd jumps. Left alone these become fake momentum signals. We
   remove what is provably wrong and *flag* the rest rather than silently
   winsorising, so the cleaning report is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantedge.logging_config import get_logger

log = get_logger(__name__)

# A single-day move beyond this is treated as a suspected bad tick rather
# than a real return. Genuine 50%+ daily moves in S&P 500 names are rare and
# usually accompanied by a corporate action the adjustment should have caught.
MAX_ABS_DAILY_RETURN = 0.5
MIN_PRICE = 0.01


@dataclass
class CleaningReport:
    """What cleaning actually did — surfaced in job telemetry."""

    rows_in: int = 0
    rows_out: int = 0
    dropped_null_price: int = 0
    dropped_nonpositive_price: int = 0
    dropped_bad_ohlc: int = 0
    dropped_duplicate: int = 0
    flagged_extreme_return: int = 0
    tickers_in: int = 0
    tickers_out: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "dropped_null_price": self.dropped_null_price,
            "dropped_nonpositive_price": self.dropped_nonpositive_price,
            "dropped_bad_ohlc": self.dropped_bad_ohlc,
            "dropped_duplicate": self.dropped_duplicate,
            "flagged_extreme_return": self.flagged_extreme_return,
            "tickers_in": self.tickers_in,
            "tickers_out": self.tickers_out,
            "retention_pct": round(100 * self.rows_out / self.rows_in, 2)
            if self.rows_in
            else 0.0,
            "notes": self.notes,
        }


def adjust_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Scale OHLC by the adjustment factor implied by ``adj_close``.

    yfinance adjusts close but leaves open/high/low raw, so the ratio
    ``adj_close / close`` is applied to the rest of the bar. Without this,
    an overnight gap computed from a raw open against an adjusted close is
    pure fiction on any day following a split or dividend.
    """
    out = df.copy()
    if "adj_close" not in out.columns:
        return out

    factor = np.where(
        (out["close"] > 0) & out["adj_close"].notna(),
        out["adj_close"] / out["close"],
        1.0,
    )
    for col in ("open", "high", "low"):
        if col in out.columns:
            out[col] = out[col] * factor
    out["close"] = out["adj_close"].where(out["adj_close"].notna(), out["close"])
    return out


def clean_ohlcv(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Validate, adjust and sort a long-format OHLCV frame.

    Returns the cleaned frame plus a report of every action taken.
    """
    report = CleaningReport(rows_in=len(df))
    if df.empty:
        return df, report

    report.tickers_in = df["ticker"].nunique()
    out = df.copy()

    # --- structural ---------------------------------------------------
    before = len(out)
    out = out.drop_duplicates(subset=["ticker", "date"], keep="last")
    report.dropped_duplicate = before - len(out)

    before = len(out)
    out = out.dropna(subset=["close"])
    report.dropped_null_price = before - len(out)

    # --- adjustment (before validity checks so bounds apply to adj prices)
    out = adjust_prices(out)

    # --- validity -------------------------------------------------------
    before = len(out)
    out = out[out["close"] >= MIN_PRICE]
    report.dropped_nonpositive_price = before - len(out)

    # high must bound low, and both must bound the traded prices.
    if {"high", "low", "open"}.issubset(out.columns):
        before = len(out)
        valid = (
            (out["high"] >= out["low"])
            & (out["high"] >= out["close"])
            & (out["low"] <= out["close"])
            & (out["open"] > 0)
        )
        # Missing OHLC (some vendors omit them) should not delete the row.
        valid = valid | out[["high", "low", "open"]].isna().any(axis=1)
        out = out[valid]
        report.dropped_bad_ohlc = before - len(out)

    if "volume" in out.columns:
        out["volume"] = out["volume"].fillna(0).clip(lower=0)

    # --- derived --------------------------------------------------------
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    out["returns"] = out.groupby("ticker", observed=True)["close"].pct_change()

    extreme = out["returns"].abs() > MAX_ABS_DAILY_RETURN
    report.flagged_extreme_return = int(extreme.sum())
    if report.flagged_extreme_return:
        # Neutralise the return but keep the price row: a suspect tick should
        # not propagate into momentum, yet dropping the bar would create a
        # hole in the series.
        out.loc[extreme, "returns"] = np.nan
        report.notes.append(
            f"{report.flagged_extreme_return} daily returns beyond "
            f"±{MAX_ABS_DAILY_RETURN:.0%} nulled as suspected bad ticks"
        )

    if "volume" in out.columns:
        out["dollar_volume"] = out["close"] * out["volume"]

    report.rows_out = len(out)
    report.tickers_out = out["ticker"].nunique()

    log.info(
        "clean.done rows_in=%s rows_out=%s dropped=%s flagged=%s",
        report.rows_in,
        report.rows_out,
        report.rows_in - report.rows_out,
        report.flagged_extreme_return,
    )
    return out, report


def to_price_panel(df: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """Long format -> wide panel (index=date, columns=ticker).

    The vectorized engine and every factor operate on this shape.
    """
    panel = df.pivot_table(index="date", columns="ticker", values=field, aggfunc="last")
    panel.index = pd.to_datetime(panel.index)
    return panel.sort_index()
