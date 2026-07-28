"""Ingestion orchestration: universe -> fetch -> clean -> persist.

Every entry point here runs inside ``track_job`` so the System Health screen
reports what actually happened rather than a decorative constant.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from quantedge.config import settings
from quantedge.db.models import OhlcvClean, Security, UniverseMembership
from quantedge.db.session import session_scope
from quantedge.ingestion.base import DataSource
from quantedge.ingestion.cleaning import clean_ohlcv
from quantedge.ingestion.telemetry import track_job
from quantedge.ingestion.universe import build_universe, members_as_of
from quantedge.ingestion.yfinance_source import YFinanceSource
from quantedge.logging_config import get_logger

log = get_logger(__name__)

CHUNK = 5_000


def get_source(name: str | None = None) -> DataSource:
    """Resolve the configured provider. Polygon slots in behind this seam."""
    name = (name or settings.data_source).lower()
    if name == "yfinance":
        return YFinanceSource()
    raise ValueError(f"unknown data source: {name!r}")


def history_start(years: int | None = None) -> date:
    years = years or settings.history_years
    return date.today() - timedelta(days=int(years * 365.25))


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------


def refresh_universe(years: int | None = None) -> dict:
    """Rebuild point-in-time membership and the securities reference table."""
    start = history_start(years)

    with track_job("universe_refresh") as ctx:
        current, membership = build_universe(start)

        with session_scope() as s:
            # Membership is fully derived; rebuild rather than merge.
            s.execute(delete(UniverseMembership))
            s.add_all(
                [
                    UniverseMembership(
                        ticker=r.ticker,
                        index_name="SP500",
                        start_date=r.start_date,
                        end_date=r.end_date,
                    )
                    for r in membership.itertuples()
                ]
            )

            active = set(current["ticker"])
            for r in current.itertuples():
                s.merge(
                    Security(
                        ticker=r.ticker,
                        name=r.name,
                        sector=r.sector,
                        industry=r.industry,
                        is_active=True,
                    )
                )
            # Ever-members that are no longer in the index still need a row so
            # sector lookups work for historical holdings.
            for t in set(membership["ticker"]) - active:
                s.merge(Security(ticker=t, is_active=False))

        ctx.rows_written = len(membership)
        ctx.tickers_processed = int(membership["ticker"].nunique())
        ctx.details = {
            "current_members": len(current),
            "ever_members": int(membership["ticker"].nunique()),
            "removed_in_window": int(membership["end_date"].notna().sum()),
            "history_start": start.isoformat(),
        }

    log.info("universe.refreshed %s", ctx.details)
    return ctx.details


def load_membership() -> pd.DataFrame:
    with session_scope() as s:
        rows = s.scalars(select(UniverseMembership)).all()
        return pd.DataFrame(
            [
                {"ticker": r.ticker, "start_date": r.start_date, "end_date": r.end_date}
                for r in rows
            ]
        )


def universe_tickers() -> list[str]:
    """Every ticker that was ever a member in the window, plus the benchmark."""
    m = load_membership()
    tickers = sorted(m["ticker"].unique()) if not m.empty else []
    if settings.benchmark_ticker not in tickers:
        tickers.append(settings.benchmark_ticker)
    return tickers


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------


def _upsert_clean(rows: list[dict]) -> int:
    """Bulk upsert into ohlcv_clean, chunked to keep statements bounded."""
    written = 0
    with session_scope() as s:
        for i in range(0, len(rows), CHUNK):
            chunk = rows[i : i + CHUNK]
            stmt = pg_insert(OhlcvClean).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=["ticker", "date"],
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "volume": stmt.excluded.volume,
                    "dollar_volume": stmt.excluded.dollar_volume,
                    "returns": stmt.excluded.returns,
                },
            )
            s.execute(stmt)
            written += len(chunk)
    return written


def ingest_prices(
    tickers: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    source_name: str | None = None,
) -> dict:
    """Fetch, clean and persist OHLCV for the universe."""
    start = start or history_start()
    end = end or date.today()
    tickers = tickers or universe_tickers()

    if not tickers:
        raise RuntimeError("universe is empty — run refresh_universe() first")

    with track_job("ohlcv_ingest") as ctx:
        source = get_source(source_name)
        log.info(
            "ingest.start tickers=%s start=%s end=%s source=%s",
            len(tickers), start, end, source.name,
        )

        result = source.fetch_ohlcv(tickers, start, end)
        cleaned, report = clean_ohlcv(result.data)

        rows: list[dict] = []
        if not cleaned.empty:
            for r in cleaned.itertuples():
                rows.append(
                    {
                        "ticker": r.ticker,
                        "date": r.date,
                        "open": float(r.open) if pd.notna(r.open) else float(r.close),
                        "high": float(r.high) if pd.notna(r.high) else float(r.close),
                        "low": float(r.low) if pd.notna(r.low) else float(r.close),
                        "close": float(r.close),
                        "volume": float(getattr(r, "volume", 0) or 0),
                        "dollar_volume": float(getattr(r, "dollar_volume", 0) or 0),
                        "returns": float(r.returns) if pd.notna(r.returns) else None,
                    }
                )

        written = _upsert_clean(rows)

        ctx.rows_written = written
        ctx.tickers_processed = len(result.succeeded)
        ctx.bytes_processed = int(cleaned.memory_usage(deep=True).sum()) if not cleaned.empty else 0
        ctx.details = {
            "source": result.source,
            "requested": len(tickers),
            "succeeded": len(result.succeeded),
            "failed": len(result.failed),
            "fetch_success_rate": round(result.success_rate * 100, 2),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "cleaning": report.as_dict(),
            # Keep a bounded sample; the full list would bloat the row.
            "failed_sample": dict(list(result.failed.items())[:20]),
        }

    log.info("ingest.done rows=%s tickers=%s", ctx.rows_written, ctx.tickers_processed)
    return ctx.details


def ingest_incremental(lookback_days: int = 7) -> dict:
    """Daily top-up: refetch a short window and upsert.

    Overlapping the window means late corrections and restatements land
    without a full backfill.
    """
    return ingest_prices(start=date.today() - timedelta(days=lookback_days))


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def load_prices(
    tickers: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Read cleaned OHLCV back out as a long-format frame."""
    with session_scope() as s:
        stmt = select(OhlcvClean)
        if tickers:
            stmt = stmt.where(OhlcvClean.ticker.in_(tickers))
        if start:
            stmt = stmt.where(OhlcvClean.date >= start)
        if end:
            stmt = stmt.where(OhlcvClean.date <= end)

        rows = s.scalars(stmt.order_by(OhlcvClean.ticker, OhlcvClean.date)).all()
        return pd.DataFrame(
            [
                {
                    "ticker": r.ticker,
                    "date": r.date,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                    "dollar_volume": r.dollar_volume,
                    "returns": r.returns,
                }
                for r in rows
            ]
        )


def coverage_stats() -> dict:
    """Row/ticker/date coverage — feeds the System Health screen."""
    with session_scope() as s:
        n_rows = s.scalar(select(func.count()).select_from(OhlcvClean)) or 0
        n_tickers = s.scalar(select(func.count(func.distinct(OhlcvClean.ticker)))) or 0
        d_min = s.scalar(select(func.min(OhlcvClean.date)))
        d_max = s.scalar(select(func.max(OhlcvClean.date)))
        n_sec = s.scalar(select(func.count()).select_from(Security)) or 0

    years = round((d_max - d_min).days / 365.25, 2) if d_min and d_max else 0.0
    return {
        "total_rows": n_rows,
        "n_tickers": n_tickers,
        "n_securities": n_sec,
        "first_date": d_min.isoformat() if d_min else None,
        "last_date": d_max.isoformat() if d_max else None,
        "years_of_history": years,
    }


def universe_as_of(as_of: date) -> list[str]:
    """Point-in-time membership lookup used by the backtest."""
    return members_as_of(load_membership(), as_of)
