"""yfinance implementation of :class:`DataSource`.

Chosen as the default because it needs no API key and can serve a full
S&P 500 × 6-year backfill. Batched, throttled and Parquet-cached so a repeat
run never re-hits the provider.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from quantedge.config import settings
from quantedge.ingestion.base import DataSource, FetchResult
from quantedge.logging_config import get_logger

log = get_logger(__name__)


class YFinanceSource(DataSource):
    name = "yfinance"

    def __init__(self, cache_dir: str | None = None, use_cache: bool = True) -> None:
        self.cache_dir = Path(cache_dir or settings.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache

    # -- internals ------------------------------------------------------

    def _cache_path(self, tickers: list[str], start: date, end: date) -> Path:
        # Hash keeps the filename bounded regardless of batch size.
        key = f"{len(tickers)}_{hash(tuple(sorted(tickers))) & 0xFFFFFFFF:08x}"
        return self.cache_dir / f"ohlcv_{key}_{start}_{end}.parquet"

    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=settings.retry_backoff_seconds, max=30),
        reraise=True,
    )
    def _download(self, batch: list[str], start: date, end: date) -> pd.DataFrame:
        # auto_adjust=False keeps raw Close alongside Adj Close: the audit
        # table stores what the vendor said, factors use the adjusted series.
        return yf.download(
            batch,
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="column",
        )

    @staticmethod
    def _to_long(df: pd.DataFrame, batch: list[str]) -> pd.DataFrame:
        """Normalise yfinance output to long format with canonical columns."""
        if df.empty:
            return pd.DataFrame()

        # Single ticker comes back with flat columns; make it uniform.
        if not isinstance(df.columns, pd.MultiIndex):
            df = pd.concat({batch[0]: df}, axis=1)
            df.columns = df.columns.swaplevel(0, 1)

        long = df.stack(level=1, future_stack=True)
        long.index.names = ["date", "ticker"]

        rename = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        long = long.rename(columns=rename)

        keep = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in long.columns]
        long = long[keep].reset_index()

        # Drop rows where the provider returned nothing usable.
        long = long.dropna(subset=["close"], how="any")
        long["date"] = pd.to_datetime(long["date"]).dt.date
        return long

    # -- interface ------------------------------------------------------

    def fetch_ohlcv(
        self, tickers: list[str], start: date, end: date
    ) -> FetchResult:
        cache = self._cache_path(tickers, start, end)
        if self.use_cache and cache.exists():
            log.info("cache.hit path=%s", cache.name)
            data = pd.read_parquet(cache)
            present = set(data["ticker"].unique()) if not data.empty else set()
            return FetchResult(
                data=data,
                succeeded=sorted(present),
                failed={t: "no data" for t in tickers if t not in present},
                source=self.name,
            )

        frames: list[pd.DataFrame] = []
        succeeded: list[str] = []
        failed: dict[str, str] = {}

        for i in range(0, len(tickers), settings.batch_size):
            batch = tickers[i : i + settings.batch_size]
            n = i // settings.batch_size + 1
            try:
                raw = self._download(batch, start, end)
                long = self._to_long(raw, batch)
                if not long.empty:
                    frames.append(long)
                    got = set(long["ticker"].unique())
                else:
                    got = set()
                succeeded.extend(sorted(got))
                for t in batch:
                    if t not in got:
                        failed[t] = "no data returned"
                log.info(
                    "fetch.batch n=%s size=%s ok=%s rows=%s",
                    n, len(batch), len(got), len(long),
                )
            except Exception as exc:
                # One bad batch must not abort a 500-ticker backfill.
                for t in batch:
                    failed[t] = f"{type(exc).__name__}: {exc}"[:200]
                log.warning("fetch.batch_failed n=%s error=%s", n, exc)

            if i + settings.batch_size < len(tickers):
                time.sleep(settings.request_delay_seconds)

        data = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(
                columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
            )
        )

        if self.use_cache and not data.empty:
            data.to_parquet(cache, index=False)
            log.info("cache.write path=%s rows=%s", cache.name, len(data))

        return FetchResult(
            data=data, succeeded=succeeded, failed=failed, source=self.name
        )

    def fetch_metadata(self, tickers: list[str]) -> pd.DataFrame:
        """Sector/industry lookup. Slow (one call per ticker), so cached."""
        cache = self.cache_dir / f"metadata_{len(tickers)}.parquet"
        if self.use_cache and cache.exists():
            return pd.read_parquet(cache)

        rows = []
        for t in tickers:
            try:
                info = yf.Ticker(t).info
                rows.append(
                    {
                        "ticker": t,
                        "name": info.get("longName") or info.get("shortName"),
                        "sector": info.get("sector"),
                        "industry": info.get("industry"),
                        "exchange": info.get("exchange"),
                    }
                )
            except Exception as exc:
                log.debug("metadata.failed ticker=%s error=%s", t, exc)
                rows.append({"ticker": t, "name": None, "sector": None,
                             "industry": None, "exchange": None})
            time.sleep(0.05)

        df = pd.DataFrame(rows)
        if self.use_cache and not df.empty:
            df.to_parquet(cache, index=False)
        return df
