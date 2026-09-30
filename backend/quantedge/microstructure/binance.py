"""Binance public market-data archives.

data.binance.vision publishes daily zip files for USDⓈ-M futures, each with a
SHA-256 ``.CHECKSUM`` alongside. Two are used here:

* ``bookTicker`` — every change to the best bid/ask price or size, with
  exchange timestamps. That is a complete L1 event stream, which is exactly
  what order-flow imbalance needs: OFI is defined on consecutive changes at the
  touch, and a sampled snapshot feed would silently drop most of them.
* ``aggTrades`` — every trade, with the aggressor side. Trade flow and, more
  importantly, trade-through prices for the limit-order fill simulation.

The bookTicker archive was discontinued after 2024-03-30, so research windows
must fall between 2023-05-16 and that date.

Files are large (50-350MB zipped per symbol-day for bookTicker) and never
change once published, so each is downloaded once, verified, and kept.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from quantedge.logging_config import get_logger

log = get_logger(__name__)

BASE_URL = "https://data.binance.vision/data/futures/um/daily"
KINDS = ("bookTicker", "aggTrades")

#: First and last day for which Binance published futures bookTicker files.
BOOK_TICKER_FIRST = date(2023, 5, 16)
BOOK_TICKER_LAST = date(2024, 3, 30)

BOOK_COLUMNS = {
    "best_bid_price": "bid_px",
    "best_bid_qty": "bid_qty",
    "best_ask_price": "ask_px",
    "best_ask_qty": "ask_qty",
    "transaction_time": "ts_ms",
}
TRADE_COLUMNS = {
    "price": "price",
    "quantity": "qty",
    "transact_time": "ts_ms",
    "is_buyer_maker": "is_buyer_maker",
}


class ChecksumMismatch(RuntimeError):
    """A download does not match the digest Binance published for it."""


def date_range(start: date, end: date) -> list[date]:
    days = (end - start).days
    if days < 0:
        raise ValueError(f"end {end} is before start {start}")
    return [start + timedelta(days=i) for i in range(days + 1)]


def archive_name(symbol: str, kind: str, day: date) -> str:
    return f"{symbol}-{kind}-{day.isoformat()}.zip"


def archive_url(symbol: str, kind: str, day: date) -> str:
    return f"{BASE_URL}/{kind}/{symbol}/{archive_name(symbol, kind, day)}"


def sha256_of(path: Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(block):
            digest.update(chunk)
    return digest.hexdigest()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, max=30), reraise=True)
def _get(url: str, dest: Path | None = None, timeout: float = 60.0) -> str | None:
    with requests.get(url, stream=dest is not None, timeout=timeout) as resp:
        resp.raise_for_status()
        if dest is None:
            return resp.text
        # Write to a temporary name and rename on completion, so an interrupted
        # download can never be mistaken for a finished one on the next run.
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        tmp.replace(dest)
    return None


def download(symbol: str, kind: str, day: date, cache_dir: Path) -> Path:
    """Fetch one archive into ``cache_dir`` and verify it against its checksum.

    A verified file already on disk is not fetched again. A file that fails
    verification is deleted before raising, so a retry starts clean.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown archive kind {kind!r}; expected one of {KINDS}")
    if kind == "bookTicker" and not BOOK_TICKER_FIRST <= day <= BOOK_TICKER_LAST:
        raise ValueError(
            f"Binance published futures bookTicker only for "
            f"{BOOK_TICKER_FIRST}..{BOOK_TICKER_LAST}; {day} is outside that window"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / archive_name(symbol, kind, day)
    checksum_file = dest.with_suffix(".zip.CHECKSUM")
    url = archive_url(symbol, kind, day)

    if not checksum_file.exists():
        text = _get(url + ".CHECKSUM")
        checksum_file.write_text(text or "")
    expected = checksum_file.read_text().split()[0].strip().lower()

    if dest.exists() and sha256_of(dest) == expected:
        return dest

    log.info("micro.download url=%s", url)
    _get(url, dest=dest, timeout=300.0)

    actual = sha256_of(dest)
    if actual != expected:
        dest.unlink(missing_ok=True)
        raise ChecksumMismatch(f"{dest.name}: expected {expected}, got {actual}")
    return dest


def _has_header(path: Path) -> bool:
    """Older Binance archives ship without a header row; newer ones have one."""
    first = pd.read_csv(path, compression="zip", nrows=1, header=None)
    return not str(first.iloc[0, 0]).replace(".", "").isdigit()


def read_book_ticker(path: Path, chunksize: int = 2_000_000) -> Iterator[pd.DataFrame]:
    """Stream a bookTicker archive in chunks.

    A liquid symbol produces 15-40 million quote updates a day. Reading that in
    one go costs several gigabytes, so callers get it in bounded chunks and are
    expected to aggregate as they go.
    """
    names = ["update_id", *BOOK_COLUMNS, "event_time"]
    header = 0 if _has_header(path) else None
    reader = pd.read_csv(
        path, compression="zip", header=header, names=names,
        usecols=list(BOOK_COLUMNS), chunksize=chunksize,
        dtype={c: "float64" for c in BOOK_COLUMNS if c != "transaction_time"}
        | {"transaction_time": "int64"},
    )
    for chunk in reader:
        yield chunk.rename(columns=BOOK_COLUMNS)[list(BOOK_COLUMNS.values())]


def read_agg_trades(path: Path) -> pd.DataFrame:
    """Load an aggTrades archive. Small enough (~1-3M rows/day) to read whole."""
    names = [
        "agg_trade_id", "price", "quantity", "first_trade_id",
        "last_trade_id", "transact_time", "is_buyer_maker",
    ]
    header = 0 if _has_header(path) else None
    df = pd.read_csv(
        path, compression="zip", header=header, names=names,
        usecols=list(TRADE_COLUMNS),
        dtype={"price": "float64", "quantity": "float64", "transact_time": "int64"},
    )
    df = df.rename(columns=TRADE_COLUMNS)
    # The flag arrives as the strings "true"/"false" in some files and as
    # booleans in others; normalise rather than trust the parser's guess.
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).str.lower().eq("true")
    return df[list(TRADE_COLUMNS.values())]
