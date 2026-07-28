"""Data-source abstraction.

Every provider implements the same narrow interface so the rest of the system
never learns which vendor supplied a bar. yfinance is the default because it
needs no key; Polygon slots in behind the same protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

# Canonical column set every source must return.
OHLCV_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


@dataclass
class FetchResult:
    """Outcome of one fetch, including partial failures.

    Failures are data, not exceptions: a 500-ticker backfill where 3 symbols
    are delisted should still succeed, and the telemetry should say so.
    """

    data: pd.DataFrame
    succeeded: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    source: str = ""

    @property
    def success_rate(self) -> float:
        total = len(self.succeeded) + len(self.failed)
        return len(self.succeeded) / total if total else 0.0

    @property
    def n_rows(self) -> int:
        return len(self.data)


class DataSource(ABC):
    """Interface all market-data providers implement."""

    name: str = "base"

    @abstractmethod
    def fetch_ohlcv(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> FetchResult:
        """Return a long-format frame indexed by (ticker, date).

        Columns must match ``OHLCV_COLUMNS``. Implementations should never
        raise for a single bad ticker — record it in ``failed`` instead.
        """

    @abstractmethod
    def fetch_metadata(self, tickers: list[str]) -> pd.DataFrame:
        """Return reference data (sector, industry, name) indexed by ticker."""

    def health_check(self) -> bool:
        """Cheap probe used by the System Health screen."""
        try:
            today = date.today()
            res = self.fetch_ohlcv(["SPY"], today.replace(year=today.year - 1), today)
            return res.n_rows > 0
        except Exception:
            return False
