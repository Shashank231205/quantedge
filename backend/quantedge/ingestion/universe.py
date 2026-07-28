"""S&P 500 universe construction with point-in-time membership.

Survivorship bias is the single most common way a backtest flatters itself:
if you run today's index members over the last six years, you have quietly
excluded every company that was dropped for performing badly, and your
returns are overstated.

We avoid that by reconstructing history. Wikipedia publishes both the current
constituent list and a dated table of additions/removals. Replaying those
changes backwards from today yields the membership set as of any past date,
including names that have since been removed.

Where a removed ticker no longer has price data available from the provider,
that is recorded rather than silently dropped, so the residual bias is
measurable instead of invisible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from io import StringIO

import pandas as pd
import requests

from quantedge.logging_config import get_logger

log = get_logger(__name__)

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) QuantEdge/0.1 research"


@dataclass
class MembershipChange:
    effective_date: date
    added: str | None
    removed: str | None
    reason: str


def _normalise_ticker(t: str) -> str:
    """Yahoo uses ``BRK-B`` where the index publishes ``BRK.B``."""
    return str(t).strip().upper().replace(".", "-")


def _parse_date(value: str) -> date | None:
    for fmt in ("%B %d, %Y", "%Y-%m-%d", "%b %d, %Y"):
        try:
            return datetime.strptime(str(value).strip(), fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def fetch_wikipedia_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (current constituents, historical changes)."""
    resp = requests.get(WIKI_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    if len(tables) < 2:
        raise RuntimeError(f"expected >=2 tables on the page, found {len(tables)}")
    return tables[0], tables[1]


def parse_current_constituents(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "ticker": df["Symbol"].map(_normalise_ticker),
            "name": df["Security"],
            "sector": df["GICS Sector"],
            "industry": df["GICS Sub-Industry"],
            "date_added": df["Date added"].map(_parse_date),
        }
    )
    return out.drop_duplicates(subset="ticker").reset_index(drop=True)


def parse_changes(df: pd.DataFrame) -> list[MembershipChange]:
    """Flatten the MultiIndex add/remove table into dated events."""
    # Columns arrive as ('Effective Date','Effective Date'), ('Added','Ticker'), ...
    flat = df.copy()
    flat.columns = [
        "_".join(str(p) for p in col).strip() if isinstance(col, tuple) else str(col)
        for col in flat.columns
    ]

    def pick(*fragments: str) -> str | None:
        for c in flat.columns:
            low = c.lower()
            if all(f in low for f in fragments):
                return c
        return None

    c_date = pick("effective", "date")
    c_add = pick("added", "ticker")
    c_rem = pick("removed", "ticker")
    c_why = pick("reason")

    if not (c_date and c_add and c_rem):
        raise RuntimeError(f"unexpected changes-table columns: {flat.columns.tolist()}")

    changes: list[MembershipChange] = []
    for _, row in flat.iterrows():
        eff = _parse_date(row[c_date])
        if eff is None:
            continue  # header repeat or malformed row
        added = row[c_add]
        removed = row[c_rem]
        changes.append(
            MembershipChange(
                effective_date=eff,
                added=_normalise_ticker(added) if pd.notna(added) else None,
                removed=_normalise_ticker(removed) if pd.notna(removed) else None,
                reason=re.sub(r"\[\d+\]", "", str(row[c_why])) if c_why else "",
            )
        )
    return changes


def build_membership_intervals(
    current: pd.DataFrame,
    changes: list[MembershipChange],
    history_start: date,
) -> pd.DataFrame:
    """Reconstruct (ticker, start_date, end_date) intervals back to ``history_start``.

    Algorithm: start from today's members, walk the change log from newest to
    oldest, and invert each event.

      * A ticker *added* on date D was not a member before D -> its interval
        starts at D.
      * A ticker *removed* on date D was a member before D -> re-open an
        interval ending at D.

    ``end_date`` of NaT/None means "still a member".
    """
    today = date.today()

    # `pending[t]` is the end_date of the interval we are currently tracing
    # backwards for ticker t. None means "still a member today".
    pending: dict[str, date | None] = {t: None for t in current["ticker"]}
    intervals: list[dict] = []

    def close(ticker: str, start: date, end: date | None) -> None:
        # Ignore intervals that finished before our window opened.
        if end is not None and end <= history_start:
            return
        start = max(start, history_start)
        # Degenerate spans (added and removed the same day, or an inverted
        # pair from a malformed source row) carry no membership.
        if end is not None and end <= start:
            return
        intervals.append(
            {"ticker": ticker, "start_date": start, "end_date": end}
        )

    # Walk newest -> oldest, inverting each event.
    for ch in sorted(changes, key=lambda c: c.effective_date, reverse=True):
        if ch.effective_date > today:
            continue  # announced but not yet effective

        if ch.added is not None and ch.added in pending:
            # Membership began on this date, so the interval we were tracing
            # for this ticker starts here and is now fully determined.
            # Only meaningful if we were actually tracing this ticker — an
            # 'added' event for a ticker we are not tracing describes a
            # membership spell that already closed, and the matching
            # 'removed' event will open it when we reach it.
            end = pending.pop(ch.added)
            close(ch.added, ch.effective_date, end)

        if ch.removed is not None:
            # Was a member up to this date. Begin tracing a new (earlier)
            # interval that ends here; its start emerges from an older
            # 'added' event, or defaults to the window start.
            #
            # If we are already tracing an interval for this ticker, the
            # ticker was removed, later re-added, and removed again. The
            # interval currently being traced must start no earlier than the
            # date we are about to close at, so emit it bounded below by this
            # event rather than by history_start (which would overlap).
            if ch.removed in pending:
                close(ch.removed, ch.effective_date, pending[ch.removed])
            # A spell that ended at or before the window start contributes
            # nothing; don't trace it (it would otherwise be emitted by the
            # final sweep as a spurious full-window interval).
            if ch.effective_date > history_start:
                pending[ch.removed] = ch.effective_date
            else:
                pending.pop(ch.removed, None)

    # Note: we deliberately process events older than ``history_start`` too.
    # An 'added' event before the window start resolves the true start of a
    # spell that extends into the window; ``close`` clamps it to the window.

    # Anything still pending was a member from before the window opened.
    for ticker, end in pending.items():
        close(ticker, history_start, end)

    df = pd.DataFrame(intervals)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["ticker", "start_date", "end_date"])
    return df.sort_values(["ticker", "start_date"]).reset_index(drop=True)


def members_as_of(membership: pd.DataFrame, as_of: date) -> list[str]:
    """Tickers that were index members on ``as_of``."""
    started = membership["start_date"] <= as_of
    not_ended = membership["end_date"].isna() | (membership["end_date"] > as_of)
    return sorted(membership.loc[started & not_ended, "ticker"].unique())


def build_universe(history_start: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and assemble (constituents, membership intervals)."""
    log.info("universe.fetch source=wikipedia")
    cur_raw, chg_raw = fetch_wikipedia_tables()

    current = parse_current_constituents(cur_raw)
    changes = parse_changes(chg_raw)
    log.info(
        "universe.parsed current=%s changes=%s", len(current), len(changes)
    )

    membership = build_membership_intervals(current, changes, history_start)

    n_delisted = int(membership["end_date"].notna().sum())
    log.info(
        "universe.built intervals=%s ever_members=%s removed_in_window=%s",
        len(membership),
        membership["ticker"].nunique(),
        n_delisted,
    )
    return current, membership
