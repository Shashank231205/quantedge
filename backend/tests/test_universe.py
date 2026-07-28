"""Point-in-time membership reconstruction.

Survivorship bias is the highest-leverage correctness issue in the whole
project: get it wrong and every downstream return figure is overstated. These
tests pin the interval algebra using hand-built change logs, so a regression
shows up here rather than as a suspiciously good Sharpe.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quantedge.ingestion.universe import (
    MembershipChange,
    _normalise_ticker,
    _parse_date,
    build_membership_intervals,
    members_as_of,
)

WINDOW_START = date(2019, 7, 1)


def _current(*tickers: str) -> pd.DataFrame:
    return pd.DataFrame({"ticker": list(tickers)})


def _no_overlaps(membership: pd.DataFrame) -> bool:
    for _, g in membership.groupby("ticker"):
        if len(g) < 2:
            continue
        g = g.sort_values("start_date")
        for i in range(len(g) - 1):
            end = g.iloc[i]["end_date"]
            if end is None or pd.isna(end) or end > g.iloc[i + 1]["start_date"]:
                return False
    return True


class TestTickerNormalisation:
    def test_dot_becomes_dash(self):
        # The index publishes BRK.B; Yahoo expects BRK-B.
        assert _normalise_ticker("BRK.B") == "BRK-B"

    def test_whitespace_and_case(self):
        assert _normalise_ticker("  aapl ") == "AAPL"


class TestDateParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("June 30, 2026", date(2026, 6, 30)),
            ("2019-07-01", date(2019, 7, 1)),
        ],
    )
    def test_supported_formats(self, raw, expected):
        assert _parse_date(raw) == expected

    def test_unparseable_returns_none(self):
        # Repeated header rows arrive as junk; they must be skipped, not raise.
        assert _parse_date("Effective Date") is None


class TestMembershipIntervals:
    def test_member_throughout_window_has_open_interval(self):
        m = build_membership_intervals(_current("AAA"), [], WINDOW_START)
        assert len(m) == 1
        assert m.iloc[0]["start_date"] == WINDOW_START
        assert m.iloc[0]["end_date"] is None

    def test_added_mid_window_starts_at_add_date(self):
        added = date(2021, 3, 1)
        changes = [MembershipChange(added, "AAA", None, "added")]
        m = build_membership_intervals(_current("AAA"), changes, WINDOW_START)
        assert len(m) == 1
        assert m.iloc[0]["start_date"] == added
        assert m.iloc[0]["end_date"] is None

    def test_removed_ticker_is_retained_with_end_date(self):
        """The whole point: a company dropped from the index must survive."""
        removed = date(2022, 6, 21)
        changes = [MembershipChange(removed, None, "OLD", "market cap")]
        m = build_membership_intervals(_current("AAA"), changes, WINDOW_START)

        old = m[m["ticker"] == "OLD"]
        assert len(old) == 1, "removed ticker must still appear in the universe"
        assert old.iloc[0]["start_date"] == WINDOW_START
        assert old.iloc[0]["end_date"] == removed

    def test_add_before_window_is_clamped_not_duplicated(self):
        """Regression: a pre-window add previously produced a second interval.

        IR was added 2020-03-02 and also back in 2010. The 2010 event must not
        manufacture an extra full-window interval.
        """
        changes = [
            MembershipChange(date(2020, 3, 2), "IR", None, "re-added"),
            MembershipChange(date(2010, 11, 17), "IR", None, "old add"),
        ]
        m = build_membership_intervals(_current("IR"), changes, WINDOW_START)
        assert len(m) == 1
        assert m.iloc[0]["start_date"] == date(2020, 3, 2)
        assert _no_overlaps(m)

    def test_removed_ticker_with_old_add_has_no_phantom_interval(self):
        """Regression: UA was removed in 2022 and had a 2016 add event.

        The stale add must not resurrect it as a currently-open member.
        """
        changes = [
            MembershipChange(date(2022, 6, 21), None, "UA", "market cap"),
            MembershipChange(date(2016, 4, 8), "UA", None, "old add"),
        ]
        m = build_membership_intervals(_current("AAA"), changes, WINDOW_START)
        ua = m[m["ticker"] == "UA"]
        assert len(ua) == 1
        assert ua.iloc[0]["end_date"] == date(2022, 6, 21)
        assert _no_overlaps(m)

    def test_spell_ending_before_window_is_dropped(self):
        changes = [MembershipChange(date(2019, 1, 1), None, "GONE", "before window")]
        m = build_membership_intervals(_current("AAA"), changes, WINDOW_START)
        assert "GONE" not in set(m["ticker"])

    def test_future_dated_change_ignored(self):
        future = date.today().replace(year=date.today().year + 5)
        changes = [MembershipChange(future, "SOON", None, "announced")]
        m = build_membership_intervals(_current("AAA"), changes, WINDOW_START)
        assert "SOON" not in set(m["ticker"])

    def test_same_day_add_and_remove_pair(self):
        d = date(2021, 9, 20)
        changes = [MembershipChange(d, "NEW", "OUT", "swap")]
        m = build_membership_intervals(_current("NEW"), changes, WINDOW_START)

        assert m[m["ticker"] == "NEW"].iloc[0]["start_date"] == d
        out = m[m["ticker"] == "OUT"]
        assert out.iloc[0]["end_date"] == d
        assert _no_overlaps(m)


class TestMembersAsOf:
    @pytest.fixture
    def membership(self):
        return pd.DataFrame(
            [
                {"ticker": "ALWAYS", "start_date": WINDOW_START, "end_date": None},
                {"ticker": "LEFT", "start_date": WINDOW_START, "end_date": date(2022, 1, 1)},
                {"ticker": "JOINED", "start_date": date(2023, 1, 1), "end_date": None},
            ]
        )

    def test_before_any_change(self, membership):
        assert members_as_of(membership, date(2021, 1, 1)) == ["ALWAYS", "LEFT"]

    def test_after_removal(self, membership):
        assert members_as_of(membership, date(2022, 6, 1)) == ["ALWAYS"]

    def test_after_addition(self, membership):
        assert members_as_of(membership, date(2024, 1, 1)) == ["ALWAYS", "JOINED"]

    def test_removal_date_is_exclusive(self, membership):
        """A name removed on D is not a member on D."""
        assert "LEFT" not in members_as_of(membership, date(2022, 1, 1))
