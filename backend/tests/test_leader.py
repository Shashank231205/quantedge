"""Scheduler leader election.

The unit tests drive two electors against an in-memory lock server that
behaves like a Postgres advisory lock: one holder at a time, released when the
holder's session dies. The integration test does the same against a real
database, killing the leader's backend with ``pg_terminate_backend`` — the
closest thing to pulling the plug on a machine that a test can do.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text

from quantedge.ingestion.leader import LeaderElector, PostgresAdvisoryLock


class LockServer:
    def __init__(self) -> None:
        self.holder: str | None = None


class FakeLock:
    def __init__(self, server: LockServer, name: str) -> None:
        self.server, self.name, self.alive = server, name, True

    def try_acquire(self) -> bool:
        if self.alive and self.server.holder in (None, self.name):
            self.server.holder = self.name
            return True
        return False

    def still_held(self) -> bool:
        return self.alive and self.server.holder == self.name

    def release(self) -> None:
        if self.server.holder == self.name:
            self.server.holder = None

    def crash(self) -> None:
        """The session dies; the server frees the lock, as Postgres does."""
        self.alive = False
        self.release()


def elector(lock, events: list[str]) -> LeaderElector:
    return LeaderElector(
        lock,
        on_elected=lambda: events.append(f"{lock.name}:elected"),
        on_demoted=lambda: events.append(f"{lock.name}:demoted"),
        retry_seconds=0,
        name=lock.name,
    )


class TestElection:
    def test_exactly_one_leader(self):
        server, events = LockServer(), []
        a, b = elector(FakeLock(server, "a"), events), elector(FakeLock(server, "b"), events)
        for _ in range(5):
            a.step()
            b.step()
        assert (a.is_leader, b.is_leader) == (True, False)
        assert events == ["a:elected"]

    def test_standby_takes_over_when_leader_dies(self):
        server, events = LockServer(), []
        lock_a = FakeLock(server, "a")
        a, b = elector(lock_a, events), elector(FakeLock(server, "b"), events)
        a.step()
        b.step()

        lock_a.crash()
        a.step()  # the old leader notices on its next heartbeat and stands down
        b.step()  # the standby acquires on its next poll

        assert (a.is_leader, b.is_leader) == (False, True)
        assert events == ["a:elected", "a:demoted", "b:elected"]

    def test_shutdown_releases_for_the_standby(self):
        import threading

        server, events = LockServer(), []
        a, b = elector(FakeLock(server, "a"), events), elector(FakeLock(server, "b"), events)
        stop = threading.Event()
        a.step()
        stop.set()
        a.run(stop)  # exits immediately and releases in its finally block
        assert server.holder is None
        assert b.step()

    def test_pooler_urls_are_refused(self):
        with pytest.raises(ValueError, match="direct database connection"):
            PostgresAdvisoryLock("postgresql+psycopg://u:p@ep-x-pooler.neon.tech/db", key=1)


@pytest.mark.integration
class TestPostgresAdvisoryLock:
    """Needs a live database; CI provides one."""

    @pytest.fixture
    def url(self):
        url = os.environ.get("DATABASE_URL")
        if not url:
            pytest.skip("DATABASE_URL not set")
        try:
            with create_engine(url).connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception:
            pytest.skip("database unreachable")
        return url

    def test_failover_when_leader_backend_is_killed(self, url):
        key = 990_001
        a, b = PostgresAdvisoryLock(url, key), PostgresAdvisoryLock(url, key)
        try:
            assert a.try_acquire()
            assert not b.try_acquire()
            assert a.still_held()

            leader_pid = a._conn.execute(text("SELECT pg_backend_pid()")).scalar()
            a._conn.commit()
            with create_engine(url).connect() as admin:
                admin.execute(text("SELECT pg_terminate_backend(:p)"), {"p": leader_pid})
                admin.commit()

            assert not a.still_held()
            assert b.try_acquire()
            assert b.still_held()
        finally:
            a.release()
            b.release()

    def test_release_hands_over_immediately(self, url):
        key = 990_002
        a, b = PostgresAdvisoryLock(url, key), PostgresAdvisoryLock(url, key)
        try:
            assert a.try_acquire()
            a.release()
            assert b.try_acquire()
        finally:
            a.release()
            b.release()
