"""Leader election for redundant schedulers.

Running two schedulers is how the pipeline survives losing a machine; running
two schedulers that both fire every job is how it double-writes. So every
scheduler instance competes for one Postgres advisory lock, and only the
holder runs jobs. The others poll, and take over when the holder goes away.

Why an advisory lock: the database is already a hard dependency, it needs no
extra infrastructure, and a *session-level* lock is released by Postgres
itself when the holding connection dies. A crashed or partitioned leader
therefore cannot keep the lock — there is no lease to expire and no clock to
trust. Takeover time is bounded by how quickly Postgres notices the dead
connection plus one polling interval.

Two caveats, stated plainly:

* The lock must be taken over a direct connection. A transaction-mode pooler
  (PgBouncer, Neon's ``-pooler`` hosts) can hand the session to another
  client between statements, so :class:`PostgresAdvisoryLock` refuses such URLs.
* There is no fencing token. A leader that loses its connection mid-job
  finishes that job while the new leader may start the next. Every job here
  writes through idempotent upserts, so the overlap costs duplicate work, not
  corrupt data — that property is what makes this design sufficient.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Protocol

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.pool import NullPool

from quantedge.logging_config import get_logger
from quantedge.observability.metrics import LEADER_TRANSITIONS, SCHEDULER_LEADER

log = get_logger(__name__)


class LeaderLock(Protocol):
    def try_acquire(self) -> bool: ...
    def still_held(self) -> bool: ...
    def release(self) -> None: ...


class PostgresAdvisoryLock:
    """A session-level ``pg_try_advisory_lock`` on a dedicated connection."""

    def __init__(self, database_url: str, key: int) -> None:
        if "-pooler." in database_url or "pgbouncer=true" in database_url:
            raise ValueError(
                "leader election needs a direct database connection; "
                "a transaction pooler cannot hold a session-level lock"
            )
        # NullPool: closing the connection must really close it, so the lock
        # goes with it. A pooled connection returned to the pool would keep
        # holding the lock on behalf of nobody.
        self._engine: Engine = create_engine(
            database_url, poolclass=NullPool, connect_args={"connect_timeout": 5}
        )
        self._key = key
        self._conn: Connection | None = None

    def try_acquire(self) -> bool:
        conn = None
        try:
            conn = self._engine.connect()
            got = bool(
                conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": self._key}).scalar()
            )
            # Session locks outlive the transaction; committing avoids leaving
            # the connection idle-in-transaction, which servers may time out.
            conn.commit()
        except Exception as exc:
            log.warning("leader.acquire_failed error=%s", exc)
            if conn is not None:
                conn.close()
            return False
        if got:
            self._conn = conn
        else:
            conn.close()
        return got

    def still_held(self) -> bool:
        """True only if the connection is alive *and* still holds the lock."""
        if self._conn is None:
            return False
        try:
            # A bigint key is stored split across classid (high 32 bits) and
            # objid (low 32 bits), with objsubid 1 marking the bigint form.
            held = self._conn.execute(
                text(
                    "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' "
                    "AND pid = pg_backend_pid() AND granted "
                    "AND classid = :hi AND objid = :lo AND objsubid = 1"
                ),
                {"hi": (self._key >> 32) & 0xFFFFFFFF, "lo": self._key & 0xFFFFFFFF},
            ).scalar()
            self._conn.commit()
        except Exception as exc:
            log.warning("leader.heartbeat_failed error=%s", exc)
            self._drop()
            return False
        if not held:
            self._drop()
        return bool(held)

    def release(self) -> None:
        if self._conn is None:
            return
        try:
            self._conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": self._key})
            self._conn.commit()
        except Exception:
            log.debug("leader.unlock_failed; closing the connection releases it")
        finally:
            self._drop()

    def _drop(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                log.debug("leader.close_failed", exc_info=True)
            self._conn = None


class LeaderElector:
    """Polls a :class:`LeaderLock` and reports transitions through callbacks.

    ``step`` is one round of the loop and is what the tests drive; ``run``
    repeats it until ``stop`` is set. Callbacks run on the elector's thread.
    """

    def __init__(
        self,
        lock: LeaderLock,
        on_elected: Callable[[], None],
        on_demoted: Callable[[], None],
        retry_seconds: float = 5.0,
        name: str = "scheduler",
    ) -> None:
        self._lock = lock
        self._on_elected = on_elected
        self._on_demoted = on_demoted
        self._retry = retry_seconds
        self._name = name
        self.is_leader = False

    def step(self) -> bool:
        if self.is_leader:
            if not self._lock.still_held():
                self._demote("lost")
        elif self._lock.try_acquire():
            self.is_leader = True
            SCHEDULER_LEADER.set(1)
            LEADER_TRANSITIONS.labels(transition="elected").inc()
            log.info("leader.elected instance=%s", self._name)
            self._on_elected()
        return self.is_leader

    def _demote(self, reason: str) -> None:
        self.is_leader = False
        SCHEDULER_LEADER.set(0)
        LEADER_TRANSITIONS.labels(transition="demoted").inc()
        log.warning("leader.demoted instance=%s reason=%s", self._name, reason)
        self._on_demoted()

    def run(self, stop: threading.Event) -> None:
        try:
            while not stop.is_set():
                self.step()
                stop.wait(self._retry)
        finally:
            if self.is_leader:
                self._lock.release()
                self._demote("shutdown")
