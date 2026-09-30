"""Engine and session management."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from quantedge.config import settings


def connect_args(url: str) -> dict:
    """Driver options for the configured database.

    Transaction-mode poolers (Neon's ``-pooler`` hosts, PgBouncer) hand each
    transaction to whichever server connection is free, so a statement that
    psycopg prepared on one connection may not exist on the next. Turning
    server-side preparation off makes the app safe behind either kind of URL.
    """
    args: dict = {"connect_timeout": 5}
    if "-pooler." in url or "pgbouncer=true" in url:
        args["prepare_threshold"] = None
    return args


engine = create_engine(
    settings.database_url,
    echo=settings.db_echo,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    # Without a bound, a connection attempt to an unreachable host waits on the
    # OS TCP timeout — minutes — and every readiness probe queued behind it
    # times out too. Five seconds lets the probe report the outage instead.
    connect_args=connect_args(settings.database_url),
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Transactional scope for scripts and jobs."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
