"""SQLAlchemy ORM models.

Design notes:
  * ``ohlcv_raw`` is an immutable audit trail of exactly what the provider
    returned. ``ohlcv_clean`` is the derived, adjusted series everything
    downstream reads. Keeping both means a data-quality question can always
    be answered after the fact.
  * ``job_runs`` and ``api_request_log`` are not decoration: they are the
    real source for the pipeline-uptime and p95-latency figures shown on the
    System Health screen. No hardcoded telemetry anywhere in this project.
  * ``universe_membership`` carries start/end dates so the backtest can ask
    "who was in the index on date t" rather than assuming today's members
    were always members (survivorship bias).
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------


class Security(Base):
    __tablename__ = "securities"

    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(255))
    sector: Mapped[str | None] = mapped_column(String(64), index=True)
    industry: Mapped[str | None] = mapped_column(String(128))
    exchange: Mapped[str | None] = mapped_column(String(32))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    first_date: Mapped[date | None] = mapped_column(Date)
    last_date: Mapped[date | None] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class UniverseMembership(Base):
    """Point-in-time index membership.

    ``end_date IS NULL`` means still a member. Recording removals is what lets
    a backtest include names that were later delisted or dropped.
    """

    __tablename__ = "universe_membership"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    index_name: Mapped[str] = mapped_column(String(32), default="SP500", index=True)
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)

    __table_args__ = (
        Index("ix_membership_lookup", "index_name", "start_date", "end_date"),
    )


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------


class OhlcvRaw(Base):
    """Immutable as-fetched provider output."""

    __tablename__ = "ohlcv_raw"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    date: Mapped[date] = mapped_column(Date)
    open: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    close: Mapped[float | None] = mapped_column(Float)
    adj_close: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32))
    ingested_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("ticker", "date", "source", name="uq_raw_ticker_date_source"),
    )


class OhlcvClean(Base):
    """Split/dividend-adjusted series. The single source of truth for factors."""

    __tablename__ = "ohlcv_clean"

    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    dollar_volume: Mapped[float | None] = mapped_column(Float)
    returns: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (
        Index("ix_clean_date", "date"),
        Index("ix_clean_ticker_date", "ticker", "date"),
    )


# ---------------------------------------------------------------------------
# Factors
# ---------------------------------------------------------------------------


class FactorValue(Base):
    __tablename__ = "factor_values"

    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    factor_name: Mapped[str] = mapped_column(String(32), primary_key=True)
    raw_value: Mapped[float | None] = mapped_column(Float)
    zscore: Mapped[float | None] = mapped_column(Float)
    rank_pct: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (Index("ix_factor_date_name", "date", "factor_name"),)


class FactorIC(Base):
    """Information Coefficient time series, per factor and forward horizon."""

    __tablename__ = "factor_ic"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    factor_name: Mapped[str] = mapped_column(String(32), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    horizon_days: Mapped[int] = mapped_column(Integer)
    ic: Mapped[float | None] = mapped_column(Float)
    rank_ic: Mapped[float | None] = mapped_column(Float)
    n_obs: Mapped[int] = mapped_column(Integer)

    __table_args__ = (
        UniqueConstraint("factor_name", "date", "horizon_days", name="uq_ic"),
    )


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128))
    engine_type: Mapped[str] = mapped_column(String(16))  # naive | vectorized
    config: Mapped[dict] = mapped_column(JSON)

    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    is_walk_forward: Mapped[bool] = mapped_column(Boolean, default=False)

    # Full metric payload. Kept as JSON so the suite can grow without a
    # migration per metric; headline figures are also columns for querying.
    metrics: Mapped[dict] = mapped_column(JSON)
    sharpe: Mapped[float | None] = mapped_column(Float, index=True)
    sharpe_oos: Mapped[float | None] = mapped_column(Float)
    max_drawdown: Mapped[float | None] = mapped_column(Float)
    total_return: Mapped[float | None] = mapped_column(Float)
    win_rate: Mapped[float | None] = mapped_column(Float)

    runtime_ms: Mapped[float | None] = mapped_column(Float)
    n_trades: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )


class WalkForwardFold(Base):
    __tablename__ = "walk_forward_folds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("backtest_runs.id"), index=True)
    fold_index: Mapped[int] = mapped_column(Integer)
    train_start: Mapped[date] = mapped_column(Date)
    train_end: Mapped[date] = mapped_column(Date)
    test_start: Mapped[date] = mapped_column(Date)
    test_end: Mapped[date] = mapped_column(Date)
    sharpe_is: Mapped[float | None] = mapped_column(Float)
    sharpe_oos: Mapped[float | None] = mapped_column(Float)
    return_oos: Mapped[float | None] = mapped_column(Float)
    max_drawdown_oos: Mapped[float | None] = mapped_column(Float)
    metrics: Mapped[dict | None] = mapped_column(JSON)


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("backtest_runs.id"), index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    side: Mapped[str] = mapped_column(String(8))  # LONG | SHORT
    entry_date: Mapped[date] = mapped_column(Date, index=True)
    exit_date: Mapped[date | None] = mapped_column(Date)
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float | None] = mapped_column(Float)
    quantity: Mapped[float] = mapped_column(Float)
    weight: Mapped[float | None] = mapped_column(Float)
    pnl_pct: Mapped[float | None] = mapped_column(Float)
    pnl_abs: Mapped[float | None] = mapped_column(Float)
    costs: Mapped[float | None] = mapped_column(Float)
    holding_days: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), default="CLOSED")

    __table_args__ = (Index("ix_trade_run_pnl", "run_id", "pnl_pct"),)


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("backtest_runs.id"), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    equity: Mapped[float] = mapped_column(Float)
    returns: Mapped[float | None] = mapped_column(Float)
    drawdown: Mapped[float | None] = mapped_column(Float)
    gross_exposure: Mapped[float | None] = mapped_column(Float)
    net_exposure: Mapped[float | None] = mapped_column(Float)
    n_positions: Mapped[int | None] = mapped_column(Integer)
    positions: Mapped[dict | None] = mapped_column(JSON)
    is_oos: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (Index("ix_snapshot_run_date", "run_id", "date"),)


# ---------------------------------------------------------------------------
# Telemetry — real, not decorative
# ---------------------------------------------------------------------------


class JobRun(Base):
    """One row per scheduled job execution. Source of the uptime figure."""

    __tablename__ = "job_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_name: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)  # RUNNING|SUCCESS|FAILED
    started_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    duration_ms: Mapped[float | None] = mapped_column(Float)
    rows_written: Mapped[int] = mapped_column(Integer, default=0)
    bytes_processed: Mapped[int] = mapped_column(BigInteger, default=0)
    tickers_processed: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    details: Mapped[dict | None] = mapped_column(JSON)


class ApiRequestLog(Base):
    """Per-request latency. Source of the real p95 on System Health."""

    __tablename__ = "api_request_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    endpoint: Mapped[str] = mapped_column(String(128), index=True)
    method: Mapped[str] = mapped_column(String(8))
    status_code: Mapped[int] = mapped_column(Integer)
    latency_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )


class AppLog(Base):
    """Persisted application log lines, surfaced in the UI log panels."""

    __tablename__ = "app_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    level: Mapped[str] = mapped_column(String(16), index=True)
    source: Mapped[str] = mapped_column(String(64), index=True)
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )


class RiskEvent(Base):
    """Breaches of a risk limit — powers the Risk Monitor alert banner."""

    __tablename__ = "risk_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(48), index=True)
    severity: Mapped[str] = mapped_column(String(16))
    message: Mapped[str] = mapped_column(Text)
    details: Mapped[dict | None] = mapped_column(JSON)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )


class Conversation(Base):
    """One INU AI chat thread, listed in the history sidebar."""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # Derived from the opening question rather than asked for, so a thread is
    # identifiable in the sidebar without the user naming it.
    title: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), index=True
    )


class ChatMessage(Base):
    """A single turn. Assistant rows also record how the answer was produced.

    Keeping the model, provider and tool list per message means a reader can
    see which free model answered and what platform data it consulted --
    the same provenance the rest of the app exposes.
    """

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    content: Mapped[str] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(80))
    provider: Mapped[str | None] = mapped_column(String(32))
    tools_used: Mapped[dict | None] = mapped_column(JSON)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    #: Set when the turn carried an attachment, so the thread can be replayed
    #: with its context intact.
    attachment_name: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )


class FactorSnapshot(Base):
    """Precomputed factor scores, one row per ticker per as-of date.

    Computing the composite requires the whole 838k-row price panel in memory,
    which peaks near 270MB on top of a ~240MB library baseline -- more than a
    512MB instance can hold. The pipeline writes the result here once, and the
    API serves rows.

    This is also simply the right shape: recomputing six years of factors on
    every page view was work the data had already done.
    """

    __tablename__ = "factor_snapshots"
    __table_args__ = (
        UniqueConstraint("as_of", "ticker", name="uq_factor_snapshot"),
        Index("ix_factor_snapshot_as_of", "as_of", "composite_score"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    as_of: Mapped[date] = mapped_column(Date, index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    composite_score: Mapped[float] = mapped_column(Float)
    #: Per-factor ranks, keyed by factor name. Stored as JSON so adding a
    #: factor does not require a migration.
    components: Mapped[dict | None] = mapped_column(JSON)
    bias: Mapped[str] = mapped_column(String(8))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class FactorDiagnostic(Base):
    """IC, decay and cross-factor correlation, computed once by the pipeline."""

    __tablename__ = "factor_diagnostics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    as_of: Mapped[date] = mapped_column(Date, index=True)
    #: What this row holds: "ic", "correlation", or "ticker_detail".
    kind: Mapped[str] = mapped_column(String(32), index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
