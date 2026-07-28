"""Structured logging with an in-memory ring buffer.

The UI log panels read from this buffer (and from the ``app_logs`` table for
history). Everything shown in those panels is a real log line emitted by this
application — there is no synthetic log generator anywhere in the project.
"""

from __future__ import annotations

import contextlib
import logging
from collections import deque
from datetime import UTC, datetime
from typing import Any

from quantedge.config import settings

_LOG_BUFFER: deque[dict[str, Any]] = deque(maxlen=settings.log_buffer_size)


class RingBufferHandler(logging.Handler):
    """Keeps the most recent records in memory for the /system/logs endpoint."""

    def emit(self, record: logging.LogRecord) -> None:
        # Logging must never raise into the caller's control flow.
        with contextlib.suppress(Exception):
            _LOG_BUFFER.append(
                {
                    "timestamp": datetime.fromtimestamp(
                        record.created, tz=UTC
                    ).isoformat(),
                    "level": record.levelname,
                    "source": record.name.replace("quantedge.", ""),
                    "message": record.getMessage(),
                }
            )


def get_recent_logs(
    limit: int = 100, level: str | None = None
) -> list[dict[str, Any]]:
    """Most-recent-first log records, optionally filtered by level.

    ``level="ERRORS"`` and ``"WARNINGS"`` map to the UI filter buttons.
    """
    records = list(_LOG_BUFFER)
    if level and level.upper() not in ("ALL", ""):
        wanted = level.upper()
        if wanted == "ERRORS":
            keep = {"ERROR", "CRITICAL"}
        elif wanted == "WARNINGS":
            keep = {"WARNING"}
        else:
            keep = {wanted}
        records = [r for r in records if r["level"] in keep]
    return list(reversed(records))[:limit]


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if any(isinstance(h, RingBufferHandler) for h in root.handlers):
        return  # already configured

    root.setLevel(level)

    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(console)
    root.addHandler(RingBufferHandler())

    # yfinance is chatty about individual ticker failures; we track those
    # ourselves in job telemetry.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    logging.getLogger("peewee").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
