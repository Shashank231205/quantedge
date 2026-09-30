"""Request correlation.

Every request gets an ID — the caller's ``X-Request-ID`` if it sent one (the
load balancer does), otherwise a fresh one — held in a context variable for
the life of the request. The logging filter stamps it onto every record, so
one request's lines can be pulled out of the interleaved output of several
replicas with a single Loki query, and the ID is echoed back in the response
so a user reporting an error can quote it.
"""

from __future__ import annotations

import logging
import re
import uuid
from contextvars import ContextVar

REQUEST_ID_HEADER = "X-Request-ID"

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

#: Accept only IDs that are safe to put in a log line and a header verbatim.
_VALID_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def resolve_request_id(incoming: str | None) -> str:
    if incoming and _VALID_ID.match(incoming):
        return incoming
    return uuid.uuid4().hex


class RequestIdFilter(logging.Filter):
    """Adds ``request_id`` to every record; ``-`` outside a request."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "-"
        return True
