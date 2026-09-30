"""Shared API dependencies: auth and cached data access."""

from __future__ import annotations

import hmac
import time
from typing import Any

from fastapi import Header, HTTPException, status

from quantedge.config import settings


async def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    """Header-based authentication.

    Deliberately simple — the PRD asks for authenticated endpoints, not an
    identity provider. Swapping this for OAuth touches only this function.
    """
    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return x_api_key


async def require_api_key_or_bearer(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> str:
    """Same key as :func:`require_api_key`, also accepted as a bearer token.

    Webhook senders such as Alertmanager can set an Authorization header but
    not arbitrary ones, so machine-to-machine endpoints take either form.
    """
    bearer = authorization[7:] if authorization and authorization.startswith("Bearer ") else None
    supplied = x_api_key or bearer
    if not supplied or not hmac.compare_digest(supplied, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return supplied


class TTLCache:
    """Small in-process cache.

    Loading the full price panel takes several seconds; without this, every
    dashboard poll would re-read 837k rows and the p95 latency target would
    be unreachable.
    """

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self.ttl = ttl_seconds
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        stamped, value = entry
        if time.time() - stamped > self.ttl:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.time(), value)

    def clear(self) -> None:
        self._store.clear()

    @property
    def keys(self) -> list[str]:
        return list(self._store)


cache = TTLCache(ttl_seconds=300.0)
