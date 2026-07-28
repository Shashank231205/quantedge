"""WebSocket event push.

Events are emitted when something genuinely happens — a job finishes, a
backtest completes, a risk limit is breached. There is no synthetic tick
generator: on daily OHLCV data there is nothing truthful to emit between
bars, and inventing one would undermine every other number on the screen.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from quantedge.logging_config import get_logger

log = get_logger(__name__)

router = APIRouter()


class ConnectionManager:
    def __init__(self) -> None:
        self.active: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self.active.append(ws)
        log.info("ws.connected clients=%s", len(self.active))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            if ws in self.active:
                self.active.remove(ws)
        log.info("ws.disconnected clients=%s", len(self.active))

    async def broadcast(self, event: dict) -> None:
        payload = json.dumps(
            {**event, "timestamp": datetime.now(UTC).isoformat()}, default=str
        )
        dead: list[WebSocket] = []
        for ws in list(self.active):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)


manager = ConnectionManager()


async def emit(event_type: str, data: dict | None = None) -> None:
    """Publish an event to every connected client."""
    await manager.broadcast({"type": event_type, "data": data or {}})


@router.websocket("/ws/events")
async def events(websocket: WebSocket) -> None:
    """Event stream plus a periodic heartbeat carrying live telemetry."""
    await manager.connect(websocket)
    try:
        await websocket.send_text(
            json.dumps(
                {
                    "type": "connected",
                    "data": {"message": "QUANTEDGE event stream"},
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
        )

        while True:
            # The heartbeat doubles as a live-telemetry push, so the System
            # Health panels update without the client polling.
            await asyncio.sleep(15)

            from quantedge.api.middleware import latency_stats
            from quantedge.ingestion.telemetry import compute_uptime

            await websocket.send_text(
                json.dumps(
                    {
                        "type": "heartbeat",
                        "data": {
                            "latency": latency_stats(window=200),
                            "uptime": compute_uptime(days=30),
                        },
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                    default=str,
                )
            )
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as exc:
        log.warning("ws.error %s", exc)
        await manager.disconnect(websocket)
