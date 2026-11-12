from __future__ import annotations

import asyncio
import json
from collections import defaultdict

from fastapi import WebSocket


class Hub:
    def __init__(self) -> None:
        self._rooms: dict[str, set[WebSocket]] = defaultdict(set)
        self.loop: asyncio.AbstractEventLoop | None = None

    def attach(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    async def join(self, ride_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self._rooms[ride_id].add(ws)

    def leave(self, ride_id: str, ws: WebSocket) -> None:
        self._rooms[ride_id].discard(ws)

    async def _send(self, ride_id: str, event: dict) -> None:
        dead = []
        payload = json.dumps(event)
        for ws in list(self._rooms.get(ride_id, ())):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.leave(ride_id, ws)

    def publish(self, ride_id: str, event: dict) -> None:
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._send(ride_id, event), self.loop)


hub = Hub()
