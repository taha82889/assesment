"""Broadcast events to WebSocket clients (optional: pip install websockets)."""

from __future__ import annotations

import asyncio
import threading

from ..events import Event


class WebSocketSink:
    name = "websocket"

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):  # noqa: S104
        import websockets  # fail early if missing

        self._ws = websockets
        self.host, self.port = host, port
        self._clients: set = set()
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="ws-sink", daemon=True)
        self._thread.start()
        self._ready.wait(5)

    def _serve(self) -> None:
        asyncio.set_event_loop(self._loop)

        async def handler(conn):
            self._clients.add(conn)
            try:
                await conn.wait_closed()
            finally:
                self._clients.discard(conn)

        async def main():
            self._server = await self._ws.serve(handler, self.host, self.port)
            self._ready.set()
            await asyncio.Future()

        try:
            self._loop.run_until_complete(main())
        except asyncio.CancelledError:
            pass

    def emit(self, event: Event) -> None:
        msg = event.model_dump_json(exclude_none=True)

        async def send():
            for c in list(self._clients):
                try:
                    await asyncio.wait_for(c.send(msg), 1.0)  # a slow client never stalls others
                except Exception:  # noqa: BLE001
                    self._clients.discard(c)

        asyncio.run_coroutine_threadsafe(send(), self._loop)

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
