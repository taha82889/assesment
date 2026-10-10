"""Sink interface and a non-blocking wrapper (bounded queue, retry with backoff)."""

from __future__ import annotations

import logging
import queue
import threading
from typing import Protocol

from ..events import Event

log = logging.getLogger(__name__)


class Sink(Protocol):
    name: str

    def emit(self, event: Event) -> None:
        """Deliver one event; raise on failure so the wrapper can retry."""

    def close(self) -> None: ...


class AsyncSink:
    """Never blocks the pipeline: events go to a bounded queue; the oldest are dropped if a
    consumer is too slow. A worker thread delivers with retry and exponential backoff."""

    def __init__(
        self,
        inner: Sink,
        max_queue: int = 10_000,
        retries: int = 5,
        base_delay: float = 0.5,
        max_delay: float = 10.0,
    ):
        self.inner, self.retries = inner, retries
        self.base_delay, self.max_delay = base_delay, max_delay
        self.name = f"async:{inner.name}"
        self.dropped = self.failed = self.delivered = 0
        self._q: queue.Queue[Event | None] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name=f"sink-{inner.name}", daemon=True)
        self._t.start()

    def emit(self, event: Event) -> None:
        while True:
            try:
                self._q.put_nowait(event)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()  # drop oldest
                    self.dropped += 1
                except queue.Empty:
                    pass

    def _deliver(self, event: Event) -> None:
        delay = self.base_delay
        for attempt in range(self.retries + 1):
            try:
                self.inner.emit(event)
                self.delivered += 1
                return
            except Exception as exc:  # noqa: BLE001 - any sink failure is retried
                if attempt == self.retries or self._stop.is_set():
                    self.failed += 1
                    log.error("sink %s gave up on %s: %s", self.inner.name, event.id, exc)
                    return
                log.warning("sink %s failed (%s); retry in %.1fs", self.inner.name, exc, delay)
                self._stop.wait(delay)
                delay = min(delay * 2, self.max_delay)

    def _run(self) -> None:
        while True:
            ev = self._q.get()
            if ev is None:
                return
            self._deliver(ev)

    @property
    def depth(self) -> int:
        return self._q.qsize()

    def close(self, timeout: float = 5.0) -> None:
        """Flush what is queued (bounded by timeout), then stop."""
        self._q.put(None)
        self._t.join(timeout)
        self._stop.set()
        self.inner.close()
