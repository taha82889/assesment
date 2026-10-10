from __future__ import annotations

from ..config import Settings
from .base import AsyncSink, Sink
from .stdout import StdoutSink


def build_sinks(settings: Settings) -> list[Sink]:
    """SINK=stdout,redis,websocket -> wrapped, non-blocking sinks."""
    out: list[Sink] = []
    for name in settings.sinks:
        if name == "stdout":
            out.append(AsyncSink(StdoutSink()))
        elif name == "redis":
            from .redis_stream import RedisStreamSink

            out.append(AsyncSink(RedisStreamSink.from_url(settings.redis_url)))
        elif name == "websocket":
            from .websocket import WebSocketSink

            out.append(AsyncSink(WebSocketSink()))
        else:
            raise ValueError(f"unknown sink {name!r}")
    return out


__all__ = ["AsyncSink", "Sink", "StdoutSink", "build_sinks"]
