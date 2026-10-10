from __future__ import annotations

from ..events import Event

STREAM = "sco:events"
MAXLEN = 100_000


class RedisStreamSink:
    name = "redis"

    def __init__(self, client, stream: str = STREAM, maxlen: int = MAXLEN):
        self.r, self.stream, self.maxlen = client, stream, maxlen

    @classmethod
    def from_url(cls, url: str) -> RedisStreamSink:
        import redis

        return cls(redis.Redis.from_url(url, socket_timeout=5))

    def emit(self, event: Event) -> None:
        self.r.xadd(
            self.stream,
            {
                "id": event.id,
                "type": event.type,
                "till": event.till,
                "data": event.model_dump_json(exclude_none=True),
            },
            maxlen=self.maxlen,
            approximate=True,
        )

    def close(self) -> None:
        try:
            self.r.close()
        except Exception:  # noqa: BLE001
            pass
