import json
import threading
import time

from sco_ocr.events import Event, event_id, export_schema
from sco_ocr.sinks.base import AsyncSink
from sco_ocr.sinks.redis_stream import RedisStreamSink


def ev(n=0):
    return Event(id=event_id("SCO 1", "7", n), till="SCO 1", type="item_added")


class Flaky:
    name = "flaky"

    def __init__(self, fail_first=2):
        self.fail, self.got = fail_first, []

    def emit(self, e):
        if self.fail > 0:
            self.fail -= 1
            raise ConnectionError("down")
        self.got.append(e)

    def close(self):
        pass


def test_retry_with_backoff_then_delivers():
    inner = Flaky(2)
    s = AsyncSink(inner, base_delay=0.01)
    s.emit(ev())
    s.close()
    assert len(inner.got) == 1 and s.failed == 0


def test_gives_up_after_retries_without_blocking():
    inner = Flaky(10_000)
    s = AsyncSink(inner, retries=2, base_delay=0.001)
    t0 = time.time()
    s.emit(ev())
    assert time.time() - t0 < 0.1  # emit never waits on delivery
    s.close()
    assert s.failed == 1


def test_slow_consumer_drops_oldest_not_pipeline():
    gate = threading.Event()

    class Slow(Flaky):
        def emit(self, e):
            gate.wait(2)
            super().emit(e)

    inner = Slow(0)
    s = AsyncSink(inner, max_queue=3)
    t0 = time.time()
    for i in range(50):
        s.emit(ev(i))
    assert time.time() - t0 < 0.5
    gate.set()
    s.close()
    assert s.dropped > 0
    assert inner.got[-1].id == ev(49).id  # newest survives


class FakeRedis:
    def __init__(self):
        self.calls = []

    def xadd(self, stream, fields, maxlen=None, approximate=None):
        self.calls.append((stream, fields, maxlen, approximate))

    def close(self):
        pass


def test_redis_stream_sink_uses_capped_xadd():
    r = FakeRedis()
    RedisStreamSink(r).emit(ev())
    stream, fields, maxlen, approx = r.calls[0]
    assert (stream, maxlen, approx) == ("sco:events", 100_000, True)
    assert json.loads(fields["data"])["type"] == "item_added"


def test_schema_export(tmp_path):
    p = export_schema(tmp_path / "e.schema.json")
    assert "transaction_completed" in p.read_text()
