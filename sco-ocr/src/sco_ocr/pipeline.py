"""Frame -> events. Synchronous core (testable); main.py drives it from the grabber."""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .capture import is_corrupt_tile
from .change import ChangeDetector
from .config import Layout
from .events import Event, event_id
from .layout import FrameSplitter
from .parse import parse_rows
from .sinks.base import Sink
from .state import StateStore, TillState, TillTracker, TrackerCfg

log = logging.getLogger(__name__)
WATCHED = ("status_text", "meta", "items", "total", "to_pay")


class Reader(Protocol):
    def read_status(self, img: np.ndarray, till: int = 0): ...
    def read_meta(self, img: np.ndarray, till: int = 0): ...
    def read_items(self, img: np.ndarray, till: int = 0): ...
    def read_amount(self, img: np.ndarray, till: int = 0): ...


class CropArchive:
    """Optional crop retention (GDPR: off by default, bounded when on)."""

    def __init__(self, mode: str, directory: Path, max_files: int = 500):
        self.mode, self.dir, self.max_files = mode, directory / "lowconf", max_files

    def save(self, img: np.ndarray, tag: str, low_conf: bool) -> None:
        if self.mode == "off" or (self.mode == "lowconf" and not low_conf):
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.dir / f"{time.time_ns()}_{tag}.png"), img)
        files = sorted(self.dir.glob("*.png"))
        for old in files[: max(0, len(files) - self.max_files)]:
            old.unlink(missing_ok=True)


class Pipeline:
    def __init__(
        self,
        layout: Layout,
        reader: Reader,
        sinks: list[Sink],
        store: StateStore | None = None,
        cfg: TrackerCfg | None = None,
        archive: CropArchive | None = None,
        workers: int = 4,
    ):
        self.layout, self.reader, self.sinks, self.store = layout, reader, sinks, store
        self.cfg = cfg or TrackerCfg()
        self.archive = archive or CropArchive("off", Path("captures"))
        self.splitter = FrameSplitter(layout)
        self.detector = ChangeDetector(layout.change.thresholds, layout.change.stable_frames)
        self._shape: tuple[int, ...] | None = None
        self._now = datetime.now(UTC)
        saved = store.load() if store else {}
        self.trackers: dict[int, TillTracker] = {}
        for n in range(1, layout.grid.cols * layout.grid.rows - len(layout.grid.unused) + 1):
            self.trackers[n] = TillTracker(n, saved.get(n), self.cfg, clock=lambda: self._now)
        self._confirm: Counter[tuple[int, str]] = Counter()  # extra voting reads still owed
        self.stats: Counter[str] = Counter()
        self.degraded = False
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(workers, thread_name_prefix="tile") if workers > 1 else None

    # -- public ------------------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray, ts: datetime | None = None) -> list[Event]:
        self._now = ts or datetime.now(UTC)
        if frame.shape != self._shape:
            self.splitter.calibrate(frame)
            self._shape = frame.shape
        self.stats["frames"] += 1
        events: list[Event] = []
        corrupt = 0
        jobs = []
        for tile in self.splitter.tiles:
            if not tile.used or tile.till is None:
                continue
            rois = self.splitter.roi_images(frame, tile)
            if is_corrupt_tile(rois["items"], self.layout.corruption):
                corrupt += 1
                self.stats["corrupt_tile_skips"] += 1
                continue  # never feed a damaged tile to detection, OCR or state
            jobs.append((tile.till, rois))
        # One task per tile: a tile's tracker/detector state is touched by one thread only.
        # ONNX inference releases the GIL, so tiles genuinely overlap.
        if self._pool:
            results = list(self._pool.map(lambda j: self._process_tile(*j), jobs))
        else:
            results = [self._process_tile(*j) for j in jobs]
        for r in results:
            events += r
        events += self._stream_health(corrupt)
        for ev in events:
            self.stats[f"event:{ev.type}"] += 1
            for sink in self.sinks:
                sink.emit(ev)
        if events and self.store:
            self.store.save({n: t.state for n, t in self.trackers.items()})
        return events

    # -- internals ---------------------------------------------------------------------------
    def _due(self, till: int, roi: str, img: np.ndarray) -> bool:
        key = (till, roi)
        if self.detector.update(till, roi, img):
            self._confirm[key] = self.cfg.votes - 1  # extra reads so values can be voted on
            return True
        if self._confirm[key] > 0:
            self._confirm[key] -= 1
            return True
        return False

    def _process_tile(self, till: int, rois: dict[str, np.ndarray]) -> list[Event]:
        tr, rd, ev = self.trackers[till], self.reader, []
        for roi in WATCHED:
            if not self._due(till, roi, rois[roi]):
                continue
            with self._lock:
                self.stats[f"ocr:{roi}"] += 1
            img = rois[roi]
            if roi == "status_text":
                ev += tr.on_status(rd.read_status(img, till))
            elif roi == "meta":
                ev += tr.on_meta(*rd.read_meta(img, till))
            elif roi == "items":
                rows = rd.read_items(img, till)
                lines = parse_rows(rows)
                low = any(ln.kind == "item" and ln.conf < self.cfg.min_price_conf for ln in lines)
                self.archive.save(img, f"sco{till}_items", low)
                ev += tr.on_items(lines)
            elif roi == "total":
                ev += tr.on_total(*rd.read_amount(img, till))
            else:
                ev += tr.on_to_pay(*rd.read_amount(img, till))
        return ev

    def _stream_health(self, corrupt: int) -> list[Event]:
        used = len(self.trackers)
        bad = corrupt >= max(2, used // 4)
        if bad == self.degraded:
            return []
        self.degraded = bad
        kind = "stream_degraded" if bad else "stream_recovered"
        seq = self.stats["stream_seq"] = self.stats["stream_seq"] + 1
        return [
            Event(
                id=event_id("stream", None, int(self._now.timestamp() * 1000) + seq),
                ts=self._now,
                till="stream",
                type=kind,
                detail=f"{corrupt}/{used} tiles corrupt",
            )
        ]

    def external_stream_event(self, degraded: bool, detail: str) -> Event:
        """For capture-level problems (reconnects, no frames) reported by main.py."""
        self.stats["stream_seq"] += 1
        ts = datetime.now(UTC)
        ev = Event(
            id=event_id("stream", None, int(ts.timestamp() * 1000) + self.stats["stream_seq"]),
            ts=ts,
            till="stream",
            type="stream_degraded" if degraded else "stream_recovered",
            detail=detail,
        )
        for sink in self.sinks:
            sink.emit(ev)
        return ev

    def close(self) -> None:
        if self._pool:
            self._pool.shutdown(wait=True)

    def states(self) -> dict[int, TillState]:
        return {n: t.state for n, t in self.trackers.items()}
