"""RTSP grabber: own thread, latest-frame slot, fps throttle, reconnect, corruption filter."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Iterator

import cv2
import numpy as np

from .config import CorruptionCfg

# Must be set before the first VideoCapture is opened.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

log = logging.getLogger(__name__)


def blockiness(gray: np.ndarray, block: int = 8) -> float:
    """Ratio of mean horizontal gradient on 8px block boundaries vs. elsewhere.

    ~1.0 for natural content; well above 1 when codec damage leaves hard block edges.
    """
    g = gray.astype(np.float32)
    dx = np.abs(np.diff(g, axis=1)).mean(axis=0)  # gradient between col i and i+1
    if dx.size < block * 2:
        return 1.0
    idx = np.arange(dx.size)
    on = dx[(idx % block) == block - 1]
    off = dx[(idx % block) != block - 1]
    return float(on.mean() / max(off.mean(), 1e-3))


def interior_detail(gray: np.ndarray, block: int = 8) -> float:
    """Laplacian variance ignoring pixels next to 8px block seams (which damage creates)."""
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    ys, xs = np.indices(lap.shape)
    seam = ((xs % block) >= block - 1) | ((xs % block) == 0)
    seam |= ((ys % block) >= block - 1) | ((ys % block) == 0)
    inner = lap[~seam]
    return float(inner.var()) if inner.size else 0.0


def streak_stats(gray: np.ndarray) -> tuple[float, float]:
    """(fraction of dark pixels, horizontal/vertical gradient ratio).

    The UI panels are light with sparse text; decode damage produces large dark
    smears with strong vertical streaks. Independent of the codec block grid, so it
    survives rescaling.
    """
    g = gray.astype(np.float32)
    dark = float((g < 150).mean())
    dx = np.abs(np.diff(g, axis=1)).mean()
    dy = np.abs(np.diff(g, axis=0)).mean()
    return dark, float(dx / max(dy, 1e-3))


def is_corrupt_tile(tile: np.ndarray, cfg: CorruptionCfg) -> bool:
    """Flag decode damage: (blocky AND smeared) OR (large dark smear with vertical streaks).

    Pass the light content area of a tile (e.g. the items ROI), not the whole tile.
    """
    gray = tile if tile.ndim == 2 else cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    if blockiness(gray) > cfg.blockiness_ratio and interior_detail(gray) < cfg.min_laplacian_var:
        return True
    dark, ratio = streak_stats(gray)
    return dark > cfg.max_dark_fraction and ratio > cfg.streak_ratio


class RtspGrabber:
    """Reads in a background thread; consumers only ever see the newest frame."""

    def __init__(self, url: str, fps: float = 3.0, max_backoff: float = 30.0):
        if not url:
            raise ValueError("RTSP_URL is empty; set it in .env")
        self.url, self.interval, self.max_backoff = url, 1.0 / fps, max_backoff
        self.reconnects = 0
        self.last_frame_ts = 0.0
        self._slot: tuple[int, np.ndarray] | None = None
        self._seq = 0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rtsp-grabber", daemon=True)

    def start(self) -> RtspGrabber:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                log.warning("rtsp open failed; retry in %.0fs", backoff)
                cap.release()
                self._stop.wait(backoff)
                backoff = min(backoff * 2, self.max_backoff)
                self.reconnects += 1
                continue
            log.info("rtsp connected")
            backoff, next_emit = 1.0, 0.0
            while not self._stop.is_set():
                ok = cap.grab()  # always drain so the buffer never lags behind live
                if not ok:
                    break
                now = time.monotonic()
                if now < next_emit:
                    continue
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    break
                next_emit = now + self.interval
                self.last_frame_ts = time.time()
                with self._cond:
                    self._seq += 1
                    self._slot = (self._seq, frame)  # size-1 queue: overwrite stale
                    self._cond.notify_all()
            cap.release()
            self.reconnects += 1
            log.warning("rtsp stream lost (reconnect #%d)", self.reconnects)

    def frames(self, timeout: float = 15.0) -> Iterator[np.ndarray]:
        last = 0
        while not self._stop.is_set():
            with self._cond:
                if not self._cond.wait_for(
                    lambda last=last: (
                        self._stop.is_set() or bool(self._slot and self._slot[0] != last)
                    ),
                    timeout,
                ):
                    log.warning("no frame for %.0fs", timeout)
                    continue
                if self._stop.is_set() or self._slot is None:
                    return
                last, frame = self._slot
            yield frame

    def read(self, timeout: float = 15.0) -> np.ndarray | None:
        return next(self.frames(timeout), None)
