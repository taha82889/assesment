"""Crop black bars, split the frame into tiles, and resolve sub-ROIs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Layout, Rect


@dataclass(frozen=True)
class Box:
    """Pixel rectangle: [x0, x1) x [y0, y1)."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    def crop(self, img: np.ndarray) -> np.ndarray:
        return img[self.y0 : self.y1, self.x0 : self.x1]


@dataclass(frozen=True)
class Tile:
    index: int  # 0-based position in the grid (row-major)
    till: int | None  # 1-based SCO number, None for unused positions
    box: Box  # relative to the bar-cropped frame
    rois: dict[str, Box]  # relative to the tile
    used: bool


def _span(profile: np.ndarray, thr: float, max_cut: float) -> tuple[int, int]:
    n = len(profile)
    lit = np.flatnonzero(profile > thr)
    if lit.size == 0:
        return 0, n
    lo, hi = int(lit[0]), int(lit[-1]) + 1
    limit = int(n * max_cut)
    return min(lo, limit), max(hi, n - limit)


def detect_crop(frame: np.ndarray, dark_threshold: float = 12.0, max_cut: float = 0.2) -> Box:
    """Find the video area by mean intensity of columns/rows (bars are ~black)."""
    gray = frame if frame.ndim == 2 else frame.mean(axis=2)
    x0, x1 = _span(gray.mean(axis=0), dark_threshold, max_cut)
    y0, y1 = _span(gray.mean(axis=1), dark_threshold, max_cut)
    return Box(x0, y0, x1, y1)


def frac_box(rect: Rect, w: int, h: int) -> Box:
    x0 = round(rect.x * w)
    y0 = round(rect.y * h)
    x1 = min(w, max(x0 + 1, round((rect.x + rect.w) * w)))
    y1 = min(h, max(y0 + 1, round((rect.y + rect.h) * h)))
    return Box(x0, y0, x1, y1)


def split_tiles(cropped_shape: tuple[int, int], layout: Layout) -> list[Tile]:
    """Tile geometry for a bar-cropped frame of (height, width).

    All tiles have identical integer size; leftover pixels at the right/bottom
    of the grid area are dropped.
    """
    fh, fw = cropped_shape[:2]
    g = layout.grid
    area = frac_box(g.bounds, fw, fh)
    tw, th = area.w // g.cols, area.h // g.rows
    if tw < 1 or th < 1:
        raise ValueError("frame too small for the grid")
    rois = {name: frac_box(r, tw, th) for name, r in layout.rois.items()}
    tiles = []
    for i in range(g.cols * g.rows):
        r, c = divmod(i, g.cols)
        used = i not in g.unused
        x0, y0 = area.x0 + c * tw, area.y0 + r * th
        till = i + 1 - sum(1 for u in g.unused if u < i) if used else None
        tiles.append(Tile(i, till, Box(x0, y0, x0 + tw, y0 + th), rois, used))
    return tiles


class FrameSplitter:
    """Crops bars once (re-detected on demand) and yields tile / ROI images."""

    def __init__(self, layout: Layout):
        self.layout = layout
        self._crop: Box | None = None
        self._tiles: list[Tile] | None = None

    def calibrate(self, frame: np.ndarray) -> None:
        c = self.layout.crop
        self._crop = detect_crop(frame, c.dark_threshold, c.min_fraction)
        self._tiles = split_tiles((self._crop.h, self._crop.w), self.layout)

    @property
    def crop(self) -> Box:
        if self._crop is None:
            raise RuntimeError("call calibrate(frame) first")
        return self._crop

    @property
    def tiles(self) -> list[Tile]:
        if self._tiles is None:
            raise RuntimeError("call calibrate(frame) first")
        return self._tiles

    def video(self, frame: np.ndarray) -> np.ndarray:
        return self.crop.crop(frame)

    def tile_image(self, frame: np.ndarray, tile: Tile) -> np.ndarray:
        return tile.box.crop(self.video(frame))

    def roi_images(self, frame: np.ndarray, tile: Tile) -> dict[str, np.ndarray]:
        timg = self.tile_image(frame, tile)
        return {name: box.crop(timg) for name, box in tile.rois.items()}
