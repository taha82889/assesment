"""Per-ROI change detection with stability debounce.

An ROI is reported as changed only when it differs from the last *accepted* image and
then holds the same new content for ``stable_frames`` consecutive frames (so in-flight
redraws and compression shimmer don't trigger OCR).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SMALL = (64, 32)  # w, h comparison size


def signature(img: np.ndarray) -> np.ndarray:
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, SMALL, interpolation=cv2.INTER_AREA).astype(np.float32)


def phash(sig: np.ndarray) -> int:
    """64-bit perceptual hash (DCT of an 32x32 resize)."""
    s = cv2.resize(sig, (32, 32), interpolation=cv2.INTER_AREA)
    d = cv2.dct(s)[:8, :8]
    bits = (d > np.median(d[1:, :])).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


@dataclass
class _Slot:
    accepted: np.ndarray | None = None  # signature last handed to OCR
    candidate: np.ndarray | None = None
    streak: int = 0


@dataclass(frozen=True)
class ChangedROI:
    tile: int
    roi: str
    image: np.ndarray


class ChangeDetector:
    def __init__(self, thresholds: dict[str, float], stable_frames: int = 2, hash_bits: int = 4):
        self.thresholds, self.stable_frames, self.hash_bits = thresholds, stable_frames, hash_bits
        self._slots: dict[tuple[int, str], _Slot] = {}

    def _differs(self, a: np.ndarray, b: np.ndarray, roi: str) -> bool:
        # Mean abs diff is the primary signal; the hash catches small but structural edits
        # (one extra text line) that barely move the mean.
        return mad(a, b) > self.thresholds[roi] or hamming(phash(a), phash(b)) > self.hash_bits

    def update(self, tile: int, roi: str, img: np.ndarray) -> bool:
        """Feed the current crop; True exactly once when a stable change is confirmed."""
        slot = self._slots.setdefault((tile, roi), _Slot())
        sig = signature(img)
        if slot.accepted is None:  # first sight: needs one OCR to establish state
            slot.accepted, slot.candidate, slot.streak = sig, sig, 1
            return True
        if not self._differs(sig, slot.accepted, roi):
            slot.candidate, slot.streak = None, 0
            return False
        if slot.candidate is not None and not self._differs(sig, slot.candidate, roi):
            slot.streak += 1
        else:
            slot.candidate, slot.streak = sig, 1
        if slot.streak >= self.stable_frames:
            slot.accepted, slot.candidate, slot.streak = sig, None, 0
            return True
        return False

    def forget(self, tile: int) -> None:
        for k in [k for k in self._slots if k[0] == tile]:
            del self._slots[k]
