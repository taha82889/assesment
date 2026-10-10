"""Cheap, non-OCR-first signals: status classification from the status button.

There is no LED-based status on this screen: each tile's state is the text on its
status button. The vocabulary is closed, so (1) template-match the button crop against
known crops, and (2) fall back to OCR text + fuzzy match. Anything unsure is ``unknown``
and must never produce a status_changed event.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz

STATUSES = ("closed", "idle", "scanning", "paying", "ended", "assistance", "unknown")


def normalize(text: str) -> str:
    """Lowercase, strip accents/punctuation/whitespace: OCR drops spaces unpredictably."""
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", t)


@dataclass(frozen=True)
class StatusResult:
    status: str
    score: float  # 0..100
    margin: float  # lead over the runner-up status


class StatusMatcher:
    def __init__(
        self,
        vocabulary: dict[str, list[str]],
        min_score: float = 65,
        min_margin: float = 8,
        min_len: int = 6,
    ):
        self.entries = [(st, normalize(p)) for st, phrases in vocabulary.items() for p in phrases]
        self.min_score, self.min_margin, self.min_len = min_score, min_margin, min_len

    def match(self, text: str) -> StatusResult:
        q = normalize(text)
        if len(q) < self.min_len:
            return StatusResult("unknown", 0.0, 0.0)
        best: dict[str, float] = {}
        for st, phrase in self.entries:
            # ratio penalises length mismatch; partial_ratio tolerates a truncated read
            s = max(fuzz.ratio(q, phrase), 0.9 * fuzz.partial_ratio(q, phrase))
            best[st] = max(best.get(st, 0.0), s)
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        (st, top), second = ranked[0], (ranked[1][1] if len(ranked) > 1 else 0.0)
        if top < self.min_score or top - second < self.min_margin:
            return StatusResult("unknown", top, top - second)
        return StatusResult(st, top, top - second)


class StatusTemplates:
    """Nearest-template classifier over small grayscale crops of the status button.

    Templates live in ``<dir>/<status>__<anything>.png``; build them with
    tools/build_status_templates.py from labelled real frames.
    """

    SIZE = (96, 24)  # w, h

    def __init__(self, directory: Path | str | None = None, min_corr: float = 0.85):
        self.items: list[tuple[str, np.ndarray]] = []
        self.min_corr = min_corr
        if directory and Path(directory).is_dir():
            for p in sorted(Path(directory).glob("*.png")):
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.items.append((p.stem.split("__")[0], self._prep(img)))

    @classmethod
    def _prep(cls, img: np.ndarray) -> np.ndarray:
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, cls.SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
        return (g - g.mean()) / (g.std() + 1e-3)

    def classify(self, img: np.ndarray) -> StatusResult | None:
        if not self.items:
            return None
        q = self._prep(img)
        best: dict[str, float] = {}
        for st, t in self.items:
            corr = float((q * t).mean())  # normalised correlation of equal-size crops
            best[st] = max(best.get(st, -1.0), corr)
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        st, top = ranked[0]
        if top < self.min_corr:
            return None
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        return StatusResult(st, top * 100, (top - second) * 100)
