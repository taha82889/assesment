"""OCR engines and a tile reader (status, meta, item rows, amounts)."""

from __future__ import annotations

import re
from typing import NamedTuple, Protocol

import cv2
import numpy as np

from .parse import RowRead, parse_amount, parse_meta, parse_price
from .signals import StatusMatcher, StatusResult, StatusTemplates


class Box(NamedTuple):
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    conf: float


def group_rows(boxes: list[Box]) -> list[list[Box]]:
    """Cluster text boxes into visual rows by vertical centre."""
    if not boxes:
        return []
    h = float(np.median([b.y1 - b.y0 for b in boxes])) or 1.0
    rows: list[list[Box]] = []
    for b in sorted(boxes, key=lambda b: (b.y0 + b.y1) / 2):
        yc = (b.y0 + b.y1) / 2
        if rows and abs(yc - np.mean([(r.y0 + r.y1) / 2 for r in rows[-1]])) < 0.6 * h:
            rows[-1].append(b)
        else:
            rows.append([b])
    return [sorted(r, key=lambda b: b.x0) for r in rows]


_TAIL_PRICE = re.compile(r"^(.*?)([-\u2212]?\s?[€E]?\s?\d{1,5}\s?[.,]\s?\d{2})\s*$")


def _looks_numeric(text: str) -> bool:
    chars = [c for c in text if not c.isspace()]
    digits = sum(c.isdigit() for c in chars)
    return digits >= 2 and digits * 2 >= len(chars)


def split_name_price(row: list[Box], width: float, price_x: float) -> RowRead:
    """Right-most box that looks like an amount is the price; the rest is the name."""
    last = row[-1]
    if len(row) > 1 and last.x0 >= price_x * width and _looks_numeric(last.text):
        name = " ".join(b.text for b in row[:-1])
        return RowRead(name, float(np.mean([b.conf for b in row[:-1]])), last.text, last.conf)
    text = " ".join(b.text for b in row)
    m = _TAIL_PRICE.match(text)
    if m and parse_price(m.group(2)) is not None:  # name and price fused in one box
        return RowRead(
            m.group(1),
            float(np.mean([b.conf for b in row])),
            m.group(2),
            float(np.mean([b.conf for b in row])),
        )
    return RowRead(text, float(np.mean([b.conf for b in row])), "", 0.0)


def strip_currency(gray: np.ndarray, gap: int = 3, max_glyph: int = 7) -> np.ndarray | None:
    """Drop the leading euro sign: a narrow glyph followed by a space-sized gap.

    OCR reads the sign as '6', 'E', 'C'... and loses the comma next to it, so the digits
    are re-read on their own. Field borders (1px lines) are ignored. Returns None when the
    crop holds no text at all.
    """
    g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    g = g.astype(np.float32)
    trim = 2 if g.shape[0] > 8 else 0  # horizontal border lines of the field
    ink = (np.abs(g - np.median(g)) > 20)[trim : g.shape[0] - trim]
    cols = ink.sum(axis=0) >= min(3, max(1, ink.shape[0] // 4))
    runs, start = [], None
    for x, v in enumerate(list(cols) + [False]):
        if v and start is None:
            start = x
        elif not v and start is not None:
            runs.append([start, x - 1])
            start = None
    runs = [r for r in runs if r[1] - r[0] >= 1]  # a 1px run is a border, not a glyph
    merged: list[list[int]] = []
    for r in runs:  # glyphs closer than `gap` belong together
        if merged and r[0] - merged[-1][1] - 1 < gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    if not merged:
        return None
    has_sign = len(merged) >= 2 and merged[0][1] - merged[0][0] + 1 <= max_glyph
    first = merged[1][0] if has_sign else merged[0][0]
    return g.astype(np.uint8)[:, max(first - 1, 0) :]


class OcrEngine(Protocol):
    def read_text(self, img: np.ndarray) -> tuple[str, float]:
        """Return (text, mean confidence 0..1); ('', 0.0) when nothing is found."""


def preprocess(img: np.ndarray, scale: int = 3, mode: str = "gray") -> np.ndarray:
    """Upscale (cubic), grayscale, optional Otsu binarisation, light sharpen."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(g, (0, 0), 1.0)
    g = cv2.addWeighted(g, 1.5, blur, -0.5, 0)  # unsharp mask
    if mode == "otsu":
        g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        if g.mean() < 127:  # keep dark text on light background
            g = 255 - g
    pad = 6 * scale
    g = cv2.copyMakeBorder(g, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def segment_rows(
    img: np.ndarray,
    min_height: int = 4,
    min_gap: int = 2,
    ink_threshold: float = 15,
    border: int = 3,
    min_ink_px: int = 3,
) -> list[tuple[int, int]]:
    """Text rows by horizontal projection profile (ink = deviation from the background)."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = g.astype(np.float32)
    # Faint blue/red lines (subtotal, promo) sit only ~15-60 grey levels off the background,
    # and the panel border would otherwise light up every row, so trim it.
    ink = (np.abs(g - np.median(g)) > ink_threshold)[:, border : g.shape[1] - border]
    profile = ink.sum(axis=1)
    on = profile >= min_ink_px
    rows, start, gap = [], None, 0
    for y, v in enumerate(on):
        if v:
            if start is None:
                start = y
            gap = 0
        elif start is not None:
            gap += 1
            if gap > min_gap:
                rows.append((start, y - gap + 1))
                start = None
    if start is not None:
        rows.append((start, len(on)))
    pad = 2
    return [(max(0, a - pad), min(g.shape[0], b + pad)) for a, b in rows if b - a >= min_height]


def _rapidocr(device: str, use_det: bool):
    """Build a RapidOCR instance with the angle classifier off (UI text is horizontal;
    identical output, ~12% faster) and the requested execution device."""
    import os
    import tempfile

    import rapidocr_onnxruntime as pkg
    import yaml
    from rapidocr_onnxruntime import RapidOCR

    if device not in ("cpu", "cuda"):
        raise ValueError(f"OCR device must be cpu or cuda, got {device!r}")
    if device == "cuda":
        import onnxruntime

        if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
            raise RuntimeError(
                "OCR_DEVICE=cuda but onnxruntime has no CUDAExecutionProvider. "
                "Install onnxruntime-gpu (and matching CUDA/cuDNN), not onnxruntime."
            )
    base = os.path.dirname(pkg.__file__)
    with open(os.path.join(base, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["Global"].update(use_angle_cls=False, use_text_det=use_det)
    for key in ("Det", "Cls", "Rec"):
        cfg[key]["model_path"] = os.path.join(base, cfg[key]["model_path"])
        cfg[key]["use_cuda"] = device == "cuda"
    if "rec_keys_path" in cfg["Rec"] and cfg["Rec"]["rec_keys_path"]:
        cfg["Rec"]["rec_keys_path"] = os.path.join(base, cfg["Rec"]["rec_keys_path"])
    if device == "cuda":
        cfg["Rec"]["rec_batch_num"] = 16  # more rows per recogniser batch on GPU
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(cfg, tmp)
    try:
        return RapidOCR(config_path=tmp.name)
    finally:
        os.unlink(tmp.name)


class RapidOcrEngine:
    """PaddleOCR (PP-OCR) models served through onnxruntime (pip install rapidocr-onnxruntime).

    Currently PP-OCRv3 models (the version bundled with rapidocr-onnxruntime 1.2.x).
    """

    def __init__(self, mode: str = "gray", scale: int = 3, device: str = "cpu"):
        self._ocr = _rapidocr(device, use_det=True)
        self.mode, self.scale, self.device = mode, scale, device
        # recogniser-only instance: ~40x faster, only usable on tight single-line crops
        self._line = _rapidocr(device, use_det=False)

    def read_line(self, img: np.ndarray) -> tuple[str, float]:
        result, _ = self._line(preprocess(img, self.scale, self.mode))
        if not result:
            return "", 0.0
        return result[0][1], float(result[0][2])

    def read_boxes(self, img: np.ndarray) -> list[Box]:
        """One detect+recognise pass; boxes in the coordinates of ``img``."""
        result, _ = self._ocr(preprocess(img, self.scale, self.mode))
        pad = 6  # preprocess() adds 6*scale px of border; subtract it, then undo the scale
        out = []
        for pts, text, conf in result or []:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            k = self.scale
            out.append(
                Box(
                    min(xs) / k - pad,
                    min(ys) / k - pad,
                    max(xs) / k - pad,
                    max(ys) / k - pad,
                    text,
                    float(conf),
                )
            )
        return out

    def read_text(self, img: np.ndarray) -> tuple[str, float]:
        result, _ = self._ocr(preprocess(img, self.scale, self.mode))
        if not result:
            return "", 0.0
        boxes = sorted(result, key=lambda r: min(p[0] for p in r[0]))  # left to right
        text = " ".join(r[1] for r in boxes)
        return text, float(np.mean([float(r[2]) for r in boxes]))


class TesseractEngine:
    """Fallback for comparison (needs the tesseract binary with nld+fra)."""

    def __init__(self, lang: str = "nld+fra", scale: int = 3):
        import pytesseract

        self._t, self.lang, self.scale = pytesseract, lang, scale

    def read_text(self, img: np.ndarray) -> tuple[str, float]:
        d = self._t.image_to_data(
            preprocess(img, self.scale, "otsu"),
            lang=self.lang,
            config="--psm 7",
            output_type=self._t.Output.DICT,
        )
        words = [
            (w, float(c))
            for w, c in zip(d["text"], d["conf"], strict=True)
            if w.strip() and float(c) >= 0
        ]
        if not words:
            return "", 0.0
        return " ".join(w for w, _ in words), float(np.mean([c for _, c in words])) / 100


class TileReader:
    """Reads the ROIs of a tile. The pipeline depends only on these four methods.

    ``till`` is context for diagnostics/test doubles; the real reader ignores it.
    """

    def __init__(
        self,
        name_engine: OcrEngine,
        matcher: StatusMatcher,
        templates: StatusTemplates | None = None,
        price_engine: OcrEngine | None = None,
        price_split: float = 0.70,
    ):
        self.names = name_engine
        self.prices = price_engine or name_engine
        self.matcher, self.templates, self.price_split = matcher, templates, price_split

    def read_status(self, img: np.ndarray, till: int = 0) -> StatusResult:
        if self.templates is not None:
            hit = self.templates.classify(img)
            if hit is not None:
                return hit
        text, _ = self.names.read_text(img)
        return self.matcher.match(text)

    def read_meta(self, img: np.ndarray, till: int = 0) -> tuple[str | None, str | None, float]:
        text, conf = self.names.read_text(img)
        basket, txn = parse_meta(text)
        return basket, txn, conf

    def read_items(self, img: np.ndarray, till: int = 0) -> list[RowRead]:
        if hasattr(self.names, "read_boxes"):  # one OCR call for the whole panel
            boxes = self.names.read_boxes(img)
            return [
                split_name_price(r, img.shape[1], self.price_split - 0.1) for r in group_rows(boxes)
            ]
        rows = []
        for y0, y1 in segment_rows(img):
            row = img[y0:y1]
            split = int(row.shape[1] * self.price_split)
            name, nconf = self.names.read_text(row[:, :split])
            ptext, pconf = self.prices.read_text(row[:, split:])
            rows.append(RowRead(name, nconf, ptext, pconf))
        return rows

    def read_amount(self, img: np.ndarray, till: int = 0) -> tuple[float | None, float]:
        return self._read_money(img)

    def _read_money(self, img: np.ndarray) -> tuple[float | None, float]:
        """Digits-only read of an amount crop (currency sign removed)."""
        digits = strip_currency(img)
        if digits is None:
            return None, 0.0
        reads = []
        if hasattr(self.prices, "read_line"):
            reads.append(self.prices.read_line(digits))
        reads.append(self.prices.read_text(digits))
        parsed = [(parse_amount(t), c) for t, c in reads if t]
        parsed = [(v, c) for v, c in parsed if v is not None]
        if not parsed:
            return None, 0.0
        best_v, best_c = max(parsed, key=lambda p: p[1])
        if len(parsed) == 2 and parsed[0][0] == parsed[1][0]:
            best_c = min(1.0, best_c + 0.1)  # two independent reads agree
        return best_v, best_c


def make_default_reader(
    layout, templates_dir: str | None = None, device: str = "cpu"
) -> TileReader:
    """One grayscale RapidOCR engine for everything (Otsu hurt price reads on real frames)."""
    from .signals import StatusMatcher, StatusTemplates

    return TileReader(
        RapidOcrEngine("gray", device=device),
        StatusMatcher(layout.status_vocabulary),
        StatusTemplates(templates_dir) if templates_dir else None,
    )
