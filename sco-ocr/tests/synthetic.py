"""Synthetic supervisor-screen frame for tests (no real transaction data).

Drawn from the layout ROI fractions so tests stay consistent with config/layout.yaml.
"""

from __future__ import annotations

import cv2
import numpy as np

from sco_ocr.config import Layout, load_layout

VW, VH, BAR = 1318, 740, 141  # video area between black bars (matches the real screen)
W, H = VW + 2 * BAR, VH


def make_frame(
    items_per_till: dict[int, list[tuple[str, str]]] | None = None,
    layout: Layout | None = None,
    status: dict[int, str] | None = None,
) -> np.ndarray:
    from sco_ocr.layout import frac_box

    layout = layout or load_layout()
    items_per_till = items_per_till or {}
    frame = np.zeros((H, W, 3), np.uint8)
    video = frame[:, BAR : BAR + VW]
    video[:] = (205, 210, 205)
    g = layout.grid
    area = frac_box(g.bounds, VW, VH)
    tw, th = area.w // g.cols, area.h // g.rows
    for i in range(g.cols * g.rows):
        r, c = divmod(i, g.cols)
        tile = video[
            area.y0 + r * th : area.y0 + (r + 1) * th, area.x0 + c * tw : area.x0 + (c + 1) * tw
        ]
        cv2.rectangle(tile, (1, 1), (tw - 4, th - 4), (160, 165, 160), 1)
        boxes = {n: frac_box(rc, tw, th) for n, rc in layout.rois.items()}
        used = i not in g.unused
        for n in ("items", "total", "to_pay", "status_text"):
            b = boxes[n]
            cv2.rectangle(tile, (b.x0, b.y0), (b.x1 - 1, b.y1 - 1), (235, 235, 235), -1)
            cv2.rectangle(tile, (b.x0, b.y0), (b.x1 - 1, b.y1 - 1), (150, 150, 150), 1)
        if not used:
            continue
        till = i + 1
        b = boxes["title"]
        cv2.putText(tile, f"SCO {till}", (b.x0, b.y1 - 2), 0, 0.4, (0, 0, 0), 1)
        b = boxes["meta"]
        cv2.putText(tile, f"# {till}  T: {7400 + till}", (b.x0, b.y1 - 2), 0, 0.35, (0, 0, 0), 1)
        b = boxes["status_text"]
        cv2.putText(
            tile,
            (status or {}).get(till, "Bezig met scannen"),
            (b.x0 + 4, b.y1 - 8),
            0,
            0.35,
            (0, 0, 0),
            1,
        )
        b = boxes["items"]
        for k, (name, price) in enumerate(items_per_till.get(till, [])):
            y = b.y0 + 12 + k * 14
            cv2.putText(tile, name, (b.x0 + 3, y), 0, 0.33, (0, 0, 0), 1)
            cv2.putText(tile, price, (b.x1 - 50, y), 0, 0.33, (0, 0, 0), 1)
        b = boxes["total"]
        cv2.putText(tile, "EUR 0,00", (b.x0 + 5, b.y1 - 4), 0, 0.35, (0, 0, 0), 1)
    return frame
