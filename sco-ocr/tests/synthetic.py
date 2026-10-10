"""Synthetic supervisor-screen frame for tests (no real transaction data)."""

from __future__ import annotations

import cv2
import numpy as np

W, H, BAR = 1920, 1080, 240  # 1440x1080 video area between black bars


def make_frame(items_per_till: dict[int, list[tuple[str, str]]] | None = None) -> np.ndarray:
    items_per_till = items_per_till or {}
    frame = np.zeros((H, W, 3), np.uint8)
    vw = W - 2 * BAR
    cols, rows = 5, 3
    tw, th = vw // cols, H // rows
    for i in range(cols * rows):
        r, c = divmod(i, cols)
        x0, y0 = BAR + c * tw, r * th
        tile = frame[y0 : y0 + th, x0 : x0 + tw]
        tile[:] = (235, 235, 235)
        cv2.rectangle(tile, (0, 0), (tw - 1, th - 1), (150, 150, 150), 1)
        if i >= 12:  # unused tiles stay blank
            continue
        till = i + 1
        led = (0, 170, 0) if till % 4 else (0, 0, 200)
        cv2.circle(tile, (int(tw * 0.06), int(th * 0.08)), 9, led, -1)
        cv2.putText(tile, f"SCO {till}", (int(tw * 0.12), int(th * 0.10)), 0, 0.6, (0, 0, 0), 2)
        cv2.putText(
            tile, f"#{till} T:{7400 + till}", (int(tw * 0.42), int(th * 0.10)), 0, 0.5, (0, 0, 0), 1
        )
        cv2.rectangle(
            tile,
            (int(tw * 0.02), int(th * 0.15)),
            (int(tw * 0.98), int(th * 0.27)),
            (200, 160, 60),
            -1,
        )
        cv2.putText(
            tile, "Bezig met scannen", (int(tw * 0.1), int(th * 0.235)), 0, 0.55, (255, 255, 255), 1
        )
        cv2.rectangle(
            tile,
            (int(tw * 0.64), int(th * 0.29)),
            (int(tw * 0.98), int(th * 0.79)),
            (0, 220, 255),
            -1,
        )  # yellow panel
        for k, (name, price) in enumerate(items_per_till.get(till, [])):
            y = int(th * 0.33) + k * 18
            cv2.putText(tile, name, (int(tw * 0.03), y), 0, 0.4, (0, 0, 0), 1)
            cv2.putText(tile, price, (int(tw * 0.48), y), 0, 0.4, (0, 0, 0), 1)
        cv2.putText(tile, "Totaal 0.00", (int(tw * 0.03), int(th * 0.92)), 0, 0.5, (0, 0, 0), 1)
    return frame
