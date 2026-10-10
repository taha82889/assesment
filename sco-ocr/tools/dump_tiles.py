"""Export every tile and sub-ROI crop for visual verification.

python tools/dump_tiles.py reference.png [--out captures/tiles]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from sco_ocr.config import DEFAULT_LAYOUT, load_layout
from sco_ocr.layout import FrameSplitter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--layout", default=str(DEFAULT_LAYOUT))
    ap.add_argument("--out", default="captures/tiles")
    a = ap.parse_args()

    frame = cv2.imread(a.image)
    if frame is None:
        raise SystemExit(f"cannot read {a.image}")
    sp = FrameSplitter(load_layout(a.layout))
    sp.calibrate(frame)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"video area: {sp.crop}")
    overlay = sp.video(frame).copy()
    for t in sp.tiles:
        color = (0, 200, 0) if t.used else (0, 0, 255)
        cv2.rectangle(overlay, (t.box.x0, t.box.y0), (t.box.x1 - 1, t.box.y1 - 1), color, 2)
        if not t.used:
            continue
        d = out / f"sco{t.till:02d}"
        d.mkdir(exist_ok=True)
        cv2.imwrite(str(d / "tile.png"), sp.tile_image(frame, t))
        for name, img in sp.roi_images(frame, t).items():
            cv2.imwrite(str(d / f"{name}.png"), img)
        for b in t.rois.values():
            cv2.rectangle(
                overlay,
                (t.box.x0 + b.x0, t.box.y0 + b.y0),
                (t.box.x0 + b.x1 - 1, t.box.y0 + b.y1 - 1),
                (255, 0, 0),
                1,
            )
    cv2.imwrite(str(out / "overlay.png"), overlay)
    print(f"wrote crops + overlay.png to {out}")


if __name__ == "__main__":
    main()
