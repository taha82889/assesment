"""Interactive calibration. Needs a GUI build of OpenCV (pip install opencv-python).

  python tools/calibrate.py reference.png

Keys:  g = grid area mode   1-7 = led,title,meta,status_text,items,total,to_pay
       drag with the mouse to (re)define the selected rectangle
       s = save config/layout.yaml     q/Esc = quit
 - grid mode: drag over the area holding all 15 tile slots (in video-area pixels)
 - ROI modes: drag inside ANY tile; the rect is stored relative to that tile
Red outlines are unused tiles, blue boxes are sub-ROIs, shown on every tile.
"""

from __future__ import annotations

import sys

import cv2

from sco_ocr.config import DEFAULT_LAYOUT, ROI_NAMES, Rect, load_layout, save_layout
from sco_ocr.layout import FrameSplitter

MODES = ["grid", *ROI_NAMES]


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    frame = cv2.imread(sys.argv[1])
    if frame is None:
        raise SystemExit(f"cannot read {sys.argv[1]}")
    layout = load_layout()
    state = {"mode": "grid", "drag": None}

    def render() -> None:
        sp = FrameSplitter(layout)
        sp.calibrate(frame)
        img = sp.video(frame).copy()
        for t in sp.tiles:
            col = (0, 200, 0) if t.used else (0, 0, 255)
            cv2.rectangle(img, (t.box.x0, t.box.y0), (t.box.x1 - 1, t.box.y1 - 1), col, 2)
            for n, b in t.rois.items():
                c = (0, 140, 255) if n == state["mode"] else (255, 0, 0)
                cv2.rectangle(
                    img,
                    (t.box.x0 + b.x0, t.box.y0 + b.y0),
                    (t.box.x0 + b.x1 - 1, t.box.y0 + b.y1 - 1),
                    c,
                    1,
                )
        cv2.putText(img, f"mode: {state['mode']}", (10, 24), 0, 0.7, (0, 0, 255), 2)
        cv2.imshow("calibrate", img)
        state["sp"] = sp

    def on_mouse(ev: int, x: int, y: int, *_: object) -> None:
        sp = state["sp"]
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["drag"] = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP and state["drag"]:
            (x0, y0), state["drag"] = state["drag"], None
            x0, x1, y0, y1 = min(x0, x), max(x0, x), min(y0, y), max(y0, y)
            if x1 - x0 < 3 or y1 - y0 < 3:
                return
            vw, vh = sp.crop.w, sp.crop.h
            if state["mode"] == "grid":
                layout.grid.bounds = Rect(x=x0 / vw, y=y0 / vh, w=(x1 - x0) / vw, h=(y1 - y0) / vh)
            else:
                t = next(
                    (
                        t
                        for t in sp.tiles
                        if t.box.x0 <= x0 < t.box.x1 and t.box.y0 <= y0 < t.box.y1
                    ),
                    None,
                )
                if t is None:
                    return
                tx0, ty0 = x0 - t.box.x0, y0 - t.box.y0
                w, h = min(x1 - x0, t.box.w - tx0), min(y1 - y0, t.box.h - ty0)
                layout.rois[state["mode"]] = Rect(
                    x=tx0 / t.box.w, y=ty0 / t.box.h, w=w / t.box.w, h=h / t.box.h
                )
            render()

    render()
    cv2.setMouseCallback("calibrate", on_mouse)
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("g"):
            state["mode"] = "grid"
            render()
        elif ord("1") <= k <= ord("7"):
            state["mode"] = ROI_NAMES[k - ord("1")]
            render()
        elif k == ord("s"):
            save_layout(layout, DEFAULT_LAYOUT)
            print(f"saved {DEFAULT_LAYOUT}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
