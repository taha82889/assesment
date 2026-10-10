"""Run the full pipeline on a recorded video (or a still image) and print the events.

ffmpeg -rtsp_transport tcp -i "$RTSP_URL" -t 600 -c copy sample.mp4
python tools/replay.py sample.mp4 [--fps 3] [--max-frames N]
"""

from __future__ import annotations

import argparse
import time

import cv2

from sco_ocr.config import load_layout
from sco_ocr.ocr import make_default_reader
from sco_ocr.pipeline import Pipeline
from sco_ocr.sinks.stdout import StdoutSink


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    layout = load_layout()
    pipe = Pipeline(layout, make_default_reader(layout), [StdoutSink()])
    cap = cv2.VideoCapture(a.video)
    step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or a.fps) / a.fps))
    i = used = 0
    t0 = time.time()
    while True:
        ok = cap.grab()
        if not ok:
            break
        i += 1
        if i % step:
            continue
        _, frame = cap.retrieve()
        pipe.process_frame(frame)
        used += 1
        if a.max_frames and used >= a.max_frames:
            break
    mins = max((time.time() - t0) / 60, 1e-9)
    ocr = sum(v for k, v in pipe.stats.items() if k.startswith("ocr:"))
    print(f"# frames={used} ocr_calls={ocr} ({ocr / mins:.0f}/min wall) stats={dict(pipe.stats)}")


if __name__ == "__main__":
    main()
