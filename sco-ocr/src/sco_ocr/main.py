"""CLI entrypoint.

python -m sco_ocr.main --save-frame ref.png     grab one reference frame
python -m sco_ocr.main --run                    stream events to the configured sinks
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys

import cv2

from .capture import RtspGrabber
from .config import Settings, load_layout
from .state import JsonFileStore


def run(s: Settings, layout_path: str | None, state_path: str) -> int:
    from .ocr import make_default_reader
    from .pipeline import CropArchive, Pipeline
    from .sinks import build_sinks

    layout = load_layout(layout_path) if layout_path else load_layout()
    sinks = build_sinks(s)
    pipe = Pipeline(
        layout,
        make_default_reader(layout),
        sinks,
        JsonFileStore(state_path),
        archive=CropArchive(s.crop_retention, s.capture_dir),
    )
    grabber = RtspGrabber(s.rtsp_url, s.fps).start()
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    seen_reconnects, degraded = 0, False
    try:
        for frame in grabber.frames():
            if stop["now"]:
                break
            if grabber.reconnects != seen_reconnects:
                seen_reconnects = grabber.reconnects
                pipe.external_stream_event(True, f"reconnect #{seen_reconnects}")
                degraded = True
            elif degraded:
                pipe.external_stream_event(False, "frames flowing again")
                degraded = False
            pipe.process_frame(frame)
    finally:  # graceful shutdown: stop capture, flush sinks, persist state
        grabber.stop()
        pipe.close()
        for sink in sinks:
            sink.close()
        JsonFileStore(state_path).save(pipe.states())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sco_ocr")
    p.add_argument("--save-frame", metavar="PATH", help="grab one reference frame and exit")
    p.add_argument("--run", action="store_true", help="consume the stream and emit events")
    p.add_argument("--layout", help="path to layout.yaml (default: config/layout.yaml)")
    p.add_argument("--state", default="state.json", help="persisted per-till state")
    args = p.parse_args(argv)
    s = Settings()
    logging.basicConfig(level=s.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.save_frame:
        grabber = RtspGrabber(s.rtsp_url, s.fps).start()
        try:
            frame = grabber.read(timeout=20)
        finally:
            grabber.stop()
        if frame is None:
            print("no frame received", file=sys.stderr)
            return 1
        cv2.imwrite(args.save_frame, frame)
        print(f"saved {args.save_frame} ({frame.shape[1]}x{frame.shape[0]})")
        return 0
    if args.run:
        return run(s, args.layout, args.state)
    p.error("choose --save-frame or --run")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
