"""CLI entrypoint. Phase 1: --save-frame. Pipeline arrives in later phases."""

from __future__ import annotations

import argparse
import logging
import sys

import cv2

from .capture import RtspGrabber
from .config import Settings


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sco_ocr")
    p.add_argument("--save-frame", metavar="PATH", help="grab one reference frame and exit")
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

    p.error("pipeline not implemented yet; use --save-frame")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
