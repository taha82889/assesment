# sco-ocr

Real-time OCR event stream for a self-checkout supervisor screen (see the build brief).

## Status
- [x] Phase 0 scaffold
- [x] Phase 1 capture (code + corruption filter; **not yet run against the live stream**)
- [x] Phase 2 layout, `tools/calibrate.py`, `tools/dump_tiles.py` (ROI fractions are guesses until calibrated on a real frame)
- [ ] Phases 3-10

## Setup
    pip install -e '.[dev]'
    cp .env.example .env   # set RTSP_URL; never commit .env
    python -m sco_ocr.main --save-frame reference.png
    python tools/dump_tiles.py reference.png     # inspect captures/tiles/overlay.png
    pip install opencv-python && python tools/calibrate.py reference.png   # GUI, writes config/layout.yaml
    pytest && ruff check .

Without the stream: `PYTHONPATH=. python tools/make_fixture.py synthetic.png` creates a fake frame.

## Privacy
The feed contains transaction data (GDPR). Confirm you are authorised to process it. Raw frames and
crops are not stored unless `CROP_RETENTION` is `lowconf` or `all`.
