# sco-ocr

Real-time OCR event stream for a self-checkout supervisor screen (see the build brief).

## Status
- [x] Phase 0 scaffold
- [x] Phase 1 capture (code + corruption filter; **not yet run against the live stream**)
- [x] Phase 2 layout, `tools/calibrate.py`, `tools/dump_tiles.py` (ROI fractions calibrated by hand against one real reference frame; re-check on the raw stream)
- [x] Phase 3 status from the status-button text (fuzzy vocabulary match + optional template classifier). **No LED signal is used**; the LED is not reliable on this screen.
- [x] Phase 4 change detection with debounce (`change.py`)
- [ ] Phases 5-10 (OCR engine wrapper, parsing/validation, state/events, sinks, replay, ops)

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

## OCR findings so far
On a degraded VLC screenshot, plain PaddleOCR-model reads (via `rapidocr-onnxruntime`) were poor: status text came out as
`Kassagesoten`/`Satgmatscannen`, and price fields were frequently wrong or empty. Hence: status uses a fuzzy/closed vocabulary,
unknown reads are never guessed, and prices must pass sum validation + temporal voting before any event is emitted.
Evaluate on the raw RTSP frames (not screenshots) before trusting these numbers.
