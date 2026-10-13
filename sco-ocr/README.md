# sco-ocr

Real-time OCR event stream for a self-checkout supervisor screen (see the build brief).

## Status
- [x] Phase 0 scaffold
- [x] Phase 1 capture (code + corruption filter; **not yet run against the live stream**)
- [x] Phase 2 layout, `tools/calibrate.py`, `tools/dump_tiles.py` (ROI fractions calibrated by hand against one real reference frame; re-check on the raw stream)
- [x] Phase 3 status from the status-button text (fuzzy vocabulary match + optional template classifier). **No LED signal is used**; the LED is not reliable on this screen.
- [x] Phase 4 change detection with debounce (`change.py`)
- [x] Phase 5 OCR (`ocr.py`): RapidOCR/PaddleOCR-model engine, optional Tesseract, one detect+recognise pass per items panel, euro-sign stripping and digits-only amount reads
- [x] Phase 6 parsing (`parse.py`) + temporal voting + confidence gates. **No sum-vs-total validation (dropped by request)**
- [x] Phase 7 per-till state machine and events (`state.py`), persisted so restarts don't re-emit
- [x] Phase 8 events + sinks (stdout, Redis Stream, WebSocket) behind a non-blocking retrying wrapper; JSON schema in `schema/`
- [x] Phase 9 (partial) `tools/replay.py`; end-to-end tests on synthetic frames. **No golden set / metrics yet** (needs recorded footage)
- [x] Phase 10 (partial) Dockerfile, compose, graceful shutdown, optional crop retention. **Not done:** `/healthz`, Prometheus, JSON logging

## Setup
    pip install -e '.[dev]'
    cp .env.example .env   # set RTSP_URL; never commit .env
    python -m sco_ocr.main --save-frame reference.png
    python tools/dump_tiles.py reference.png     # inspect captures/tiles/overlay.png
    pip install opencv-python && python tools/calibrate.py reference.png   # GUI, writes config/layout.yaml
    pytest && ruff check .        # `pytest -m slow` also runs real OCR on captures/reference.png
    python -m sco_ocr.main --run  # stream events to the sinks in SINK (stdout/redis/websocket)

Without the stream: `PYTHONPATH=. python tools/make_fixture.py synthetic.png` creates a fake frame.

## Privacy
The feed contains transaction data (GDPR). Confirm you are authorised to process it. Raw frames and
crops are not stored unless `CROP_RETENTION` is `lowconf` or `all`.

## Accuracy: what is and isn't protecting prices
There is no total-vs-items check, so a wrong price is stopped only by: a price-confidence gate (default 0.6),
a strict 2-decimal format, rejection of non-positive item prices, and N-agreeing-reads voting. Voting helps little
against a *deterministic* misread of identical pixels, so OCR quality on the raw stream is what matters.
Measured on one degraded VLC screenshot (not the raw stream): amount fields 7/9 correct, the 2 wrong ones at conf <= 0.70;
item names are often garbled and many item rows are dropped for lack of a trustworthy price. Re-measure on raw frames,
and consider a recogniser fine-tuned on this fixed UI font if the numbers stay short of the target.

Latency: an OCR call costs ~0.3-0.6 s on 4 CPU cores; a change on one till triggers ~3 ROIs x 2 voting reads, so ~2-3 s.
Idle screens cost ~0 OCR calls.

## Edge device / GPU
Set `OCR_DEVICE=cuda` and install `onnxruntime-gpu` (instead of `onnxruntime`) with matching CUDA/cuDNN. Startup fails
loudly if the CUDA provider is missing; it never silently falls back to CPU. **The GPU path is untested** (no GPU in the
development sandbox). CPU measurements: ~0.5 s per OCR call; the angle classifier is disabled (identical output, ~12% faster);
lowering the 3x upscale is faster but changes the recognised text, so it is not done.
Planned for speed on GPU: batch all changed ROI crops of a frame into one call, recogniser-only reads for fixed fields,
PP-OCRv5 models, and ultimately a small recogniser trained on this screen's fixed font.
