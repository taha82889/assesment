"""End to end: synthetic frames (real pixel changes) + a scripted reader."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from sco_ocr.config import load_layout
from sco_ocr.parse import RowRead
from sco_ocr.pipeline import CropArchive, Pipeline
from sco_ocr.signals import StatusResult
from sco_ocr.state import JsonFileStore
from tests.synthetic import make_frame


class ScriptedReader:
    """Returns what the 'screen' currently shows per till; counts calls."""

    def __init__(self):
        self.truth: dict[int, dict] = {}
        self.calls = 0

    def _t(self, till):
        self.calls += 1
        return self.truth.get(till, {})

    def read_status(self, img, till=0):
        return StatusResult(self._t(till).get("status", "scanning"), 95, 40)

    def read_meta(self, img, till=0):
        t = self._t(till)
        return t.get("basket"), t.get("txn"), 0.9

    def read_items(self, img, till=0):
        return [
            RowRead(n, 0.9, f"€ {p:.2f}".replace(".", ","), 0.9)
            for n, p in self._t(till).get("items", [])
        ]

    def read_amount(self, img, till=0):
        return self._t(till).get("total", 0.0), 0.9


class Sink:
    name = "list"

    def __init__(self):
        self.events = []

    def emit(self, e):
        self.events.append(e)

    def close(self):
        pass


@pytest.fixture()
def rig(tmp_path):
    reader, sink = ScriptedReader(), Sink()
    pipe = Pipeline(load_layout(), reader, [sink], JsonFileStore(tmp_path / "s.json"))
    return pipe, reader, sink


def show(pipe, reader, truth, frames=3):
    reader.truth = truth
    items = {t: v.get("items", []) for t, v in truth.items()}
    status = {
        t: {"scanning": "Bezig met scannen", "ended": "Transactie beeindigd"}[v["status"]]
        for t, v in truth.items()
        if "status" in v
    }
    out = []
    for _ in range(frames):
        frame = make_frame(
            {t: [(n, f"{p:.2f}") for n, p in its] for t, its in items.items()}, status=status
        )
        out += pipe.process_frame(frame)
    return out


def test_idle_screen_causes_no_further_ocr(rig):
    pipe, reader, _ = rig
    show(pipe, reader, {}, frames=4)
    settled = reader.calls
    show(pipe, reader, {}, frames=20)
    assert reader.calls == settled


def test_item_event_only_for_changed_till_and_within_few_frames(rig):
    pipe, reader, sink = rig
    show(pipe, reader, {}, frames=4)
    sink.events.clear()
    ev = show(pipe, reader, {3: {"items": [("500G GRAN CHOC-AMA", 3.35)], "txn": "7403"}}, frames=5)
    added = [e for e in ev if e.type == "item_added"]
    assert [(e.till, e.item.name, e.item.price) for e in added] == [
        ("SCO 3", "500G GRAN CHOC-AMA", 3.35)
    ]
    assert {e.till for e in ev} == {"SCO 3"}


def test_full_flow_started_added_total_completed(rig):
    pipe, reader, _ = rig
    show(pipe, reader, {}, frames=4)
    ev = show(pipe, reader, {5: {"items": [("MELK 1L", 1.19)], "txn": "9001", "total": 1.19}}, 5)
    ev += show(
        pipe,
        reader,
        {5: {"items": [("MELK 1L", 1.19), ("BROOD", 2.5)], "txn": "9001", "total": 3.69}},
        5,
    )
    ev += show(
        pipe,
        reader,
        {
            5: {
                "items": [("MELK 1L", 1.19), ("BROOD", 2.5)],
                "txn": "9001",
                "total": 3.69,
                "status": "ended",
            }
        },
        5,
    )
    kinds = [e.type for e in ev]
    assert kinds.count("transaction_started") == 1 and kinds.count("item_added") == 2
    assert (
        "total_changed" in kinds
        and kinds[-1] == "transaction_completed"
        or "transaction_completed" in kinds
    )
    assert len({e.id for e in ev}) == len(ev)


def test_restart_does_not_reemit(tmp_path):
    layout = load_layout()
    truth = {4: {"items": [("MELK 1L", 1.19), ("BROOD", 2.5)], "txn": "9100"}}
    r1, s1 = ScriptedReader(), Sink()
    p1 = Pipeline(layout, r1, [s1], JsonFileStore(tmp_path / "s.json"))
    show(p1, r1, truth, 6)
    assert sum(e.type == "item_added" for e in s1.events) == 2
    r2, s2 = ScriptedReader(), Sink()
    p2 = Pipeline(layout, r2, [s2], JsonFileStore(tmp_path / "s.json"))  # restart
    show(p2, r2, truth, 6)
    assert not [e for e in s2.events if e.type in ("item_added", "transaction_started")]


def test_corrupt_tile_never_produces_events(rig):
    pipe, reader, sink = rig
    reader.truth = {11: {"items": [("GARBAGE", 9.99)]}}
    show(pipe, reader, {}, frames=4)
    sink.events.clear()
    frames = []
    for _ in range(5):
        f = make_frame()
        sp = pipe.splitter
        t = sp.tiles[10]
        x0, y0 = (
            sp.crop.x0 + t.box.x0 + t.rois["items"].x0,
            sp.crop.y0 + t.box.y0 + t.rois["items"].y0,
        )
        b = t.rois["items"]
        rng = np.random.default_rng(1)
        blk = np.repeat(rng.integers(30, 120, (b.h // 8 + 1, b.w // 8 + 1)), 8, 0)
        blk = np.repeat(blk, 8, 1)[: b.h, : b.w].astype(np.uint8)
        f[y0 : y0 + b.h, x0 : x0 + b.w] = blk[..., None]
        frames.append(f)
    ev = []
    for f in frames:
        ev += pipe.process_frame(f)
    assert not [e for e in ev if e.till == "SCO 11"]
    assert pipe.stats["corrupt_tile_skips"] >= 5


def test_crop_retention_off_writes_nothing(tmp_path):
    a = CropArchive("off", tmp_path)
    a.save(np.zeros((4, 4, 3), np.uint8), "x", low_conf=True)
    assert not (tmp_path / "lowconf").exists()
    b = CropArchive("lowconf", tmp_path, max_files=2)
    for _ in range(4):
        b.save(np.zeros((4, 4, 3), np.uint8), "x", low_conf=True)
    b.save(np.zeros((4, 4, 3), np.uint8), "x", low_conf=False)
    assert len(list((tmp_path / "lowconf").glob("*.png"))) == 2


REF = Path(__file__).resolve().parents[1] / "captures" / "reference.png"


@pytest.mark.slow
@pytest.mark.skipif(not REF.exists(), reason="local real reference frame not present")
def test_real_ocr_smoke_on_reference_frame(layout):
    pytest.importorskip("rapidocr_onnxruntime")
    from sco_ocr.ocr import make_default_reader

    sink = Sink()
    pipe = Pipeline(layout, make_default_reader(layout), [sink])
    f = cv2.imread(str(REF))
    for _ in range(3):
        pipe.process_frame(f)
    # SCO 11 (garbled) must stay silent no matter what the OCR thinks
    assert not [e for e in sink.events if e.till == "SCO 11"]
    assert pipe.stats["corrupt_tile_skips"] >= 3
