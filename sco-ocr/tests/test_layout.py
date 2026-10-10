import numpy as np
import pytest
from pydantic import ValidationError

from sco_ocr.config import Layout, Rect, load_layout
from sco_ocr.layout import FrameSplitter, detect_crop
from tests.synthetic import BAR, H, W


def test_detect_crop_removes_black_bars(frame):
    box = detect_crop(frame)
    assert abs(box.x0 - BAR) <= 1 and abs(box.x1 - (W - BAR)) <= 1
    assert (box.y0, box.y1) == (0, H)


def test_crop_is_bounded_on_all_black_frame():
    box = detect_crop(np.zeros((100, 200, 3), np.uint8))
    assert (box.x0, box.x1, box.y0, box.y1) == (0, 200, 0, 100)


def test_twelve_equal_tiles_with_rois_in_bounds(frame, layout):
    sp = FrameSplitter(layout)
    sp.calibrate(frame)
    used = [t for t in sp.tiles if t.used]
    assert len(sp.tiles) == 15 and len(used) == 12
    assert [t.till for t in used] == list(range(1, 13))
    assert len({(t.box.w, t.box.h) for t in sp.tiles}) == 1
    for t in sp.tiles:
        for box in t.rois.values():
            assert 0 <= box.x0 < box.x1 <= t.box.w and 0 <= box.y0 < box.y1 <= t.box.h
        assert t.box.x1 <= sp.crop.w and t.box.y1 <= sp.crop.h


def test_tile_crops_have_expected_content(frame, layout):
    sp = FrameSplitter(layout)
    sp.calibrate(frame)
    tile3 = sp.tiles[2]
    rois = sp.roi_images(frame, tile3)
    assert rois["items"].std() > 5  # has text
    assert sp.roi_images(frame, sp.tiles[12])["items"].std() < 5  # unused: blank


def test_rect_validation():
    with pytest.raises(ValidationError):
        Rect(x=0.8, y=0, w=0.5, h=0.1)


def test_layout_requires_all_rois():
    with pytest.raises(ValidationError):
        Layout(rois={"led": Rect(x=0, y=0, w=0.1, h=0.1)})


def test_default_layout_loads():
    assert load_layout().grid.cols == 5
