from pathlib import Path

import cv2
import numpy as np
import pytest

from sco_ocr.capture import blockiness, is_corrupt_tile
from sco_ocr.config import CorruptionCfg
from sco_ocr.layout import FrameSplitter


def _blocky(seed=0):
    rng = np.random.default_rng(seed)
    small = rng.integers(100, 160, (20, 20)).astype(np.uint8)
    return cv2.resize(small, (160, 160), interpolation=cv2.INTER_NEAREST)


def test_blocky_smear_flagged():
    assert blockiness(_blocky()) > 1.6
    assert is_corrupt_tile(_blocky(), CorruptionCfg())


def test_clean_text_tile_not_flagged(frame):
    assert not is_corrupt_tile(frame[100:400, 400:600], CorruptionCfg())


REF = Path(__file__).resolve().parents[1] / "captures" / "reference.png"


@pytest.mark.skipif(not REF.exists(), reason="local real reference frame (gitignored) not present")
def test_real_reference_only_sco11_is_corrupt(layout):
    f = cv2.imread(str(REF))
    sp = FrameSplitter(layout)
    sp.calibrate(f)
    bad = [
        t.till
        for t in sp.tiles
        if t.used and is_corrupt_tile(sp.roi_images(f, t)["items"], layout.corruption)
    ]
    assert bad == [11]
