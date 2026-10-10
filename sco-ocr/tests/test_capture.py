import cv2
import numpy as np

from sco_ocr.capture import blockiness, is_corrupt_tile
from sco_ocr.config import CorruptionCfg


def _blocky(seed=0):
    rng = np.random.default_rng(seed)
    small = rng.integers(100, 160, (20, 20)).astype(np.uint8)
    return cv2.resize(small, (160, 160), interpolation=cv2.INTER_NEAREST)


def test_blocky_smear_flagged():
    assert blockiness(_blocky()) > 1.6
    assert is_corrupt_tile(_blocky(), CorruptionCfg())


def test_clean_text_tile_not_flagged(frame):
    tile = frame[0:360, 240:528]
    assert not is_corrupt_tile(tile, CorruptionCfg())
