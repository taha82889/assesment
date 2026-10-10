import numpy as np
import pytest

from sco_ocr.change import ChangeDetector

TH = {"items": 4.0, "total": 4.0}


def _panel(lines=0, seed=None):
    img = np.full((120, 160, 3), 235, np.uint8)
    import cv2

    for k in range(lines):
        cv2.putText(img, f"ITEM NUMBER {k} 1.{k}9", (4, 14 + 14 * k), 0, 0.4, (0, 0, 0), 1)
    if seed is not None:  # sensor/compression noise
        n = np.random.default_rng(seed).normal(0, 1.5, img.shape)
        img = np.clip(img + n, 0, 255).astype(np.uint8)
    return img


def test_first_sight_fires_once_then_idle_is_silent():
    d = ChangeDetector(TH)
    assert d.update(0, "items", _panel(2))
    assert not any(d.update(0, "items", _panel(2, seed=i)) for i in range(30))


def test_noise_below_threshold_never_triggers():
    d = ChangeDetector(TH)
    d.update(0, "items", _panel(3, seed=0))
    assert not any(d.update(0, "items", _panel(3, seed=i)) for i in range(1, 50))


def test_new_line_fires_after_stable_frames_only():
    d = ChangeDetector(TH, stable_frames=2)
    d.update(0, "items", _panel(2))
    assert not d.update(0, "items", _panel(3))  # first frame of change: debounce
    assert d.update(0, "items", _panel(3))  # second identical frame: confirmed
    assert not d.update(0, "items", _panel(3))  # no repeat


def test_flicker_does_not_fire():
    d = ChangeDetector(TH, stable_frames=2)
    d.update(0, "items", _panel(2))
    fired = [d.update(0, "items", _panel(n)) for n in (3, 2, 3, 2, 4, 2)]
    assert not any(fired)


def test_only_changed_tile_fires():
    d = ChangeDetector(TH)
    for t in range(3):
        d.update(t, "items", _panel(1))
    out = []
    for _ in range(2):
        for t in range(3):
            out.append((t, d.update(t, "items", _panel(2 if t == 1 else 1))))
    assert [t for t, fired in out if fired] == [1]


@pytest.mark.parametrize("stable", [1, 3])
def test_stable_frames_configurable(stable):
    d = ChangeDetector(TH, stable_frames=stable)
    d.update(0, "items", _panel(1))
    res = [d.update(0, "items", _panel(4)) for _ in range(stable)]
    assert res == [False] * (stable - 1) + [True]
