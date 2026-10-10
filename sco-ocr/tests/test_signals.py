import pytest

from sco_ocr.config import load_layout
from sco_ocr.signals import StatusMatcher, normalize


@pytest.fixture(scope="module")
def matcher():
    return StatusMatcher(load_layout().status_vocabulary)


# Strings are real OCR output from a degraded reference frame (see captures/ probe).
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Kassagesoten", "closed"),
        ("Kassa gesloten", "closed"),
        ("Satgmatscannen", "scanning"),
        ("Bezig met scannen", "scanning"),
        ("Tiansctebeink", "ended"),
        ("Transactie beëindigd", "ended"),
        ("Betaling bezig", "paying"),
    ],
)
def test_status_from_noisy_ocr(matcher, text, expected):
    assert matcher.match(text).status == expected


@pytest.mark.parametrize("text", ["", "ab", "123", "arce", "amenr", "Betarn ba", "Hetain28e20 zz"])
def test_garbage_is_unknown_never_a_guess(matcher, text):
    assert matcher.match(text).status == "unknown"


def test_normalize():
    assert normalize("Transactie beëindigd!") == "transactiebeeindigd"
