import pytest

from sco_ocr.config import load_layout
from tests.synthetic import make_frame


@pytest.fixture(scope="session")
def layout():
    return load_layout()


@pytest.fixture()
def frame():
    return make_frame({3: [("500G GRAN CHOC-AMA", "3.35")]})
