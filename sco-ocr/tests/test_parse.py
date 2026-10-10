import pytest

from sco_ocr.parse import RowRead, classify, parse_meta, parse_price, parse_rows, same_name


@pytest.mark.parametrize(
    "text,expected",
    [
        ("€ 3,35", 3.35),
        ("€3.35", 3.35),
        ("C 10,59", 10.59),
        ("-37,87", -37.87),
        ("€ 123,73", 123.73),
        ("E 1,53", 1.53),
        ("123", None),
        ("", None),
        (".00", None),
        ("abc", None),
    ],
)
def test_parse_price(text, expected):
    assert parse_price(text) == expected


@pytest.mark.parametrize(
    "name,kind",
    [
        ("500G GRAN CHOC-AMA", "item"),
        ("TOTAAL", "subtotal"),
        ("TOTAL", "subtotal"),
        ("NUTRI-B", "discount"),
        ("PROMO", "promo"),
        ("SuperPlus kaart", "loyalty"),
        ("KREDIETKAART", "payment"),
        ("STROPKES MOSTERD", "item"),
    ],
)
def test_classify(name, kind):
    assert classify(name) == kind


def test_parse_rows_keeps_special_lines_and_drops_noise():
    rows = [
        RowRead("500G GRAN CHOC-AMA", 0.8, "€ 3,35", 0.9),
        RowRead("TOTAAL", 0.8, "€ 162,62", 0.8),
        RowRead("PROMO", 0.7, "-37,87", 0.7),
        RowRead("", 0.0, "", 0.0),
        RowRead("MYSTERY ITEM", 0.9, "garbled", 0.2),  # no valid price -> never an item
        RowRead("ZECLITSPAORGAN", 0.6, "-0.84", 0.6),  # negative 'item' price is an OCR error
    ]
    lines = parse_rows(rows)
    assert [(ln.kind, ln.price) for ln in lines] == [
        ("item", 3.35),
        ("subtotal", 162.62),
        ("promo", -37.87),
    ]


def test_same_name_tolerates_ocr_variants():
    assert same_name("500G GRAN CHOC-AMA", "500G GRAN CHOC-ASA")
    assert not same_name("MELK 1L", "STROPKES MOSTERD")


def test_parse_meta():
    assert parse_meta("# 37  T: 7427") == ("37", "7427")
    assert parse_meta("#  0") == ("0", None)
