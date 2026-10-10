"""OCR text -> items / prices / classified lines, plus cross-read helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from .signals import normalize

PRICE_RE = re.compile(r"-?\s?\d{1,5}[.,]\d{2}")
# Characters OCR commonly confuses with digits inside a price crop.
_DIGITISE = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "D": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "S": "5",
        "s": "5",
        "Z": "2",
        "B": "8",
        "G": "6",
    }
)

# Non-item lines on the screen (normalised keywords), classified instead of dropped.
LINE_KINDS: dict[str, tuple[str, ...]] = {
    "subtotal": ("totaal", "subtotaal"),
    "promo": ("promo", "korting"),
    "discount": ("nutri", "voordeel", "bon"),
    "loyalty": ("superplus", "klantenkaart"),
    "payment": ("krediet", "bancontact", "contant", "cash", "maestro", "visa", "mastercard"),
}


@dataclass(frozen=True)
class RowRead:
    name: str
    name_conf: float
    price_text: str
    price_conf: float


@dataclass(frozen=True)
class Line:
    kind: str  # item | subtotal | promo | discount | loyalty | payment
    name: str
    price: float | None
    conf: float  # confidence of the price read (name conf for priceless lines)


def parse_price(text: str) -> float | None:
    """'€ 3,35' / 'C3.35' / 'S,B5' -> 3.35.  None when no valid amount is present."""
    t = text.translate(_DIGITISE) if not re.search(r"\d[.,]\d", text) else text
    t = re.sub(r"[^\d.,\-]", "", t.replace("−", "-").replace("–", "-"))
    m = PRICE_RE.search(t)
    if not m:
        return None
    return round(float(m.group().replace(" ", "").replace(",", ".")), 2)


def parse_amount(text: str) -> float | None:
    """Lenient amount parser for crops with the currency sign already removed.

    Prices on this screen always carry exactly two decimals, so when OCR drops the
    separator ('153', '3.35' read as '335') the last two digits are the cents.
    """
    strict = parse_price(text)
    if strict is not None:
        return strict
    neg = "-" in text or "−" in text
    digits = re.sub(r"\D", "", text.translate(_DIGITISE))
    if not 3 <= len(digits) <= 7:
        return None
    value = round(float(f"{int(digits[:-2])}.{digits[-2:]}"), 2)
    return -value if neg else value


def classify(name: str) -> str:
    n = normalize(name)
    if not n:
        return "item"
    for kind, words in LINE_KINDS.items():
        for w in words:
            # prefix match, or a whole-word OCR variant ("TOTAL" for "TOTAAL")
            if n.startswith(w) or (len(n) >= 4 and fuzz.ratio(w, n) >= 80):
                return kind
    return "item"


def parse_rows(rows: list[RowRead]) -> list[Line]:
    out = []
    for r in rows:
        name = re.sub(r"\s+", " ", r.name).strip(" .:-_|")
        price = parse_price(r.price_text)
        if len(normalize(name)) < 3 and price is None:
            continue  # blank / noise row
        kind = classify(name)
        if kind == "item" and (price is None or price <= 0):
            continue  # unreadable or non-positive price (discounts are their own line kinds)
        out.append(Line(kind, name, price, r.price_conf if price is not None else r.name_conf))
    return out


def same_name(a: str, b: str, threshold: float = 80) -> bool:
    """OCR variants of one row must not look like a new item."""
    na, nb = normalize(a), normalize(b)
    return bool(na and nb) and fuzz.ratio(na, nb) >= threshold


def parse_meta(text: str) -> tuple[str | None, str | None]:
    """'# 37  T: 7427' -> ('37', '7427')."""
    basket = re.search(r"#\s*(\d+)", text)
    txn = re.search(r"[TtIil1]\s*[:;]\s*(\d{2,})", text)
    return (basket.group(1) if basket else None, txn.group(1) if txn else None)
