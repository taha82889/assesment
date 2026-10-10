import pytest

from sco_ocr.parse import Line
from sco_ocr.signals import StatusResult
from sco_ocr.state import JsonFileStore, TillTracker


def item(name, price, conf=0.8):
    return Line("item", name, price, conf)


def types(events):
    return [e.type for e in events]


def feed_items(tr, lines, times=2):
    out = []
    for _ in range(times):
        out += tr.on_items(lines)
    return out


def st(s, score=90):
    return StatusResult(s, score, 30)


@pytest.fixture()
def tr():
    return TillTracker(3)


def test_item_needs_two_agreeing_reads(tr):
    assert tr.on_items([item("MELK 1L", 1.19)]) == []
    ev = tr.on_items([item("MELK 1L", 1.19)])
    assert types(ev) == ["transaction_started", "item_added"]
    assert ev[1].item.name == "MELK 1L" and ev[1].item.price == 1.19
    assert tr.on_items([item("MELK 1L", 1.19)]) == []  # no repeat


def test_single_misread_price_never_emitted(tr):
    assert tr.on_items([item("MELK 1L", 7.19)]) == []
    assert tr.on_items([item("MELK 1L", 1.19)]) == []  # reads disagree -> still nothing
    assert types(tr.on_items([item("MELK 1L", 1.19)])) == ["transaction_started", "item_added"]


def test_high_confidence_accepted_in_one_read(tr):
    assert types(tr.on_items([item("MELK 1L", 1.19, conf=0.97)])) == [
        "transaction_started",
        "item_added",
    ]


def test_low_confidence_price_discarded(tr):
    assert feed_items(tr, [item("MELK 1L", 1.19, conf=0.3)], times=5) == []


def test_name_variants_are_not_new_items(tr):
    feed_items(tr, [item("500G GRAN CHOC-AMA", 3.35)])
    ev = feed_items(tr, [item("500G GRAN CHOC-ASA", 3.35)], times=3)
    assert ev == []
    assert len(tr.state.items) == 1


def test_same_item_scanned_twice_counts_twice(tr):
    feed_items(tr, [item("MELK 1L", 1.19)])
    ev = feed_items(tr, [item("MELK 1L", 1.19), item("MELK 1L", 1.19)])
    assert types(ev) == ["item_added"]
    assert len(tr.state.items) == 2


def test_scrolled_out_rows_are_not_removals(tr):
    a, b, c = item("AAA BREAD", 1.0), item("BBB CHEESE", 2.0), item("CCC WATER", 3.0)
    feed_items(tr, [a, b])
    ev = feed_items(tr, [b, c])  # list scrolled: a left the view, c arrived
    assert types(ev) == ["item_added"] and ev[0].item.name == "CCC WATER"
    assert len(tr.state.items) == 3  # nothing removed


def test_special_lines_never_become_items(tr):
    lines = [Line("subtotal", "TOTAAL", 162.62, 0.9), Line("promo", "PROMO", -37.87, 0.9)]
    assert feed_items(tr, lines, times=3) == []


def test_status_first_silent_then_changes_need_votes(tr):
    assert tr.on_status(st("scanning")) == []
    assert tr.on_status(st("scanning")) == []  # adopted silently on cold start
    assert tr.state.status == "scanning"
    assert tr.on_status(st("paying")) == []
    assert types(tr.on_status(st("paying"))) == ["status_changed"]


def test_unknown_or_weak_status_ignored(tr):
    tr.on_status(st("scanning"))
    tr.on_status(st("scanning"))
    assert tr.on_status(StatusResult("unknown", 40, 0)) == []
    assert tr.on_status(st("ended", score=50)) == []
    assert tr.state.status == "scanning"


def test_offline_online(tr):
    for s in ("scanning", "scanning", "closed", "closed"):
        ev = tr.on_status(st(s))
    assert types(ev) == ["status_changed", "till_offline"]
    for s in ("idle", "idle"):
        ev = tr.on_status(st(s))
    assert types(ev) == ["status_changed", "till_online"]


def test_total_changed_first_value_silent(tr):
    assert tr.on_total(0.0, 0.8) == []
    assert tr.on_total(0.0, 0.8) == []
    assert tr.on_total(3.35, 0.8) == []
    ev = tr.on_total(3.35, 0.8)
    assert types(ev) == ["total_changed"] and ev[0].total == 3.35


def test_full_transaction_lifecycle(tr):
    for s in ("scanning", "scanning"):
        tr.on_status(st(s))
    tr.on_meta("37", "7427", 0.8)
    tr.on_meta("37", "7427", 0.8)
    feed_items(tr, [item("MELK 1L", 1.19)])
    assert tr.state.txn_id == "7427"
    for s in ("paying", "paying", "ended"):
        tr.on_status(st(s))
    ev = tr.on_status(st("ended"))
    assert "transaction_completed" in types(ev)
    # completing twice is impossible
    assert tr.on_status(st("ended")) == []
    # screen clears -> basket resets, next customer starts a new transaction
    ev = feed_items(tr, [], times=2)
    assert ev == [] and tr.state.items == [] and not tr.state.started
    ev = feed_items(tr, [item("BROOD", 2.5)])
    assert types(ev) == ["transaction_started", "item_added"]


def test_txn_id_change_completes_old_and_resets(tr):
    feed_items(tr, [item("MELK 1L", 1.19)])
    for _ in range(2):
        tr.on_meta("1", "7000", 0.8)
    ev = []
    for _ in range(2):
        ev += tr.on_meta("2", "7001", 0.8)
    assert "transaction_completed" in types(ev)
    assert tr.state.txn_id == "7001" and tr.state.items == []


def test_event_ids_unique_and_restart_does_not_reemit(tmp_path):
    t1 = TillTracker(3)
    ev = feed_items(t1, [item("MELK 1L", 1.19), item("BROOD", 2.5)])
    assert len({e.id for e in ev}) == len(ev)
    store = JsonFileStore(tmp_path / "state.json")
    store.save({3: t1.state})
    t2 = TillTracker(3, store.load()[3])  # process restart
    assert feed_items(t2, [item("MELK 1L", 1.19), item("BROOD", 2.5)], times=3) == []
    new = feed_items(t2, [item("MELK 1L", 1.19), item("BROOD", 2.5), item("KAAS", 4.0)])
    assert types(new) == ["item_added"]
    assert new[0].id not in {e.id for e in ev}  # seq continues, ids never collide


def test_low_confidence_total_is_ignored(tr):
    for _ in range(4):
        assert tr.on_total(5.55, 0.4) == []
    assert tr.state.total is None
