"""Per-till state machine: temporal voting, confidence gating, diffing into events.

There is no total-vs-items validation. Price accuracy therefore rests on (1) a price
confidence gate, (2) a strict amount format, and (3) a value being seen in N consecutive
reads (or at very high confidence) before it is accepted.
"""

from __future__ import annotations

import json
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from .events import Event, ItemModel, event_id
from .parse import Line, same_name
from .signals import StatusResult


@dataclass(frozen=True)
class TrackerCfg:
    votes: int = 2  # consecutive agreeing reads required
    high_conf: float = 0.93  # a single read at/above this is accepted immediately
    min_price_conf: float = 0.6  # price reads below this are discarded outright
    min_status_score: float = 65


class StateItem(BaseModel):
    name: str
    price: float


class TillState(BaseModel):
    till: int
    status: str | None = None
    basket_no: str | None = None
    txn_id: str | None = None
    items: list[StateItem] = Field(default_factory=list)
    total: float | None = None
    to_pay: float | None = None
    online: bool | None = None
    started: bool = False
    completed: bool = False
    seq: int = 0  # per-till event counter (persisted: ids stay stable across restarts)


class StateStore(Protocol):
    def load(self) -> dict[int, TillState]: ...
    def save(self, states: dict[int, TillState]) -> None: ...


class JsonFileStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)

    def load(self) -> dict[int, TillState]:
        if not self.path.exists():
            return {}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return {int(k): TillState.model_validate(v) for k, v in raw.items()}

    def save(self, states: dict[int, TillState]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: v.model_dump() for k, v in states.items()}), encoding="utf-8")
        tmp.replace(self.path)  # atomic


class RedisStateStore:
    def __init__(self, client, key: str = "sco:state"):
        self.r, self.key = client, key

    def load(self) -> dict[int, TillState]:
        raw = self.r.hgetall(self.key)
        return {int(k): TillState.model_validate_json(v) for k, v in raw.items()}

    def save(self, states: dict[int, TillState]) -> None:
        if states:
            self.r.hset(self.key, mapping={str(k): v.model_dump_json() for k, v in states.items()})


class Voter:
    """Accept a scalar once it was read N times in a row, or once at high confidence."""

    def __init__(self, votes: int, high_conf: float):
        self.votes, self.high = votes, high_conf
        self._last: object = None
        self._count = 0

    def feed(self, value: object, conf: float) -> object | None:
        if value is None:
            self._last, self._count = None, 0
            return None
        self._count = self._count + 1 if value == self._last else 1
        self._last = value
        return value if self._count >= self.votes or conf >= self.high else None


Key = tuple[int, float]  # (cluster id, price)


class RowVoter:
    """Votes on the multiset of item rows across consecutive reads.

    Rows are matched by content (fuzzy name + exact price), never by position, because the
    panel scrolls on long baskets and OCR varies the spelling of the same row.
    """

    def __init__(self, votes: int, high_conf: float):
        self.votes, self.high = votes, high_conf
        self._reads: deque[Counter[Key]] = deque(maxlen=votes)
        self._names: list[tuple[str, float]] = []  # cluster id -> (display name, best conf)

    def seed(self, items: list[StateItem]) -> None:
        for it in items:
            self._cluster(it.name, it.price, 1.0)

    def _cluster(self, name: str, price: float, conf: float) -> int:
        for i, (n, _) in enumerate(self._names):
            if same_name(n, name):
                if conf > self._names[i][1]:
                    self._names[i] = (name, conf)
                return i
        self._names.append((name, conf))
        return len(self._names) - 1

    def reset(self) -> None:
        self._reads.clear()
        self._names.clear()

    def display(self, key: Key) -> str:
        return self._names[key[0]][0]

    def feed(self, lines: list[Line]) -> tuple[Counter[Key], bool]:
        """Return (accepted rows multiset, list_confirmed_empty)."""
        current: Counter[Key] = Counter()
        sure: Counter[Key] = Counter()
        for ln in lines:
            if ln.kind != "item" or ln.price is None:
                continue
            key = (self._cluster(ln.name, ln.price, ln.conf), ln.price)
            current[key] += 1
            if ln.conf >= self.high:
                sure[key] += 1
        self._reads.append(current)
        accepted = Counter(sure)
        if len(self._reads) == self.votes:
            agreed = Counter(self._reads[0])
            for r in list(self._reads)[1:]:
                agreed &= r
            for k, n in agreed.items():
                accepted[k] = max(accepted[k], n)
        empty = len(self._reads) == self.votes and not any(self._reads)
        return accepted, empty


class TillTracker:
    def __init__(
        self,
        till: int,
        state: TillState | None = None,
        cfg: TrackerCfg | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        cfg = cfg or TrackerCfg()
        self.cfg, self.clock = cfg, clock
        self.state = state or TillState(till=till)
        self.label = f"SCO {till}"
        self._status = Voter(cfg.votes, 101)  # statuses need N agreeing reads, no shortcut
        self._basket = Voter(cfg.votes, cfg.high_conf)
        self._txn = Voter(cfg.votes, cfg.high_conf)
        self._total = Voter(cfg.votes, cfg.high_conf)
        self._to_pay = Voter(cfg.votes, cfg.high_conf)
        self._rows = RowVoter(cfg.votes, cfg.high_conf)
        self._rows.seed(self.state.items)

    # -- helpers ---------------------------------------------------------------------------
    def _emit(self, type_: str, conf: float | None = None, **kw) -> Event:
        st = self.state
        st.seq += 1
        return Event(
            id=event_id(self.label, st.txn_id, st.seq),
            ts=self.clock(),
            till=self.label,
            txn_id=st.txn_id,
            type=type_,
            total=kw.pop("total", st.total),
            status=kw.pop("status", st.status),
            confidence=conf,
            **kw,
        )

    def _clear_basket(self) -> None:
        st = self.state
        st.items, st.started, st.completed = [], False, False
        st.txn_id = st.basket_no = None
        st.total = st.to_pay = 0.0
        self._rows.reset()

    def _complete(self, detail: str) -> list[Event]:
        st = self.state
        if not st.started or st.completed:
            return []
        st.completed = True
        return [self._emit("transaction_completed", detail=detail)]

    # -- observations ----------------------------------------------------------------------
    def on_status(self, res: StatusResult) -> list[Event]:
        if res.status == "unknown" or res.score < self.cfg.min_status_score:
            return []
        value = self._status.feed(res.status, res.score / 100)
        st, events = self.state, []
        if value is None or value == st.status:
            return events
        prev, st.status = st.status, str(value)
        if prev is None:  # first sighting after a cold start: adopt silently
            st.online = st.status != "closed"
            return events
        events.append(self._emit("status_changed", prev_status=prev))
        if st.status == "closed" and st.online is not False:
            st.online = False
            events.append(self._emit("till_offline"))
            events += self._complete("till closed")
        elif prev == "closed" and st.status != "closed":
            st.online = True
            events.append(self._emit("till_online"))
        if st.status == "ended":
            events += self._complete("payment finished")
        return events

    def on_meta(self, basket: str | None, txn: str | None, conf: float) -> list[Event]:
        st, events = self.state, []
        if (b := self._basket.feed(basket, conf)) is not None:
            st.basket_no = str(b)
        t = self._txn.feed(txn, conf)
        if t is not None and str(t) != st.txn_id:
            if st.txn_id is not None:  # a different transaction id replaced the old one
                events += self._complete("transaction id changed")
                self._clear_basket()
            st.txn_id = str(t)
        return events

    def on_items(self, lines: list[Line]) -> list[Event]:
        st, events = self.state, []
        good = [ln for ln in lines if ln.kind != "item" or (ln.conf >= self.cfg.min_price_conf)]
        accepted, empty = self._rows.feed(good)
        if empty and st.items:
            events += self._complete("basket cleared")
            self._clear_basket()
            return events
        known = Counter((self._rows._cluster(i.name, i.price, 1.0), i.price) for i in st.items)
        for key, n in (accepted - known).items():
            for _ in range(n):
                if not st.started:
                    st.started, st.completed = True, False
                    events.append(self._emit("transaction_started"))
                item = StateItem(name=self._rows.display(key), price=key[1])
                st.items.append(item)
                events.append(self._emit("item_added", item=ItemModel(**item.model_dump())))
        return events

    def on_total(self, value: float | None, conf: float) -> list[Event]:
        return self._amount(self._total, "total", value, conf)

    def on_to_pay(self, value: float | None, conf: float) -> list[Event]:
        return self._amount(self._to_pay, "to_pay", value, conf)

    def _amount(self, voter: Voter, field: str, value: float | None, conf: float) -> list[Event]:
        if value is not None and conf < self.cfg.min_price_conf:
            value = None  # too uncertain to count as a read at all
        accepted = voter.feed(value, conf)
        if accepted is None:
            return []
        prev = getattr(self.state, field)
        setattr(self.state, field, float(accepted))  # type: ignore[arg-type]
        if field == "total" and prev is not None and prev != accepted:
            return [self._emit("total_changed", conf=conf)]
        return []
