"""Event models, idempotent ids and JSON schema export."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

EventType = Literal[
    "transaction_started",
    "item_added",
    "item_removed",
    "total_changed",
    "status_changed",
    "transaction_completed",
    "till_offline",
    "till_online",
    "stream_degraded",
    "stream_recovered",
]


class ItemModel(BaseModel):
    name: str
    price: float


class Event(BaseModel):
    id: str
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    till: str  # "SCO 3"; "stream" for stream-level events
    txn_id: str | None = None
    type: EventType
    item: ItemModel | None = None
    total: float | None = None
    status: str | None = None
    prev_status: str | None = None
    confidence: float | None = None
    detail: str | None = None


def event_id(till: str, txn_id: str | None, seq: int) -> str:
    """sha1(till + txn_id + seq): stable across restarts, so consumers can dedupe."""
    return hashlib.sha1(f"{till}|{txn_id or ''}|{seq}".encode()).hexdigest()


def export_schema(path: Path | str = "schema/event.schema.json") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(Event.model_json_schema(), indent=2) + "\n", encoding="utf-8")
    return p


if __name__ == "__main__":
    print(export_schema())
