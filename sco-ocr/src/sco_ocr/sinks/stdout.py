from __future__ import annotations

import sys

from ..events import Event


class StdoutSink:
    name = "stdout"

    def emit(self, event: Event) -> None:
        sys.stdout.write(event.model_dump_json(exclude_none=True) + "\n")
        sys.stdout.flush()

    def close(self) -> None:
        pass
