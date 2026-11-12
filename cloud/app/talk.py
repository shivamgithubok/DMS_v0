"""Guardian Speak flag. Stays on until the guardian cancels."""

from __future__ import annotations

import threading

_LOCK = threading.Lock()
_ON: set[str] = set()


def note(ride_id: str) -> None:
    with _LOCK:
        _ON.add(ride_id)


def wanted(ride_id: str) -> bool:
    with _LOCK:
        return ride_id in _ON


def clear(ride_id: str | None = None) -> None:
    with _LOCK:
        if ride_id is None:
            _ON.clear()
        else:
            _ON.discard(ride_id)
