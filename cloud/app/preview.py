"""In-memory dummy video stills so Guardian can be tested without a LiveKit SFU."""

from __future__ import annotations

import threading
import time

_LOCK = threading.Lock()
_FRAMES: dict[str, dict[str, dict]] = {}


def put(ride_id: str, track: str, image: bytes, mime: str = "image/jpeg") -> None:
    with _LOCK:
        _FRAMES.setdefault(ride_id, {})[track] = {
            "image": image,
            "mime": mime or "image/jpeg",
            "at": int(time.time() * 1000),
        }


def get(ride_id: str, track: str) -> dict | None:
    with _LOCK:
        side = _FRAMES.get(ride_id) or {}
        row = side.get(track)
        return dict(row) if row else None


_STREAMS: dict[str, dict] = {}


def put_stream(ride_id: str, image: bytes, audio: bytes, samples: int, rate: int, width: int, height: int) -> None:
    with _LOCK:
        _STREAMS[ride_id] = {
            "image": image,
            "audio": audio,
            "samples": int(samples),
            "rate": int(rate),
            "width": int(width),
            "height": int(height),
            "at": int(time.time() * 1000),
            "seq": int((_STREAMS.get(ride_id) or {}).get("seq", 0)) + 1,
        }


def stream(ride_id: str) -> dict | None:
    with _LOCK:
        row = _STREAMS.get(ride_id)
        return dict(row) if row else None
