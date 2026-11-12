"""Cabin consent for watch and talk.

Sharing means a guardian may ask. The stream starts only after Accept.
"""

from __future__ import annotations

import threading
import time
import uuid

from sqlalchemy.orm import Session

from .models import AccessAudit

_LOCK = threading.Lock()
_PENDING: dict[tuple[str, str], dict] = {}
_LIVE: dict[tuple[str, str], dict] = {}


def _key(ride_id: str, kind: str) -> tuple[str, str]:
    return ride_id, kind


def pending(ride_id: str, kind: str) -> dict | None:
    with _LOCK:
        row = _PENDING.get(_key(ride_id, kind))
        return dict(row) if row else None


def live(ride_id: str, kind: str) -> dict | None:
    with _LOCK:
        row = _LIVE.get(_key(ride_id, kind))
        return dict(row) if row else None


def ask(ride_id: str, kind: str, guardian_id: str, name: str) -> dict:
    row = {
        "rideId": ride_id,
        "guardianId": guardian_id,
        "name": name,
        "kind": kind,
        "at": int(time.time() * 1000),
    }
    with _LOCK:
        current = _LIVE.get(_key(ride_id, kind))
        if current and current["guardianId"] == guardian_id:
            return dict(current)
        _PENDING[_key(ride_id, kind)] = row
    return row


def accept(ride_id: str, kind: str) -> dict | None:
    with _LOCK:
        row = _PENDING.pop(_key(ride_id, kind), None)
        if row is None:
            return None
        _LIVE[_key(ride_id, kind)] = row
        return dict(row)


def deny(ride_id: str, kind: str) -> dict | None:
    with _LOCK:
        row = _PENDING.pop(_key(ride_id, kind), None)
        return dict(row) if row else None


def clear(ride_id: str | None = None, kind: str | None = None) -> None:
    with _LOCK:
        if ride_id is None:
            _PENDING.clear()
            _LIVE.clear()
            return
        kinds = (kind,) if kind else ("watch", "talk")
        for item in kinds:
            _PENDING.pop(_key(ride_id, item), None)
            _LIVE.pop(_key(ride_id, item), None)


def audit(db: Session, ride_id: str, guardian_id: str, kind: str, action: str, actor: str) -> None:
    db.add(
        AccessAudit(
            id=f"audit-{uuid.uuid4().hex[:12]}",
            ride_id=ride_id,
            guardian_id=guardian_id,
            kind=kind,
            action=action,
            at=int(time.time() * 1000),
            actor=actor,
        )
    )
