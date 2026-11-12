"""Delete clip files and rows after their lifetime. Default is 30 days."""

from __future__ import annotations

import logging
import threading
import time

from sqlalchemy import select
from sqlalchemy.orm import Session

from .clips import delete_clip_files
from .models import MediaClip

log = logging.getLogger("vms.retention")


def retain_clips(db: Session, now_ms: int | None = None) -> int:
    """Remove clips whose expires_at has passed. A clip with no expiry stays."""
    now = int(now_ms if now_ms is not None else time.time() * 1000)
    rows = db.scalars(select(MediaClip).where(MediaClip.expires_at.is_not(None), MediaClip.expires_at <= now)).all()
    for row in rows:
        delete_clip_files(row)
        db.delete(row)
    return len(rows)


def start_retention(interval_s: float = 3600) -> None:
    def loop() -> None:
        from .db import SessionLocal

        while True:
            try:
                with SessionLocal() as db:
                    removed = retain_clips(db)
                    db.commit()
                if removed:
                    log.info("expired %d clip(s)", removed)
            except Exception:
                log.exception("clip retention failed")
            time.sleep(interval_s)

    threading.Thread(target=loop, name="clip-retention", daemon=True).start()
