"""Guardian-started recordings and snaps. Alert clips stay in media_clips."""

from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from .clips import media_root
from .models import Guardian, GuardianRecording, GuardianSnapshot, LiveFrame, Ride


def _guardian(db: Session, ride_id: str, guardian_id: str) -> Guardian:
    row = db.get(Guardian, guardian_id)
    if row is None or row.ride_id != ride_id:
        raise LookupError("guardian not found")
    return row


def active_recording(db: Session, ride_id: str) -> GuardianRecording | None:
    return db.scalars(
        select(GuardianRecording).where(
            GuardianRecording.ride_id == ride_id,
            GuardianRecording.status == "recording",
        )
    ).first()


def start_recording(db: Session, ride: Ride, guardian_id: str) -> GuardianRecording:
    _guardian(db, ride.id, guardian_id)
    current = active_recording(db, ride.id)
    if current is not None:
        return current
    row = GuardianRecording(
        id=f"rec-{uuid.uuid4().hex[:10]}",
        ride_id=ride.id,
        guardian_id=guardian_id,
        status="recording",
        started_at=int(time.time() * 1000),
    )
    db.add(row)
    return row


def cancel_recording(db: Session, ride_id: str, rec_id: str) -> GuardianRecording:
    row = db.get(GuardianRecording, rec_id)
    if row is None or row.ride_id != ride_id:
        raise LookupError("recording not found")
    if row.status == "recording":
        row.status = "stopping"
        row.ended_at = int(time.time() * 1000)
    return row


def save_recording_video(db: Session, ride: Ride, rec_id: str, video: bytes) -> GuardianRecording:
    row = db.get(GuardianRecording, rec_id)
    if row is None or row.ride_id != ride.id:
        raise LookupError("recording not found")
    folder = media_root() / ride.id / "recordings" / row.id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "clip.mp4"
    path.write_bytes(video)
    row.path = str(path)
    row.status = "ready"
    row.ended_at = row.ended_at or int(time.time() * 1000)
    return row


def list_recordings(db: Session, ride_id: str, guardian_id: str = "") -> list[GuardianRecording]:
    rows = db.scalars(select(GuardianRecording).where(GuardianRecording.ride_id == ride_id)).all()
    if guardian_id:
        rows = [row for row in rows if row.guardian_id == guardian_id]
    rows.sort(key=lambda row: row.started_at or 0, reverse=True)
    return rows


def recording_public(row: GuardianRecording) -> dict:
    ready = row.status == "ready" and bool(row.path) and Path(row.path).is_file()
    return {
        "id": row.id,
        "rideId": row.ride_id,
        "guardianId": row.guardian_id,
        "status": row.status,
        "startedAt": row.started_at,
        "endedAt": row.ended_at,
        "videoUrl": f"/v1/media/recordings/{row.id}/clip.mp4" if ready else None,
        "audio": True,
    }


def recording_file(db: Session, rec_id: str) -> Path | None:
    row = db.get(GuardianRecording, rec_id)
    if row is None or not row.path:
        return None
    path = Path(row.path)
    return path if path.is_file() else None


def take_snapshot(db: Session, ride: Ride, guardian_id: str, channel: str) -> GuardianSnapshot:
    _guardian(db, ride.id, guardian_id)
    live = db.scalars(
        select(LiveFrame).where(LiveFrame.ride_id == ride.id, LiveFrame.channel == channel)
    ).first()
    if live is None or not live.path or not Path(live.path).is_file():
        raise FileNotFoundError("no frame yet")
    snap_id = f"snap-{uuid.uuid4().hex[:10]}"
    folder = media_root() / ride.id / "snapshots"
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{snap_id}.jpg"
    shutil.copyfile(live.path, dest)
    row = GuardianSnapshot(
        id=snap_id,
        ride_id=ride.id,
        guardian_id=guardian_id,
        channel=channel,
        at=int(live.at or time.time() * 1000),
        path=str(dest),
    )
    db.add(row)
    return row


def list_snapshots(db: Session, ride_id: str, guardian_id: str = "") -> list[GuardianSnapshot]:
    rows = db.scalars(select(GuardianSnapshot).where(GuardianSnapshot.ride_id == ride_id)).all()
    if guardian_id:
        rows = [row for row in rows if row.guardian_id == guardian_id]
    rows.sort(key=lambda row: row.at or 0, reverse=True)
    return rows


def snapshot_public(row: GuardianSnapshot) -> dict:
    return {
        "id": row.id,
        "rideId": row.ride_id,
        "guardianId": row.guardian_id,
        "channel": row.channel,
        "at": row.at,
        "url": f"/v1/media/snapshots/{row.id}/snap.jpg",
    }


def snapshot_file(db: Session, snap_id: str) -> tuple[GuardianSnapshot, Path] | None:
    row = db.get(GuardianSnapshot, snap_id)
    if row is None or not row.path:
        return None
    path = Path(row.path)
    if not path.is_file():
        return None
    return row, path


def recording_command(db: Session, ride_id: str) -> dict:
    """What the car should do on its next poll."""
    active = active_recording(db, ride_id)
    stopping = db.scalars(
        select(GuardianRecording).where(
            GuardianRecording.ride_id == ride_id,
            GuardianRecording.status == "stopping",
        )
    ).first()
    return {
        "recording": None if active is None else {"id": active.id, "startedAt": active.started_at, "guardianId": active.guardian_id},
        "recordingUpload": None if stopping is None else stopping.id,
    }
