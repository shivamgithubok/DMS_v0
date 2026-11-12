"""Alert clips and latest DMS/CMS stills. Bytes live under MEDIA_DIR, not in SQL."""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .models import Alert, LiveFrame, MediaClip, Ride, Trip


def media_root() -> Path:
    path = Path(settings.media_dir)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def channel_name(value: str) -> str:
    name = (value or "").strip().lower()
    if name in {"driver", "dms"}:
        return "dms"
    if name in {"cabin", "cms", "passenger"}:
        return "cms"
    return ""


def save_live(db: Session, ride: Ride, channel: str, image: bytes, at: int | None = None) -> LiveFrame:
    when = int(at or time.time() * 1000)
    folder = media_root() / ride.id / "live"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{channel}.jpg"
    path.write_bytes(image)
    row = db.scalars(
        select(LiveFrame).where(LiveFrame.ride_id == ride.id, LiveFrame.channel == channel)
    ).first()
    if row is None:
        row = LiveFrame(ride_id=ride.id, channel=channel, at=when, path=str(path))
        db.add(row)
    else:
        row.at = when
        row.path = str(path)
    return row


def live_public(row: LiveFrame) -> dict:
    return {
        "rideId": row.ride_id,
        "channel": row.channel,
        "at": row.at,
        "url": f"/v1/rides/{row.ride_id}/frames/live/image?channel={row.channel}",
    }


def read_live(row: LiveFrame) -> bytes:
    return Path(row.path).read_bytes()


def save_clip(
    db: Session,
    ride: Ride,
    channel: str,
    frames: list[tuple[int, bytes]],
    alert_id: str = "",
    video: bytes = b"",
    start_at: int | None = None,
    end_at: int | None = None,
) -> MediaClip:
    if not frames and not video:
        raise ValueError("clip needs a video or at least one frame")
    clip_id = f"clip-{uuid.uuid4().hex[:10]}"
    folder = media_root() / ride.id / clip_id
    folder.mkdir(parents=True, exist_ok=True)
    stored = []
    for index, (at, image) in enumerate(frames):
        name = f"{index}.jpg"
        (folder / name).write_bytes(image)
        stored.append({"at": int(at), "file": name})
    stored.sort(key=lambda item: item["at"])
    if video:
        (folder / "clip.mp4").write_bytes(video)
    if stored:
        start, end = stored[0]["at"], stored[-1]["at"]
    else:
        start, end = int(start_at or 0), int(end_at or start_at or 0)
    active = db.scalars(
        select(Trip).where(Trip.ride_id == ride.id, Trip.status == "active").order_by(Trip.started_at.desc())
    ).first()
    row = MediaClip(
        id=clip_id,
        ride_id=ride.id,
        alert_id=alert_id,
        channel=channel,
        start_at=start,
        end_at=end,
        frames_json=json.dumps(stored),
        trip_started_at=active.started_at if active is not None else None,
        expires_at=int(time.time() * 1000) + int(settings.clip_ttl_days) * 86_400_000,
    )
    db.add(row)
    if alert_id:
        alert = db.scalars(select(Alert).where(Alert.id == alert_id, Alert.ride_id == ride.id)).first()
        if alert is not None:
            alert.clip_id = clip_id
            from .store import write_story

            write_story(alert, row)
    return row


def clip_public(row: MediaClip) -> dict:
    try:
        frames = json.loads(row.frames_json or "[]")
    except json.JSONDecodeError:
        frames = []
    return {
        "id": row.id,
        "rideId": row.ride_id,
        "alertId": row.alert_id or None,
        "channel": row.channel,
        "startAt": row.start_at,
        "endAt": row.end_at,
        "frames": [
            {
                "at": item["at"],
                "url": f"/v1/media/{row.id}/{item['file']}",
            }
            for item in frames
            if item.get("file")
        ],
        "videoUrl": f"/v1/media/{row.id}/clip.mp4" if (media_root() / row.ride_id / row.id / "clip.mp4").is_file() else None,
        "expiresAt": row.expires_at,
    }


def clip_expired(row: MediaClip, now_ms: int | None = None) -> bool:
    if not row.expires_at:
        return False
    now = int(now_ms if now_ms is not None else time.time() * 1000)
    return now >= int(row.expires_at)


def delete_clip_files(row: MediaClip) -> None:
    folder = media_root() / row.ride_id / row.id
    if folder.is_dir():
        shutil.rmtree(folder, ignore_errors=True)


def clip_file(row: MediaClip, name: str) -> Path | None:
    if "/" in name or name.startswith("."):
        return None
    path = media_root() / row.ride_id / row.id / name
    if not path.is_file():
        return None
    return path
