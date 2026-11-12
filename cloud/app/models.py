from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import BigInteger, Boolean, DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Ride(Base):
    __tablename__ = "rides"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    device_id: Mapped[str] = mapped_column(String(128), default="", index=True)
    passenger: Mapped[str] = mapped_column(String(128), default="")
    safety_status: Mapped[str] = mapped_column(String(32), default="safe")
    driver_state: Mapped[str] = mapped_column(String(32), default="attentive")
    device_state: Mapped[str] = mapped_column(String(32), default="offline")
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    heading: Mapped[float | None] = mapped_column(Float, nullable=True)
    speed_kmh: Mapped[float | None] = mapped_column(Float, nullable=True)
    eta_min: Mapped[int | None] = mapped_column(Integer, nullable=True)
    score: Mapped[int] = mapped_column(Integer, default=98)
    share_link: Mapped[str] = mapped_column(String(256), default="")
    cabin_json: Mapped[str] = mapped_column(Text, default="[]")
    analysis_json: Mapped[str] = mapped_column(Text, default="{}")
    publisher_at: Mapped[int] = mapped_column(BigInteger, default=0)
    status: Mapped[str] = mapped_column(String(16), default="active")
    started_at: Mapped[int] = mapped_column(BigInteger, default=0)
    ended_at: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Alert(Base):
    __tablename__ = "alerts"
    __table_args__ = (UniqueConstraint("id", name="alerts_id_unique"),)

    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(128), index=True)
    ride_id: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(64), default="suspicious_activity")
    orin_type: Mapped[str] = mapped_column(String(64), default="")
    severity: Mapped[str] = mapped_column(String(32), default="medium")
    title: Mapped[str] = mapped_column(String(256), default="")
    detail: Mapped[str] = mapped_column(Text, default="")
    at: Mapped[int] = mapped_column(BigInteger, default=0)
    location_label: Mapped[str] = mapped_column(String(128), default="")
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    origin: Mapped[str] = mapped_column(String(32), default="vehicle")
    evidence_id: Mapped[str] = mapped_column(String(128), default="")
    channel: Mapped[str] = mapped_column(String(16), default="")
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    clip_id: Mapped[str] = mapped_column(String(64), default="")
    trip_started_at: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    story: Mapped[str] = mapped_column(Text, default="")


class Guardian(Base):
    __tablename__ = "guardians"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ride_id: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(128))
    relationship: Mapped[str] = mapped_column(String(64), default="")
    phone: Mapped[str] = mapped_column(String(64), default="")
    sharing: Mapped[bool] = mapped_column(Boolean, default=True)
    can_notify: Mapped[bool] = mapped_column(Boolean, default=True)
    can_watch: Mapped[bool] = mapped_column(Boolean, default=False)
    can_talk: Mapped[bool] = mapped_column(Boolean, default=False)
    last_notified_at: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    push_token: Mapped[str] = mapped_column(String(512), default="")


class AccessAudit(Base):
    """Who asked to watch or talk, and who in the cabin accepted or denied."""

    __tablename__ = "access_audit"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ride_id: Mapped[str] = mapped_column(String(64), index=True)
    guardian_id: Mapped[str] = mapped_column(String(64), default="")
    kind: Mapped[str] = mapped_column(String(16), default="")
    action: Mapped[str] = mapped_column(String(16), default="")
    at: Mapped[int] = mapped_column(BigInteger, default=0)
    actor: Mapped[str] = mapped_column(String(64), default="")


class MediaClip(Base):
    __tablename__ = "media_clips"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ride_id: Mapped[str] = mapped_column(String(64), index=True)
    alert_id: Mapped[str] = mapped_column(String(128), default="", index=True)
    channel: Mapped[str] = mapped_column(String(16), default="dms")
    start_at: Mapped[int] = mapped_column(BigInteger, default=0)
    end_at: Mapped[int] = mapped_column(BigInteger, default=0)
    frames_json: Mapped[str] = mapped_column(Text, default="[]")
    trip_started_at: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    expires_at: Mapped[int | None] = mapped_column(BigInteger, nullable=True)


class LiveFrame(Base):
    __tablename__ = "live_frames"
    __table_args__ = (UniqueConstraint("ride_id", "channel", name="live_frame_ride_channel"),)

    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ride_id: Mapped[str] = mapped_column(String(64), index=True)
    channel: Mapped[str] = mapped_column(String(16))
    at: Mapped[int] = mapped_column(BigInteger, default=0)
    path: Mapped[str] = mapped_column(String(512), default="")


class Trip(Base):
    """One journey under a ride. ride-1842 stays the phone's ride id. started_at separates trips."""

    __tablename__ = "trips"
    __table_args__ = (UniqueConstraint("ride_id", "started_at", name="trip_ride_started"),)

    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ride_id: Mapped[str] = mapped_column(String(64), index=True)
    public_id: Mapped[str | None] = mapped_column(String(64), nullable=True, unique=True)
    started_at: Mapped[int] = mapped_column(BigInteger, index=True)
    ended_at: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="active")
    device_id: Mapped[str] = mapped_column(String(128), default="", index=True)
    safety_status: Mapped[str] = mapped_column(String(32), default="safe")
    score: Mapped[int] = mapped_column(Integer, default=98)
    alert_count: Mapped[int] = mapped_column(Integer, default=0)
    start_lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    start_lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    end_lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    end_lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    link_expires_at: Mapped[int | None] = mapped_column(BigInteger, nullable=True)


class Place(Base):
    """A saved point, such as home. Arrival uses it only with a live fix."""

    __tablename__ = "places"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ride_id: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(128), default="Home")
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    radius_m: Mapped[float] = mapped_column(Float, default=80)


class TripPoint(Base):
    __tablename__ = "trip_points"
    __table_args__ = (UniqueConstraint("ride_id", "started_at", "at", name="trip_point_at"),)

    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ride_id: Mapped[str] = mapped_column(String(64), index=True)
    started_at: Mapped[int] = mapped_column(BigInteger, index=True)
    at: Mapped[int] = mapped_column(BigInteger)
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    fix: Mapped[str] = mapped_column(String(16), default="live")


class GuardianRecording(Base):
    """Video the guardian started. Not an alert clip."""

    __tablename__ = "guardian_recordings"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ride_id: Mapped[str] = mapped_column(String(64), index=True)
    guardian_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), default="recording")
    started_at: Mapped[int] = mapped_column(BigInteger, default=0)
    ended_at: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    path: Mapped[str] = mapped_column(String(512), default="")


class GuardianSnapshot(Base):
    """Still the guardian saved. Not the rotating live JPEG."""

    __tablename__ = "guardian_snapshots"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ride_id: Mapped[str] = mapped_column(String(64), index=True)
    guardian_id: Mapped[str] = mapped_column(String(64), index=True)
    channel: Mapped[str] = mapped_column(String(16), default="dms")
    at: Mapped[int] = mapped_column(BigInteger, default=0)
    path: Mapped[str] = mapped_column(String(512), default="")


class Notice(Base):
    __tablename__ = "notices"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ride_id: Mapped[str] = mapped_column(String(64), index=True)
    at: Mapped[int] = mapped_column(BigInteger, default=0)
    title: Mapped[str] = mapped_column(String(128), default="")
    body: Mapped[str] = mapped_column(Text, default="")
    guardian_names: Mapped[str] = mapped_column(Text, default="")
    origin: Mapped[str] = mapped_column(String(32), default="vehicle")
