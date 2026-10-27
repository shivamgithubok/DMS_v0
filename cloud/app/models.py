from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, UniqueConstraint
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
    at: Mapped[int] = mapped_column(Integer, default=0)
    location_label: Mapped[str] = mapped_column(String(128), default="")
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    origin: Mapped[str] = mapped_column(String(32), default="vehicle")
    evidence_id: Mapped[str] = mapped_column(String(128), default="")


class Guardian(Base):
    __tablename__ = "guardians"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ride_id: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(128))
    relationship: Mapped[str] = mapped_column(String(64), default="")
    phone: Mapped[str] = mapped_column(String(64), default="")
    sharing: Mapped[bool] = mapped_column(Boolean, default=True)
    last_notified_at: Mapped[int | None] = mapped_column(Integer, nullable=True)
    push_token: Mapped[str] = mapped_column(String(512), default="")


class Notice(Base):
    __tablename__ = "notices"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ride_id: Mapped[str] = mapped_column(String(64), index=True)
    at: Mapped[int] = mapped_column(Integer, default=0)
    title: Mapped[str] = mapped_column(String(128), default="")
    body: Mapped[str] = mapped_column(Text, default="")
    guardian_names: Mapped[str] = mapped_column(Text, default="")
    origin: Mapped[str] = mapped_column(String(32), default="vehicle")
