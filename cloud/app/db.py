from __future__ import annotations

import time
from pathlib import Path

from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import Session, sessionmaker

from .config import settings
from .models import Base, Guardian, Ride


def _ensure_sqlite_parent(url: str) -> None:
    """sqlite:///./data/local-vms.db fails when the folder does not exist yet."""
    if not url.startswith("sqlite"):
        return
    raw = url.split("?", 1)[0]
    prefix = "sqlite:///"
    if not raw.startswith(prefix):
        return
    location = raw[len(prefix):]
    if location in ("", ":memory:"):
        return
    parent = Path(location).expanduser().parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)


_ensure_sqlite_parent(settings.database_url)

connect_args = {}
if settings.database_url.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(settings.database_url, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


# Columns that store Unix epoch milliseconds. Postgres INTEGER is 32-bit and
# rejects a current timestamp. SQLite INTEGER is already 64-bit.
EPOCH_MS_COLUMNS = (
    ("rides", "started_at"),
    ("rides", "ended_at"),
    ("rides", "publisher_at"),
    ("alerts", "at"),
    ("live_frames", "at"),
    ("notices", "at"),
    ("media_clips", "start_at"),
    ("media_clips", "end_at"),
    ("media_clips", "expires_at"),
    ("guardians", "last_notified_at"),
)


def init_db() -> None:
    Base.metadata.create_all(engine)
    _ensure_columns()
    widen_epoch_columns()
    with SessionLocal() as db:
        seed(db)
        db.commit()


def _ensure_columns() -> None:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        if "rides" in tables:
            cols = {c["name"] for c in inspector.get_columns("rides")}
            if "analysis_json" not in cols:
                conn.execute(text("ALTER TABLE rides ADD COLUMN analysis_json TEXT DEFAULT '{}'"))
            if "publisher_at" not in cols:
                conn.execute(text("ALTER TABLE rides ADD COLUMN publisher_at INTEGER DEFAULT 0"))
            if "status" not in cols:
                conn.execute(text("ALTER TABLE rides ADD COLUMN status VARCHAR(16) DEFAULT 'active'"))
            if "started_at" not in cols:
                conn.execute(text("ALTER TABLE rides ADD COLUMN started_at INTEGER DEFAULT 0"))
            if "ended_at" not in cols:
                conn.execute(text("ALTER TABLE rides ADD COLUMN ended_at INTEGER"))
        if "alerts" in tables:
            cols = {c["name"] for c in inspector.get_columns("alerts")}
            if "channel" not in cols:
                conn.execute(text("ALTER TABLE alerts ADD COLUMN channel VARCHAR(16) DEFAULT ''"))
            if "confidence" not in cols:
                conn.execute(text("ALTER TABLE alerts ADD COLUMN confidence FLOAT"))
            if "clip_id" not in cols:
                conn.execute(text("ALTER TABLE alerts ADD COLUMN clip_id VARCHAR(64) DEFAULT ''"))
            if "trip_started_at" not in cols:
                conn.execute(text("ALTER TABLE alerts ADD COLUMN trip_started_at BIGINT"))
            if "story" not in cols:
                conn.execute(text("ALTER TABLE alerts ADD COLUMN story TEXT DEFAULT ''"))
        if "guardians" in tables:
            cols = {c["name"] for c in inspector.get_columns("guardians")}
            if "can_notify" not in cols:
                conn.execute(text("ALTER TABLE guardians ADD COLUMN can_notify BOOLEAN DEFAULT 0"))
                conn.execute(text(
                    "UPDATE guardians SET can_notify = CASE WHEN sharing THEN 1 ELSE 0 END"
                ))
            if "can_watch" not in cols:
                conn.execute(text("ALTER TABLE guardians ADD COLUMN can_watch BOOLEAN DEFAULT 0"))
            if "can_talk" not in cols:
                conn.execute(text("ALTER TABLE guardians ADD COLUMN can_talk BOOLEAN DEFAULT 0"))
        if "trips" in tables:
            cols = {c["name"] for c in inspector.get_columns("trips")}
            if "public_id" not in cols:
                conn.execute(text("ALTER TABLE trips ADD COLUMN public_id VARCHAR(64)"))
            if "link_expires_at" not in cols:
                conn.execute(text("ALTER TABLE trips ADD COLUMN link_expires_at BIGINT"))
            conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS trips_public_id ON trips (public_id) WHERE public_id IS NOT NULL"
            ))
        if "media_clips" in tables:
            cols = {c["name"] for c in inspector.get_columns("media_clips")}
            if "trip_started_at" not in cols:
                conn.execute(text("ALTER TABLE media_clips ADD COLUMN trip_started_at BIGINT"))
            if "expires_at" not in cols:
                conn.execute(text("ALTER TABLE media_clips ADD COLUMN expires_at BIGINT"))
                conn.execute(text(
                    "UPDATE media_clips SET expires_at = end_at + 2592000000 "
                    "WHERE expires_at IS NULL AND end_at IS NOT NULL AND end_at > 0"
                ))


def widen_epoch_columns() -> None:
    """Widen existing Postgres INTEGER epoch columns. New databases already use BIGINT."""
    if engine.dialect.name != "postgresql":
        return
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table, column in EPOCH_MS_COLUMNS:
            if table not in tables:
                continue
            found = {col["name"]: col for col in inspector.get_columns(table)}
            current = found.get(column)
            if current is None or "BIGINT" in str(current["type"]).upper():
                continue
            conn.execute(text(f'ALTER TABLE "{table}" ALTER COLUMN "{column}" TYPE BIGINT'))


def seed(db: Session) -> None:
    ride_id = settings.default_ride_id
    ride = db.get(Ride, ride_id)
    if ride is None:
        db.add(
            Ride(
                id=ride_id,
                device_id=settings.default_device_id,
                passenger=settings.default_passenger if settings.seed_demo else "",
                share_link=f"vms://trip/{ride_id}",
                status="active",
                started_at=int(time.time() * 1000),
            )
        )
    if not settings.seed_demo:
        return
    existing = db.scalars(select(Guardian).where(Guardian.ride_id == ride_id)).all()
    if existing:
        return
    db.add_all(
        [
            Guardian(
                id="g-priya",
                ride_id=ride_id,
                name="Priya Shah",
                relationship="Mother",
                phone="+91 98450 11008",
                sharing=True,
                can_notify=True,
            ),
            Guardian(
                id="g-kabir",
                ride_id=ride_id,
                name="Kabir Shah",
                relationship="Brother",
                phone="+91 98450 11042",
                sharing=True,
                can_notify=True,
            ),
            Guardian(
                id="g-desk",
                ride_id=ride_id,
                name="North Desk",
                relationship="Fleet desk",
                phone="+91 80 4120 0188",
                sharing=False,
                can_notify=False,
            ),
        ]
    )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
