from __future__ import annotations

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from .config import settings
from .models import Base, Guardian, Ride

connect_args = {}
if settings.database_url.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(settings.database_url, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        seed(db)
        db.commit()


def seed(db: Session) -> None:
    ride_id = settings.default_ride_id
    ride = db.get(Ride, ride_id)
    if ride is None:
        db.add(
            Ride(
                id=ride_id,
                device_id=settings.default_device_id,
                passenger=settings.default_passenger,
                share_link=f"vms://trip/{ride_id}",
            )
        )
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
            ),
            Guardian(
                id="g-kabir",
                ride_id=ride_id,
                name="Kabir Shah",
                relationship="Brother",
                phone="+91 98450 11042",
                sharing=True,
            ),
            Guardian(
                id="g-desk",
                ride_id=ride_id,
                name="North Desk",
                relationship="Fleet desk",
                phone="+91 80 4120 0188",
                sharing=False,
            ),
        ]
    )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
