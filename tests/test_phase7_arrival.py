"""Phase 7: a live fix can arrive home. A fixed pin cannot, and the trip link dies."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
sys.path.insert(0, str(ROOT))

HOME = {"id": "home", "name": "Home", "lat": 12.91, "lon": 77.61, "radius_m": 80}


def _feed(watch, source, fix, lat, lon, at_s, fresh=True):
    return watch.consider(source=source, fix=fix, fresh=fresh, lat=lat, lon=lon, at_s=at_s)


def test_fixed_gps_does_not_arrive_or_stop():
    from dms.config import DEFAULTS, Config
    from dms.gps import GpsFeed, PlaceWatch, route_point
    from dms.state import GpsState

    watch = PlaceWatch([HOME])
    assert _feed(watch, "fixed", "fixed", HOME["lat"], HOME["lon"], 0) == []
    assert _feed(watch, "nmea", "live", HOME["lat"], HOME["lon"], 0, fresh=False) == []
    assert route_point("fixed", "fixed", 12.9, 77.6) is None
    assert route_point("nmea", "live", None, None) is None

    shared = type("Box", (), {"gps": None})()
    gps = dict(DEFAULTS["gps"])
    gps["lat"] = None
    gps["lon"] = None
    feed = GpsFeed(Config({"gps": gps}), shared)
    feed._publish()
    assert shared.gps.available is False
    assert shared.gps.fix == "none"
    assert isinstance(shared.gps, GpsState)


def test_nmea_track_into_a_saved_place_arrives_once():
    from dms.gps import PlaceWatch

    watch = PlaceWatch([HOME])
    seen = []
    for at_s, lat in ((0, 12.95), (10, 12.93), (20, 12.910)):
        seen += _feed(watch, "nmea", "live", lat, 77.61, at_s)
    assert [item["type"] for item in seen] == ["ARRIVED"]
    assert _feed(watch, "nmea", "live", 12.910, 77.61, 30) == []

    stop = PlaceWatch([])
    stopped = []
    stopped += _feed(stop, "nmea", "live", 12.90, 77.61, 0)
    stopped += _feed(stop, "nmea", "live", 12.902, 77.61, 10)
    for step in range(1, 12):
        stopped += _feed(stop, "nmea", "live", 12.902, 77.61, 10 + step * 10)
    assert stopped == []
    stopped += _feed(stop, "nmea", "live", 12.902, 77.61, 130)
    assert [item["type"] for item in stopped] == ["UNEXPECTED_STOP"]
    assert _feed(stop, "nmea", "live", 12.902, 77.61, 140) == []


def test_history_keeps_only_a_live_track_and_pages_once(tmp_path, monkeypatch):
    from dms.alerts import AlertBus
    from dms.config import Config
    from dms.history import TripHistory
    from dms.state import GpsState, SharedState

    monkeypatch.setattr(
        "dms.history.get_json",
        lambda url, timeout: {"places": [{**HOME, "radiusM": 80}]},
    )
    shared = SharedState()
    bus = AlertBus("orin-nano-01", cooldown_s=0)
    cfg = Config({
        "device_id": "orin-nano-01",
        "history": {"ride_id": "ride-1842", "db": str(tmp_path / "history.sqlite3"), "point_interval_s": 1},
        "gps": {"source": "nmea", "stale_s": 15, "places": []},
        "webrtc": {"api_url": "http://127.0.0.1:9", "device_key": "dev-media-key"},
    })
    history = TripHistory(cfg, shared, bus)
    history._started_at = history.store.open_session(1_000)
    history._place_poll = 0
    shared.gps = GpsState(enabled=True, available=True, fix="live", source="nmea", lat=12.95, lon=77.61, at=1_000)
    history._sample()
    shared.gps = GpsState(enabled=True, available=True, fix="live", source="nmea", lat=12.91, lon=77.61, at=3_000)
    history._sample()
    assert [alert.type for alert in bus.history] == ["ARRIVED"]
    shared.gps = GpsState(enabled=True, available=True, fix="fixed", source="fixed", lat=1.0, lon=2.0, at=9_000)
    history._sample()
    count = history.store.db.execute(
        "SELECT COUNT(*) AS n FROM trip_points WHERE started_at = ?",
        (history._started_at,),
    ).fetchone()["n"]
    assert count == 2
    from app.push import should_notify

    class _Row:
        orin_type = "ARRIVED"
        kind = "arrival"

    assert should_notify(_Row()) is True
    history.store.close()


def test_expired_link_cannot_open_video_and_fixed_point_is_not_a_route(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine, func, select
    from sqlalchemy.orm import sessionmaker

    from app.config import settings
    from app.models import Base, Guardian, Ride, Trip, TripPoint
    from app.store import alert_row, record_point

    monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path}/vms.db")
    monkeypatch.setattr(settings, "auth_required", False)
    monkeypatch.setattr(settings, "media_device_key", "dev-media-key")
    monkeypatch.setattr(settings, "require_device_key", False)
    from app import consent, db as cloud_db
    from app.db import get_db
    from app.main import app

    consent.clear()
    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
    previous_engine, previous_session = cloud_db.engine, cloud_db.SessionLocal
    cloud_db.engine = engine
    cloud_db.SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    Base.metadata.create_all(engine)
    with cloud_db.SessionLocal() as db:
        ride = Ride(id="ride-1842", device_id="orin-nano-01", analysis_json="{}", score=98)
        db.add(ride)
        db.add(Guardian(id="g-priya", ride_id="ride-1842", name="Priya Shah", sharing=True, can_notify=True))
        db.add(Trip(
            ride_id="ride-1842", public_id="trip-live0000001", started_at=5_000,
            status="active", device_id="orin-nano-01",
        ))
        db.add(Trip(
            ride_id="ride-1842", public_id="trip-deadbeef0001", started_at=1_000,
            ended_at=2_000, status="ended", link_expires_at=3_000, device_id="orin-nano-01",
        ))
        db.commit()
        record_point(db, ride, 12.9, 77.6, 4_000, "fixed")
        record_point(db, ride, 12.91, 77.61, 6_000, "live")
        row = alert_row(db, ride, {
            "id": "arrived-5000-home",
            "type": "ARRIVED",
            "severity": "info",
            "description": "Arrived at Home",
            "epoch_ms": 6_000,
        })
        assert row is not None
        assert row.severity == "info"
        assert ride.score == 98
        points = db.scalar(select(func.count()).select_from(TripPoint))
        db.commit()
    assert points == 1

    def _override():
        session = cloud_db.SessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _override
    try:
        from unittest.mock import patch

        with patch("app.mqtt_ingest.ingest.start"):
            with TestClient(app) as client:
                shared = client.post("/v1/rides/ride-1842/share")
                assert shared.status_code == 200
                assert shared.json()["link"] == "vms://trip/trip-live0000001"
                assert shared.json()["expiresAt"] is None
                dead = client.get("/v1/rides/trip-deadbeef0001/video", params={"guardianId": "g-priya"})
                assert dead.status_code == 410
                living = client.get("/v1/rides/trip-live0000001/video", params={"guardianId": "g-priya"})
                assert living.status_code == 403
                saved = client.post("/v1/rides/ride-1842/places", json={"name": "Home", "lat": 12.91, "lon": 77.61})
                assert saved.status_code == 200
                places = client.get("/v1/devices/orin-nano-01/places", params={"key": "dev-media-key"})
                assert places.status_code == 200
                assert places.json()["places"][0]["name"] == "Home"
                assert client.get("/v1/devices/orin-nano-01/places", params={"key": "nope"}).status_code == 403
    finally:
        app.dependency_overrides.clear()
        consent.clear()
        cloud_db.engine = previous_engine
        cloud_db.SessionLocal = previous_session
