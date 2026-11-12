"""Speak token, push delivery, trip alert filter, and GPS on the alert pin."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("jwt")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
sys.path.insert(0, str(ROOT))


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/vms.db")
    monkeypatch.setenv("MEDIA_DIR", str(tmp_path / "media"))
    monkeypatch.setenv("LIVEKIT_URL", "dummy://local")
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "devsecret-change-me-please-32b")
    monkeypatch.setenv("MEDIA_DEVICE_KEY", "dev-media-key")
    from app.config import settings

    settings.database_url = f"sqlite:///{tmp_path}/vms.db"
    settings.media_dir = str(tmp_path / "media")
    settings.livekit_url = "dummy://local"
    settings.livekit_api_key = "devkey"
    settings.livekit_api_secret = "devsecret-change-me-please-32b"
    settings.media_device_key = "dev-media-key"
    settings.fcm_server_key = "test-fcm-key"
    from app import db as cloud_db
    from app.models import Base, Guardian, Ride

    engine = __import__("sqlalchemy").create_engine(settings.database_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = __import__("sqlalchemy.orm", fromlist=["sessionmaker"]).sessionmaker(bind=engine, expire_on_commit=False)
    cloud_db.engine = engine
    cloud_db.SessionLocal = Session
    with Session() as db:
        db.add(Ride(id="ride-1842", device_id="orin-nano-01", analysis_json="{}", publisher_at=0, lat=12.9, lon=77.6))
        db.add(Guardian(id="g-priya", ride_id="ride-1842", name="Priya Shah", sharing=True, push_token="ExponentPushToken[abc]"))
        db.commit()
    from app.db import get_db
    from app.main import app

    def _override():
        session = cloud_db.SessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _override
    from app import consent

    consent.clear()
    return app, cloud_db


def test_trip_alerts_and_gps_pin(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.models import Alert, Trip
    from app.store import alert_public, alert_row
    from sqlalchemy import select

    app, cloud_db = _client(tmp_path, monkeypatch)
    with patch("app.mqtt_ingest.ingest.start"):
        with TestClient(app) as client:
            with cloud_db.SessionLocal() as db:
                ride = db.get(__import__("app.models", fromlist=["Ride"]).Ride, "ride-1842")
                db.add(Trip(ride_id="ride-1842", started_at=1000, ended_at=5000, status="ended", device_id="orin-nano-01"))
                db.add(Trip(ride_id="ride-1842", started_at=8000, ended_at=None, status="active", device_id="orin-nano-01"))
                inside = alert_row(db, ride, {"id": "a-in", "type": "DISTRACTION", "severity": "high", "epoch_ms": 2000, "lat": 12.91, "lon": 77.61})
                alert_row(db, ride, {"id": "a-out", "type": "DISTRACTION", "severity": "medium", "epoch_ms": 9000})
                db.commit()
                outside = db.scalars(select(Alert).where(Alert.id == "a-out")).first()
                assert alert_public(inside)["coordinate"] == {"latitude": 12.91, "longitude": 77.61}
                assert alert_public(outside)["coordinate"]["latitude"] == 12.9

            listed = client.get("/v1/rides/ride-1842/alerts", params={"startedAt": 1000})
            assert listed.status_code == 200
            assert [item["id"] for item in listed.json()["alerts"]] == ["a-in"]
            unknown = client.get("/v1/rides/ride-1842/alerts", params={"startedAt": 42})
            assert unknown.status_code == 200 and unknown.json()["alerts"] == []
            history = client.get("/v1/history", params={"deviceId": "orin-nano-01"})
            first = next(row for row in history.json()["rides"] if row["startedAt"] == 1000)
            assert first["alertCount"] == 1
    app.dependency_overrides.clear()


def test_speak_token_and_push(tmp_path, monkeypatch):
    import jwt
    from fastapi.testclient import TestClient

    from app.config import settings

    app, cloud_db = _client(tmp_path, monkeypatch)
    sent = []

    def _capture(targets, body):
        sent.append((targets, body))

    monkeypatch.setattr("app.push.deliver", _capture)
    with patch("app.mqtt_ingest.ingest.start"):
        with TestClient(app) as client:
            blocked = client.post("/v1/rides/ride-1842/video/talk")
            assert blocked.status_code == 409

            settings.livekit_url = "ws://127.0.0.1:7880"
            client.post("/v1/rides/ride-1842/guardians/g-priya/capabilities", json={"talk": True})
            waiting = client.post("/v1/rides/ride-1842/video/talk", params={"guardianId": "g-priya"})
            assert waiting.status_code == 403
            assert client.post(
                "/v1/devices/orin-nano-01/consent",
                json={"key": "dev-media-key", "kind": "talk", "decision": "accept"},
            ).status_code == 200
            started = client.post("/v1/rides/ride-1842/video/talk", params={"guardianId": "g-priya"})
            assert started.status_code == 200
            body = started.json()
            assert body["tracks"] == ["guardian-mic"]
            assert body["room"] == "ride-1842"
            claims = jwt.decode(body["token"], settings.livekit_api_secret, algorithms=["HS256"], issuer="devkey")
            assert claims["video"]["canPublish"] is True
            assert claims["video"]["canSubscribe"] is False
            assert claims["video"]["canPublishSources"] == ["microphone"]

            token = client.post("/v1/devices/orin-nano-01/media-token", json={"key": "dev-media-key"})
            assert token.json()["talk"] is True
            assert token.json()["speakerToken"]
            speaker = jwt.decode(token.json()["speakerToken"], settings.livekit_api_secret, algorithms=["HS256"], issuer="devkey")
            assert speaker["video"]["canPublish"] is False
            assert speaker["video"]["canSubscribe"] is True

            stopped = client.post("/v1/rides/ride-1842/video/talk/cancel")
            assert stopped.json()["talk"] is False
            idle = client.post("/v1/devices/orin-nano-01/media-token", json={"key": "dev-media-key"})
            assert idle.json()["talk"] is False

            sos = client.post("/v1/rides/ride-1842/sos", json={"passenger": "Meera", "latitude": 12.9, "longitude": 77.6})
            assert sos.status_code == 200
            assert sos.json()["pushed"] == 1
            for _ in range(20):
                if sent:
                    break
                __import__("time").sleep(0.05)
            assert sent[0][0] == ["ExponentPushToken[abc]"]
            assert sent[0][1]["kind"] == "sos"
    app.dependency_overrides.clear()
