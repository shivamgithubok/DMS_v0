"""Phase 1: phone routes lock when asked, and epoch milliseconds fit."""

import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
sys.path.insert(0, str(ROOT))


def test_health_hides_broker_details():
    from app.main import health

    body = health()
    assert body == {"ok": True, "mqtt": {"connected": body["mqtt"]["connected"]}}
    assert "host" not in body["mqtt"]
    assert "error" not in body["mqtt"]
    assert "ride" not in body


def test_postgres_epoch_columns_are_bigint():
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateTable

    from app.models import Alert, Guardian, LiveFrame, MediaClip, Notice, Ride

    dialect = postgresql.dialect()
    expected = {
        Alert: "at",
        Ride: "started_at",
        LiveFrame: "at",
        Notice: "at",
        MediaClip: "start_at",
        Guardian: "last_notified_at",
    }
    for model, column in expected.items():
        ddl = str(CreateTable(model.__table__).compile(dialect=dialect))
        line = next(part for part in ddl.splitlines() if part.strip().startswith(column))
        assert "BIGINT" in line.upper()


def test_sqlite_stores_current_epoch_ms(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.models import Alert, Base

    engine = create_engine(f"sqlite:///{tmp_path}/epoch.db")
    Base.metadata.create_all(engine)
    now_ms = int(time.time() * 1000)
    assert now_ms > 2**31 - 1
    Session = sessionmaker(bind=engine)
    with Session() as db:
        db.add(Alert(id="a-now", ride_id="ride-1842", at=now_ms))
        db.commit()
        stored = db.get(Alert, 1)
        assert stored is not None
        assert stored.at == now_ms


def test_seed_demo_off_skips_guardian_phones(tmp_path, monkeypatch):
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import sessionmaker

    from app.config import settings
    from app.db import seed
    from app.models import Base, Guardian, Ride

    monkeypatch.setattr(settings, "seed_demo", False)
    engine = create_engine(f"sqlite:///{tmp_path}/seed.db")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    settings.seed_demo = False
    with Session() as db:
        seed(db)
        db.commit()
        guardians = db.scalars(select(Guardian)).all()
        ride = db.get(Ride, settings.default_ride_id)
    assert guardians == []
    assert ride is not None
    assert ride.passenger == ""
    assert ride.started_at > 2**31 - 1


def test_require_device_key_rejects_dev_default(monkeypatch, tmp_path):
    from dms.config import load_config

    monkeypatch.setenv("DMS_REQUIRE_DEVICE_KEY", "1")
    monkeypatch.delenv("DMS_MEDIA_DEVICE_KEY", raising=False)
    monkeypatch.delenv("MEDIA_DEVICE_KEY", raising=False)
    with pytest.raises(ValueError, match="dev-media-key"):
        load_config(None, {"data_dir": str(tmp_path)})


def test_require_device_key_accepts_a_real_key(monkeypatch, tmp_path):
    from dms.config import load_config

    monkeypatch.setenv("DMS_REQUIRE_DEVICE_KEY", "1")
    monkeypatch.setenv("DMS_MEDIA_DEVICE_KEY", "vehicle-secret")
    cfg = load_config(None, {"data_dir": str(tmp_path)})
    assert cfg.webrtc.device_key == "vehicle-secret"


def _api(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_DEVICE_KEY", "dev-media-key")
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.config import settings

    url = f"sqlite:///{tmp_path}/api.db"
    monkeypatch.setattr(settings, "database_url", url)
    monkeypatch.setattr(settings, "media_device_key", "dev-media-key")
    monkeypatch.setattr(settings, "auth_required", False)
    monkeypatch.setattr(settings, "require_device_key", False)
    monkeypatch.setattr(settings, "seed_demo", False)
    monkeypatch.setattr(settings, "guardian_api_key", "guardian-secret")
    monkeypatch.setattr(settings, "session_secret", "session-secret-at-least-32-bytes")
    from app import db as cloud_db
    from app.models import Base

    engine = create_engine(url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    monkeypatch.setattr(cloud_db, "engine", engine)
    monkeypatch.setattr(cloud_db, "SessionLocal", sessionmaker(bind=engine, expire_on_commit=False))
    from app.main import app

    return app


def test_auth_off_still_reads_ride_and_posts_sos(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    app = _api(tmp_path, monkeypatch)
    with patch("app.mqtt_ingest.ingest.start"):
        with TestClient(app) as client:
            ride = client.get("/v1/rides/ride-1842")
            sos = client.post(
                "/v1/rides/ride-1842/sos",
                json={"passenger": "Meera", "latitude": 12.97, "longitude": 77.59},
            )
    assert ride.status_code == 200
    assert sos.status_code == 200
    assert sos.json()["alert"]["kind"] == "sos"


def test_auth_on_blocks_phone_and_allows_a_session(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.config import settings

    app = _api(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "auth_required", True)
    monkeypatch.setattr(settings, "require_device_key", True)
    monkeypatch.setattr(settings, "media_device_key", "vehicle-secret")
    with patch("app.mqtt_ingest.ingest.start"):
        with TestClient(app) as client:
            assert client.get("/v1/rides/ride-1842").status_code == 401
            assert client.get("/v1/rides/ride-1842/guardians").status_code == 401
            assert client.get("/v1/rides/ride-1842/video").status_code == 401
            assert client.post("/v1/rides/ride-1842/video/talk").status_code == 401
            assert client.post("/v1/rides/ride-1842/guardians/g-priya/sharing").status_code == 401
            assert client.post("/v1/devices/g-priya/push-token", json={"token": "ExponentPushToken[x]"}).status_code == 401
            assert client.post("/v1/devices/orin-nano-01/media-token", json={"key": "dev-media-key"}).status_code == 403
            vehicle = client.post("/v1/devices/orin-nano-01/media-token", json={"key": "vehicle-secret"})
            assert vehicle.status_code == 200
            denied = client.post("/v1/session", json={"key": "nope", "guardianId": "g-priya"})
            assert denied.status_code == 401
            session = client.post("/v1/session", json={"key": "guardian-secret", "guardianId": "g-priya"})
            assert session.status_code == 200
            token = session.json()["token"]
            allowed = client.get("/v1/rides/ride-1842", headers={"Authorization": f"Bearer {token}"})
            assert allowed.status_code == 200
            with pytest.raises(Exception):
                with client.websocket_connect("/v1/rides/ride-1842/live"):
                    pass
            with client.websocket_connect(f"/v1/rides/ride-1842/live?token={token}") as ws:
                ws.close()
