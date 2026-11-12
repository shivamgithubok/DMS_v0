"""Guardian record / stop / snap, separate from alert clips."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
sys.path.insert(0, str(ROOT))


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/vms.db")
    monkeypatch.setenv("MEDIA_DIR", str(tmp_path / "media"))
    monkeypatch.setenv("LIVEKIT_URL", "dummy://local")
    monkeypatch.setenv("MEDIA_DEVICE_KEY", "dev-media-key")
    from app.config import settings

    settings.database_url = f"sqlite:///{tmp_path}/vms.db"
    settings.media_dir = str(tmp_path / "media")
    settings.livekit_url = "dummy://local"
    settings.media_device_key = "dev-media-key"
    from app import db as cloud_db
    from app.models import Base, Guardian, Ride

    engine = __import__("sqlalchemy").create_engine(settings.database_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = __import__("sqlalchemy.orm", fromlist=["sessionmaker"]).sessionmaker(bind=engine, expire_on_commit=False)
    cloud_db.engine = engine
    cloud_db.SessionLocal = Session
    with Session() as db:
        db.add(Ride(id="ride-1842", device_id="orin-nano-01", analysis_json="{}", publisher_at=0))
        db.add(Guardian(id="g-priya", ride_id="ride-1842", name="Priya Shah", sharing=True))
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
    return app


def test_guardian_record_snap_and_list(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    app = _client(tmp_path, monkeypatch)
    jpeg = b"\xff\xd8\xff" + b"x" * 40
    video = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 32
    with patch("app.mqtt_ingest.ingest.start"):
        with TestClient(app) as client:
            from app.clips import save_live
            from app.db import SessionLocal
            from app.models import Ride

            with SessionLocal() as db:
                save_live(db, db.get(Ride, "ride-1842"), "dms", jpeg, 1761560000000)
                db.commit()

            started = client.post("/v1/rides/ride-1842/recordings", json={"action": "start", "guardianId": "g-priya"})
            assert started.status_code == 200
            rec_id = started.json()["id"]
            assert started.json()["status"] == "recording"
            assert started.json()["audio"] is True
            assert started.json()["videoUrl"] is None

            token = client.post("/v1/devices/orin-nano-01/media-token", json={"key": "dev-media-key"})
            assert token.json()["recording"]["id"] == rec_id

            stopped = client.post(f"/v1/rides/ride-1842/recordings/{rec_id}/cancel")
            assert stopped.json()["status"] == "stopping"
            assert client.post("/v1/devices/orin-nano-01/media-token", json={"key": "dev-media-key"}).json()["recording"] is None

            uploaded = client.post(
                f"/v1/devices/orin-nano-01/recordings/{rec_id}",
                json={"key": "dev-media-key", "video_b64": __import__("base64").b64encode(video).decode()},
            )
            assert uploaded.status_code == 200
            assert uploaded.json()["status"] == "ready"
            assert uploaded.json()["videoUrl"].endswith("/clip.mp4")

            listed = client.get("/v1/rides/ride-1842/recordings", params={"guardianId": "g-priya"})
            assert listed.json()["recordings"][0]["id"] == rec_id
            played = client.get(uploaded.json()["videoUrl"])
            assert played.status_code == 200
            assert played.content[:8] == video[:8]

            snap = client.post("/v1/rides/ride-1842/snapshots", json={"guardianId": "g-priya", "channel": "dms"})
            assert snap.status_code == 200
            shots = client.get("/v1/rides/ride-1842/snapshots", params={"guardianId": "g-priya"})
            assert shots.json()["snapshots"][0]["id"] == snap.json()["id"]
            image = client.get(snap.json()["url"])
            assert image.status_code == 200
            assert image.content[:2] == b"\xff\xd8"
    app.dependency_overrides.clear()
