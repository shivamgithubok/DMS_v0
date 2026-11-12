import sys
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
sys.path.insert(0, str(ROOT))

from app.models import Base, Guardian, Ride  # noqa: E402
from app.store import alert_row  # noqa: E402
from dms.webrtc_publisher import dummy_frame, jpeg_b64  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402


def test_history_live_frames_and_alert_clip(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_DIR", str(tmp_path / "media"))
    monkeypatch.setenv("MEDIA_DEVICE_KEY", "dev-media-key")
    from app.config import settings

    settings.database_url = f"sqlite:///{tmp_path}/vms.db"
    settings.media_dir = str(tmp_path / "media")
    settings.media_device_key = "dev-media-key"
    from app import db as cloud_db

    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    cloud_db.engine = engine
    cloud_db.SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    with cloud_db.SessionLocal() as db:
        db.add(Ride(id="ride-1842", device_id="orin-nano-01", status="active", started_at=2000, analysis_json="{}"))
        db.add(Ride(id="ride-old", device_id="orin-nano-01", status="ended", started_at=1000, ended_at=1500, analysis_json="{}"))
        db.add(Guardian(id="g-priya", ride_id="ride-1842", name="Priya", sharing=True))
        db.add(Guardian(id="g-old", ride_id="ride-old", name="Priya", sharing=True))
        ride = db.get(Ride, "ride-1842")
        alert_row(db, ride, {"id": "alert-1", "type": "DISTRACTION", "channel": "dms", "epoch_ms": 2500})
        db.commit()

    from app.db import get_db
    from app.main import app
    from fastapi.testclient import TestClient

    def _override():
        session = cloud_db.SessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _override
    driver = jpeg_b64(dummy_frame("driver", seq=1))
    cabin = jpeg_b64(dummy_frame("cabin", seq=2))
    with patch("app.mqtt_ingest.ingest.start"):
        with TestClient(app) as client:
            hist = client.get("/v1/history", params={"deviceId": "orin-nano-01"})
            assert hist.status_code == 200
            ids = [row["id"] for row in hist.json()["rides"]]
            assert ids == ["ride-1842", "ride-old"]
            assert hist.json()["rides"][0]["alertCount"] == 1
            assert hist.json()["rides"][1]["status"] == "ended"

            posted = client.post(
                "/v1/devices/orin-nano-01/video/frame",
                json={"key": "dev-media-key", "track": "driver", "at": 3000, "image_b64": driver},
            )
            assert posted.status_code == 200
            assert posted.json()["channel"] == "dms"
            client.post(
                "/v1/devices/orin-nano-01/video/frame",
                json={"key": "dev-media-key", "track": "cabin", "at": 3001, "image_b64": cabin},
            )
            live = client.get("/v1/rides/ride-1842/frames/live", params={"channel": "dms"})
            assert live.status_code == 200
            assert live.json()["at"] == 3000
            image = client.get("/v1/rides/ride-1842/frames/live/image", params={"channel": "cms"})
            assert image.status_code == 200 and image.content[:2] == b"\xff\xd8"

            clip = client.post(
                "/v1/devices/orin-nano-01/clips",
                json={
                    "key": "dev-media-key",
                    "channel": "dms",
                    "alert_id": "alert-1",
                    "frames": [
                        {"at": 2400, "image_b64": driver},
                        {"at": 2600, "image_b64": cabin},
                    ],
                },
            )
            assert clip.status_code == 200
            body = clip.json()
            assert body["channel"] == "dms"
            assert body["startAt"] == 2400 and body["endAt"] == 2600
            assert body["alertId"] == "alert-1"
            still = client.get(body["frames"][0]["url"])
            assert still.status_code == 200 and still.content[:2] == b"\xff\xd8"
            alerts = client.get("/v1/rides/ride-1842/alerts", params={"channel": "dms"})
            assert alerts.json()["alerts"][0]["clipId"] == body["id"]

            with cloud_db.SessionLocal() as db:
                row = db.get(Guardian, "g-priya")
                row.sharing = False
                db.commit()
            assert client.get("/v1/rides/ride-1842/frames/live", params={"channel": "dms"}).status_code == 403
            assert client.get(f"/v1/rides/ride-1842/clips/{body['id']}").status_code == 403
    app.dependency_overrides.clear()
