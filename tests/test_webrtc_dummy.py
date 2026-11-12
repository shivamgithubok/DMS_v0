import base64
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("jwt")
pytest.importorskip("fastapi")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
sys.path.insert(0, str(ROOT))

from app.models import Base, Guardian, Ride  # noqa: E402
from dms.webrtc_publisher import dummy_frame, jpeg_b64  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/vms.db")
    monkeypatch.setenv("LIVEKIT_URL", "dummy://local")
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "devsecret-change-me-please-32b")
    monkeypatch.setenv("MEDIA_DEVICE_KEY", "dev-media-key")
    from app.config import settings

    settings.database_url = f"sqlite:///{tmp_path}/vms.db"
    settings.livekit_url = "dummy://local"
    settings.livekit_api_key = "devkey"
    settings.livekit_api_secret = "devsecret-change-me-please-32b"
    settings.media_device_key = "dev-media-key"
    from app import db as cloud_db

    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    cloud_db.engine = engine
    cloud_db.SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    with cloud_db.SessionLocal() as db:
        db.add(Ride(id="ride-1842", device_id="orin-nano-01", analysis_json="{}", publisher_at=0))
        db.add(Guardian(id="g-priya", ride_id="ride-1842", name="Priya Shah", sharing=True))
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
    from app import consent

    consent.clear()
    return app, TestClient, cloud_db


def test_dummy_watch_flow_with_synthetic_frames(tmp_path, monkeypatch):
    import jwt
    from app.config import settings

    app, TestClient, cloud_db = _client(tmp_path, monkeypatch)
    with patch("app.mqtt_ingest.ingest.start"):
        with TestClient(app) as client:
            denied = client.post(
                "/v1/devices/orin-nano-01/media-token",
                json={"key": "wrong"},
            )
            assert denied.status_code == 403

            pub = client.post(
                "/v1/devices/orin-nano-01/media-token",
                json={"key": "dev-media-key"},
            )
            assert pub.status_code == 200
            pub_body = pub.json()
            assert pub_body["mode"] == "dummy" and pub_body["room"] == "ride-1842"
            pub_claims = jwt.decode(
                pub_body["token"], settings.livekit_api_secret, algorithms=["HS256"], issuer="devkey"
            )
            assert pub_claims["sub"] == "camera-orin-nano-01"
            assert pub_claims["video"]["canPublish"] is False
            assert pub_claims["video"]["canSubscribe"] is False

            driver = jpeg_b64(dummy_frame("driver", seq=1))
            cabin = jpeg_b64(dummy_frame("cabin", seq=1))
            assert client.post(
                "/v1/devices/orin-nano-01/video/frame",
                json={"key": "dev-media-key", "track": "driver", "image_b64": driver},
            ).json()["ok"] is True
            assert client.post(
                "/v1/devices/orin-nano-01/video/frame",
                json={"key": "dev-media-key", "track": "cabin", "image_b64": cabin},
            ).status_code == 200

            client.post("/v1/rides/ride-1842/guardians/g-priya/capabilities", json={"watch": True})
            assert client.get("/v1/rides/ride-1842/video", params={"guardianId": "g-priya"}).status_code == 403
            assert client.post(
                "/v1/devices/orin-nano-01/consent",
                json={"key": "dev-media-key", "kind": "watch", "decision": "accept"},
            ).status_code == 200
            watch = client.get("/v1/rides/ride-1842/video", params={"guardianId": "g-priya"})
            assert watch.status_code == 200
            body = watch.json()
            assert body["publisherOnline"] is True
            assert body["stream"] == "jpeg"
            assert body["tracks"] == ["driver", "cabin"]
            assert "video/frame" in body["publishTo"]
            view_claims = jwt.decode(
                body["token"], settings.livekit_api_secret, algorithms=["HS256"], issuer="devkey"
            )
            assert view_claims["sub"].startswith("viewer-")
            assert view_claims["video"]["canPublish"] is False
            assert view_claims["video"]["canSubscribe"] is True
            assert view_claims["video"]["room"] == "ride-1842"

            still = client.get("/v1/rides/ride-1842/video/preview", params={"track": "driver"})
            assert still.status_code == 200
            assert still.content[:2] == b"\xff\xd8"
            assert len(still.content) > 100
            cabin_still = client.get("/v1/rides/ride-1842/video/preview", params={"track": "cabin"})
            assert cabin_still.status_code == 200
            assert cabin_still.content != still.content

            with cloud_db.SessionLocal() as db:
                row = db.get(Guardian, "g-priya")
                row.sharing = False
                db.commit()
            assert client.get("/v1/rides/ride-1842/video", params={"guardianId": "g-priya"}).status_code == 403
            assert client.get("/v1/rides/ride-1842/video/preview").status_code == 403
    app.dependency_overrides.clear()


def test_livekit_watch_names_the_cabin_video_track(tmp_path, monkeypatch):
    app, TestClient, _cloud_db = _client(tmp_path, monkeypatch)
    from app.config import settings

    settings.livekit_url = "ws://127.0.0.1:7880"
    with patch("app.mqtt_ingest.ingest.start"):
        with TestClient(app) as client:
            idle = client.post("/v1/devices/orin-nano-01/media-token", json={"key": "dev-media-key"})
            assert idle.status_code == 200
            assert idle.json()["publish"] is False
            assert idle.json()["stream"] == "jpeg"
            assert idle.json()["tracks"] == ["driver", "cabin"]

            client.post("/v1/rides/ride-1842/guardians/g-priya/capabilities", json={"watch": True})
            assert client.get("/v1/rides/ride-1842/video", params={"guardianId": "g-priya"}).status_code == 403
            assert client.post(
                "/v1/devices/orin-nano-01/consent",
                json={"key": "dev-media-key", "kind": "watch", "decision": "accept"},
            ).status_code == 200
            watch = client.get("/v1/rides/ride-1842/video", params={"guardianId": "g-priya"})
            assert watch.status_code == 200
            body = watch.json()
            assert body["mode"] == "livekit"
            assert body["stream"] == "webrtc"
            assert body["publish"] is True
            assert body["publishTo"] == "livekit"
            assert body["tracks"] == ["cabin"]
            assert body["video"]["camera"] == "driver"
            assert body["video"]["width"] == 640
            assert body["video"]["height"] == 360
            assert body["video"]["fps"] == 12
            assert body["audio"]["track"] == "cabin"

            pub = client.post("/v1/devices/orin-nano-01/media-token", json={"key": "dev-media-key"})
            assert pub.json()["publish"] is True
            assert pub.json()["stream"] == "webrtc"
            assert pub.json()["tracks"] == ["cabin"]
            assert pub.json()["publishTo"] == "livekit"
    app.dependency_overrides.clear()


def test_dummy_jpeg_is_valid():
    raw = base64.b64decode(jpeg_b64(dummy_frame("driver", seq=9)))
    assert raw[:2] == b"\xff\xd8"
    assert dummy_frame("driver").shape[0] == 180
