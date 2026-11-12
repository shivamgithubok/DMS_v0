"""Local cabin stream: stills keep flowing, video+audio start only after a guardian watches."""

import base64
import socket
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
sys.path.insert(0, str(ROOT))

from app.models import Base, Guardian, Ride  # noqa: E402
from dms.config import Config  # noqa: E402
from dms.state import SharedState  # noqa: E402
from dms.webrtc_publisher import WebRtcPublisher, stitch_cabin  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402


class _Grab:
    def __init__(self, image: np.ndarray) -> None:
        self.image = image

    def latest(self):
        return self


class _Ring:
    def __init__(self, value: float) -> None:
        self.total = 1600
        self._samples = np.full(1600, value, dtype=np.float32)

    def latest(self, n: int):
        return self._samples[:n].copy(), 0.0


class _Audio:
    def __init__(self) -> None:
        self.ring16 = _Ring(0.5)
        self.ring = self.ring16


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/vms.db")
    monkeypatch.setenv("LIVEKIT_URL", "local://cabin")
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "devsecret-change-me-please-32b")
    monkeypatch.setenv("MEDIA_DEVICE_KEY", "dev-media-key")
    from app.config import settings

    settings.database_url = f"sqlite:///{tmp_path}/vms.db"
    settings.livekit_url = "local://cabin"
    settings.livekit_api_key = "devkey"
    settings.livekit_api_secret = "devsecret-change-me-please-32b"
    settings.media_device_key = "dev-media-key"
    from app import db as cloud_db
    from app import watch

    watch.clear()
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

    def _override():
        session = cloud_db.SessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _override
    from app import consent

    consent.clear()
    return app, watch


def test_stitch_puts_driver_on_the_left():
    driver = np.zeros((100, 80, 3), dtype=np.uint8)
    driver[:] = (0, 0, 220)
    passenger = np.zeros((100, 80, 3), dtype=np.uint8)
    passenger[:] = (0, 200, 0)
    frame = stitch_cabin(driver, passenger)
    assert frame.shape == (360, 640, 3)
    assert frame[:, :320, 2].mean() > 150
    assert frame[:, 320:, 1].mean() > 150


def test_local_stream_starts_when_guardian_watches(tmp_path, monkeypatch):
    import uvicorn
    from fastapi.testclient import TestClient

    app, watch = _app(tmp_path, monkeypatch)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    with patch("app.mqtt_ingest.ingest.start"):
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.time() + 5
        while not server.started and time.time() < deadline:
            time.sleep(0.02)
        assert server.started

        client = TestClient(app)
        idle = client.post("/v1/devices/orin-nano-01/media-token", json={"key": "dev-media-key"})
        assert idle.status_code == 200
        assert idle.json()["mode"] == "local"
        assert idle.json()["publish"] is False

        driver = np.zeros((480, 640, 3), dtype=np.uint8)
        driver[:] = (0, 0, 220)
        passenger = np.zeros((480, 640, 3), dtype=np.uint8)
        passenger[:] = (0, 200, 0)

        class _Capture:
            pass

        capture = _Capture()
        capture.driver = _Grab(driver)
        capture.passenger = _Grab(passenger)
        cfg = Config(
            {
                "device_id": "orin-nano-01",
                "webrtc": {
                    "enabled": True,
                    "dummy": False,
                    "api_url": f"http://127.0.0.1:{port}",
                    "device_key": "dev-media-key",
                    "interval_s": 0.1,
                    "timeout_s": 2.0,
                },
            }
        )
        publisher = WebRtcPublisher(cfg, capture, SharedState(), audio=_Audio())
        publisher.start()
        try:
            assert client.get("/v1/rides/ride-1842/video/preview", params={"track": "driver"}).status_code == 403
            assert client.get("/v1/rides/ride-1842/video/live").status_code == 403
            client.post("/v1/rides/ride-1842/guardians/g-priya/capabilities", json={"watch": True})
            assert client.get("/v1/rides/ride-1842/video", params={"guardianId": "g-priya"}).status_code == 403
            assert client.post(
                "/v1/devices/orin-nano-01/consent",
                json={"key": "dev-media-key", "kind": "watch", "decision": "accept"},
            ).status_code == 200
            still = _wait_status(lambda: client.get("/v1/rides/ride-1842/video/preview", params={"track": "driver"}))
            assert still.status_code == 200
            watch_body = client.get("/v1/rides/ride-1842/video", params={"guardianId": "g-priya"}).json()
            assert watch_body["publish"] is True
            assert watch_body["mode"] == "local"

            live_resp = _wait_status(lambda: client.get("/v1/rides/ride-1842/video/live"))
            live = live_resp.json()
            raw = base64.b64decode(live["image_b64"])
            frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            assert watch_body["stream"] == "jpeg"
            assert frame is not None and frame.shape == (360, 640, 3)
            assert frame[:, :, 2].mean() > 100
            assert frame[:, :, 1].mean() < 40
            pcm = np.frombuffer(base64.b64decode(live["audio_b64"]), dtype=np.int16)
            assert live["samples"] == pcm.size
            assert live["rate"] == 16000
            assert abs(int(pcm.mean())) > 1000
            cancelled = client.post("/v1/rides/ride-1842/video/cancel")
            assert cancelled.status_code == 200
            assert cancelled.json()["publish"] is False
            stopped = client.post("/v1/devices/orin-nano-01/media-token", json={"key": "dev-media-key"})
            assert stopped.json()["publish"] is False
            deadline = time.time() + 3
            while time.time() < deadline and publisher.shared.webrtc.publishing:
                time.sleep(0.05)
            assert publisher.shared.webrtc.publishing is False
            assert client.get("/v1/rides/ride-1842/video/live").status_code == 403
        finally:
            publisher.stop()
            publisher.join(timeout=2)
            server.should_exit = True
            thread.join(timeout=2)
    app.dependency_overrides.clear()


def _wait_status(call, ok: int = 200, timeout: float = 4.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = call()
        if last.status_code == ok:
            return last
        time.sleep(0.05)
    raise AssertionError(
        f"timed out waiting for HTTP {ok}, last={getattr(last, 'status_code', None)} {getattr(last, 'text', '')[:240]}"
    )
