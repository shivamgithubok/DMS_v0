"""Phase 2: each start is its own trip, and a second clip waits instead of being dropped."""

import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
sys.path.insert(0, str(ROOT))


def test_two_starts_are_two_trips(tmp_path):
    from dms.history import HistoryStore

    store = HistoryStore(tmp_path / "history.sqlite3", "ride-1842", "orin-nano-01")
    store.open_session(1_000)
    store.open_session(2_000)
    rows = store.db.execute(
        "SELECT started_at, status, public_id FROM trips ORDER BY started_at"
    ).fetchall()
    assert [row["started_at"] for row in rows] == [1_000, 2_000]
    assert rows[0]["status"] == "ended"
    assert rows[1]["status"] == "active"
    assert rows[0]["public_id"].startswith("trip-")
    assert rows[1]["public_id"].startswith("trip-")
    assert rows[0]["public_id"] != rows[1]["public_id"]
    published = [item["id"] for item in store.snapshot()["trips"]]
    assert published == [rows[1]["public_id"], rows[0]["public_id"]]
    store.close()


def test_second_alert_is_queued_while_the_first_upload_is_blocked(tmp_path):
    from dms.alerts import Alert, AlertBus
    from dms.clip_recorder import ClipRecorder
    from dms.config import Config

    entered = threading.Event()
    release = threading.Event()
    seen: list[str] = []

    def slow_post(url, body, timeout):
        seen.append(body["alert_id"])
        if len(seen) == 1:
            entered.set()
            assert release.wait(5)
        return {"id": "clip-" + body["alert_id"]}

    class _Grab:
        def latest(self):
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            image[:] = (30, 40, 200)
            return type("Frame", (), {"image": image})()

    capture = type("Cap", (), {"driver": _Grab(), "passenger": _Grab()})()
    cfg = Config(
        {
            "device_id": "orin-nano-01",
            "webrtc": {"api_url": "http://127.0.0.1:9", "device_key": "dev-media-key", "timeout_s": 2.0},
        }
    )
    bus = AlertBus("orin-nano-01", cooldown_s=0)
    recorder = ClipRecorder(
        cfg, capture, bus, before_ms=4000, after_ms=0, sample_s=10,
        queue_path=tmp_path / "clips.sqlite3",
    )
    recorder._sample()
    first = Alert(type="DROWSINESS", category="vision", confidence=0.9, channel="dms", description="drowsy")
    second = Alert(type="SCREAM", category="audio", confidence=0.9, channel="dms", description="scream")
    with patch("dms.clip_recorder.post_json", slow_post), patch(
        "dms.clip_recorder.encode_clip_mp4", return_value=b"\x00\x00\x00\x18ftyp"
    ):
        bus.emit(first)
        assert entered.wait(5), "first clip never reached the upload"
        assert recorder.queue.count() >= 1
        bus.emit(second)
        assert recorder.queue.count() >= 2
        release.set()
        deadline = time.time() + 5
        while time.time() < deadline and recorder.uploaded < 2:
            time.sleep(0.02)
    assert recorder.uploaded == 2
    assert seen == [first.id, second.id]
    assert recorder.queue.count() == 0
    recorder.stop()


def test_invalid_config_fails_before_cameras(tmp_path):
    from dms.config import load_config

    bad = tmp_path / "bad.yaml"
    bad.write_text("mqtt:\n  port: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mqtt.port"):
        load_config(bad)
    loaded = load_config(None, overrides={"fleet_label": "kept"})
    assert loaded["mqtt"]["port"] == 1883
    assert loaded["fleet_label"] == "kept"


def test_outbox_drops_status_and_coaching_before_distress(tmp_path):
    from dms.mqtt_publisher import Outbox

    box = Outbox(tmp_path / "outbox.sqlite3", max_rows=1)
    box.put("dms/dev/alert/vision", json.dumps({"type": "DROWSINESS"}), 1)
    box.put("dms/dev/alert/vision", json.dumps({"type": "PHONE_USAGE"}), 1)
    box.put("dms/dev/status", json.dumps({"state": "online"}), 1)
    left = [json.loads(row[0]) for row in box.db.execute("SELECT payload FROM outbox ORDER BY id")]
    assert left == [{"type": "DROWSINESS"}]
    box.db.close()


def test_phone_opens_the_alias_and_the_trip_id(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlalchemy")
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from dms.history import HistoryStore

    store = HistoryStore(tmp_path / "history.sqlite3", "ride-1842", "orin-nano-01")
    store.open_session(1_000)
    store.open_session(5_000)
    snapshot = store.snapshot()
    store.close()

    from app.config import settings
    from app import db as cloud_db
    from app.db import get_db
    from app.main import app
    from app.models import Base
    from app.store import apply_trips
    from fastapi.testclient import TestClient

    monkeypatch.setattr(settings, "auth_required", False)
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path}/vms.db")
    previous_engine = cloud_db.engine
    previous_session = cloud_db.SessionLocal
    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
    cloud_db.engine = engine
    cloud_db.SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    Base.metadata.create_all(engine)

    def _override():
        session = cloud_db.SessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _override
    try:
        with patch("app.mqtt_ingest.ingest.start"):
            with TestClient(app) as client:
                with cloud_db.SessionLocal() as db:
                    apply_trips(db, "dms/orin-nano-01/trips", snapshot)
                    db.commit()
                history = client.get("/v1/history", params={"deviceId": "orin-nano-01"})
                assert history.status_code == 200
                rows = history.json()["rides"]
                assert len(rows) == 2
                assert all(row["id"].startswith("trip-") for row in rows)
                assert rows[0]["rideId"] == "ride-1842"
                alias = client.get("/v1/rides/ride-1842")
                assert alias.status_code == 200
                assert alias.json()["id"] == "ride-1842"
                trip_id = rows[0]["id"]
                opened = client.get(f"/v1/rides/{trip_id}")
                assert opened.status_code == 200
                body = opened.json()
                assert body["id"] == "ride-1842"
                assert body["tripId"] == trip_id
                assert body["startedAt"] == rows[0]["startedAt"]
    finally:
        app.dependency_overrides.clear()
        cloud_db.engine = previous_engine
        cloud_db.SessionLocal = previous_session
