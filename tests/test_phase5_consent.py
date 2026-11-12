"""Phase 5: sharing may ask. Watch and talk start only after the cabin accepts."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
sys.path.insert(0, str(ROOT))


def test_watch_and_talk_wait_for_cabin_accept(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("jwt")
    import jwt
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import sessionmaker

    from app.config import settings
    from app.models import AccessAudit, Base, Guardian, Ride

    monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path}/vms.db")
    monkeypatch.setattr(settings, "livekit_url", "ws://127.0.0.1:7880")
    monkeypatch.setattr(settings, "livekit_api_key", "devkey")
    monkeypatch.setattr(settings, "livekit_api_secret", "devsecret-change-me-please-32b")
    monkeypatch.setattr(settings, "media_device_key", "dev-media-key")
    monkeypatch.setattr(settings, "auth_required", False)
    from app import consent, db as cloud_db
    from app.db import get_db
    from app.main import app

    consent.clear()
    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
    cloud_db.engine = engine
    cloud_db.SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    Base.metadata.create_all(engine)
    with cloud_db.SessionLocal() as db:
        db.add(Ride(id="ride-1842", device_id="orin-nano-01", analysis_json="{}"))
        db.add(Guardian(id="g-priya", ride_id="ride-1842", name="Priya Shah", sharing=True, can_notify=True))
        db.commit()

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
                listed = client.get("/v1/rides/ride-1842/guardians")
                priya = listed.json()["guardians"][0]
                assert priya["sharing"] is True
                assert priya["notify"] is True
                assert priya["watch"] is False
                assert priya["talk"] is False

                client.post("/v1/rides/ride-1842/guardians/g-priya/capabilities", json={"watch": True, "talk": True})
                asked = client.get("/v1/rides/ride-1842/video", params={"guardianId": "g-priya"})
                assert asked.status_code == 403
                quiet = client.post("/v1/devices/orin-nano-01/media-token", json={"key": "dev-media-key"})
                assert quiet.json()["publish"] is False
                claims = jwt.decode(
                    quiet.json()["token"], settings.livekit_api_secret, algorithms=["HS256"], issuer="devkey"
                )
                assert claims["video"]["canPublish"] is False
                assert quiet.json()["watchRequest"]["name"] == "Priya Shah"
                assert quiet.json()["talk"] is False

                accepted = client.post(
                    "/v1/devices/orin-nano-01/consent",
                    json={"key": "dev-media-key", "kind": "watch", "decision": "accept"},
                )
                assert accepted.status_code == 200
                again = client.post(
                    "/v1/devices/orin-nano-01/consent",
                    json={"key": "dev-media-key", "kind": "watch", "decision": "accept"},
                )
                assert again.status_code == 409
                opened = client.get("/v1/rides/ride-1842/video", params={"guardianId": "g-priya"})
                assert opened.status_code == 200
                live = client.post("/v1/devices/orin-nano-01/media-token", json={"key": "dev-media-key"})
                assert live.json()["publish"] is True
                live_claims = jwt.decode(
                    live.json()["token"], settings.livekit_api_secret, algorithms=["HS256"], issuer="devkey"
                )
                assert live_claims["video"]["canPublish"] is True

                waiting = client.post("/v1/rides/ride-1842/video/talk", params={"guardianId": "g-priya"})
                assert waiting.status_code == 403
                client.post(
                    "/v1/devices/orin-nano-01/consent",
                    json={"key": "dev-media-key", "kind": "talk", "decision": "accept"},
                )
                spoken = client.post("/v1/rides/ride-1842/video/talk", params={"guardianId": "g-priya"})
                assert spoken.status_code == 200
                assert spoken.json()["token"]

                with cloud_db.SessionLocal() as db:
                    rows = db.scalars(select(AccessAudit).order_by(AccessAudit.at)).all()
                actions = [(row.kind, row.action, row.actor) for row in rows]
                assert ("watch", "ask", "g-priya") in actions
                assert ("watch", "accept", "cabin") in actions
                assert sum(1 for row in rows if row.kind == "watch" and row.action == "accept") == 1
    finally:
        app.dependency_overrides.clear()
        consent.clear()


def test_cabin_accept_and_deny_are_hit_targets():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from dms.alerts import AlertBus
    from dms.state import SharedState
    from dms.ui.dashboard import Dashboard

    app = QApplication.instance() or QApplication([])
    shared = SharedState()
    dash = Dashboard(shared, AlertBus("orin-nano-01"), cfg=None, fps=5)
    dash.show()
    dash.activateWindow()
    dash.setFocus()
    app.processEvents()
    assert dash.timer.isActive()
    shared.consent.watch_request = "Priya"
    accept = dash._consent_rects()[0][0].center()
    assert dash.hit(accept.x(), accept.y()) == "watch:accept"
    assert shared.consent.live is True
    assert shared.consent.watch_request == ""
    shared.consent.talk_request = "Kabir"
    deny = dash._consent_rects()[1][0].center()
    assert dash.hit(deny.x(), deny.y()) == "talk:deny"
    assert shared.consent.talking is False
    QTest.keyClick(dash, Qt.Key.Key_2)
    app.processEvents()
    if dash.mode != "cms":
        dash._shortcuts[1].activated.emit()
    assert dash.mode == "cms"
    dash._shortcuts[2].activated.emit()
    assert dash.mode == "split"
    dash._shortcuts[0].activated.emit()
    assert dash.mode == "dms"
    dash.close()
    app.processEvents()
