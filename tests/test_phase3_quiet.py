"""Phase 3: the family is paged for distress. Coaching stays on the vehicle."""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
sys.path.insert(0, str(ROOT))


def test_phone_use_is_stored_and_not_paged(tmp_path, monkeypatch):
    pytest.importorskip("sqlalchemy")
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.models import Base, Ride, Trip
    from app.push import should_notify
    from app.store import alert_public, alert_row

    engine = create_engine(f"sqlite:///{tmp_path}/quiet.db")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as db:
        ride = Ride(id="ride-1842", device_id="orin-nano-01", lat=12.9, lon=77.6)
        db.add(ride)
        db.add(Trip(
            ride_id="ride-1842", started_at=1000, status="active", device_id="orin-nano-01",
            public_id="trip-quiet",
        ))
        db.flush()
        phone = alert_row(db, ride, {
            "id": "phone-1", "type": "PHONE_USAGE", "severity": "high",
            "description": "Phone usage while driving", "epoch_ms": 1500,
        })
        scream = alert_row(db, ride, {
            "id": "scream-1", "type": "SCREAM", "severity": "high",
            "description": "Screaming detected in cabin", "epoch_ms": 1600,
        })
        db.commit()
        shown = [alert_public(phone), alert_public(scream)]
        assert [row["id"] for row in shown] == ["phone-1", "scream-1"]
        assert shown[0]["audience"] == "cabin"
        assert shown[1]["audience"] == "family"
        assert should_notify(phone) is False
        assert should_notify(scream) is True

        from app import push

        sent = []

        class _Now(threading.Thread):
            def start(self):
                self.run()

        monkeypatch.setattr(push.threading, "Thread", _Now)
        monkeypatch.setattr(push, "tokens_for", lambda _db, _ride_id: ["ExponentPushToken[abc]"])
        monkeypatch.setattr(push, "deliver", lambda targets, body: sent.append(body))
        assert push.notify(db, ride.id, scream) == 1
        assert sent[0]["tripId"] == "trip-quiet"
        assert sent[0]["rideId"] == "ride-1842"


def test_keyword_stays_in_the_cabin():
    from dms.alerts import ABUSIVE_LANGUAGE, AUDIO, Alert

    body = Alert(type=ABUSIVE_LANGUAGE, category=AUDIO, confidence=0.8, description="keyword").payload()
    assert body["audience"] == "cabin"


def test_covered_frame_is_camera_blocked_not_driver_absent():
    from dms.alerts import CAMERA_BLOCKED, DRIVER_ABSENT, AlertBus
    from dms.realtime_detector import DriverMonitor, Hold
    from dms.state import DriverState

    def detector():
        det = DriverMonitor.__new__(DriverMonitor)
        det.rt = SimpleNamespace(yaw_threshold_deg=30.0, alert_cooldown_s=15.0)
        det.holds = {
            name: Hold(999) for name in (
                "eyes_closed", "head_turned", "gaze_away", "looking_down", "phone",
                "drowsy", "smoking", "drinking", "seatbelt_off",
            )
        }
        det.holds["no_face"] = Hold(5)
        det.holds["camera_blocked"] = Hold(5)
        det.trace = SimpleNamespace(change=lambda *a, **k: None, rise=lambda *a, **k: None)
        det.seatbelt = None
        det.camera = SimpleNamespace(stats=SimpleNamespace(connected=True))
        det.bus = AlertBus("orin-nano-01", cooldown_s=0)
        det.snapshots = None
        det.eye_closure = []
        det.model = None
        det._last_preds = None
        det._phone_why = det._drink_why = det._smoke_why = ""
        det._drink_conf = det._smoke_conf = 0.0
        return det

    covered = detector()
    black = np.zeros((48, 64, 3), dtype=np.uint8)
    driver = DriverState(face_found=False, seatbelt="Unknown")
    covered._rules(driver, black, 0.0)
    assert covered.bus.history == []
    covered._rules(driver, black, 5.0)
    assert [alert.type for alert in covered.bus.history] == [CAMERA_BLOCKED]
    assert DRIVER_ABSENT not in covered.bus.counts

    empty = detector()
    scene = np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    absent = DriverState(face_found=False, seatbelt="Unknown")
    empty._rules(absent, scene, 0.0)
    empty._rules(absent, scene, 5.0)
    assert [alert.type for alert in empty.bus.history] == [DRIVER_ABSENT]


def test_vlm_still_cannot_raise_a_driver_alert():
    from dms.vlm_worker import assess

    text = (
        '{"dms": {"scene": "looking at a phone", "hazards": ["driver_distracted"], "confidence": 0.95},'
        ' "cms": {"scene": "quiet cabin", "hazards": [], "confidence": 0.95}}'
    )
    _verdict, decision, routed = assess(text)
    assert "driver_distracted" not in decision.kept
    assert routed == []
