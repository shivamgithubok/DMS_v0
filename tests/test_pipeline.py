"""End-to-end driver channel on the reference frame (phone in hand, eyes off road)."""

import time
from pathlib import Path

import pytest

from dms import alerts as A
from dms.capture import Camera
from dms.config import ROOT, load_config
from dms.realtime_detector import DriverMonitor
from dms.state import SharedState

FRAME = ROOT / "assets" / "demo" / "driver_reference.jpg"


@pytest.fixture(scope="module")
def run():
    cfg = load_config(None, {"realtime": {"snapshot_alerts": False}})
    cam = Camera("driver", str(FRAME), fps=30).start()
    shared, bus = SharedState(), A.AlertBus("test")
    mon = DriverMonitor(cfg, cam, shared, bus)
    mon.start()
    time.sleep(7)
    mon.stop()
    cam.stop()
    return shared.driver, bus


def test_face_landmarks_and_identity(run):
    ds, _ = run
    assert ds.face_found and ds.landmarks is not None
    assert ds.identity.known and ds.identity.driver_id == "4827"


def test_phone_and_gaze(run):
    ds, _ = run
    assert ds.phone and ds.phone_box is not None
    assert ds.gaze_away and ds.gaze == "Left"


def test_seatbelt_visible(run):
    assert run[0].seatbelt == "On"


def test_distraction_alerts(run):
    _, bus = run
    reasons = {a.reason for a in bus.history if a.type == A.DISTRACTION}
    assert {"phone_usage", "gaze_away"} <= reasons
    payload = bus.history[0].payload()
    assert payload["type"] == "DISTRACTION" and "snapshot_path" in payload and payload["category"] == "vision"


def test_realtime_rate(run):
    assert run[0].fps >= 15.0, f"{run[0].fps:.1f} FPS"
