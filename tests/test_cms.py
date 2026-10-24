"""Cabin seat assignment, alert holds, VLM routing, and the CMS MQTT topic."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from dms import alerts as A
from dms.cms import CabinMonitor, CabinRules, build_cabin, point_in_polygon
from dms.config import load_config
from dms.state import CabinState, SeatState
from dms.vlm_worker import assess


def _seats():
    return load_config(None)["cms"]["seats"]


def test_point_lands_in_the_configured_seat():
    seats = _seats()
    assert point_in_polygon(0.10, 0.20, seats[1]["polygon"])  # rear_left
    cabin = build_cabin([(80, 60, 140, 120)], [], 640, 480, seats, 0.18)
    assert cabin.seats[1].name == "rear_left"
    assert cabin.seats[1].occupant == "child"
    assert cabin.seats[1].belt == "Unknown"
    assert cabin.child_present is True
    assert cabin.adult_present is False
    assert all(s.belt == "Unknown" for s in cabin.seats)


def test_adult_in_cabin_blocks_child_present():
    seats = _seats()
    child = (80, 60, 140, 120)          # rear left, short face
    adult = (80, 300, 220, 460)         # front band, tall face
    cabin = build_cabin([child, adult], [], 640, 480, seats, 0.18)
    assert cabin.adult_present is True
    assert cabin.child_present is False
    assert cabin.occupants == 2
    front = next(s for s in cabin.seats if s.name == "front_passenger")
    assert front.occupant == "adult"


def test_object_on_rear_right_and_holds():
    seats = _seats()
    # Center of the box sits in rear_right (x > 0.67, y < 0.55).
    bag = ((460, 40, 560, 160), 0.8, "backpack")
    cabin = build_cabin([], [bag], 640, 480, seats, 0.18)
    obj = cabin.objects[0]
    assert obj.seat == "rear_right"
    rules = CabinRules(child_s=3.0, object_s=2.0, occupancy_s=1.0)
    assert rules.step(cabin, 0.0) == []
    fired = rules.step(cabin, 2.0)
    assert fired[0][0] == A.OBJECT_ON_SEAT
    assert fired[0][1] == "rear_right:backpack"
    assert rules.step(cabin, 3.0) == []


def test_child_alert_is_rising_edge_and_not_left_behind():
    seats = _seats()
    cabin = build_cabin([(80, 60, 140, 120)], [], 640, 480, seats, 0.18)
    rules = CabinRules(3.0, 2.0, 1.0)
    assert rules.step(cabin, 0.0) == []
    fired = rules.step(cabin, 3.0)
    assert len(fired) == 1
    assert fired[0][0] == A.CHILD_PRESENT
    assert "left behind" not in fired[0][2].lower()
    assert rules.step(cabin, 4.0) == []


def test_occupancy_change_after_a_baseline():
    seats = [
        SeatState("rear_left", "empty"),
        SeatState("rear_right", "empty"),
    ]
    rules = CabinRules(3.0, 2.0, 1.0)
    empty = CabinState(seats=seats)
    assert rules.step(empty, 0.0) == []
    moved = CabinState(seats=[SeatState("rear_left", "child"), SeatState("rear_right", "empty")],
                       child_present=True)
    assert rules.step(moved, 0.2) == []
    kinds = [item[0] for item in rules.step(moved, 1.3)]
    assert kinds == [A.OCCUPANCY_CHANGE]


def test_vlm_cabin_categories_are_not_a_driver_alert():
    _, _, routed = assess("A passenger is leaning into the front seat and grabbing the driver.")
    assert routed == [(A.PASSENGER_AGGRESSIVE, "passenger_aggressive", "cms")]
    _, _, routed = assess("A bag on the dashboard is blocking the driver's view.")
    assert routed == [(A.OBJECT_ON_SEAT, "vlm_blocking", "cms")]
    _, decision, routed = assess("UNSAFE: the driver is holding a phone and a toddler is alone, unattended.")
    kinds = {item[0] for item in routed}
    assert A.VLM_ALERT not in kinds and A.CHILD_PRESENT in kinds
    assert any("detector owns" in item for item in decision.suppressed)
    assert assess("SAFE: the driver is attentive.")[2] == []


def test_cms_alerts_use_the_cms_topic(tmp_path):
    cfg = load_config(None, {"device_id": "cab", "data_dir": str(tmp_path), "mqtt": {"heartbeat_s": 60}})
    shared = __import__("dms.state", fromlist=["SharedState"]).SharedState()
    bus = A.AlertBus("cab", cooldown_s=0)
    from dms.mqtt_publisher import MqttPublisher

    pub = MqttPublisher(cfg, bus, shared)
    try:
        bus.emit(A.Alert(type=A.CHILD_PRESENT, category=A.VISION, channel="cms", confidence=0.6,
                         reason="child_no_adult", description="Child present in rear left, no adult visible",
                         extra={"seats": [{"name": "rear_left", "occupant": "child", "belt": "Unknown"}]}))
        bus.emit(A.Alert(type=A.DISTRACTION, category=A.VISION, confidence=0.9, reason="gaze"))
        pub._drain_bus()
        rows = pub.outbox.due(10, 0, set())
        topics = {topic: json.loads(payload) for _, topic, payload, _ in rows}
        assert f"cms/cab/alert" in topics
        assert topics["cms/cab/alert"]["channel"] == "cms"
        assert topics["cms/cab/alert"]["type"] == "CHILD_PRESENT"
        assert "seats" in topics["cms/cab/alert"]["details"]
        assert topics["dms/cab/alert/vision"]["type"] == "DISTRACTION"
        assert topics["dms/cab/alert/vision"]["channel"] == "dms"
    finally:
        pub.outbox.close()


def test_monitor_process_on_a_blank_frame(tmp_path):
    cfg = load_config(None, {
        "device_id": "cab",
        "data_dir": str(tmp_path),
        "realtime": {"phone_detector": ""},
    })
    from dms.state import SharedState

    shared = SharedState()
    bus = A.AlertBus("cab", cooldown_s=0)

    class _Cam:
        pass

    monitor = CabinMonitor(cfg, _Cam(), shared, bus)
    assert monitor.objects is None
    image = np.zeros((480, 640, 3), np.uint8)
    state = monitor.process(image, 1.0, 15.0, connected=True)
    assert state.connected is True
    assert state.occupants == 0
    assert shared.passenger is state
    assert all(s.belt == "Unknown" for s in state.seats)


def test_dashboard_modes_render(tmp_path):
    """Qt and TensorRT in one process segfault on this machine, so render out of process."""
    import subprocess
    import sys

    out = tmp_path / "frames"
    out.mkdir()
    code = f"""
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import numpy as np
from PySide6.QtWidgets import QApplication
QApplication([])
from dms import alerts as A
from dms.cms import build_cabin
from dms.config import load_config
from dms.state import SharedState
from dms.ui.dashboard import render_png

shared = SharedState()
frame = np.zeros((480, 640, 3), np.uint8)
frame[:] = (30, 40, 50)
shared.driver.frame = frame
shared.driver.seq = 1
shared.driver.width = 640
shared.driver.height = 480
shared.driver.fps = 20
seats = load_config(None)["cms"]["seats"]
cabin = build_cabin([(80, 60, 140, 120)], [((460, 40, 560, 160), 0.7, "backpack")], 640, 480, seats, 0.18)
cabin.frame = frame
cabin.frame_ts = 1.0
cabin.connected = True
cabin.fps = 15
shared.passenger = cabin
bus = A.AlertBus("ui", cooldown_s=0)
bus.emit(A.Alert(type=A.CHILD_PRESENT, category=A.VISION, channel="cms", confidence=0.6,
                 reason="child_no_adult", description="Child present in rear left, no adult visible"))
for mode in ("dms", "cms", "split"):
    render_png(shared, bus, {str(out)!r} + "/" + mode + ".png", width=1280, height=720, mode=mode)

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from dms.ui.dashboard import Dashboard
window = Dashboard(shared, bus, fps=1)
window.timer.stop()
window.resize(1600, 893)
scale, ox, oy = window._view_transform()
for mode in ("cms", "split", "dms"):
    rect = next(r for r, m, _t in window._mode_rects() if m == mode)
    center = rect.center()
    local = QPointF(ox + center.x() * scale, oy + center.y() * scale)
    event = QMouseEvent(QEvent.Type.MouseButtonPress, local, local, Qt.MouseButton.LeftButton,
                        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    window.mousePressEvent(event)
    assert window.mode == mode, window.mode
"""
    proc = subprocess.run([sys.executable, "-c", code], cwd=os.getcwd(), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    for mode in ("dms", "cms", "split"):
        path = out / f"{mode}.png"
        assert path.stat().st_size > 10_000
