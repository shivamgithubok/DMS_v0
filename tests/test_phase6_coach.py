"""Phase 6: one coaching line, and a known driver's eye baseline survives a restart."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
sys.path.insert(0, str(ROOT))


def test_known_driver_keeps_his_threshold_across_a_restart(tmp_path):
    from dms.driver_id import DriverRoster
    from dms.realtime_detector import calibrated_ear

    path = tmp_path / "roster.json"
    path.write_text(json.dumps([
        {"driver_id": "4827", "name": "Asha", "embedding": [0.0] * 128, "ear_baseline": 0.25},
    ]))
    first = DriverRoster(path)
    assert first.baseline_for("4827") == 0.25
    reloaded = DriverRoster(path)
    threshold, _open, saved = calibrated_ear([], 0.21, True, reloaded.baseline_for("4827"), True)
    assert threshold == pytest.approx(0.18)
    assert saved == pytest.approx(0.25)

    unknown, _open, saved_unknown = calibrated_ear([0.4] * 80, 0.21, True, 0.25, False)
    assert unknown == pytest.approx(0.21)
    assert saved_unknown is None

    reloaded.remember_baseline("4827", 0.30)
    assert DriverRoster(path).baseline_for("4827") == 0.3


def test_phone_use_still_chimes_and_does_not_page(tmp_path, monkeypatch):
    monkeypatch.setattr("dms.local_alarm.data_path", lambda *parts: tmp_path.joinpath(*parts))
    from dms.alerts import PHONE_USAGE, VISION, Alert
    from dms.local_alarm import LocalAlarm

    alarm = LocalAlarm()
    alarm.player = "aplay"
    with patch("dms.local_alarm.subprocess.Popen") as popen:
        alarm(Alert(type=PHONE_USAGE, category=VISION, confidence=0.9, description="Phone usage while driving"))
    assert popen.called

    from app.push import should_notify

    class _Row:
        orin_type = "PHONE_USAGE"
        kind = "distraction"

    assert should_notify(_Row()) is False


def test_coach_is_the_default_and_the_old_layout_still_draws(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from dms.alerts import PHONE_USAGE, VISION, Alert, AlertBus
    from dms.state import SharedState
    from dms.ui.dashboard import Dashboard, render_png

    _app = QApplication.instance() or QApplication([])
    shared = SharedState()
    bus = AlertBus("orin-nano-01", cooldown_s=0)
    dash = Dashboard(shared, bus, cfg=None, fps=5)
    assert dash.mode == "coach"
    bus.emit(Alert(type=PHONE_USAGE, category=VISION, confidence=0.9, description="Phone usage while driving"))
    coach = tmp_path / "coach.png"
    instrument = tmp_path / "dms.png"
    render_png(shared, bus, str(coach), width=960, height=536, mode="coach")
    render_png(shared, bus, str(instrument), width=960, height=536, mode="dms")
    assert coach.stat().st_size > 1000
    assert instrument.stat().st_size > 1000
    dash._shortcuts[0].activated.emit()
    assert dash.mode == "dms"
    dash.close()
