"""Regression checks for the rules that raise or suppress an alert.

No camera, no model file, no broker. Run this after every change:

    python tests/test_decisions.py

A failing name is the rule that changed. The log test checks that a decision
line lands in the file with the module and line that wrote it.
"""

from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from dms.alerts import DISTRACTION, OBJECT_ON_SEAT, Alert, AlertBus
from dms.cms import CabinRules
from dms.config import load_config
from dms.logutil import StateLog, configure_logging
from dms.objects import COCO_CUP, COCO_PHONE
from dms.realtime_detector import (
    Hold,
    ObjectWorker,
    assess_drink,
    assess_phone,
    assess_smoke,
    choose_distraction,
    drowsy_evidence,
    eyes_are_closed,
    measured_closure,
    mouth_zone,
    perclos,
    view_blocked,
)
from dms.seatbelt import SeatbeltDetector
from dms.state import CabinObject, CabinState, SeatState

FACE = (200.0, 100.0, 300.0, 250.0)  # x1, y1, x2, y2
DET = 0.35
MODEL = 0.50


def setUpModule():
    # Keep alert lines out of the console until a test installs the real handlers.
    quiet = logging.getLogger("dms")
    quiet.addHandler(logging.NullHandler())
    quiet.propagate = False


def _inside(zone, pad: float = 4.0):
    x1, y1, x2, y2 = zone
    return (x1 + pad, y1 + pad, x2 - pad, y2 - pad)


class PhoneDrinkSmoke(unittest.TestCase):
    def test_phone_needs_a_box_in_the_hand_or_at_the_ear(self):
        ear = (140.0, 130.0, 200.0, 220.0)
        using, conf, box, why = assess_phone(FACE, [(ear, 0.8, COCO_PHONE)], [], 0.1, DET, MODEL)
        self.assertTrue(using)
        self.assertEqual(why, "ok")
        self.assertEqual(box, ear)
        self.assertGreaterEqual(conf, 0.8)

        lap = (240.0, 450.0, 280.0, 500.0)
        using, _, _, why = assess_phone(FACE, [(lap, 0.9, COCO_PHONE)], [], 0.99, DET, MODEL)
        self.assertFalse(using)
        self.assertEqual(why, "not_in_hand")

        using, conf, box, why = assess_phone(FACE, [(lap, 0.9, COCO_PHONE)], [lap], 0.2, DET, MODEL)
        self.assertTrue(using)
        self.assertEqual(why, "ok")
        self.assertEqual(box, lap)
        self.assertAlmostEqual(conf, 0.9)

    def test_activity_score_and_a_hand_at_the_ear_do_not_invent_a_phone(self):
        hand = (140.0, 130.0, 200.0, 220.0)
        using, _, _, why = assess_phone(FACE, [], [hand], 0.99, DET, MODEL)
        self.assertFalse(using)
        self.assertEqual(why, "no_box")
        using, _, _, why = assess_phone(None, [(hand, 0.9, COCO_PHONE)], [hand], 0.9, DET, MODEL)
        self.assertEqual(why, "no_face")

    def test_weak_phone_box_is_ignored(self):
        ear = (140.0, 130.0, 200.0, 220.0)
        _, _, _, why = assess_phone(FACE, [(ear, 0.1, COCO_PHONE)], [], 0.9, DET, MODEL)
        self.assertEqual(why, "no_box")

    def test_drink_needs_a_cup_at_the_mouth_and_a_hand(self):
        cup = _inside(mouth_zone(FACE))
        drinking, conf, why = assess_drink(FACE, [(cup, 0.7, COCO_CUP)], [cup], 0.0, DET, MODEL)
        self.assertTrue(drinking)
        self.assertEqual(why, "ok")
        self.assertAlmostEqual(conf, 0.7)

        drinking, _, why = assess_drink(FACE, [], [], 0.99, DET, MODEL)
        self.assertFalse(drinking)
        self.assertEqual(why, "no_box")

        lap = (240.0, 450.0, 280.0, 500.0)
        _, _, why = assess_drink(FACE, [(lap, 0.9, COCO_CUP)], [lap], 0.9, DET, MODEL)
        self.assertEqual(why, "no_box")

        _, _, why = assess_drink(FACE, [(cup, 0.9, COCO_CUP)], [], 0.9, DET, MODEL)
        self.assertEqual(why, "not_in_hand")

    def test_smoke_needs_the_cigarette_label_and_a_hand_at_the_mouth(self):
        hand = _inside(mouth_zone(FACE))
        smoking, conf, why = assess_smoke(FACE, [hand], "Cigarette", 0.8, MODEL)
        self.assertTrue(smoking)
        self.assertEqual(why, "ok")
        self.assertAlmostEqual(conf, 0.8)
        _, _, why = assess_smoke(FACE, [], "Cigarette", 0.9, MODEL)
        self.assertEqual(why, "no_hand")
        _, _, why = assess_smoke(FACE, [hand], "None", 0.9, MODEL)
        self.assertEqual(why, "not_cigarette")
        _, _, why = assess_smoke(FACE, [hand], "Cigarette", 0.2, MODEL)
        self.assertEqual(why, "not_cigarette")


class EyeAndDistraction(unittest.TestCase):
    def test_eyes_closed_uses_ear_only(self):
        self.assertFalse(eyes_are_closed(True, None, 0.21))
        self.assertFalse(eyes_are_closed(False, 0.05, 0.21))
        self.assertTrue(eyes_are_closed(True, 0.10, 0.21))
        self.assertFalse(eyes_are_closed(True, 0.30, 0.21))

    def test_perclos_counts_a_lost_mesh_only_when_the_model_is_sure(self):
        self.assertIsNone(measured_closure(False, None, 0.21, 0.99))
        self.assertTrue(measured_closure(True, 0.10, 0.21, 0.0))
        self.assertFalse(measured_closure(True, 0.30, 0.21, 0.99))  # EAR wins
        self.assertTrue(measured_closure(True, None, 0.21, 0.85))
        self.assertIsNone(measured_closure(True, None, 0.21, 0.50))

    def test_perclos_waits_for_a_real_span_and_ignores_normal_blinking(self):
        self.assertIsNone(perclos([(0.0, True), (5.0, True)], 5.0))
        samples = [(0.0, False), (10.0, False), (20.0, True), (25.0, True)]
        self.assertAlmostEqual(perclos(samples, 25.0), 0.5)
        blinks = [(float(t), t % 20 == 0) for t in range(0, 25)]
        self.assertLess(perclos(blinks, 24.0), 0.15)
        self.assertFalse(drowsy_evidence(0.10, False, True))
        self.assertTrue(drowsy_evidence(0.20, False, False))
        self.assertFalse(drowsy_evidence(None, True, False))
        self.assertTrue(drowsy_evidence(None, True, True))

    def test_one_distraction_reason_and_eyes_win(self):
        self.assertEqual(choose_distraction(True, True, True, True), "eyes_closed")
        self.assertEqual(choose_distraction(False, True, True, True), "phone_usage")
        self.assertEqual(choose_distraction(False, False, True, True), "looking_down")
        self.assertEqual(choose_distraction(False, False, False, True), "gaze_away")
        self.assertIsNone(choose_distraction(False, False, False, False))

    def test_flat_frame_is_a_covered_camera(self):
        self.assertTrue(view_blocked(np.zeros((48, 64, 3), np.uint8)))
        self.assertTrue(view_blocked(np.full((48, 64, 3), 255, np.uint8)))
        self.assertFalse(view_blocked(None))
        frame = np.zeros((48, 64, 3), np.uint8)
        frame[:, :32] = 40
        frame[:, 32:] = 180
        self.assertFalse(view_blocked(frame))


class Timers(unittest.TestCase):
    def test_hold_fires_after_its_window_and_resets_when_the_gap_is_zero(self):
        hold = Hold(2.0)
        hold.update(True, 0.0)
        self.assertFalse(hold.fired(1.9))
        self.assertTrue(hold.fired(2.0))
        hold.update(False, 2.1)
        self.assertFalse(hold.active)
        self.assertFalse(hold.fired(3.0))

    def test_hold_bridges_one_missed_detector_cycle(self):
        hold = Hold(2.0, gap_s=0.5)
        hold.update(True, 0.0)
        hold.update(True, 1.5)
        hold.update(False, 1.6)  # inside the gap: the 2 s clock keeps running
        self.assertTrue(hold.active)
        self.assertFalse(hold.fired(1.6))
        hold.update(True, 1.9)
        self.assertTrue(hold.fired(2.0))
        dropped = Hold(2.0, gap_s=0.5)
        dropped.update(True, 0.0)
        dropped.update(False, 1.0)
        dropped.update(False, 1.6)  # gap exceeded
        self.assertFalse(dropped.active)


class Seatbelt(unittest.TestCase):
    def test_off_waits_for_a_full_window_and_a_blind_chest_is_unknown(self):
        belt = SeatbeltDetector(window=30)
        for _ in range(10):
            belt.votes.append(False)
        self.assertEqual(belt.state(), "Unknown")
        for _ in range(20):
            belt.votes.append(False)
        self.assertEqual(belt.state(), "Off")

        on = SeatbeltDetector(window=30)
        for _ in range(12):
            on.votes.append(True)
        self.assertEqual(on.state(), "On")
        on.update(None, None)
        on.update(None, None)
        self.assertEqual(on.state(), "On")  # one or two misses keep the last answer
        on.update(None, None)
        self.assertEqual(on.state(), "Unknown")
        self.assertEqual(len(on.votes), 12)  # votes stay for the next good frame


class AlertsAndCabin(unittest.TestCase):
    def test_cooldown_is_per_reason_and_checked_under_the_lock(self):
        bus = AlertBus("orin-test", cooldown_s=10.0)
        first = Alert(type=DISTRACTION, category="vision", confidence=0.9, reason="phone_usage")
        self.assertTrue(bus.emit(first))
        again = Alert(type=DISTRACTION, category="vision", confidence=0.9, reason="phone_usage")
        self.assertFalse(bus.emit(again))
        self.assertFalse(bus.accepts(DISTRACTION, "phone_usage", 10.0))
        other = Alert(type=DISTRACTION, category="vision", confidence=0.8, reason="gaze_away")
        self.assertTrue(bus.accepts(DISTRACTION, "gaze_away", 10.0))
        self.assertTrue(bus.emit(other))
        self.assertEqual(bus.counts[DISTRACTION], 2)

    def test_child_and_object_alerts_wait_out_their_hold(self):
        rules = CabinRules(3.0, 2.0, 1.0)
        child = CabinState(child_present=True, seats=[SeatState("rear_left", occupant="child")])
        self.assertEqual(rules.step(child, 0.0), [])
        self.assertEqual(rules.step(child, 2.9), [])
        raised = rules.step(child, 3.0)
        self.assertEqual(raised[0][0], "CHILD_PRESENT")
        self.assertEqual(rules.step(child, 3.2), [])

        objects = CabinRules(3.0, 2.0, 1.0)
        cabin = CabinState(objects=[CabinObject("backpack", (0, 0, 10, 10), 0.8, "rear_left")])
        self.assertEqual(objects.step(cabin, 0.0), [])
        raised = objects.step(cabin, 2.0)
        self.assertEqual(raised[0][0], OBJECT_ON_SEAT)
        self.assertEqual(objects.step(cabin, 2.2), [])

    def test_defaults_match_the_live_rule_times(self):
        cfg = load_config(None)
        self.assertEqual(cfg.realtime.eyes_closed_s, 2.5)
        self.assertEqual(cfg.realtime.phone_s, 2.0)
        self.assertEqual(cfg.realtime.no_face_s, 5.0)
        self.assertEqual(cfg.realtime.seatbelt_off_s, 10.0)
        self.assertTrue(cfg.log_file.endswith("dms.log"))


class LogFile(unittest.TestCase):
    def test_decision_line_is_written_with_its_source(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dms.log"
            configure_logging("INFO", path)
            state = StateLog(logging.getLogger("dms.rules"))
            state.change("seatbelt", "Off", "seatbelt %s votes=%d", "Off", 30, level=logging.INFO)
            state.change("seatbelt", "Off", "seatbelt %s votes=%d", "Off", 30, level=logging.INFO)
            state.rise("view_blocked", True, "flat frame; driver-absent timer paused")
            state.rise("view_blocked", True, "flat frame; driver-absent timer paused")
            state.rise("view_blocked", False, "flat frame; driver-absent timer paused")
            logging.getLogger("dms.rules").debug("needle-debug")
            text = path.read_text(encoding="utf-8")
        self.assertEqual(text.count("seatbelt Off votes=30"), 1)
        self.assertIn("flat frame; driver-absent timer paused", text)
        self.assertIn("view_blocked cleared", text)
        self.assertIn("needle-debug", text)
        self.assertIn("test_decisions.py", text)
        self.assertIn("dms.rules", text)

    def test_object_snapshot_is_one_consistent_update(self):
        worker = ObjectWorker(None, None, max_hz=30)
        objects, hands, ts, ready = worker.snapshot()
        self.assertFalse(ready)
        self.assertEqual(objects, [])
        with worker._cond:
            worker.objects = [(1, 2, 3, 4, 0.9)]
            worker.hands = [(5, 6, 7, 8, 0.8)]
            worker.ts = 12.5
            worker._ready = True
        objects, hands, ts, ready = worker.snapshot()
        self.assertTrue(ready)
        self.assertEqual(ts, 12.5)
        self.assertEqual(objects[0][0], 1)
        self.assertEqual(hands[0][0], 5)
        objects.append("mutated-by-reader")
        fresh, _, _, _ = worker.snapshot()
        self.assertEqual(len(fresh), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
