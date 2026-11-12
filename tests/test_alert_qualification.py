"""The alerts other than phone need the same kind of agreement before they publish."""

import numpy as np

from dms.objects import COCO_CUP
from dms.realtime_detector import (
    assess_drink, assess_smoke, choose_distraction, drowsy_evidence, eyes_are_closed, perclos, view_blocked,
)
from dms.seatbelt import SeatbeltDetector

FACE = (200, 100, 400, 400)
DET = 0.35
MODEL = 0.55


def test_eyes_closed_needs_a_measured_ear():
    assert eyes_are_closed(True, 0.10, 0.21) is True
    assert eyes_are_closed(True, 0.30, 0.21) is False
    assert eyes_are_closed(True, None, 0.21) is False
    assert eyes_are_closed(False, 0.10, 0.21) is False


def test_one_distraction_card_and_no_head_turn_alone():
    assert choose_distraction(True, True, True, True) == "eyes_closed"
    assert choose_distraction(False, True, True, True) == "phone_usage"
    assert choose_distraction(False, False, True, True) == "looking_down"
    assert choose_distraction(False, False, False, True) == "gaze_away"
    assert choose_distraction(False, False, False, False) is None


def test_drowsiness_needs_perclos_or_an_agreed_yawn():
    short = [(t, True) for t in range(10)]
    assert perclos(short, now=10) is None

    blinks = [(float(t), t % 20 == 0) for t in range(25)]
    closure = perclos(blinks, now=24)
    assert closure is not None and closure < 0.15
    assert drowsy_evidence(closure, yawning=False, model_drowsy=True) is False

    heavy = [(float(t), t % 4 == 0) for t in range(25)]
    assert drowsy_evidence(perclos(heavy, now=24), yawning=False, model_drowsy=False) is True
    assert drowsy_evidence(0.02, yawning=True, model_drowsy=False) is False
    assert drowsy_evidence(0.02, yawning=True, model_drowsy=True) is True


def test_drinking_needs_a_cup_at_the_mouth_and_a_hand():
    cup = (240, 400, 340, 520)
    hand = (230, 390, 360, 540)
    using, conf = assess_drink(FACE, [(cup, 0.80, COCO_CUP)], [hand], 0.0, DET, MODEL)
    assert using is True and conf == 0.80
    assert assess_drink(FACE, [(cup, 0.80, COCO_CUP)], [], 0.90, DET, MODEL) == (False, 0.0)
    assert assess_drink(FACE, [], [], 0.90, DET, MODEL) == (False, 0.0)


def test_smoking_needs_the_label_and_a_hand_at_the_mouth():
    hand = (240, 300, 360, 500)
    smoking, conf = assess_smoke(FACE, [hand], "Cigarette", 0.80, MODEL)
    assert smoking is True and conf == 0.80
    assert assess_smoke(FACE, [], "Cigarette", 0.80, MODEL) == (False, 0.0)
    assert assess_smoke(FACE, [hand], "None", 0.90, MODEL) == (False, 0.0)


def test_seatbelt_off_waits_for_a_full_window():
    belt = SeatbeltDetector(window=30)
    for _ in range(10):
        belt.votes.append(False)
    assert belt.state() == "Unknown"
    for _ in range(20):
        belt.votes.append(False)
    assert belt.state() == "Off"
    belt.votes.clear()
    for _ in range(30):
        belt.votes.append(True)
    assert belt.state() == "On"


def test_a_covered_camera_is_not_an_empty_seat():
    assert view_blocked(np.zeros((48, 64, 3), dtype=np.uint8)) is True
    assert view_blocked(np.full((48, 64, 3), 255, dtype=np.uint8)) is True
    cabin = np.zeros((48, 64, 3), dtype=np.uint8)
    cabin[:, :32] = (40, 50, 60)
    cabin[:, 32:] = (180, 160, 140)
    assert view_blocked(cabin) is False
