import json

import numpy as np
import pytest

from dms.state import DriverState, PassengerState, SeatState
from dms.vlm_worker import assess, compose, parse_vlm, scene_prompt, state_card


@pytest.mark.parametrize("text, unsafe, cats", [
    ("SAFE: the driver is attentive.", False, []),
    ("No unsafe behavior is visible in the cabin.", False, []),
    ("The driver is not distracted and no passenger is aggressive.", False, []),
    ("UNSAFE: The driver is holding a phone while driving.", True, ["driver_distracted"]),
    ("A passenger is leaning into the front seat and grabbing the driver.", True, ["passenger_aggressive"]),
    ("A toddler is alone in the back seat, unattended.", True, ["child_unattended"]),
    ("A bag on the dashboard is blocking the driver's view.", True, ["objects_blocking_driver"]),
    ("UNSAFE: something looks wrong.", True, ["other"]),
])
def test_parse(text, unsafe, cats):
    verdict = parse_vlm(text)
    assert verdict.unsafe is unsafe
    for c in cats:
        assert c in verdict.categories
    if not unsafe:
        assert verdict.categories == []


def test_think_tags_are_stripped_before_the_verdict():
    verdict = parse_vlm("<think>maybe the phone is a reflection</think>\nUNSAFE: the driver is holding a phone while driving.")
    assert verdict.unsafe is True
    assert "driver_distracted" in verdict.categories
    assert "<think>" not in verdict.summary


def test_json_scene_is_preferred_over_prose_words():
    raw = json.dumps({
        "scene": "A passenger is leaning toward the driver.",
        "hazards": ["passenger_aggressive", "driver_distracted"],
        "confidence": 0.81,
    })
    verdict = parse_vlm("<think>ignore</think>\n" + raw)
    assert verdict.scene.startswith("A passenger")
    assert verdict.confidence == pytest.approx(0.81)
    assert verdict.categories == ["passenger_aggressive", "driver_distracted"]
    assert "<think>" not in verdict.summary


def test_policy_keeps_cabin_hazards_and_drops_driver_state():
    text = json.dumps({
        "scene": "Passenger is grabbing toward the driver.",
        "hazards": ["driver_distracted", "passenger_aggressive"],
        "confidence": 0.82,
    })
    _verdict, decision, routed = assess(text)
    assert decision.kept == ["passenger_aggressive"]
    assert any("detector owns" in item for item in decision.suppressed)
    assert [item[0] for item in routed] == ["PASSENGER_AGGRESSIVE"]
    assert decision.scene.startswith("Passenger")
    assert decision.line.startswith("Policy: kept passenger aggressive")


def test_policy_suppresses_a_child_hazard_when_an_adult_is_measured():
    cabin = PassengerState(connected=True, adult_present=True, child_present=False, adults=1,
                           seats=[SeatState("front_passenger", "adult")])
    text = json.dumps({"scene": "A child seems alone.", "hazards": ["child_unattended"], "confidence": 0.9})
    _verdict, decision, routed = assess(text, cabin=cabin)
    assert decision.kept == []
    assert routed == []
    assert "adult in cabin" in decision.suppressed[0]
    assert "no extra" not in decision.line


def test_policy_keeps_a_child_hazard_the_cabin_monitor_already_sees():
    cabin = PassengerState(connected=True, child_present=True, children=1, seats=[SeatState("rear_left", "child")])
    text = json.dumps({"scene": "Child alone in the rear.", "hazards": ["child_unattended"], "confidence": 0.8})
    _verdict, decision, routed = assess(text, cabin=cabin)
    assert decision.kept == ["child_unattended"]
    assert routed[0][0] == "CHILD_PRESENT"


def test_low_confidence_json_does_not_alert():
    text = json.dumps({"scene": "Maybe a bag on the dash.", "hazards": ["objects_blocking_driver"], "confidence": 0.2})
    _verdict, decision, routed = assess(text)
    assert routed == []
    assert "confidence" in decision.suppressed[0]


def test_state_card_is_appended_to_the_scene_prompt():
    driver = DriverState(eye_state="Open", yaw=8, phone=False, seatbelt="On", distracted=False)
    cabin = PassengerState(connected=True, occupants=1, adults=1, seats=[SeatState("rear_left", "adult")])
    card = state_card(driver, cabin, None)
    assert "yaw_deg=8" in card
    assert "rear_left=adult" in card
    prompt = scene_prompt("Describe the scene.", "Left half is the driver.", driver, cabin, None)
    assert prompt.startswith("Describe the scene.")
    assert "Measurements:" in prompt
    assert "cabin=unavailable" in state_card(driver, None, None)


def test_compose_side_by_side():
    a = np.zeros((480, 640, 3), np.uint8)
    b = np.full((240, 320, 3), 255, np.uint8)
    out = compose(a, b, 896)
    assert out.shape[1] == 896 and out.shape[0] == 336
    assert compose(a, None, 448).shape == (336, 448, 3)
    assert compose(None, None, 448) is None
