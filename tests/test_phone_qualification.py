"""Phone use needs a detector box. A hand or an activity score cannot invent one."""

from dms.objects import COCO_PHONE
from dms.realtime_detector import Hold, assess_phone

FACE = (200, 100, 400, 400)  # driver's face
DET = 0.35
MODEL = 0.50


def _phone(box, score=0.80):
    return (box, score, COCO_PHONE)


def test_hand_at_ear_without_a_phone_is_not_phone_use():
    hand = (360, 140, 520, 400)  # sits in the right ear zone
    using, conf, box = assess_phone(FACE, [], [hand], 0.0, DET, MODEL)
    assert (using, conf, box) == (False, 0.0, None)


def test_activity_score_without_a_phone_box_is_not_phone_use():
    using, conf, box = assess_phone(FACE, [], [], 0.95, DET, MODEL)
    assert (using, conf, box) == (False, 0.0, None)


def test_phone_box_alone_is_not_phone_use():
    # In reach, below the face, not in an ear zone, and no hand on it.
    using, conf, box = assess_phone(FACE, [_phone((250, 520, 370, 700))], [], 0.90, DET, MODEL)
    assert (using, conf, box) == (False, 0.0, None)


def test_phone_outside_reach_is_ignored():
    far = (-800, 200, -700, 400)
    hand = (-820, 180, -680, 420)
    using, conf, box = assess_phone(FACE, [_phone(far)], [hand], 0.90, DET, MODEL)
    assert (using, conf, box) == (False, 0.0, None)


def test_weak_detector_score_is_ignored():
    phone = (250, 520, 370, 700)
    hand = (230, 500, 400, 740)
    using, conf, box = assess_phone(FACE, [_phone(phone, 0.20)], [hand], 0.90, DET, MODEL)
    assert (using, conf, box) == (False, 0.0, None)


def test_hand_on_a_real_phone_confirms_use():
    phone = (250, 520, 370, 700)
    hand = (230, 500, 400, 740)
    using, conf, box = assess_phone(FACE, [_phone(phone, 0.62)], [hand], 0.10, DET, MODEL)
    assert using is True
    assert box == phone
    assert conf == 0.62


def test_phone_at_the_ear_confirms_use_without_a_hand():
    phone = (370, 160, 500, 360)  # inside the right ear zone
    using, conf, box = assess_phone(FACE, [_phone(phone, 0.70)], [], 0.0, DET, MODEL)
    assert using is True
    assert box == phone
    assert conf == 0.70


def test_activity_score_raises_confidence_only_after_the_box_is_confirmed():
    phone = (250, 520, 370, 700)
    hand = (230, 500, 400, 740)
    using, conf, box = assess_phone(FACE, [_phone(phone, 0.40)], [hand], 0.88, DET, MODEL)
    assert using is True
    assert box == phone
    assert conf == 0.88


def test_phone_alert_waits_until_the_hold():
    phone = (370, 160, 500, 360)
    hold = Hold(2.0)
    started = 10.0
    fired = False
    for step in range(21):
        now = started + step * 0.1
        using, _, _ = assess_phone(FACE, [_phone(phone)], [], 0.0, DET, MODEL)
        hold.update(using, now)
        fired = hold.fired(now)
        if step < 20:
            assert fired is False
    assert fired is True
