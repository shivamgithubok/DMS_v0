import numpy as np
import pytest

from dms.face import SCHEMA_68, HeadPose, _MODEL_3D, eye_aspect_ratio, geometric_yaw
from dms.objects import palm_anchors
from dms.realtime_detector import Hold

CAM = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1.0]])


def _project(yaw_deg, pitch_deg):
    """Project the head model rotated by (yaw, pitch); yaw > 0 turns the face to image-left."""
    a, b = np.radians(-yaw_deg), np.radians(pitch_deg)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    flip = np.diag([1, -1, -1.0])  # model (y up, z to viewer) -> camera (y down, z forward)
    pts = (flip @ rx @ ry @ _MODEL_3D.T).T + np.array([0, 0, 800.0])
    uv = (CAM @ pts.T).T
    full = np.zeros((68, 2), np.float32)
    for k, i in enumerate(SCHEMA_68.pnp):
        full[i] = uv[k, :2] / uv[k, 2]
    return full


@pytest.mark.parametrize("yaw", [-40, -30, 0, 30, 40])
def test_head_yaw_sign_and_magnitude(yaw):
    est = HeadPose().estimate(_project(yaw, 0), SCHEMA_68, 640, 480)
    assert est is not None
    assert est[0] == pytest.approx(yaw, abs=2.0)


@pytest.mark.parametrize("pitch", [-20, 0, 20])
def test_head_pitch(pitch):
    # pitch > 0 = looking up; the synthetic rotation tips the forehead towards the camera for > 0
    est = HeadPose().estimate(_project(0, -pitch), SCHEMA_68, 640, 480)
    assert est[1] == pytest.approx(pitch, abs=2.0)


def test_geometric_yaw_agrees_in_sign():
    assert geometric_yaw(_project(30, 0), SCHEMA_68) > 10
    assert geometric_yaw(_project(-30, 0), SCHEMA_68) < -10


def test_eye_aspect_ratio_open_vs_closed():
    pts = np.zeros((68, 2), np.float32)
    idx = SCHEMA_68.left_eye
    open_eye = [(0, 0), (10, -6), (20, -6), (30, 0), (20, 6), (10, 6)]
    for i, p in zip(idx, open_eye):
        pts[i] = p
    assert eye_aspect_ratio(pts, idx) == pytest.approx(0.4, abs=1e-6)
    for i, p in zip(idx, [(0, 0), (10, -1), (20, -1), (30, 0), (20, 1), (10, 1)]):
        pts[i] = p
    assert eye_aspect_ratio(pts, idx) < 0.1


def test_palm_anchor_layout():
    a = palm_anchors()
    assert a.shape == (2016, 2)
    assert a[0] == pytest.approx([1 / 48, 1 / 48])
    assert a[-1] == pytest.approx([23 / 24, 23 / 24])


def test_hold_timing():
    h = Hold(1.5)
    assert h.update(True, 10.0) == 0.0
    assert not h.fired(11.0)
    assert h.fired(11.6)
    h.update(False, 11.7)
    assert not h.active and not h.fired(12.0)
