"""Snapshots shared between worker threads and the UI.

Workers build a fresh snapshot object and assign it in one statement (atomic under the
GIL); readers never see a half-updated record and no lock is held while rendering.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Identity:
    known: bool = False
    name: str = "Unknown"
    driver_id: str = "----"
    age: str = "--"
    match_pct: float = 0.0
    photo: np.ndarray | None = None  # BGR
    notes: str = ""


@dataclass
class DriverState:
    frame: np.ndarray | None = None
    frame_ts: float = 0.0
    seq: int = 0
    width: int = 640
    height: int = 480

    face_found: bool = False
    face_bbox: tuple | None = None  # x1, y1, x2, y2
    face_score: float = 0.0
    landmarks: np.ndarray | None = None  # (N, 2) image px
    mesh_edges: list = field(default_factory=list)  # [(i, j)] into mesh_points
    mesh_points: np.ndarray | None = None

    ear: float | None = None
    ear_threshold: float = 0.21
    eyes_open_pct: float | None = None
    eye_state: str = "--"
    blink_rate: float | None = None
    eyes_closed_s: float = 0.0

    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    head_pose: str = "--"
    head_turned_s: float = 0.0

    gaze: str = "--"  # e.g. "Left", "Forward", "Down"
    gaze_away: bool = False
    gaze_angle: float | None = None  # degrees, > 0 towards image-left
    iris: float | None = None  # 0..1 iris position in the eye opening
    gaze_vector: tuple = (0.0, 0.0)  # unit-ish (dx, dy) in image space for the arrow
    eye_crop: np.ndarray | None = None

    emotion: str = "--"
    emotion_conf: float = 0.0
    drowsy: bool = False
    yawning: bool = False
    age_group: str = "--"

    activity: str = "--"
    activity_conf: float = 0.0
    phone: bool = False
    phone_conf: float = 0.0
    phone_box: tuple | None = None
    hands: list = field(default_factory=list)  # [(x1, y1, x2, y2, score)]
    hand_position: str = "--"
    seatbelt: str = "Unknown"
    smoking: bool = False
    drinking: bool = False

    distracted: bool = False
    distraction_reasons: list = field(default_factory=list)
    overlay_alert: str = ""  # banner text over the video ("DISTRACTION DETECTED")

    identity: Identity = field(default_factory=Identity)

    model_name: str = "MobileNetV3-DMS"
    framework: str = "--"
    model_ms: float = 0.0
    pipeline_ms: float = 0.0
    stage_ms: dict = field(default_factory=dict)  # EMA per pipeline stage
    fps: float = 0.0
    modules: dict = field(default_factory=lambda: {
        "Face Tracking": "Off", "Eye Tracking": "Off", "Object Detection": "Off", "Hand Tracking": "Off",
    })


@dataclass
class SeatState:
    """One configured seat zone. Belt stays Unknown until a cabin belt check exists."""

    name: str
    occupant: str = "empty"  # empty | adult | child | unknown
    belt: str = "Unknown"
    face_bbox: tuple | None = None


@dataclass
class CabinObject:
    label: str
    bbox: tuple
    score: float = 0.0
    seat: str = ""


@dataclass
class PassengerState:
    """Cabin snapshot. The attribute on SharedState stays ``passenger`` so the UI and
    the publisher keep working; ``CabinState`` is the same type."""

    frame: np.ndarray | None = None
    frame_ts: float = 0.0
    faces: list = field(default_factory=list)  # [(x1, y1, x2, y2)]
    face_kinds: list = field(default_factory=list)  # adult | child | unknown, parallel to faces
    occupants: int = 0
    adults: int = 0
    children: int = 0
    fps: float = 0.0
    connected: bool = False
    seats: list = field(default_factory=list)  # list[SeatState]
    objects: list = field(default_factory=list)  # list[CabinObject]
    child_present: bool = False
    adult_present: bool = False


CabinState = PassengerState


@dataclass
class VlmState:
    enabled: bool = False
    status: str = "off"  # off | waiting | running | ok | error | throttled
    text: str = ""
    scene: str = ""  # one-sentence scene description
    policy: str = ""  # safety-policy line shown on the scene card
    unsafe: bool = False
    categories: list = field(default_factory=list)  # hazards the policy kept
    suppressed: list = field(default_factory=list)  # "hazard: reason"
    confidence: float = 0.0
    latency_s: float = 0.0  # last chat-completion round trip, seconds
    latency_avg_s: float = 0.0  # mean of the recent successful round trips
    interval_s: float = 0.0
    last_ts: float = 0.0
    calls: int = 0
    errors: int = 0
    model: str = ""
    dms_scene: str = ""
    cms_scene: str = ""
    dms_description: str = ""
    cms_description: str = ""
    dms: dict = field(default_factory=dict)
    cms: dict = field(default_factory=dict)


@dataclass
class AudioState:
    enabled: bool = False
    backend: str = "off"
    level_dbfs: float = -120.0
    tags: dict = field(default_factory=dict)  # top labels -> score
    last_event: str = ""
    last_event_ts: float = 0.0
    kws: str = "off"
    latency_ms: float = 0.0
    error: str = ""


@dataclass
class WebRtcState:
    enabled: bool = False
    status: str = "off"
    mode: str = ""
    api: str = ""
    url: str = ""
    room: str = ""
    seq: int = 0
    errors: int = 0
    last_error: str = ""


@dataclass
class MqttState:
    enabled: bool = False
    connected: bool = False
    broker: str = ""
    published: int = 0
    acked: int = 0
    failed: int = 0
    queued: int = 0
    success_rate: float = 100.0
    last_error: str = ""


class SharedState:
    def __init__(self) -> None:
        self.started = time.time()
        self.driver = DriverState()
        self.passenger = PassengerState()
        self.vlm = VlmState()
        self.audio = AudioState()
        self.mqtt = MqttState()
        self.webrtc = WebRtcState()
        self.system = None  # telemetry.SystemStats
        self.hardware = ""
        self.device_id = ""
