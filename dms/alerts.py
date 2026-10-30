"""Alert model and fan-out bus (detectors -> MQTT queue, UI, local alarm)."""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

log = logging.getLogger("dms.alerts")

VISION = "vision"
AUDIO = "audio"

# Alert types. DISTRACTION / VLM_ALERT / SCREAM / ABUSIVE_LANGUAGE / ALTERCATION are the
# spec's names; the rest are extra driver-state events the 8-head model provides.
DISTRACTION = "DISTRACTION"
DROWSINESS = "DROWSINESS"
PHONE_USAGE = "PHONE_USAGE"
SMOKING = "SMOKING"
DRINKING = "DRINKING"
SEATBELT_OFF = "SEATBELT_OFF"
DRIVER_ABSENT = "DRIVER_ABSENT"
VLM_ALERT = "VLM_ALERT"
SCREAM = "SCREAM"
ABUSIVE_LANGUAGE = "ABUSIVE_LANGUAGE"
ALTERCATION = "ALTERCATION"
# Cabin channel. CHILD_PRESENT is "a child is visible and no adult is", not a
# left-behind event: this device has no ignition or door signal.
CHILD_PRESENT = "CHILD_PRESENT"
OBJECT_ON_SEAT = "OBJECT_ON_SEAT"
PASSENGER_AGGRESSIVE = "PASSENGER_AGGRESSIVE"
OCCUPANCY_CHANGE = "OCCUPANCY_CHANGE"

CMS_TYPES = {CHILD_PRESENT, OBJECT_ON_SEAT, PASSENGER_AGGRESSIVE, OCCUPANCY_CHANGE}

SEVERITY = {
    DISTRACTION: "high", PHONE_USAGE: "high", DROWSINESS: "high", DRIVER_ABSENT: "high",
    SMOKING: "medium", DRINKING: "medium", SEATBELT_OFF: "medium", VLM_ALERT: "medium",
    SCREAM: "high", ALTERCATION: "high", ABUSIVE_LANGUAGE: "medium",
    CHILD_PRESENT: "high", OBJECT_ON_SEAT: "medium", PASSENGER_AGGRESSIVE: "high",
    OCCUPANCY_CHANGE: "low",
}

TITLES = {
    DISTRACTION: "Driver Distraction Detected",
    PHONE_USAGE: "Driver Distraction Detected",
    DROWSINESS: "Driver Drowsiness Detected",
    SMOKING: "Smoking Detected",
    DRINKING: "Drinking While Driving",
    SEATBELT_OFF: "Seatbelt Not Fastened",
    DRIVER_ABSENT: "Driver Not Visible",
    VLM_ALERT: "Cabin Safety Alert",
    SCREAM: "Scream Detected",
    ALTERCATION: "Cabin Altercation",
    ABUSIVE_LANGUAGE: "Abusive Language",
    CHILD_PRESENT: "Child Present",
    OBJECT_ON_SEAT: "Object On Seat",
    PASSENGER_AGGRESSIVE: "Passenger Aggressive",
    OCCUPANCY_CHANGE: "Occupancy Changed",
}


@dataclass
class Alert:
    type: str
    category: str
    confidence: float
    description: str = ""
    reason: str = ""
    snapshot_path: str | None = None
    audio_path: str | None = None
    extra: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    device_id: str = ""
    channel: str = "dms"  # dms | cms

    @property
    def severity(self) -> str:
        return SEVERITY.get(self.type, "medium")

    @property
    def title(self) -> str:
        return TITLES.get(self.type, self.type.replace("_", " ").title())

    @property
    def iso_time(self) -> str:
        return datetime.fromtimestamp(self.timestamp, timezone.utc).isoformat(timespec="milliseconds")

    def payload(self) -> dict:
        body = {
            "id": self.id,
            "device_id": self.device_id,
            "type": self.type,
            "category": self.category,
            "severity": self.severity,
            "confidence": round(float(self.confidence), 3),
            "timestamp": self.iso_time,
            "epoch_ms": int(self.timestamp * 1000),
            "description": self.description,
            "channel": self.channel,
        }
        if self.reason:
            body["reason"] = self.reason
        if self.category == VISION:
            body["snapshot_path"] = self.snapshot_path
        if self.audio_path:
            body["audio_path"] = self.audio_path
        if self.extra:
            body["details"] = self.extra
        return body


class AlertBus:
    """Debounces alerts per (type, reason) and fans them out.

    The MQTT queue is bounded only by the publisher's persistent outbox, so an alert is
    never dropped here; UI/alarm subscribers get callbacks on the emitting thread and must
    be quick.
    """

    def __init__(self, device_id: str, cooldown_s: float = 8.0, history: int = 50) -> None:
        self.device_id = device_id
        self.cooldown_s = cooldown_s
        self.mqtt_queue: "queue.Queue[Alert]" = queue.Queue()
        self.analysis_queue: "queue.Queue[tuple[str, dict]]" = queue.Queue()
        self._last: dict[tuple, float] = {}
        self._lock = threading.Lock()
        self._subscribers: list = []
        self.history: list[Alert] = []
        self._history_size = history
        self.counts: dict[str, int] = {}

    def subscribe(self, callback) -> None:
        self._subscribers.append(callback)

    def emit_analysis(self, channel: str, body: dict) -> None:
        self.analysis_queue.put((channel, body))

    def emit(self, alert: Alert, cooldown_s: float | None = None) -> bool:
        key = (alert.type, alert.reason)
        window = self.cooldown_s if cooldown_s is None else cooldown_s
        now = time.monotonic()
        with self._lock:
            last = self._last.get(key)
            if last is not None and now - last < window:
                return False
            self._last[key] = now
            alert.device_id = alert.device_id or self.device_id
            self.history.append(alert)
            if len(self.history) > self._history_size:
                del self.history[0]
            self.counts[alert.type] = self.counts.get(alert.type, 0) + 1
        self.mqtt_queue.put(alert)
        log.warning("ALERT %s %s conf=%.2f %s", alert.type, alert.reason, alert.confidence, alert.description)
        for callback in list(self._subscribers):
            try:
                callback(alert)
            except Exception:  # a broken subscriber must not stop alerting
                log.exception("alert subscriber failed")
        return True

    def latest(self, channel: str | None = None) -> Alert | None:
        with self._lock:
            if channel is None:
                return self.history[-1] if self.history else None
            for alert in reversed(self.history):
                if alert.channel == channel:
                    return alert
            return None

    def recent(self, n: int = 10) -> list[Alert]:
        with self._lock:
            return list(self.history[-n:])


def as_dict(alert: Alert) -> dict:
    return asdict(alert)
