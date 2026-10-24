"""Cabin monitor (Thread 6).

Reads the passenger camera at about 2 Hz, off the real-time driver thread. YuNet faces
and the existing NanoDet model (ONNX, no second TensorRT context) are assigned to seat
polygons from config. Passenger belts stay Unknown: the driver torso heuristic does not
transfer to a wide cabin frame. CHILD_PRESENT means a child is in a rear seat and no
adult face is visible. It is not a left-behind alarm; there is no ignition or door input.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import alerts as A
from .config import data_path, model_path
from .face import FaceDetector
from .objects import (COCO_BOTTLE, COCO_CUP, COCO_PHONE, COCO_WINE_GLASS, PhoneDetector)
from .state import CabinObject, CabinState, SeatState

log = logging.getLogger("dms.cms")

COCO_BACKPACK = 24
COCO_HANDBAG = 26
COCO_SUITCASE = 28
CMS_CLASSES = (COCO_BACKPACK, COCO_HANDBAG, COCO_SUITCASE, COCO_BOTTLE, COCO_CUP, COCO_WINE_GLASS, COCO_PHONE)
LABELS = {
    COCO_BACKPACK: "backpack", COCO_HANDBAG: "handbag", COCO_SUITCASE: "suitcase",
    COCO_BOTTLE: "bottle", COCO_CUP: "cup", COCO_WINE_GLASS: "wine glass", COCO_PHONE: "phone",
}
_RANK = {"empty": 0, "unknown": 1, "child": 2, "adult": 3}


class _Hold:
    def __init__(self, seconds: float) -> None:
        self.seconds = float(seconds)
        self.since: float | None = None

    def update(self, active: bool, now: float) -> None:
        if active:
            if self.since is None:
                self.since = now
        else:
            self.since = None

    def ready(self, now: float) -> bool:
        return self.since is not None and now - self.since >= self.seconds


def point_in_polygon(x: float, y: float, polygon) -> bool:
    """Ray cast. ``polygon`` is a sequence of (x, y) in the same space as the point."""
    inside = False
    count = len(polygon)
    if count < 3:
        return False
    j = count - 1
    for i in range(count):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if (yi > y) != (yj > y):
            denom = (yj - yi) or 1e-12
            if x < (xj - xi) * (y - yi) / denom + xi:
                inside = not inside
        j = i
    return inside


def _seat_at(nx: float, ny: float, seats) -> str:
    for seat in seats:
        if point_in_polygon(nx, ny, seat.get("polygon") or ()):
            return str(seat.get("name") or "")
    return ""


def _classify(bbox, frame_h: int, seat_name: str, adult_ratio: float) -> str:
    height = max(0, int(bbox[3]) - int(bbox[1]))
    ratio = height / float(max(1, frame_h))
    rear = seat_name.startswith("rear")
    if rear and ratio < adult_ratio:
        return "child"
    if ratio >= adult_ratio:
        return "adult"
    if seat_name == "front_passenger" and ratio >= adult_ratio * 0.67:
        return "adult"
    if ratio < adult_ratio * 0.67:
        return "child"
    return "unknown"


def build_cabin(faces, detections, width: int, height: int, seat_cfgs, adult_ratio: float) -> CabinState:
    """Assign face boxes and object detections to seats. ``faces`` are (x1, y1, x2, y2).

    ``detections`` are (bbox, score, label). Belt is always Unknown.
    """
    width = max(1, int(width))
    height = max(1, int(height))
    by_seat: dict[str, list] = {str(s.get("name")): [] for s in seat_cfgs}
    kinds: list[str] = []
    face_boxes: list[tuple] = []
    for raw in faces:
        box = tuple(int(v) for v in raw[:4])
        face_boxes.append(box)
        cx = (box[0] + box[2]) / 2.0 / width
        cy = (box[1] + box[3]) / 2.0 / height
        seat_name = _seat_at(cx, cy, seat_cfgs)
        kind = _classify(box, height, seat_name, adult_ratio)
        kinds.append(kind)
        if seat_name in by_seat:
            by_seat[seat_name].append((kind, box))

    seats: list[SeatState] = []
    for spec in seat_cfgs:
        name = str(spec.get("name") or "")
        found = by_seat.get(name) or []
        if not found:
            seats.append(SeatState(name=name, occupant="empty", belt="Unknown"))
            continue
        kind, box = max(found, key=lambda item: _RANK.get(item[0], 0))
        seats.append(SeatState(name=name, occupant=kind, belt="Unknown", face_bbox=box))

    objects: list[CabinObject] = []
    for raw, score, label in detections:
        box = tuple(float(v) for v in raw[:4])
        cx = (box[0] + box[2]) / 2.0 / width
        cy = (box[1] + box[3]) / 2.0 / height
        objects.append(CabinObject(label=str(label), bbox=box, score=float(score), seat=_seat_at(cx, cy, seat_cfgs)))

    adults = sum(1 for kind in kinds if kind == "adult")
    children = sum(1 for kind in kinds if kind == "child")
    child_in_rear = any(s.occupant == "child" for s in seats)
    return CabinState(
        faces=face_boxes, face_kinds=kinds, occupants=len(face_boxes), adults=adults, children=children,
        seats=seats, objects=objects, child_present=child_in_rear and adults == 0, adult_present=adults > 0,
    )


def seat_rows(seats) -> list[dict]:
    return [{"name": s.name, "occupant": s.occupant, "belt": s.belt} for s in seats]


def object_rows(objects) -> list[dict]:
    return [{"label": o.label, "seat": o.seat, "score": round(float(o.score), 3)} for o in objects]


def status_body(device_id: str, cabin: CabinState, state: str, queued: int) -> dict:
    return {
        "device_id": device_id,
        "state": state,
        "timestamp": time.time(),
        "occupants": cabin.occupants,
        "adults": cabin.adults,
        "children": cabin.children,
        "child_present": cabin.child_present,
        "adult_present": cabin.adult_present,
        "seats": seat_rows(cabin.seats),
        "objects": object_rows(cabin.objects),
        "camera_fps": round(float(cabin.fps), 1),
        "connected": cabin.connected,
        "queued_alerts": queued,
    }


class CabinRules:
    """Hold times and rising-edge alerts. The first occupancy sample is a baseline."""

    def __init__(self, child_s: float, object_s: float, occupancy_s: float) -> None:
        self.child = _Hold(child_s)
        self.object_s = float(object_s)
        self.occupancy_s = float(occupancy_s)
        self._child_latched = False
        self._object_holds: dict[str, _Hold] = {}
        self._object_latched: set[str] = set()
        self._stable: dict[str, str] = {}
        self._pending: dict[str, tuple[str, float]] = {}

    def step(self, cabin: CabinState, now: float) -> list[tuple]:
        """Return (type, reason, description, confidence) for alerts that just became true."""
        out = []
        active = bool(cabin.child_present)
        self.child.update(active, now)
        ready = self.child.ready(now)
        if ready and not self._child_latched:
            names = [s.name.replace("_", " ") for s in cabin.seats if s.occupant == "child"]
            where = ", ".join(names) or "rear seat"
            out.append((A.CHILD_PRESENT, "child_no_adult",
                        f"Child present in {where}, no adult visible", 0.6))
        self._child_latched = ready
        if not active:
            self._child_latched = False

        live = set()
        for obj in cabin.objects:
            if not obj.seat:
                continue
            key = f"{obj.seat}:{obj.label}"
            live.add(key)
            hold = self._object_holds.setdefault(key, _Hold(self.object_s))
            hold.update(True, now)
            if hold.ready(now) and key not in self._object_latched:
                self._object_latched.add(key)
                seat = obj.seat.replace("_", " ")
                out.append((A.OBJECT_ON_SEAT, key, f"{obj.label} on {seat}", max(0.4, min(0.95, obj.score))))
        for key, hold in self._object_holds.items():
            if key not in live:
                hold.update(False, now)
                self._object_latched.discard(key)

        for seat in cabin.seats:
            prev = self._stable.get(seat.name)
            if prev is None:
                self._stable[seat.name] = seat.occupant
                continue
            if seat.occupant == prev:
                self._pending.pop(seat.name, None)
                continue
            started = self._pending.get(seat.name)
            if started is None or started[0] != seat.occupant:
                self._pending[seat.name] = (seat.occupant, now)
                continue
            if now - started[1] >= self.occupancy_s:
                out.append((A.OCCUPANCY_CHANGE, f"{seat.name}:{prev}->{seat.occupant}",
                            f"{seat.name.replace('_', ' ')} {prev} to {seat.occupant}", 0.8))
                self._stable[seat.name] = seat.occupant
                self._pending.pop(seat.name, None)
        return out


class CabinMonitor(threading.Thread):
    def __init__(self, cfg, camera, shared, bus: A.AlertBus) -> None:
        super().__init__(name="cms-monitor", daemon=True)
        self.cfg = cfg
        self.cms = cfg.cms
        self.camera = camera
        self.shared = shared
        self.bus = bus
        self.period = float(self.cms.interval_s)
        self._halt = threading.Event()
        self.rules = CabinRules(self.cms.child_hold_s, self.cms.object_hold_s, self.cms.occupancy_hold_s)
        self.faces = self._face_detector()
        self.objects = self._object_detector()
        log.info("cms monitor every %.2fs, seats=%d, faces=%s, objects=%s",
                 self.period, len(self.cms.seats), bool(self.faces), bool(self.objects))

    def _face_detector(self):
        path = model_path(self.cfg.realtime.face_detector)
        if not Path(path).exists():
            log.warning("face detector %s missing; cabin occupancy disabled", path.name)
            return None
        try:
            return FaceDetector(path, int(self.cfg.realtime.face_detector_input), score_threshold=0.55)
        except Exception as exc:
            log.warning("face detector unavailable: %s", exc)
            return None

    def _object_detector(self):
        name = self.cfg.realtime.phone_detector
        if not name:
            return None
        path = model_path(name)
        if not Path(path).exists():
            log.warning("object detector %s missing; cabin objects disabled", path.name)
            return None
        try:
            return PhoneDetector(path, None, score_threshold=float(self.cfg.realtime.phone_detector_threshold),
                                 allow_trt=False, allow_cuda=False)
        except Exception as exc:
            log.warning("cabin object detector unavailable: %s", exc)
            return None

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        seq = 0
        while not self._halt.is_set():
            frame = self.camera.wait_next(seq, timeout=1.0)
            if frame is None:
                previous = self.shared.passenger
                self.shared.passenger = CabinState(
                    frame=previous.frame, frame_ts=previous.frame_ts, faces=previous.faces,
                    face_kinds=previous.face_kinds, occupants=previous.occupants, adults=previous.adults,
                    children=previous.children, seats=previous.seats, objects=previous.objects,
                    child_present=previous.child_present, adult_present=previous.adult_present,
                    fps=0.0, connected=False)
                continue
            seq = frame.seq
            try:
                self.process(frame.image, frame.timestamp, self.camera.stats.fps, connected=True)
            except Exception:
                log.exception("cms frame failed")
            self._halt.wait(self.period)

    def process(self, image: np.ndarray, timestamp: float, fps: float, connected: bool = True) -> CabinState:
        height, width = image.shape[:2]
        face_boxes = []
        if self.faces is not None:
            face_boxes = [f.bbox for f in self.faces.detect(image)]
        detections = []
        if self.objects is not None:
            for box, score, label in self.objects.detect(image, CMS_CLASSES):
                detections.append((box, score, LABELS.get(label, str(label))))
        state = build_cabin(face_boxes, detections, width, height, list(self.cms.seats),
                            float(self.cms.adult_face_ratio))
        state.frame = image
        state.frame_ts = timestamp
        state.fps = float(fps)
        state.connected = connected
        self.shared.passenger = state
        if connected:
            for kind, reason, text, conf in self.rules.step(state, time.monotonic()):
                self._emit(kind, reason, conf, text, image, state)
        return state

    def _emit(self, kind: str, reason: str, conf: float, text: str, image, state: CabinState) -> None:
        path = data_path("snapshots", f"{time.strftime('%Y%m%d_%H%M%S')}_{kind.lower()}.jpg")
        try:
            cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            snapshot = str(path)
        except Exception:
            log.exception("cms snapshot failed")
            snapshot = None
        self.bus.emit(A.Alert(
            type=kind, category=A.VISION, channel="cms", confidence=conf, reason=reason,
            description=text, snapshot_path=snapshot,
            extra={"seats": seat_rows(state.seats), "objects": object_rows(state.objects)}),
            cooldown_s=30.0)
