"""Real-time driver channel (spec Thread 2).

Per frame (640x480, target >= 15 FPS on Orin Nano 15 W):
  YuNet face box -> face landmarks -> EAR + solvePnP head pose
  -> 8-head MobileNetV3 driver model on the face crop (TensorRT FP16):
     drowsiness, gaze, yawn, emotion, eye_state, distraction, activity, age
Every Nth frame: NanoDet phone/cup detector, palm (hand) detector, seatbelt check.
Every ~2 s: SFace driver identification.

Temporal rules turn per-frame signals into one alert per episode:
  measured EAR below threshold for > 2.5 s                         -> eyes closed
  gaze off road for > 3 s                                          -> one distraction card
  phone box in reach, hand on it or phone at ear, for > 2 s       -> phone
  eyes closed much of the last minute, or a yawn the model agrees with -> drowsiness
  cup or bottle at the mouth with a hand on it for > 2 s          -> drinking
  cigarette score with a hand at the mouth for > 2 s              -> smoking
  seatbelt Off across a full vote window, then > 10 s             -> seatbelt
  face missing > 5 s on a usable frame                             -> driver absent
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from . import alerts as A
from .config import data_path, model_path
from .driver_id import DriverIdentifier, DriverRoster
from .face import (FaceDetector, HeadPose, MeshTopology, create_landmarker, eye_aspect_ratio, eye_crop,
                   geometric_yaw, iris_ratio)
from .inference import Timer, create_session
from .logutil import StateLog
from .objects import COCO_PHONE, DRINK_CLASSES, HandDetector, PhoneDetector, overlap
from .seatbelt import SeatbeltDetector
from .state import DriverState, Identity

log = logging.getLogger("dms.realtime")

IMG_SIZE = 224
FACE_MARGIN = 0.20
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 8-head output contract of edge_driver_model_distilled (MobileNetV3, from the reference repo).
TASKS = {
    "drowsiness": ("Alert", "Drowsy"),
    "gaze": ("Bottom-Left", "Middle-Left", "Top-Left", "Bottom-Right", "Middle-Right", "Top-Right",
             "Top-Center", "Bottom-Center"),
    "yawn": ("No Yawn", "Yawning"),
    "emotion": ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"),
    "eye_state": ("Open", "Closed"),
    "distraction": ("Safe", "Distracted"),
    "activity": ("Phone Use", "Drinking", "Cigarette", "Seatbelt", "None"),
    "age": ("Minor", "Young Adult", "Adult"),
}
OUTPUT_ORDER = list(TASKS)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class DriverModel:
    """8-head driver-state classifier with probability smoothing."""

    def __init__(self, engine: Path | None, onnx: Path | None, smooth_window: int = 5) -> None:
        self.session = create_session(onnx, engine)
        self.input = self.session.inputs[0]
        if sorted(self.session.outputs) != sorted(OUTPUT_ORDER):
            raise RuntimeError(f"unexpected driver model outputs {self.session.outputs}")
        self.history = {name: deque(maxlen=smooth_window) for name in OUTPUT_ORDER}
        provider = getattr(self.session, "provider", "TensorRT")
        self.framework = {"TensorRT": "TensorRT FP16" if getattr(self.session, "fp16", True) else "TensorRT",
                          "ORT-TensorRT": "ONNX Runtime + TensorRT", "ORT-CUDA": "ONNX Runtime (CUDA)",
                          "ORT-CPU": "ONNX Runtime"}.get(provider, provider)
        self.timer = Timer()

    @staticmethod
    def crop(frame: np.ndarray, bbox) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        mx, my = FACE_MARGIN * (x2 - x1), FACE_MARGIN * (y2 - y1)
        xa, ya = max(0, int(x1 - mx)), max(0, int(y1 - my))
        xb, yb = min(w, int(x2 + mx)), min(h, int(y2 + my))
        face = frame[ya:yb, xa:xb]
        if face.size == 0:
            face = frame
        return cv2.resize(face, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    def infer(self, face_bgr: np.ndarray) -> dict:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]
        with self.timer.measure():
            out = self.session.run({self.input: np.ascontiguousarray(tensor, dtype=np.float32)})
        preds = {}
        for name in OUTPUT_ORDER:
            probs = softmax(np.asarray(out[name], dtype=np.float32).reshape(-1))
            self.history[name].append(probs)
            avg = np.mean(self.history[name], axis=0)
            idx = int(np.argmax(avg))
            preds[name] = {"label": TASKS[name][idx], "conf": float(avg[idx]), "probs": avg}
        return preds

    def p(self, preds: dict, task: str, label: str) -> float:
        return float(preds[task]["probs"][TASKS[task].index(label)])

    def reset(self) -> None:
        for h in self.history.values():
            h.clear()


# A phone box must sit mostly inside the driver's reach, and either a hand must cover
# part of that box or the box must sit in an ear zone. The activity head never creates a box.
_PHONE_IN_REACH = 0.5
_PHONE_IN_HAND = 0.2
_PHONE_AT_EAR = 0.3
_HAND_AT_MOUTH = 0.3
# Share of the last minute the eyes were measured closed. Normal blinking stays under this.
_PERCLOS_ALERT = 0.15
_PERCLOS_WINDOW_S = 60.0
_PERCLOS_MIN_SPAN_S = 20.0
# Gaze-head labels this weak used to count as eyes off the road.
_MODEL_GAZE_AWAY = 0.70


def driver_reach(face) -> tuple:
    x1, y1, x2, y2 = face
    fw, fh = x2 - x1, y2 - y1
    return (x1 - 2.2 * fw, y1 - 0.3 * fh, x2 + 2.2 * fw, y2 + 2.6 * fh)


def ear_zones(face) -> tuple:
    x1, y1, x2, y2 = face
    fw, fh = x2 - x1, y2 - y1
    return ((x1 - 0.7 * fw, y1 + 0.1 * fh, x1 + 0.2 * fw, y2 + 0.35 * fh),
            (x2 - 0.2 * fw, y1 + 0.1 * fh, x2 + 0.7 * fw, y2 + 0.35 * fh))


def assess_phone(face, objects, hands, activity_phone: float,
                 detector_threshold: float, model_threshold: float):
    """Return (using, confidence, box, reason).

    Using a phone requires a detector box inside the driver's reach, plus a hand on
    that box or the box at the ear. ``activity_phone`` can raise confidence once a box
    is confirmed. A hand at the ear, or a high activity score, does not create a phone.
    ``reason`` is ``ok``, ``no_face``, ``no_box``, or ``not_in_hand``.
    """
    if not face:
        return False, 0.0, None, "no_face"
    reach = driver_reach(face)
    phone_box, phone_score = None, 0.0
    for box, score, cls in objects:
        if cls != COCO_PHONE or score < detector_threshold or score <= phone_score:
            continue
        if overlap(box, reach) < _PHONE_IN_REACH:
            continue
        phone_box, phone_score = box, float(score)
    if phone_box is None:
        return False, 0.0, None, "no_box"
    zones = ear_zones(face)
    at_ear = any(overlap(phone_box, zone) > _PHONE_AT_EAR for zone in zones)
    in_hand = any(overlap(phone_box, hand[:4]) > _PHONE_IN_HAND for hand in hands)
    if not (at_ear or in_hand):
        return False, 0.0, None, "not_in_hand"
    confidence = phone_score
    if activity_phone >= model_threshold:
        confidence = max(confidence, float(activity_phone))
    return True, float(min(1.0, confidence)), phone_box, "ok"


def mouth_zone(face) -> tuple:
    x1, y1, x2, y2 = face
    fw, fh = x2 - x1, y2 - y1
    return (x1 + 0.1 * fw, y1 + 0.55 * fh, x2 - 0.1 * fw, y2 + 0.45 * fh)


def eyes_are_closed(face_found: bool, ear, threshold) -> bool:
    """A measured eyelid closure. A model label with no EAR does not count."""
    return bool(face_found and ear is not None and threshold is not None and ear < threshold)


def measured_closure(face_found: bool, ear, threshold, model_p_closed) -> bool | None:
    """One PERCLOS sample, or None when this frame has no measurement.

    EAR wins when landmarks fitted. If the mesh drops (common once the eyes shut)
    a high eye-state score still counts, so the minute-long fraction is not built
    only from the frames where the eyes were open enough to fit. A weak score is
    left out rather than recorded as open.
    """
    if not face_found:
        return None
    if ear is not None and threshold is not None:
        return bool(ear < threshold)
    if model_p_closed is not None and float(model_p_closed) >= 0.8:
        return True
    return None


def choose_distraction(eyes_closed: bool, phone: bool, looking_down: bool, gaze_away: bool) -> str | None:
    """One distraction episode, one reason. A head turn by itself is not an alert."""
    if eyes_closed:
        return "eyes_closed"
    if phone:
        return "phone_usage"
    if looking_down:
        return "looking_down"
    if gaze_away:
        return "gaze_away"
    return None


def perclos(samples, now: float, window_s: float = _PERCLOS_WINDOW_S,
            min_span_s: float = _PERCLOS_MIN_SPAN_S) -> float | None:
    """Fraction of measured frames in the window whose eyes were closed."""
    recent = [(t, closed) for t, closed in samples if now - window_s <= t <= now]
    if len(recent) < 2 or recent[-1][0] - recent[0][0] < min_span_s:
        return None
    return sum(1 for _, closed in recent if closed) / len(recent)


def drowsy_evidence(closure, yawning: bool, model_drowsy: bool, minimum: float = _PERCLOS_ALERT) -> bool:
    """Drowsiness is a minute of eye closure, or a yawn the drowsiness head agrees with."""
    if closure is not None and closure >= minimum:
        return True
    return bool(yawning and model_drowsy)


def assess_drink(face, objects, hands, activity_drink: float,
                 detector_threshold: float, model_threshold: float):
    """Return (drinking, confidence, reason). A cup score or the activity head cannot create it alone."""
    if not face:
        return False, 0.0, "no_face"
    reach = driver_reach(face)
    mouth = mouth_zone(face)
    box, score = None, 0.0
    for candidate, candidate_score, cls in objects:
        if cls not in DRINK_CLASSES or candidate_score < detector_threshold or candidate_score <= score:
            continue
        if overlap(candidate, reach) < _PHONE_IN_REACH or overlap(candidate, mouth) <= 0.2:
            continue
        box, score = candidate, float(candidate_score)
    if box is None:
        return False, 0.0, "no_box"
    if not any(overlap(box, hand[:4]) > _PHONE_IN_HAND for hand in hands):
        return False, 0.0, "not_in_hand"
    confidence = score
    if activity_drink >= model_threshold:
        confidence = max(confidence, float(activity_drink))
    return True, float(min(1.0, confidence)), "ok"


def assess_smoke(face, hands, activity_label: str, activity_conf: float, model_threshold: float):
    """Return (smoking, confidence, reason). The cigarette label needs a hand at the mouth."""
    if not face:
        return False, 0.0, "no_face"
    if activity_label != "Cigarette" or activity_conf < model_threshold:
        return False, 0.0, "not_cigarette"
    mouth = mouth_zone(face)
    if not any(overlap(hand[:4], mouth) > _HAND_AT_MOUTH for hand in hands):
        return False, 0.0, "no_hand"
    return True, float(min(1.0, activity_conf)), "ok"


def view_blocked(image) -> bool:
    """A flat black or flat white frame is a covered or blinded camera, not an empty seat."""
    if image is None or getattr(image, "size", 0) == 0:
        return False
    sample = image[::8, ::8]
    if sample.size == 0:
        return False
    mean = float(sample.mean())
    std = float(sample.std())
    return std < 4.0 and (mean < 12.0 or mean > 248.0)


class Hold:
    """Tracks how long a condition has been continuously true.

    ``gap_s`` keeps the timer across a short dropout. Phone, drink, and smoke boxes
    arrive from a slower thread and the hand overlap flickers; without this, a single
    missed cycle resets a 2 s rule and the alert never fires.
    """

    def __init__(self, seconds: float, gap_s: float = 0.0) -> None:
        self.seconds = seconds
        self.gap_s = gap_s
        self.since: float | None = None
        self._gap_since: float | None = None

    def update(self, active: bool, now: float) -> float:
        if active:
            self._gap_since = None
            if self.since is None:
                self.since = now
            return now - self.since
        if self.since is None or self.gap_s <= 0:
            self.since = None
            self._gap_since = None
            return 0.0
        if self._gap_since is None:
            self._gap_since = now
        if now - self._gap_since <= self.gap_s:
            return now - self.since
        self.since = None
        self._gap_since = None
        return 0.0

    def held(self, now: float) -> float:
        return 0.0 if self.since is None else now - self.since

    @property
    def active(self) -> bool:
        return self.since is not None

    def fired(self, now: float) -> bool:
        return self.since is not None and now - self.since >= self.seconds


class SnapshotWriter:
    def __init__(self) -> None:
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="snapshot")

    def save(self, frame: np.ndarray, ds: DriverState, alert: A.Alert) -> str:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(alert.timestamp))
        path = data_path("snapshots", f"{stamp}_{alert.type.lower()}_{alert.id[:6]}.jpg")
        image = frame.copy()

        def _write():
            if ds.face_bbox:
                x1, y1, x2, y2 = map(int, ds.face_bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), (80, 220, 40), 2)
            if ds.phone_box:
                x1, y1, x2, y2 = map(int, ds.phone_box)
                cv2.rectangle(image, (x1, y1), (x2, y2), (40, 40, 230), 2)
            label = f"{alert.type} {alert.reason} {alert.confidence:.2f}"
            cv2.rectangle(image, (0, 0), (image.shape[1], 24), (20, 20, 20), -1)
            cv2.putText(image, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 85])

        self.pool.submit(_write)
        return str(path)

    def close(self) -> None:
        self.pool.shutdown(wait=True)


class ObjectWorker(threading.Thread):
    """Phone/cup and hand detectors on their own thread.

    They are the most expensive part of the driver channel on a CPU fallback, so the
    real-time loop (EAR / head pose / 8-head model, which carries the spec's timing rules)
    never waits for them: it hands over its newest frame and reads the latest boxes, which
    are held for ``hold_s``.
    """

    def __init__(self, phone_detector, hand_detector, max_hz: float = 15.0) -> None:
        super().__init__(name="object-detector", daemon=True)
        self.phone_detector = phone_detector
        self.hand_detector = hand_detector
        self.period = 1.0 / max(1.0, float(max_hz))
        self._cond = threading.Condition()
        self._frame = None
        self._halt = threading.Event()
        self.objects: list = []
        self.hands: list = []
        self.ts = 0.0  # capture timestamp of the frame the boxes belong to
        self._ready = False
        self.timer = Timer()

    def submit(self, image: np.ndarray, ts: float) -> None:
        with self._cond:
            self._frame = (image, ts)
            self._cond.notify()

    def snapshot(self) -> tuple:
        """Boxes, hands, capture time, and whether a result has been published."""
        with self._cond:
            return list(self.objects), list(self.hands), self.ts, self._ready

    def stop(self) -> None:
        self._halt.set()
        with self._cond:
            self._cond.notify()

    def run(self) -> None:
        while not self._halt.is_set():
            with self._cond:
                while self._frame is None and not self._halt.is_set():
                    self._cond.wait(0.5)
                if self._halt.is_set():
                    return
                image, ts = self._frame
                self._frame = None
            started = time.monotonic()
            with self.timer.measure():
                try:
                    objects = self.phone_detector.detect(image) if self.phone_detector else []
                    hands = self.hand_detector.detect(image) if self.hand_detector else []
                except Exception:
                    log.exception("object detector failed")
                    objects, hands = [], []
            with self._cond:
                self.objects, self.hands, self.ts = list(objects), list(hands), ts
                self._ready = True
            rest = self.period - (time.monotonic() - started)
            if rest > 0:
                self._halt.wait(rest)


class DriverMonitor(threading.Thread):
    def __init__(self, cfg, camera, shared, bus: A.AlertBus) -> None:
        super().__init__(name="realtime-detector", daemon=True)
        self.cfg = cfg
        rt = cfg.realtime
        self.rt = rt
        self.camera = camera
        self.shared = shared
        self.bus = bus
        self._halt = threading.Event()
        self.snapshots = SnapshotWriter() if rt.snapshot_alerts else None

        self.face_detector = FaceDetector(model_path(rt.face_detector), rt.face_detector_input)
        self.landmarker = create_landmarker(rt.landmarks, model_path(rt.lbf_model), model_path(rt.face_landmarker_task))
        self.model = DriverModel(model_path(rt.driver_model), model_path(rt.driver_model_onnx), rt.smooth_window)
        self.phone_detector = self._optional(PhoneDetector, rt.phone_detector, score_threshold=rt.phone_detector_threshold)
        self.hand_detector = self._optional(HandDetector, rt.hand_detector)
        self.object_worker = None
        if self.phone_detector or self.hand_detector:
            hz = float(rt.object_detector_hz)
            sessions = [d.session for d in (self.phone_detector, self.hand_detector) if d]
            if all(getattr(sess, "provider", "") == "ORT-CPU" for sess in sessions):
                hz = min(hz, 6.0)  # no GPU: keep the CPU for the real-time loop
            self.object_worker = ObjectWorker(self.phone_detector, self.hand_detector, hz)
            self.object_worker.start()
        self.identifier = None
        if Path(model_path(rt.face_recognizer)).exists():
            self.identifier = DriverIdentifier(model_path(rt.face_recognizer), DriverRoster(data_path("drivers", "roster.json")))
        self.seatbelt = SeatbeltDetector(window=30) if rt.seatbelt_check else None
        self.pose = HeadPose()
        self.topology = MeshTopology()

        gap = float(rt.object_hold_s)
        self.holds = {
            "eyes_closed": Hold(rt.eyes_closed_s),
            "head_turned": Hold(rt.head_turn_s),
            "gaze_away": Hold(rt.head_turn_s),
            "looking_down": Hold(rt.head_turn_s),
            "phone": Hold(rt.phone_s, gap_s=gap),
            "drowsy": Hold(rt.drowsy_s),
            "smoking": Hold(rt.smoking_s, gap_s=gap),
            "drinking": Hold(rt.drinking_s, gap_s=gap),
            "seatbelt_off": Hold(rt.seatbelt_off_s),
            "no_face": Hold(rt.no_face_s),
        }
        self.trace = StateLog(log)
        self._phone_why = ""
        self._drink_why = ""
        self._smoke_why = ""
        self.ear_history: deque = deque(maxlen=900)  # ~30 s of open-eye EAR samples
        self.eye_closure: deque = deque(maxlen=1800)  # ~60 s of measured open/closed eyes
        self._drink_conf = 0.0
        self._smoke_conf = 0.0
        self.blinks: deque = deque()
        self._eye_closed_since: float | None = None
        self.frame_index = 0
        self._iris: float | None = None
        self._face_row = None
        self.last_identify = 0.0
        self.last_face_ts = 0.0
        self.identity = Identity()
        self.fps_times: deque = deque(maxlen=60)
        self.pipeline_timer = Timer()
        self.stage_ms: dict[str, float] = {}
        self.started = time.monotonic()
        log.info("driver model on %s; landmarks: %s; phone detector: %s; hand detector: %s; driver id: %s",
                 self.model.framework, getattr(self.landmarker, "label", "none"),
                 bool(self.phone_detector), bool(self.hand_detector), bool(self.identifier))

    def _optional(self, cls, name, **kwargs):
        if not name:
            return None
        onnx = model_path(name)
        engine = onnx.with_suffix(".engine")
        if not onnx.exists() and not engine.exists():
            log.warning("%s not found, %s disabled", onnx.name, cls.__name__)
            return None
        try:
            return cls(onnx, engine if engine.exists() else None, **kwargs)
        except Exception as exc:
            log.warning("%s unavailable: %s", cls.__name__, exc)
            return None

    # ------------------------------------------------------------------ thread
    def stop(self) -> None:
        self._halt.set()
        if self.object_worker:
            self.object_worker.stop()

    def run(self) -> None:
        seq = 0
        while not self._halt.is_set():
            frame = self.camera.wait_next(seq, timeout=0.5)
            if frame is None:
                self._no_camera()
                continue
            seq = frame.seq
            try:
                self.process(frame)
            except Exception:
                log.exception("driver frame failed")
                time.sleep(0.05)
        if self.snapshots:
            self.snapshots.close()

    def _no_camera(self) -> None:
        ds = self.shared.driver
        if ds.frame is not None and time.monotonic() - ds.frame_ts > 2.0:
            for key in ds.modules:
                ds.modules[key] = "Off"

    # ------------------------------------------------------------------ per frame
    def process(self, frame) -> DriverState:
        t_start = time.perf_counter()
        rt = self.rt
        image = frame.image
        now = frame.timestamp
        h, w = image.shape[:2]
        self.frame_index += 1
        ds = DriverState(frame=image, frame_ts=now, seq=frame.seq, width=w, height=h,
                         ear_threshold=rt.ear_threshold, framework=self.model.framework)

        marks = [("start", time.perf_counter())]
        faces = self.face_detector.detect(image)
        marks.append(("face", time.perf_counter()))
        face = faces[0] if faces else None
        self._face_row = face.row if face else None
        preds = None
        pts = None
        if face:
            self.last_face_ts = now
            ds.face_found, ds.face_bbox, ds.face_score = True, face.bbox, face.score
            preds = self.model.infer(self.model.crop(image, face.bbox))
            marks.append(("model", time.perf_counter()))
            if self.landmarker is not None:
                pts = self.landmarker.fit(image, face, int(frame.wall_time * 1000))
                marks.append(("landmarks", time.perf_counter()))
        else:
            if now - self.last_face_ts > 1.0:
                self.model.reset()
                self.pose = HeadPose()
            if now - self.last_face_ts > 5.0:
                self.topology.reset()
                if self.identifier:
                    self.identifier.reset()
                self.identity = Identity()

        self._last_preds = preds
        ds.model_ms = self.model.timer.mean
        self._geometry(ds, image, pts, preds, now)
        self._model_heads(ds, preds)
        marks.append(("geometry", time.perf_counter()))
        self._objects(ds, image, now)
        marks.append(("objects", time.perf_counter()))
        self._identify(ds, image, face, now)
        marks.append(("identify", time.perf_counter()))
        if self.seatbelt is not None:
            if self.frame_index % 5 == 0:  # belt state changes slowly; a missing face still counts as a miss
                ds.seatbelt = self.seatbelt.update(image, face.bbox if face else None)
            else:
                ds.seatbelt = self.seatbelt.state()
        marks.append(("seatbelt", time.perf_counter()))
        ds.identity = self.identity
        p_closed = self.model.p(self._last_preds, "eye_state", "Closed") if self._last_preds else None
        sample = measured_closure(ds.face_found, ds.ear, ds.ear_threshold, p_closed)
        if sample is not None:
            self.eye_closure.append((now, sample))
        self._rules(ds, image, now)
        marks.append(("rules", time.perf_counter()))
        for (_, t0), (name, t1) in zip(marks, marks[1:]):
            prev = self.stage_ms.get(name, (t1 - t0) * 1000.0)
            self.stage_ms[name] = 0.9 * prev + 0.1 * (t1 - t0) * 1000.0
        ds.stage_ms = dict(self.stage_ms)

        ds.modules = {
            "Face Tracking": "Active" if ds.face_found else "Searching",
            "Eye Tracking": "Active" if ds.ear is not None or (preds and ds.face_found) else "Lost",
            "Object Detection": "Active" if self.phone_detector else "Off",
            "Hand Tracking": "Active" if self.hand_detector else "Off",
        }
        self.fps_times.append(time.monotonic())
        if len(self.fps_times) > 1:
            ds.fps = (len(self.fps_times) - 1) / max(1e-6, self.fps_times[-1] - self.fps_times[0])
        self.pipeline_timer.add((time.perf_counter() - t_start) * 1000.0)
        ds.pipeline_ms = self.pipeline_timer.mean
        self.shared.driver = ds
        return ds

    # ------------------------------------------------------------------ signals
    def _geometry(self, ds: DriverState, image, pts, preds, now: float) -> None:
        rt = self.rt
        if pts is None or self.landmarker is None:
            return
        schema = self.landmarker.schema
        ds.landmarks = pts
        pose = self.pose.estimate(pts, schema, ds.width, ds.height)
        if pose is not None:
            ds.yaw, ds.pitch, ds.roll = pose
            # Blend with the landmark-geometry yaw: PnP on a generic head model drifts
            # for strongly turned faces, the nose/eye-corner ratio does not.
            ds.yaw = 0.6 * ds.yaw + 0.4 * geometric_yaw(pts, schema)
        else:
            ds.yaw = geometric_yaw(pts, schema)
        ear_l = eye_aspect_ratio(pts, schema.left_eye)
        ear_r = eye_aspect_ratio(pts, schema.right_eye)
        if ds.yaw is not None and abs(ds.yaw) > 25:
            # the far eye is foreshortened; trust the near one
            ear = ear_r if ds.yaw > 0 else ear_l
        else:
            ear = (ear_l + ear_r) / 2.0
        ds.ear = ear
        if preds is not None and preds["eye_state"]["label"] == "Open" and abs(ds.yaw or 0) < 20:
            self.ear_history.append(ear)
        threshold = rt.ear_threshold
        if rt.ear_auto_calibrate and len(self.ear_history) >= 60:
            baseline = float(np.percentile(self.ear_history, 80))
            threshold = float(np.clip(0.72 * baseline, 0.12, rt.ear_threshold))
            open_ref = baseline
        else:
            open_ref = max(rt.ear_threshold / 0.72, 0.28)
        ds.ear_threshold = threshold
        ear_pct = float(np.clip((ear - 0.5 * threshold) / max(1e-3, open_ref - 0.5 * threshold), 0.0, 1.0))
        if preds is not None:
            p_open = self.model.p(preds, "eye_state", "Open")
            ds.eyes_open_pct = 100.0 * (0.5 * ear_pct + 0.5 * p_open)
        else:
            ds.eyes_open_pct = 100.0 * ear_pct
        closed = ear < threshold
        # blink = closure shorter than 0.5 s
        if closed and self._eye_closed_since is None:
            self._eye_closed_since = now
        elif not closed and self._eye_closed_since is not None:
            if now - self._eye_closed_since < 0.5:
                self.blinks.append(now)
            self._eye_closed_since = None
        while self.blinks and now - self.blinks[0] > 60.0:
            self.blinks.popleft()
        elapsed = min(60.0, now - self.started)
        ds.blink_rate = len(self.blinks) * 60.0 / elapsed if elapsed >= 10 else None
        ds.eye_state = "Closed" if closed else "Open"
        if not closed:
            kps = self._face_row[4:8].reshape(2, 2) if self._face_row is not None else None
            iris = iris_ratio(image, pts, schema, kps)
            if iris is not None:
                self._iris = iris if self._iris is None else 0.6 * self._iris + 0.4 * iris
        ds.iris = self._iris
        ds.eye_crop = eye_crop(image, pts, schema)
        mesh_pts = self.topology.points(pts, schema)
        ds.mesh_points = mesh_pts
        ds.mesh_edges = self.topology.update(mesh_pts, ds.yaw)

    def _model_heads(self, ds: DriverState, preds) -> None:
        rt = self.rt
        if preds is None:
            ds.gaze, ds.head_pose = ("--", "--") if not ds.face_found else (ds.gaze, ds.head_pose)
            return
        ds.emotion, ds.emotion_conf = preds["emotion"]["label"], preds["emotion"]["conf"]
        ds.age_group = preds["age"]["label"]
        ds.drowsy = self.model.p(preds, "drowsiness", "Drowsy") >= rt.drowsy_threshold
        ds.yawning = self.model.p(preds, "yawn", "Yawning") >= rt.yawn_threshold
        ds.activity, ds.activity_conf = preds["activity"]["label"], preds["activity"]["conf"]
        if ds.ear is None:  # no landmarks: eye state from the model head
            p_closed = self.model.p(preds, "eye_state", "Closed")
            ds.eye_state = "Closed" if p_closed >= 0.8 else "Open"
            ds.eyes_open_pct = 100.0 * (1.0 - p_closed)

        # Labels are from the driver's point of view (as in the reference HMI): with a
        # non-mirrored camera facing the driver, image-left is the driver's right.
        def side(image_left: bool) -> str:
            return ("Left" if image_left else "Right") if rt.mirror else ("Right" if image_left else "Left")

        yaw, pitch = ds.yaw, ds.pitch
        if yaw is not None:
            if abs(yaw) > 15:
                ds.head_pose = f"Turned {side(yaw > 0)}"
            elif pitch is not None and pitch < -15:
                ds.head_pose = "Looking Down"
            elif pitch is not None and pitch > 15:
                ds.head_pose = "Looking Up"
            else:
                ds.head_pose = "Forward"

        # Gaze = head yaw + eye-in-head angle (iris position); the model's gaze head decides
        # up/down and covers frames without landmarks.
        g = preds["gaze"]
        label, conf = g["label"], g["conf"]
        model_side = "Left" if "Left" in label else "Right" if "Right" in label else None
        vertical = "Up" if label.startswith("Top") else "Down" if label.startswith("Bottom") else ""
        model_away = conf >= _MODEL_GAZE_AWAY and label not in ("Middle-Left", "Middle-Right")
        angle = None
        if yaw is not None:
            eye_term = 0.0 if ds.iris is None else (0.5 - ds.iris) * 150.0  # iris 0.3 -> ~30 deg
            angle = yaw + eye_term
        ds.gaze_angle = angle
        if angle is not None and abs(angle) > rt.gaze_away_deg:
            ds.gaze, ds.gaze_away = side(angle > 0), True
        elif (pitch is not None and pitch < -rt.pitch_down_threshold_deg) or (model_away and vertical == "Down"):
            ds.gaze, ds.gaze_away = "Down", True
        elif model_away and vertical == "Up":
            ds.gaze, ds.gaze_away = "Up", True
        elif angle is None and model_away and model_side:
            ds.gaze, ds.gaze_away = model_side, True
        else:
            ds.gaze, ds.gaze_away = "Forward", False
        # arrow on the eye crop is drawn in image space
        dx = 0.0
        if ds.gaze_away and angle is not None and abs(angle) > rt.gaze_away_deg:
            dx = -1.0 if angle > 0 else 1.0
        elif ds.gaze_away and ds.gaze in ("Left", "Right"):
            dx = -1.0 if (ds.gaze == "Right") != bool(rt.mirror) else 1.0
        dy = {"Down": 1.0, "Up": -1.0}.get(ds.gaze, 0.0)
        ds.gaze_vector = (dx, dy)

    def _objects(self, ds: DriverState, image, now: float) -> None:
        rt = self.rt
        worker = self.object_worker
        objects, hands = [], []
        if worker is not None:
            worker.submit(image, now)
            objects, hands, seen, ready = worker.snapshot()
            if not ready or now - seen >= rt.object_hold_s:
                objects, hands = [], []
        ds.hands = hands

        face = ds.face_bbox
        self._drink_conf = 0.0
        self._smoke_conf = 0.0
        self._phone_why = self._drink_why = self._smoke_why = "no_face"
        if face:
            y2 = face[3]
            fh = y2 - face[1]
            zones = ear_zones(face)
            hand_at_ear = any(overlap(hb[:4], zone) > 0.3 for hb in hands for zone in zones)
            hand_on_face = any(overlap(hb[:4], face) > 0.3 for hb in hands)
            p_model = self.model.p(self._last_preds, "activity", "Phone Use") if self._last_preds else 0.0
            ds.phone, ds.phone_conf, phone_box, self._phone_why = assess_phone(
                face, objects, hands, p_model, rt.phone_detector_threshold, rt.phone_model_threshold)
            ds.phone_box = phone_box
            if ds.phone:
                ds.hand_position = "On Phone"
            elif hand_at_ear:
                ds.hand_position = "Near Ear"
            elif hand_on_face:
                ds.hand_position = "Near Face"
            elif any((hb[1] + hb[3]) / 2 > y2 + 0.45 * fh for hb in hands):
                ds.hand_position = "On Wheel"
            elif hands:
                ds.hand_position = "Free"
            else:
                ds.hand_position = "Not Visible"
            drink_activity = ds.activity_conf if ds.activity == "Drinking" else 0.0
            ds.drinking, self._drink_conf, self._drink_why = assess_drink(
                face, objects, hands, drink_activity, rt.phone_detector_threshold, rt.activity_threshold)
            ds.smoking, self._smoke_conf, self._smoke_why = assess_smoke(
                face, hands, ds.activity, ds.activity_conf, rt.activity_threshold)

    _last_preds = None

    def _identify(self, ds: DriverState, image, face, now: float) -> None:
        if not (self.identifier and face):
            if not self.identifier and face:
                self.identity = Identity(known=False, name="Not enrolled", age=ds.age_group)
            return
        interval = self.rt.identify_every_s if self.identity.known else min(0.5, self.rt.identify_every_s)
        frontal = ds.yaw is None or abs(ds.yaw) < 35
        if now - self.last_identify >= interval and frontal:
            self.last_identify = now
            try:
                self.identity = self.identifier.identify(image, face.row, ds.age_group)
            except cv2.error as exc:
                log.debug("identify failed: %s", exc)

    # ------------------------------------------------------------------ rules
    def _rules(self, ds: DriverState, image, now: float) -> None:
        rt = self.rt
        H = self.holds
        face = ds.face_found
        eyes_closed = eyes_are_closed(face, ds.ear, ds.ear_threshold)
        H["eyes_closed"].update(eyes_closed, now)
        H["head_turned"].update(face and ds.yaw is not None and abs(ds.yaw) > rt.yaw_threshold_deg, now)
        H["gaze_away"].update(face and ds.gaze_away and ds.gaze in ("Left", "Right", "Up"), now)
        H["looking_down"].update(face and ds.gaze == "Down", now)
        H["phone"].update(face and ds.phone, now)
        closure = perclos(self.eye_closure, now)
        H["drowsy"].update(face and drowsy_evidence(closure, ds.yawning, ds.drowsy), now)
        H["smoking"].update(face and ds.smoking, now)
        H["drinking"].update(face and ds.drinking, now)
        H["seatbelt_off"].update(ds.seatbelt == "Off", now)
        blocked = view_blocked(image)
        H["no_face"].update((not face) and not blocked, now)
        ds.eyes_closed_s = H["eyes_closed"].held(now)
        ds.head_turned_s = H["head_turned"].held(now)
        votes = len(self.seatbelt.votes) if self.seatbelt is not None else 0
        self.trace.change("seatbelt", ds.seatbelt, "seatbelt %s votes=%d", ds.seatbelt, votes, level=logging.INFO)
        self.trace.change("phone", self._phone_why, "phone evidence %s conf=%.2f", self._phone_why, ds.phone_conf)
        self.trace.change("drink", self._drink_why, "drink evidence %s conf=%.2f", self._drink_why, self._drink_conf)
        self.trace.change("smoke", self._smoke_why, "smoke evidence %s conf=%.2f", self._smoke_why, self._smoke_conf)
        self.trace.rise("head_turned", H["head_turned"].fired(now),
                        "head turned %.0f deg for %.1fs; recorded, not raised as its own alert",
                        abs(ds.yaw or 0.0), ds.head_turned_s)
        self.trace.rise("view_blocked", bool(blocked and not face),
                        "flat black or white frame; driver-absent timer paused")

        chosen = choose_distraction(
            H["eyes_closed"].fired(now), H["phone"].fired(now),
            H["looking_down"].fired(now), H["gaze_away"].fired(now))
        texts = {
            "eyes_closed": (1.0 - (ds.ear / ds.ear_threshold if ds.ear else 0.0) * 0.5,
                            f"Eyes closed for {ds.eyes_closed_s:.1f}s"),
            "phone_usage": (ds.phone_conf, "Phone usage while driving"),
            "looking_down": (0.7, "Looking down, eyes off road"),
            "gaze_away": (0.75, f"Eyes off road ({ds.gaze.lower()})"),
        }
        ds.distracted = chosen is not None
        ds.distraction_reasons = [chosen] if chosen else []
        self.trace.change("distraction", chosen or "", "distraction %s", chosen or "clear", level=logging.INFO)
        closure_txt = "n/a" if closure is None else f"{closure:.0%}"
        self.trace.change("perclos", closure_txt, "perclos %s", closure_txt)
        if chosen:
            conf, text = texts[chosen]
            self._emit(A.DISTRACTION, chosen, conf, text, ds, image)

        if H["drowsy"].fired(now):
            if closure is not None and closure >= _PERCLOS_ALERT:
                drowsy_text = f"Eyes closed {closure:.0%} of the last minute"
                drowsy_conf = min(1.0, 0.5 + closure)
            else:
                drowsy_text = "Repeated yawning with drowsiness signs"
                drowsy_conf = self.model.p(self._last_preds, "drowsiness", "Drowsy") if self._last_preds else 0.7
            self._emit(A.DROWSINESS, "drowsy", drowsy_conf, drowsy_text, ds, image)
        if H["smoking"].fired(now):
            self._emit(A.SMOKING, "cigarette", self._smoke_conf, "Smoking while driving", ds, image)
        if H["drinking"].fired(now):
            self._emit(A.DRINKING, "drinking", self._drink_conf, "Drinking while driving", ds, image)
        if H["seatbelt_off"].fired(now):
            self._emit(A.SEATBELT_OFF, "seatbelt", 0.7, "Seatbelt not detected", ds, image, cooldown=60.0)
        if H["no_face"].fired(now) and self.camera.stats.connected:
            self._emit(A.DRIVER_ABSENT, "no_face", 0.9, "Driver face not visible", ds, image, cooldown=30.0)

        if ds.distracted and chosen != "eyes_closed":
            ds.overlay_alert = "DISTRACTION DETECTED"
        elif (H["drowsy"].fired(now) or chosen == "eyes_closed"
              or (H["eyes_closed"].active and ds.eyes_closed_s > 1.0)):
            ds.overlay_alert = "DROWSINESS DETECTED"
        elif H["smoking"].fired(now):
            ds.overlay_alert = "SMOKING DETECTED"
        elif H["drinking"].fired(now):
            ds.overlay_alert = "DRINKING DETECTED"
        elif H["seatbelt_off"].fired(now):
            ds.overlay_alert = "SEATBELT NOT DETECTED"
        else:
            ds.overlay_alert = ""

    def _emit(self, kind: str, reason: str, conf: float, text: str, ds: DriverState, image, cooldown=None) -> None:
        alert = A.Alert(type=kind, category=A.VISION, confidence=float(np.clip(conf, 0.0, 1.0)),
                        description=text, reason=reason,
                        extra={"yaw": _r(ds.yaw), "pitch": _r(ds.pitch), "ear": _r(ds.ear, 3),
                               "gaze": ds.gaze, "driver_id": ds.identity.driver_id if ds.identity.known else None})
        cooldown = self.rt.alert_cooldown_s if cooldown is None else cooldown
        # reserve the cooldown slot first so we only write snapshots for alerts that go out
        if self.snapshots is not None and self._would_emit(alert, cooldown):
            alert.snapshot_path = self.snapshots.save(image, ds, alert)
        self.bus.emit(alert, cooldown_s=cooldown)

    def _would_emit(self, alert: A.Alert, cooldown: float) -> bool:
        return self.bus.accepts(alert.type, alert.reason, cooldown)


def _r(value, digits: int = 1):
    return None if value is None else round(float(value), digits)
