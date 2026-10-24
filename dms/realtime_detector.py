"""Real-time driver channel (spec Thread 2).

Per frame (640x480, target >= 15 FPS on Orin Nano 15 W):
  YuNet face box -> face landmarks -> EAR + solvePnP head pose
  -> 8-head MobileNetV3 driver model on the face crop (TensorRT FP16):
     drowsiness, gaze, yawn, emotion, eye_state, distraction, activity, age
Every Nth frame: NanoDet phone/cup detector, palm (hand) detector, seatbelt check.
Every ~2 s: SFace driver identification.

Temporal rules turn per-frame signals into alerts:
  EAR < threshold for > 1.5 s            -> DISTRACTION (eyes_closed)
  |head yaw| > 30 deg for > 2 s          -> DISTRACTION (head_turned)
  gaze off-road / head down for > 2 s    -> DISTRACTION (gaze_away / looking_down)
  phone in hand or hand at ear for > 1 s -> DISTRACTION (phone_usage)
  drowsy head, cigarette, drink, no belt, no driver -> their own alert types
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


class Hold:
    """Tracks how long a condition has been continuously true."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.since: float | None = None

    def update(self, active: bool, now: float) -> float:
        if active:
            if self.since is None:
                self.since = now
            return now - self.since
        self.since = None
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
        self.timer = Timer()

    def submit(self, image: np.ndarray, ts: float) -> None:
        with self._cond:
            self._frame = (image, ts)
            self._cond.notify()

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
            self.objects, self.hands, self.ts = objects, hands, ts
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

        self.holds = {
            "eyes_closed": Hold(rt.eyes_closed_s),
            "head_turned": Hold(rt.head_turn_s),
            "gaze_away": Hold(rt.head_turn_s),
            "looking_down": Hold(rt.head_turn_s),
            "phone": Hold(rt.phone_s),
            "drowsy": Hold(rt.drowsy_s),
            "smoking": Hold(rt.smoking_s),
            "drinking": Hold(rt.drinking_s),
            "seatbelt_off": Hold(rt.seatbelt_off_s),
            "no_face": Hold(rt.no_face_s),
        }
        self.ear_history: deque = deque(maxlen=900)  # ~30 s of open-eye EAR samples
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
            if self.frame_index % 5 == 0 and face:  # belt state changes slowly
                ds.seatbelt = self.seatbelt.update(image, face.bbox)
            else:
                ds.seatbelt = self.seatbelt.state()
        marks.append(("seatbelt", time.perf_counter()))
        ds.identity = self.identity
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
        model_away = conf >= 0.55 and label not in ("Middle-Left", "Middle-Right")
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
            if now - worker.ts < rt.object_hold_s:
                objects, hands = worker.objects, worker.hands
        ds.hands = hands

        face = ds.face_bbox
        phone_box, phone_det = None, 0.0
        drink_near = False
        if face:
            x1, y1, x2, y2 = face
            fw, fh = x2 - x1, y2 - y1
            reach = (x1 - 2.2 * fw, y1 - 0.3 * fh, x2 + 2.2 * fw, y2 + 2.6 * fh)  # driver's arm reach
            mouth = (x1 + 0.1 * fw, y1 + 0.55 * fh, x2 - 0.1 * fw, y2 + 0.45 * fh)
            for box, score, cls in objects:
                if overlap(box, reach) < 0.5:
                    continue  # belongs to a passenger / background
                if cls == COCO_PHONE and score > phone_det:
                    phone_box, phone_det = box, score
                elif cls in DRINK_CLASSES and overlap(box, mouth) > 0.2:
                    drink_near = True
            ear_zones = ((x1 - 0.7 * fw, y1 + 0.1 * fh, x1 + 0.2 * fw, y2 + 0.35 * fh),
                         (x2 - 0.2 * fw, y1 + 0.1 * fh, x2 + 0.7 * fw, y2 + 0.35 * fh))
            hand_at_ear = any(overlap(hb[:4], zone) > 0.3 for hb in hands for zone in ear_zones)
            hand_on_face = any(overlap(hb[:4], face) > 0.3 for hb in hands)
            phone_at_ear = phone_box is not None and any(overlap(phone_box, z) > 0.3 for z in ear_zones)
            chest = (x1 - 1.5 * fw, y2 - 0.2 * fh, x2 + 1.5 * fw, y2 + 2.0 * fh)
            hand_on_phone = phone_box is not None and (any(overlap(phone_box, hb[:4]) > 0.1 for hb in hands)
                                                       or overlap(phone_box, chest) > 0.6)
            if phone_box is not None and (phone_at_ear or hand_on_phone):
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
            p_model = self.model.p(self._last_preds, "activity", "Phone Use") if self._last_preds else 0.0
            score = max(phone_det, p_model, 0.7 if hand_at_ear else 0.0)
            ds.phone = (phone_det >= rt.phone_detector_threshold or p_model >= rt.phone_model_threshold or
                        hand_at_ear or phone_at_ear)
            ds.phone_conf = score
            if ds.phone and phone_box is None:
                # no detector box (e.g. phone hidden in the hand): box the hand nearest the face
                near = [hb for hb in hands if overlap(hb[:4], reach) > 0.5]
                if near:
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    hb = min(near, key=lambda b: abs((b[0] + b[2]) / 2 - cx) + abs((b[1] + b[3]) / 2 - cy))
                    phone_box = hb[:4]
            ds.phone_box = phone_box if ds.phone else None
            act = ds.activity
            ds.smoking = act == "Cigarette" and ds.activity_conf >= rt.activity_threshold
            ds.drinking = drink_near or (act == "Drinking" and ds.activity_conf >= rt.activity_threshold)

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
        eyes_closed = face and (ds.ear is not None and ds.ear < ds.ear_threshold or
                                ds.ear is None and ds.eye_state == "Closed")
        H["eyes_closed"].update(eyes_closed, now)
        H["head_turned"].update(face and ds.yaw is not None and abs(ds.yaw) > rt.yaw_threshold_deg, now)
        H["gaze_away"].update(face and ds.gaze_away and ds.gaze in ("Left", "Right", "Up"), now)
        H["looking_down"].update(face and ds.gaze == "Down", now)
        H["phone"].update(face and ds.phone, now)
        H["drowsy"].update(face and ds.drowsy, now)
        H["smoking"].update(face and ds.smoking, now)
        H["drinking"].update(face and ds.drinking, now)
        H["seatbelt_off"].update(ds.seatbelt == "Off", now)
        H["no_face"].update(not face, now)
        ds.eyes_closed_s = H["eyes_closed"].held(now)
        ds.head_turned_s = H["head_turned"].held(now)

        reasons = []
        if H["phone"].fired(now):
            reasons.append(("phone_usage", ds.phone_conf, "Phone usage while driving"))
        if H["eyes_closed"].fired(now):
            conf = 1.0 - (ds.ear / ds.ear_threshold if ds.ear else 0.0) * 0.5
            reasons.append(("eyes_closed", conf, f"Eyes closed for {ds.eyes_closed_s:.1f}s"))
        if H["head_turned"].fired(now):
            reasons.append(("head_turned", min(1.0, abs(ds.yaw or 0) / 45.0),
                            f"Head turned {abs(ds.yaw or 0):.0f}° for {ds.head_turned_s:.1f}s"))
        if H["gaze_away"].fired(now):
            reasons.append(("gaze_away", 0.75, f"Eyes off road ({ds.gaze.lower()})"))
        if H["looking_down"].fired(now):
            reasons.append(("looking_down", 0.7, "Looking down, eyes off road"))
        ds.distracted = bool(reasons)
        ds.distraction_reasons = [r[0] for r in reasons]
        for reason, conf, text in reasons:
            self._emit(A.DISTRACTION, reason, conf, text, ds, image)

        if H["drowsy"].fired(now):
            self._emit(A.DROWSINESS, "drowsy", self.model.p(self._last_preds, "drowsiness", "Drowsy")
                       if self._last_preds else 0.7, "Drowsiness signs detected", ds, image)
        if H["smoking"].fired(now):
            self._emit(A.SMOKING, "cigarette", ds.activity_conf, "Smoking while driving", ds, image)
        if H["drinking"].fired(now):
            self._emit(A.DRINKING, "drinking", max(ds.activity_conf, 0.6), "Drinking while driving", ds, image)
        if H["seatbelt_off"].fired(now):
            self._emit(A.SEATBELT_OFF, "seatbelt", 0.7, "Seatbelt not detected", ds, image, cooldown=60.0)
        if H["no_face"].fired(now) and self.camera.stats.connected:
            self._emit(A.DRIVER_ABSENT, "no_face", 0.9, "Driver face not visible", ds, image, cooldown=30.0)

        if ds.distracted:
            ds.overlay_alert = "DISTRACTION DETECTED"
        elif H["drowsy"].fired(now) or H["eyes_closed"].active and ds.eyes_closed_s > 1.0:
            ds.overlay_alert = "DROWSINESS DETECTED"
        elif H["smoking"].fired(now):
            ds.overlay_alert = "SMOKING DETECTED"
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
        last = self.bus._last.get((alert.type, alert.reason))
        return last is None or time.monotonic() - last >= cooldown


def _r(value, digits: int = 1):
    return None if value is None else round(float(value), digits)
