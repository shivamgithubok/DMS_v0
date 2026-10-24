"""Face detection, landmarks, Eye Aspect Ratio, head pose and the overlay mesh.

Detector: YuNet (OpenCV FaceDetectorYN, ~1 ms at 320 px on the Orin's A78AE cores).
Landmarks, in order of preference for ``landmarks: auto``:
  1. MediaPipe Face Mesh (478 points, iris-refined) from the legacy ``solutions`` API,
     which ships its model inside the wheel (mediapipe<=0.10.18, the aarch64 pin).
  2. MediaPipe Tasks FaceLandmarker with ``models/face_landmarker.task``.
  3. OpenCV Facemark LBF, 68 points (``models/lbfmodel.yaml``), opencv-contrib.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger("dms.face")


@dataclass
class FaceBox:
    bbox: tuple  # x1, y1, x2, y2 (int, image px)
    score: float
    row: np.ndarray  # raw YuNet row (15 floats) in image coords, for SFace alignment


class FaceDetector:
    def __init__(self, model_path, input_width: int = 320, score_threshold: float = 0.6) -> None:
        self.input_width = int(input_width)
        self.net = cv2.FaceDetectorYN.create(str(model_path), "", (self.input_width, self.input_width),
                                             score_threshold, 0.3, 50)
        self._size = None

    def detect(self, frame: np.ndarray) -> list[FaceBox]:
        h, w = frame.shape[:2]
        scale = self.input_width / float(w)
        small = cv2.resize(frame, (self.input_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
        size = (small.shape[1], small.shape[0])
        if size != self._size:
            self.net.setInputSize(size)
            self._size = size
        _, faces = self.net.detect(small)
        out: list[FaceBox] = []
        if faces is None:
            return out
        for row in faces:
            row = row.astype(np.float32).copy()
            row[:14] /= scale
            x, y, bw, bh = row[:4]
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(w - 1, int(x + bw)), min(h - 1, int(y + bh))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            out.append(FaceBox((x1, y1, x2, y2), float(row[14]), row))
        out.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        return out


# --------------------------------------------------------------------------- landmark schemas
@dataclass(frozen=True)
class Schema:
    name: str
    left_eye: tuple  # 6 indices p1..p6 (image-left eye)
    right_eye: tuple  # 6 indices (image-right eye)
    pnp: tuple  # nose tip, chin, image-left eye outer corner, image-right eye outer corner, mouth L, mouth R
    mesh: tuple  # subset drawn in the overlay mesh
    brows: tuple = ()  # used to extrapolate forehead points (68-pt only)


SCHEMA_68 = Schema(
    name="lbf68",
    left_eye=(36, 37, 38, 39, 40, 41),
    right_eye=(42, 43, 44, 45, 46, 47),
    pnp=(30, 8, 36, 45, 48, 54),
    mesh=tuple(range(68)),
    brows=(17, 19, 21, 22, 24, 26),
)

# Sparse, symmetric subset of the 478-point topology: oval, brows, eyes, nose, lips, cheeks.
_MP_MESH = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
    148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
    168, 6, 197, 195, 5, 4, 1, 98, 327, 2,
    61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91,
    50, 280, 205, 425, 118, 347, 151, 9,
)
SCHEMA_MP = Schema(
    name="mediapipe478",
    left_eye=(33, 160, 158, 133, 153, 144),
    right_eye=(362, 385, 387, 263, 373, 380),
    pnp=(1, 152, 33, 263, 61, 291),
    mesh=_MP_MESH,
)

# Generic head model in mm (x right, y up, z towards the camera), matching SCHEMA.pnp order.
_MODEL_3D = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)


class LbfLandmarks:
    schema = SCHEMA_68
    label = "Facemark LBF (68)"

    def __init__(self, model_path) -> None:
        if not hasattr(cv2, "face"):
            raise RuntimeError("opencv-contrib (cv2.face) is not installed")
        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel(str(model_path))

    def fit(self, frame: np.ndarray, face: FaceBox, ts_ms: int) -> np.ndarray | None:
        x1, y1, x2, y2 = face.bbox
        w, h = x2 - x1, y2 - y1
        # LBF was trained on Haar-style boxes: square, starting a little below the hairline.
        side = max(w, h)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 + 0.06 * h
        box = np.array([[cx - side / 2, cy - side / 2, side, side]], dtype=np.int32)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ok, landmarks = self.facemark.fit(gray, box)
        if not ok or not len(landmarks):
            return None
        return landmarks[0][0].astype(np.float32)


class MediaPipeLandmarks:
    schema = SCHEMA_MP

    def __init__(self, task_path=None) -> None:
        import mediapipe as mp

        self.mp = mp
        self.mesh = None
        self.landmarker = None
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self.mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.label = "MediaPipe Face Mesh (478)"
        elif task_path is not None and task_path.exists():
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(task_path)),
                running_mode=VisionTaskRunningMode.VIDEO, num_faces=1,
                min_face_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.landmarker = FaceLandmarker.create_from_options(options)
            self.label = "MediaPipe FaceLandmarker (478)"
            self._last_ts = -1
        else:
            raise RuntimeError("MediaPipe Face Mesh unavailable (no solutions API and no face_landmarker.task)")

    def fit(self, frame: np.ndarray, face: FaceBox, ts_ms: int) -> np.ndarray | None:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.mesh is not None:
            result = self.mesh.process(rgb)
            if not result.multi_face_landmarks:
                return None
            pts = result.multi_face_landmarks[0].landmark
        else:
            ts_ms = max(int(ts_ms), self._last_ts + 1)
            self._last_ts = ts_ms
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect_for_video(image, ts_ms)
            if not result.face_landmarks:
                return None
            pts = result.face_landmarks[0]
        return np.array([(p.x * w, p.y * h) for p in pts], dtype=np.float32)


def create_landmarker(mode: str, lbf_path, task_path):
    errors = []
    order = {"auto": ("mediapipe", "lbf"), "mediapipe": ("mediapipe",), "lbf": ("lbf",)}.get(mode, ("lbf",))
    for name in order:
        try:
            if name == "mediapipe":
                return MediaPipeLandmarks(task_path)
            return LbfLandmarks(lbf_path)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    log.warning("no landmark backend available (%s); EAR/head pose fall back to the model heads", "; ".join(errors))
    return None


# --------------------------------------------------------------------------- geometry
def eye_aspect_ratio(pts: np.ndarray, idx: tuple) -> float:
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-3:
        return 0.0
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * horizontal))


class HeadPose:
    """solvePnP head pose with temporal seeding for stable yaw/pitch/roll."""

    def __init__(self) -> None:
        self.rvec = None
        self.tvec = None

    def estimate(self, pts: np.ndarray, schema: Schema, width: int, height: int):
        image_pts = np.array([pts[i] for i in schema.pnp], dtype=np.float64)
        focal = float(width)
        camera = np.array([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros((4, 1))
        if self.rvec is not None:
            ok, rvec, tvec = cv2.solvePnP(_MODEL_3D, image_pts, camera, dist, self.rvec.copy(), self.tvec.copy(),
                                          useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        else:
            ok, rvec, tvec = cv2.solvePnP(_MODEL_3D, image_pts, camera, dist, flags=cv2.SOLVEPNP_EPNP)
            if ok:
                ok, rvec, tvec = cv2.solvePnP(_MODEL_3D, image_pts, camera, dist, rvec, tvec,
                                              useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or tvec[2, 0] <= 0:
            self.rvec = self.tvec = None
            return None
        self.rvec, self.tvec = rvec, tvec
        rmat, _ = cv2.Rodrigues(rvec)
        # Camera looks along +z, model z points at the camera: express in a frame where
        # yaw > 0 means the face turned towards image-left and pitch > 0 means looking up.
        sy = math.hypot(rmat[0, 0], rmat[1, 0])
        if sy > 1e-6:
            pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
            yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
            roll = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
        else:
            pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
            yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
            roll = 0.0
        pitch = _wrap180(pitch + 180.0)  # model faces +z, camera axis is +z: flip about x
        # Verified against synthetic projections (tests/test_geometry.py):
        # yaw > 0 -> face turned towards image-left, pitch > 0 -> looking up.
        return yaw, -pitch, roll


def _wrap180(angle: float) -> float:
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def geometric_yaw(pts: np.ndarray, schema: Schema) -> float:
    """Yaw proxy from nose offset between the eye outer corners, degrees (>0 = image-left)."""
    nose = pts[schema.pnp[0]]
    left = pts[schema.pnp[2]]
    right = pts[schema.pnp[3]]
    span = right[0] - left[0]
    if abs(span) < 1e-3:
        return 0.0
    ratio = (nose[0] - left[0]) / span  # 0.5 when frontal
    return float(np.clip((0.5 - ratio) * 2.0, -1.0, 1.0) * 60.0)


def iris_ratio(frame: np.ndarray, pts: np.ndarray, schema: Schema, eye_kps: np.ndarray | None = None) -> float | None:
    """Horizontal iris position inside the eye opening, 0 = image-left corner, 1 = image-right,
    ~0.5 looking straight. Uses MediaPipe iris landmarks when present (478-point mesh),
    otherwise the darkness centroid inside the eyelid polygon (68-point models), averaged
    with YuNet's eye keypoint. An eye whose keypoint falls outside its own landmark corners
    has a slipped contour and is ignored."""
    ratios, weights = [], []
    use_iris = len(pts) >= 478
    gray = None
    for n, (idx, iris_idx) in enumerate(((schema.left_eye, 468), (schema.right_eye, 473))):
        left, right = pts[idx[0]], pts[idx[3]]
        span = right[0] - left[0]
        if span < 6:
            continue
        if use_iris:
            cx = float(pts[iris_idx][0])
            weight_eye = 1.0
        else:
            if gray is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            poly = np.array([pts[i] for i in idx], dtype=np.float32)
            x1, y1 = np.floor(poly.min(axis=0)).astype(int)
            x2, y2 = np.ceil(poly.max(axis=0)).astype(int)
            if x2 - x1 < 6 or y2 - y1 < 2 or x1 < 0 or y1 < 0 or x2 >= gray.shape[1] or y2 >= gray.shape[0]:
                continue
            patch = gray[y1:y2 + 1, x1:x2 + 1].astype(np.float32)
            mask = np.zeros(patch.shape, np.uint8)
            cv2.fillConvexPoly(mask, np.round(poly - (x1, y1)).astype(np.int32), 1)
            inside = patch[mask > 0]
            if inside.size < 12:
                continue
            # sclera + iris inside a well-placed lid contour gives high contrast; a contour
            # that slipped onto skin or lashes does not -> weight eyes by that contrast
            weight_eye = float(np.std(inside)) ** 2
            cut = np.percentile(inside, 30)
            weight = np.clip(cut - patch, 0, None) * mask
            total = weight.sum()
            if total <= 0:
                continue
            cx = x1 + float((weight.sum(axis=0) * np.arange(patch.shape[1])).sum() / total)
        ratio = (cx - left[0]) / span
        if eye_kps is not None and not use_iris:
            kp_ratio = (float(eye_kps[n][0]) - left[0]) / span
            if not -0.05 <= kp_ratio <= 1.05:
                continue  # landmark contour slipped off this eye
            ratio = 0.5 * (ratio + kp_ratio)
        ratios.append(ratio)
        weights.append(weight_eye)
    if not ratios or sum(weights) <= 0:
        return None
    return float(np.clip(np.average(ratios, weights=weights), 0.0, 1.0))


def eye_crop(frame: np.ndarray, pts: np.ndarray, schema: Schema, prefer: str = "auto", out_size=(240, 150)):
    """Return a crop of the more visible eye, aspect matched to the UI panel."""
    candidates = []
    for side, idx in (("left", schema.left_eye), ("right", schema.right_eye)):
        a, b = pts[idx[0]], pts[idx[3]]
        candidates.append((np.linalg.norm(a - b), side, (a + b) / 2.0))
    candidates.sort(reverse=True)
    width, _, center = candidates[0]
    if prefer != "auto":
        for item in candidates:
            if item[1] == prefer:
                width, _, center = item
    half_w = max(12.0, width * 1.25)
    half_h = half_w * out_size[1] / out_size[0]
    h, w = frame.shape[:2]
    x1, x2 = int(center[0] - half_w), int(center[0] + half_w)
    y1, y2 = int(center[1] - half_h), int(center[1] + half_h)
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        pad = int(max(-x1, -y1, x2 - w, y2 - h, 0)) + 1
        frame = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x1, x2, y1, y2 = x1 + pad, x2 + pad, y1 + pad, y2 + pad
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, out_size, interpolation=cv2.INTER_CUBIC)


class MeshTopology:
    """Delaunay triangulation of the overlay points, frozen on the first frontal face so the
    drawn mesh is stable from frame to frame (no re-triangulation flicker)."""

    def __init__(self) -> None:
        self.edges: list[tuple[int, int]] = []
        self.frozen = False

    def reset(self) -> None:
        self.edges = []
        self.frozen = False

    def points(self, pts: np.ndarray, schema: Schema) -> np.ndarray:
        sub = pts[list(schema.mesh)]
        if schema.brows:
            sub = np.vstack([sub, _forehead(pts, schema)])
        return sub

    def update(self, mesh_pts: np.ndarray, yaw: float | None) -> list[tuple[int, int]]:
        if self.frozen and self.edges:
            return self.edges
        edges = _delaunay_edges(mesh_pts)
        if edges:
            self.edges = edges
            if yaw is not None and abs(yaw) < 12:
                self.frozen = True
        return self.edges


def _forehead(pts: np.ndarray, schema: Schema) -> np.ndarray:
    """Extrapolate forehead points above the brows (68-pt models stop at the eyebrows)."""
    brows = pts[list(schema.brows)]
    chin = pts[8]
    nose_top = pts[27]
    up = nose_top - chin
    norm = np.linalg.norm(up)
    if norm < 1e-3:
        return np.empty((0, 2), dtype=np.float32)
    up = up / norm
    lift = 0.32 * norm
    upper = brows + up * lift
    mid = brows + up * lift * 0.5
    return np.vstack([upper, mid[[1, 4]]]).astype(np.float32)


def _delaunay_edges(points: np.ndarray) -> list[tuple[int, int]]:
    if len(points) < 3:
        return []
    x1, y1 = points.min(axis=0) - 2
    x2, y2 = points.max(axis=0) + 2
    subdiv = cv2.Subdiv2D((int(x1), int(y1), int(x2 - x1) + 2, int(y2 - y1) + 2))
    index = {}
    for i, (x, y) in enumerate(points):
        key = (round(float(x), 1), round(float(y), 1))
        if key in index:
            continue
        index[key] = i
        try:
            subdiv.insert((float(x), float(y)))
        except cv2.error:
            continue
    edges = set()
    for tri in subdiv.getTriangleList():
        ids = []
        for k in range(3):
            key = (round(float(tri[2 * k]), 1), round(float(tri[2 * k + 1]), 1))
            if key not in index:
                break
            ids.append(index[key])
        if len(ids) != 3:
            continue
        for a, b in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[0], ids[2])):
            edges.add((min(a, b), max(a, b)))
    return sorted(edges)
