"""Seatbelt check from the driver camera.

Heuristic, not a learned model: a fastened belt shows up as a long, straight, diagonal
edge crossing the chest region under the face. Each frame scores the diagonal line
evidence in that region; a few seconds of votes decide On / Off. When the chest region
is out of frame the answer is Unknown (no alert is raised on Unknown).
"""

from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np


class SeatbeltDetector:
    def __init__(self, window: int = 45, on_ratio: float = 0.45, off_ratio: float = 0.12) -> None:
        self.votes: deque = deque(maxlen=window)
        self.on_ratio = on_ratio
        self.off_ratio = off_ratio
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        self.last_lines: list = []
        self._fld = None

    def reset(self) -> None:
        self.votes.clear()

    def score(self, frame: np.ndarray, face_bbox) -> float | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = face_bbox
        fw, fh = x2 - x1, y2 - y1
        # Torso band from the shoulders down; wider than the face on both sides because the
        # belt's shoulder anchor sits outside the face column (side depends on LHD/RHD and
        # camera placement).
        rx1, rx2 = int(max(0, x1 - 1.2 * fw)), int(min(w, x2 + 1.6 * fw))
        ry1, ry2 = int(min(h, y2 - 0.1 * fh)), int(min(h, y2 + 1.5 * fh))
        if ry2 - ry1 < 0.45 * fh or rx2 - rx1 < fw:
            return None  # chest not visible enough to judge
        roi = cv2.cvtColor(frame[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2GRAY)
        scale = min(1.0, 640.0 / roi.shape[1])  # full detail at 640x480 input; FLD needs it
        if scale < 1.0:
            roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        roi = self.clahe.apply(roi)
        segments = self._segments(roi)
        diag = math.hypot(*roi.shape[:2])
        kept = []
        for x_a, y_a, x_b, y_b in segments:
            angle = abs(math.degrees(math.atan2(y_b - y_a, x_b - x_a)))
            angle = min(angle, 180 - angle)
            length = math.hypot(x_b - x_a, y_b - y_a)
            if 28 <= angle <= 68 and length > 0.08 * diag:  # belt runs shoulder -> hip
                kept.append((length, angle, ((x_a / scale + rx1, y_a / scale + ry1), (x_b / scale + rx1, y_b / scale + ry1))))
        kept.sort(key=lambda k: -k[0])
        self.last_lines = [k[2] for k in kept[:4]]
        if not kept:
            return 0.0
        # A belt gives one long straight edge (usually two parallel ones). Hair, collars and
        # the wheel rim give many short or curved fragments, so score the longest edge plus
        # its best parallel partner instead of summing everything.
        best_len, best_angle, _ = kept[0]
        partner = max((k[0] for k in kept[1:] if abs(k[1] - best_angle) < 10), default=0.0)
        return min(1.0, (best_len + 0.5 * partner) / (0.30 * diag))

    def _segments(self, gray: np.ndarray):
        if self._fld is None:
            if hasattr(cv2, "ximgproc"):
                self._fld = cv2.ximgproc.createFastLineDetector(20, 1.41, 30, 60, 3, True)
            else:
                self._fld = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lines = self._fld.detect(gray)
        if isinstance(lines, tuple):  # LSD returns (lines, widths, precisions, nfa)
            lines = lines[0]
        return [] if lines is None else lines.reshape(-1, 4)

    def update(self, frame: np.ndarray, face_bbox) -> str:
        if face_bbox is None:
            return self.state()
        value = self.score(frame, face_bbox)
        if value is not None:
            self.votes.append(value >= 0.6)
        return self.state()

    def state(self) -> str:
        if len(self.votes) < max(10, self.votes.maxlen // 3):
            return "Unknown"
        ratio = sum(self.votes) / len(self.votes)
        if ratio >= self.on_ratio:
            return "On"
        if ratio <= self.off_ratio:
            return "Off"
        return "Unknown"
