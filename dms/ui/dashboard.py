"""PySide6 driver-monitoring dashboard.

Everything is painted with QPainter on a fixed 2576 x 1438 design canvas (the reference
layout) and uniformly scaled to the window, so the screen looks identical on a 7"
1024x600 cabin display, a 1080p monitor or a screenshot. The widget only reads the
immutable snapshots in ``SharedState``; it never blocks a worker thread.

Layout (design units), mode ``dms``:
  left   Driver Identification | Facial Analysis | Activity Recognition | Cabin Audio
  centre live driver camera with face mesh, ID tag, phone box, alert callout | VLM bar
  right  Real-time Video Analysis | Gaze Direction | AI Model Inference | Alert | System

Tab cycles ``dms``, ``cms`` (cabin camera, seats, objects) and ``split`` (both videos).
"""

from __future__ import annotations

import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (QBrush, QColor, QFont, QFontDatabase, QFontMetricsF, QImage, QKeySequence,
                           QLinearGradient, QPainter, QPainterPath, QPen, QShortcut)
from PySide6.QtWidgets import QWidget

from .. import alerts as A
from . import icons

W, H = 2576, 1438
FONT_DIR = Path(__file__).resolve().parent / "fonts"


class C:  # palette sampled from the reference design
    bg = QColor("#0b1219")
    panel = QColor(9, 17, 26, 222)
    panel_border = QColor(150, 182, 214, 92)
    panel_glow = QColor(120, 170, 220, 28)
    title = QColor("#f4f7fa")
    label = QColor("#c9d2db")
    dim = QColor("#7d8a97")
    green = QColor("#4ee08f")
    green_box = QColor("#22f58c")
    green_bar = QColor("#10d36b")
    green_tag = QColor("#10c160")
    red = QColor("#ff5b4f")
    red_box = QColor("#e8262f")
    red_tag = QColor("#d8232c")
    salmon = QColor("#ff8f84")
    amber = QColor("#ffb347")
    blue = QColor("#79b0ef")
    white = QColor("#f2f5f8")
    track = QColor(255, 255, 255, 26)


# (title, rect) of every panel; painted once into the cached chrome layer
LAYOUT = (
    ("Driver Identification", QRectF(24, 50, 542, 292)),
    ("Facial Analysis", QRectF(24, 360, 542, 340)),
    ("Activity Recognition", QRectF(24, 718, 542, 436)),
    ("Cabin Audio", QRectF(24, 1172, 542, 242)),
    (None, QRectF(592, 1188, 1330, 226)),
    ("Real-time Video Analysis", QRectF(1956, 50, 596, 352)),
    ("Gaze Direction", QRectF(1956, 420, 596, 254)),
    ("AI Model Inference", QRectF(1956, 692, 596, 298)),
    ("Alert", QRectF(1956, 1008, 596, 300)),
    (None, QRectF(1956, 1326, 596, 88)),
)
VIDEO_RECT = QRectF(592, 50, 1330, 1118)
MODES = ("dms", "cms", "split")
CMS_LAYOUT = (
    ("Occupancy", QRectF(24, 50, 542, 320)),
    ("Seat Map", QRectF(24, 388, 542, 400)),
    ("Objects", QRectF(24, 806, 542, 348)),
    ("Cabin Audio", QRectF(24, 1172, 542, 242)),
    (None, QRectF(592, 1188, 1330, 226)),
    ("Cameras", QRectF(1956, 50, 596, 300)),
    ("Safety", QRectF(1956, 368, 596, 380)),
    ("Alert", QRectF(1956, 766, 596, 540)),
    (None, QRectF(1956, 1326, 596, 88)),
)
SPLIT_LAYOUT = (
    ("Driver", QRectF(24, 64, 1254, 780)),
    ("Cabin", QRectF(1298, 64, 1254, 780)),
    ("Driver Alert", QRectF(24, 860, 1254, 300)),
    ("Cabin Alert", QRectF(1298, 860, 1254, 300)),
    ("Cabin Audio", QRectF(24, 1176, 1254, 238)),
    (None, QRectF(1298, 1176, 1254, 238)),
)

STATUS_COLOR = {"Active": C.green, "Searching": C.amber, "Lost": C.amber, "Off": C.dim}


def _font(size: float, weight: QFont.Weight = QFont.Weight.Medium, family: str = "Barlow") -> QFont:
    font = QFont(family)
    font.setPixelSize(max(1, int(round(size))))
    font.setWeight(weight)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    return font


def to_qimage(bgr: np.ndarray) -> QImage:
    bgr = np.ascontiguousarray(bgr)
    h, w = bgr.shape[:2]
    return QImage(bgr.data, w, h, int(bgr.strides[0]), QImage.Format.Format_BGR888).copy()


class Dashboard(QWidget):
    def __init__(self, shared, bus, cfg=None, fps: int = 30) -> None:
        super().__init__()
        self.shared = shared
        self.bus = bus
        self.cfg = cfg
        self.mode = "dms"
        self.setWindowTitle("Driver Monitoring System")
        self.setMinimumSize(960, 536)
        self.resize(1600, 893)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.family = "Barlow"
        for ttf in sorted(FONT_DIR.glob("*.ttf")):
            fid = QFontDatabase.addApplicationFont(str(ttf))
            fams = QFontDatabase.applicationFontFamilies(fid) if fid >= 0 else []
            if fams:
                self.family = fams[0]
        self.f_title = _font(31, QFont.Weight.DemiBold, self.family)
        self.f_row = _font(25, QFont.Weight.Normal, self.family)
        self.f_row_m = _font(25, QFont.Weight.Medium, self.family)
        self.f_value = _font(25, QFont.Weight.Medium, self.family)
        self.f_small = _font(21, QFont.Weight.Normal, self.family)
        self.f_tag = _font(23, QFont.Weight.Medium, self.family)
        self.f_callout = _font(27, QFont.Weight.DemiBold, self.family)
        self.f_alert = _font(30, QFont.Weight.Medium, self.family)
        self._frame_cache: dict = {}
        self._bg_image: QImage | None = None
        self._bg_ts = 0.0
        self._pan_x: float | None = None
        self._chrome: QImage | None = None
        self._chrome_size = None
        self._chrome_ts = 0.0
        self._t0 = time.monotonic()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(int(1000 / max(1, fps)))
        QShortcut(QKeySequence("F11"), self, activated=self.toggle_fullscreen)
        QShortcut(QKeySequence("Esc"), self, activated=self.showNormal)
        QShortcut(QKeySequence("Ctrl+Q"), self, activated=self.close)
        self._shortcuts = []
        for key, mode in (("1", "dms"), ("2", "cms"), ("3", "split")):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(lambda m=mode: self.set_mode(m))
            self._shortcuts.append(shortcut)
        cycle = QShortcut(QKeySequence(Qt.Key.Key_Tab), self)
        cycle.setContext(Qt.ShortcutContext.ApplicationShortcut)
        cycle.activated.connect(self.cycle_mode)
        self._shortcuts.append(cycle)

    def set_mode(self, mode: str) -> None:
        if mode not in MODES or mode == self.mode:
            return
        self.mode = mode
        self._chrome = None
        titles = {"dms": "Driver Monitoring System", "cms": "Cabin Monitoring System",
                  "split": "Driver and Cabin Monitoring"}
        self.setWindowTitle(titles[mode])
        self.update()

    def cycle_mode(self) -> None:
        self.set_mode(MODES[(MODES.index(self.mode) + 1) % len(MODES)])

    def _view_transform(self) -> tuple[float, float, float]:
        scale = min(self.width() / W, self.height() / H) or 1.0
        return scale, (self.width() - W * scale) / 2, (self.height() - H * scale) / 2

    def _to_design(self, local) -> QPointF:
        scale, ox, oy = self._view_transform()
        return QPointF((local.x() - ox) / scale, (local.y() - oy) / scale)

    def _mode_rects(self) -> list[tuple[QRectF, str]]:
        labels = (("1  DMS", "dms"), ("2  CMS", "cms"), ("3  Split", "split"))
        fm = QFontMetricsF(self.f_small)
        x = 28.0
        rects = []
        for text, mode in labels:
            width = fm.horizontalAdvance(text)
            rects.append((QRectF(x - 14, 2, width + 28, 42), mode, text))
            x += width + 36
        return rects

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            point = self._to_design(event.position())
            for rect, mode, _text in self._mode_rects():
                if rect.contains(point):
                    self.set_mode(mode)
                    event.accept()
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        point = self._to_design(event.position())
        over = any(rect.contains(point) for rect, _mode, _text in self._mode_rects())
        self.setCursor(Qt.CursorShape.PointingHandCursor if over else Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(event)

    def _layout(self):
        if self.mode == "cms":
            return CMS_LAYOUT
        if self.mode == "split":
            return SPLIT_LAYOUT
        return LAYOUT

    def toggle_fullscreen(self) -> None:
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    # ------------------------------------------------------------------ helpers
    def _qimage(self, key: str, array: np.ndarray | None, stamp) -> QImage | None:
        if array is None:
            return None
        cached = self._frame_cache.get(key)
        if cached and cached[0] == stamp:
            return cached[1]
        image = to_qimage(array)
        self._frame_cache[key] = (stamp, image)
        return image

    def _panel(self, p: QPainter, rect: QRectF, title: str | None = None) -> None:
        p.save()
        p.setPen(QPen(C.panel_glow, 6))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(rect.adjusted(-1, -1, 1, 1), 9, 9)
        p.setPen(QPen(C.panel_border, 1.6))
        p.setBrush(QBrush(C.panel))
        p.drawRoundedRect(rect, 8, 8)
        # faint top sheen
        sheen = QLinearGradient(rect.topLeft(), QPointF(rect.left(), rect.top() + 80))
        sheen.setColorAt(0, QColor(255, 255, 255, 14))
        sheen.setColorAt(1, QColor(255, 255, 255, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(sheen))
        p.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 8, 8)
        if title:
            p.setFont(self.f_title)
            p.setPen(C.title)
            p.drawText(QPointF(rect.left() + 32, rect.top() + 48), title)
        p.restore()

    def _text(self, p: QPainter, x: float, y: float, text: str, color: QColor, font: QFont,
              align: str = "left", max_w: float | None = None) -> float:
        p.setFont(font)
        p.setPen(color)
        fm = QFontMetricsF(font)
        if max_w is not None:
            text = fm.elidedText(text, Qt.TextElideMode.ElideRight, max_w)
        width = fm.horizontalAdvance(text)
        if align == "right":
            x -= width
        elif align == "center":
            x -= width / 2
        p.drawText(QPointF(x, y), text)
        return width

    def _row(self, p: QPainter, x1: float, x2: float, y: float, label: str, value: str, color: QColor) -> None:
        self._text(p, x1, y, label, C.label, self.f_row)
        self._text(p, x2, y, value, color, self.f_value, align="right")

    # ------------------------------------------------------------------ paint
    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        scale = min(self.width() / W, self.height() / H)
        size = (self.width(), self.height())
        token = (size, self.mode)
        if (self._chrome is None or self._chrome_size != token
                or (self.shared.driver.frame is not None and time.monotonic() - self._chrome_ts > 1.0)):
            self._build_chrome(size, scale)
        p.drawImage(0, 0, self._chrome)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing |
                         QPainter.RenderHint.SmoothPixmapTransform)
        p.translate((self.width() - W * scale) / 2, (self.height() - H * scale) / 2)
        p.scale(scale, scale)
        self.render_dynamic(p)
        p.end()

    def _build_chrome(self, size, scale: float) -> None:
        """Background + panel frames + titles. Rebuilt on resize and once a second (the
        blurred camera backdrop), so the per-frame paint only draws what changes."""
        image = QImage(size[0], size[1], QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(C.bg)
        p = QPainter(image)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing |
                         QPainter.RenderHint.SmoothPixmapTransform)
        p.translate((size[0] - W * scale) / 2, (size[1] - H * scale) / 2)
        p.scale(scale, scale)
        self.render_chrome(p)
        p.end()
        self._chrome, self._chrome_size, self._chrome_ts = image, (size, self.mode), time.monotonic()

    def render_canvas(self, p: QPainter) -> None:
        self.render_chrome(p)
        self.render_dynamic(p)

    def render_chrome(self, p: QPainter) -> None:
        self._bg_ts = 0.0
        source = self.shared.passenger if self.mode == "cms" else self.shared.driver
        self._background(p, source)
        for title, rect in self._layout():
            self._panel(p, rect, title)

    def render_dynamic(self, p: QPainter) -> None:
        self._modes(p)
        if self.mode == "cms":
            self._render_cms(p)
        elif self.mode == "split":
            self._render_split(p)
        else:
            self._render_dms(p)

    def _render_dms(self, p: QPainter) -> None:
        ds = self.shared.driver
        now = time.time()
        self._video(p, VIDEO_RECT, ds)
        (_, r_id), (_, r_face), (_, r_act), (_, r_audio), (_, r_vlm), (_, r_ana), (_, r_gaze), (_, r_inf), \
            (_, r_alert), (_, r_sys) = LAYOUT
        self._driver_id(p, r_id, ds)
        self._facial(p, r_face, ds)
        self._activity(p, r_act, ds)
        self._audio(p, r_audio)
        self._vlm(p, r_vlm)
        self._analysis(p, r_ana, ds)
        self._gaze(p, r_gaze, ds)
        self._inference(p, r_inf, ds)
        self._alert(p, r_alert, now)
        self._system(p, r_sys, ds)

    # ------------------------------------------------------------------ background + video
    def _background(self, p: QPainter, ds) -> None:
        frame = ds.frame
        if frame is not None and time.monotonic() - self._bg_ts > 0.5:
            self._bg_ts = time.monotonic()
            small = cv2.resize(frame, (160, int(160 * frame.shape[0] / frame.shape[1])), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (0, 0), 3.0)
            small = cv2.convertScaleAbs(small, alpha=0.55, beta=0)
            self._bg_image = to_qimage(small)
        if self._bg_image is not None:
            img = self._bg_image
            s = max(W / img.width(), H / img.height())
            p.drawImage(QRectF((W - img.width() * s) / 2, (H - img.height() * s) / 2, img.width() * s, img.height() * s), img)
            p.fillRect(QRectF(0, 0, W, H), QColor(6, 12, 18, 120))
        else:
            grad = QLinearGradient(0, 0, 0, H)
            grad.setColorAt(0, QColor("#16222c"))
            grad.setColorAt(1, QColor("#0a0f14"))
            p.fillRect(QRectF(0, 0, W, H), QBrush(grad))

    def _video_transform(self, rect: QRectF, ds):
        fw, fh = ds.width or 640, ds.height or 480
        scale = max(rect.width() / fw, rect.height() / fh)
        span_x = fw * scale - rect.width()
        span_y = fh * scale - rect.height()
        target = span_x / 2
        if ds.face_bbox is not None:  # keep the driver's face centred (smooth follow)
            cx = (ds.face_bbox[0] + ds.face_bbox[2]) / 2 * scale
            target = cx - rect.width() * 0.40
            if ds.phone_box is not None:  # keep the phone box (and its tag) in view too
                target = min(target, ds.phone_box[0] * scale - 60)
            target = float(np.clip(target, 0, span_x))
        if self._pan_x is None:
            self._pan_x = target
        self._pan_x += (target - self._pan_x) * 0.15
        ox, oy = self._pan_x, span_y / 2

        def m(x, y):
            return QPointF(rect.left() + x * scale - ox, rect.top() + y * scale - oy)
        return scale, ox, oy, m

    def _video(self, p: QPainter, rect: QRectF, ds) -> None:
        p.save()
        path = QPainterPath()
        path.addRoundedRect(rect, 6, 6)
        p.setClipPath(path)
        img = self._qimage("driver", ds.frame, ds.seq)
        if img is None:
            p.fillRect(rect, QColor("#05090d"))
            self._text(p, rect.center().x(), rect.center().y(), "Waiting for driver camera...", C.dim, self.f_alert, "center")
            p.restore()
            self._frame_border(p, rect)
            return
        scale, ox, oy, m = self._video_transform(rect, ds)
        target = QRectF(rect.left() - ox, rect.top() - oy, img.width() * scale, img.height() * scale)
        dev = p.worldTransform().m11()
        dw, dh = max(1, int(target.width() * dev)), max(1, int(target.height() * dev))
        cached = self._frame_cache.get("driver_scaled")
        if not cached or cached[0] != (ds.seq, dw, dh):
            interp = cv2.INTER_AREA if dw < ds.frame.shape[1] else cv2.INTER_LINEAR
            scaled = to_qimage(cv2.resize(ds.frame, (dw, dh), interpolation=interp))
            self._frame_cache["driver_scaled"] = ((ds.seq, dw, dh), scaled)
        else:
            scaled = cached[1]
        p.drawImage(target, scaled)
        pulse = 0.65 + 0.35 * math.sin((time.monotonic() - self._t0) * 6.0)

        face_rect = None
        if ds.face_bbox is not None:
            x1, y1, x2, y2 = ds.face_bbox
            face_rect = QRectF(m(x1, y1), m(x2, y2))
        if ds.mesh_points is not None and len(ds.mesh_points):
            pts = [m(float(x), float(y)) for x, y in ds.mesh_points]
            p.setPen(QPen(QColor(255, 255, 255, 105), 1.3))
            for a, b in ds.mesh_edges:
                if a < len(pts) and b < len(pts):
                    p.drawLine(pts[a], pts[b])
            p.setPen(Qt.PenStyle.NoPen)
            for i, pt in enumerate(pts):
                big = i % 3 == 0
                p.setBrush(QColor(255, 255, 255, 60))
                p.drawEllipse(pt, 6 if big else 4, 6 if big else 4)
                p.setBrush(QColor(255, 255, 255, 235))
                p.drawEllipse(pt, 3.2 if big else 2.2, 3.2 if big else 2.2)

        if face_rect is not None:
            ident = ds.identity
            color = C.green_box if ident.known else C.amber
            tag_fill = C.green_tag if ident.known else QColor("#c98316")
            p.setPen(QPen(color, 4))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(face_rect)
            line1 = f"Person ID-{ident.driver_id}" if ident.known else "Person ID-----"
            line2 = "Known Driver" if ident.known else ("Unknown Driver" if ident.name != "Not enrolled" else "Not Enrolled")
            fm = QFontMetricsF(self.f_tag)
            tag_w = max(fm.horizontalAdvance(line1), fm.horizontalAdvance(line2)) + 30
            tag = QRectF(face_rect.left() - 2, face_rect.top() - 66, tag_w, 64)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(tag_fill)
            p.drawRoundedRect(tag, 4, 4)
            p.drawRect(QRectF(tag.left(), tag.bottom() - 6, tag.width(), 8))
            self._text(p, tag.left() + 14, tag.top() + 27, line1, C.white, self.f_tag)
            self._text(p, tag.left() + 14, tag.top() + 55, line2, C.white, self.f_tag)

        if ds.phone and ds.phone_box is not None:
            x1, y1, x2, y2 = ds.phone_box
            box = QRectF(m(x1, y1), m(x2, y2)).adjusted(-12, -12, 12, 12)
            p.setPen(QPen(C.red_box, 4))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(box)
            fm = QFontMetricsF(self.f_tag)
            tag = QRectF(box.left() - 2, box.top() - 58, fm.horizontalAdvance("Phone Detected") + 76, 56)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(C.red_tag)
            p.drawRoundedRect(tag, 4, 4)
            p.drawRect(QRectF(tag.left(), tag.bottom() - 6, tag.width(), 8))
            icons.phone(p, QRectF(tag.left() + 14, tag.top() + 12, 28, 34), C.white, 2.6)
            self._text(p, tag.left() + 54, tag.top() + 37, "Phone Detected", C.white, self.f_tag)
            if face_rect is not None:
                p.setPen(QPen(QColor(232, 38, 47, 210), 2))
                p.drawLine(QPointF(tag.right(), tag.top() + 2), QPointF(face_rect.left(), face_rect.bottom() - face_rect.height() * 0.15))

        if ds.overlay_alert and face_rect is not None:
            fm = QFontMetricsF(self.f_callout)
            text_w = fm.horizontalAdvance(ds.overlay_alert)
            box_w, box_h = text_w + 104, 80
            bx = face_rect.right() + 70
            by = max(rect.top() + 30, face_rect.top() + face_rect.height() * 0.18)
            if bx + box_w > rect.right() - 24:
                # not enough room beside the face: sit above its top-right corner
                bx = rect.right() - 24 - box_w
                by = max(rect.top() + 24, face_rect.top() - box_h - 84)
                if bx < face_rect.right() - 40 and by + box_h > face_rect.top() - 70:
                    bx = max(rect.left() + 24, face_rect.left() - box_w - 60)
                    by = max(rect.top() + 30, face_rect.top() + face_rect.height() * 0.18)
            box = QRectF(bx, by, box_w, box_h)
            p.setPen(QPen(QColor(232, 45, 52, int(255 * pulse)), 3))
            p.setBrush(QColor(110, 16, 22, 215))
            p.drawRoundedRect(box, 5, 5)
            icons.warning(p, QRectF(box.left() + 22, box.top() + 20, 42, 38), QColor("#ff4a4a"), QColor(110, 16, 22))
            self._text(p, box.left() + 82, box.top() + 51, ds.overlay_alert, C.white, self.f_callout)
            p.setPen(QPen(QColor(232, 45, 52, 220), 2))
            if box.left() >= face_rect.right():
                p.drawLine(QPointF(box.left(), box.bottom()),
                           QPointF(face_rect.right(), face_rect.top() + face_rect.height() * 0.42))
            elif box.right() <= face_rect.left():
                p.drawLine(QPointF(box.right(), box.bottom()),
                           QPointF(face_rect.left(), face_rect.top() + face_rect.height() * 0.42))
            else:
                p.drawLine(QPointF(box.left() + 40, box.bottom()), QPointF(face_rect.right() - 20, face_rect.top()))
        elif ds.overlay_alert:
            fm = QFontMetricsF(self.f_callout)
            box = QRectF(rect.center().x() - fm.horizontalAdvance(ds.overlay_alert) / 2 - 60, rect.top() + 40,
                         fm.horizontalAdvance(ds.overlay_alert) + 120, 86)
            p.setPen(QPen(QColor(232, 45, 52, int(255 * pulse)), 3))
            p.setBrush(QColor(110, 16, 22, 215))
            p.drawRoundedRect(box, 5, 5)
            self._text(p, box.center().x(), box.top() + 55, ds.overlay_alert, C.white, self.f_callout, "center")

        if not ds.face_found and ds.frame is not None:
            self._text(p, rect.left() + 30, rect.bottom() - 30, "Searching for driver face...", C.amber, self.f_row_m)
        p.restore()
        self._frame_border(p, rect)

    def _frame_border(self, p: QPainter, rect: QRectF) -> None:
        p.save()
        p.setPen(QPen(QColor(170, 200, 230, 70), 1.6))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(rect, 6, 6)
        p.restore()

    # ------------------------------------------------------------------ left column
    def _driver_id(self, p: QPainter, r: QRectF, ds) -> None:
        ident = ds.identity
        photo_rect = QRectF(r.left() + 32, r.top() + 85, 138, 174)
        photo = ident.photo
        stamp = ident.driver_id
        if photo is None and ds.frame is not None and ds.face_bbox is not None and ds.seq % 15 == 0:
            x1, y1, x2, y2 = ds.face_bbox
            fw, fh = x2 - x1, y2 - y1
            ya, yb = max(0, int(y1 - 0.15 * fh)), min(ds.height, int(y2 + 0.1 * fh))
            cx, half = (x1 + x2) / 2, (yb - ya) * 138 / 174 / 2
            xa, xb = max(0, int(cx - half)), min(ds.width, int(cx + half))
            crop = ds.frame[ya:yb, xa:xb]
            if crop.size:
                self._live_photo = cv2.resize(crop, (138, 174))
        if photo is None:
            photo = getattr(self, "_live_photo", None)
            stamp = ("live", ds.seq // 15)
        img = self._qimage("photo", photo, stamp)
        p.save()
        clip = QPainterPath()
        clip.addRoundedRect(photo_rect, 4, 4)
        p.setClipPath(clip)
        if img is not None:
            s = max(photo_rect.width() / img.width(), photo_rect.height() / img.height())
            iw, ih = img.width() * s, img.height() * s
            p.drawImage(QRectF(photo_rect.center().x() - iw / 2, photo_rect.center().y() - ih / 2, iw, ih), img)
        else:
            p.fillRect(photo_rect, QColor("#1a2632"))
        p.restore()
        p.setPen(QPen(QColor(255, 255, 255, 40), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(photo_rect, 4, 4)
        x = r.left() + 194
        self._text(p, x, r.top() + 108, f"Name: {ident.name}", C.label, self.f_row, max_w=r.right() - x - 20)
        self._text(p, x, r.top() + 146, f"Age: {ident.age}", C.label, self.f_row)
        self._text(p, x, r.top() + 184, f"Driver ID: {ident.driver_id}", C.label, self.f_row)
        w = self._text(p, x, r.top() + 222, "Match: ", C.label, self.f_row)
        match = ident.match_pct if ds.face_found else 0.0
        color = C.green if ident.known else (C.amber if ds.face_found else C.dim)
        self._text(p, x + w, r.top() + 222, f"{match:.1f}%" if ds.face_found else "--", color, self.f_value)
        bar = QRectF(x, r.top() + 240, r.right() - x - 34, 18)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(C.track)
        p.drawRect(bar)
        p.setBrush(C.green_bar if ident.known else C.amber)
        p.drawRect(QRectF(bar.left(), bar.top(), bar.width() * min(1.0, match / 100.0), bar.height()))

    def _facial(self, p: QPainter, r: QRectF, ds) -> None:
        x1, x2 = r.left() + 33, r.right() - 36
        face = ds.face_found
        eyes = f"{ds.eyes_open_pct:.0f}%" if face and ds.eyes_open_pct is not None else "--"
        eyes_c = C.green if face and ds.eyes_open_pct is not None and ds.eye_state == "Open" else C.red if face else C.dim
        if ds.gaze in ("--",) or not face:
            gaze, gaze_c = "--", C.dim
        elif ds.gaze_away:
            gaze, gaze_c = f"{ds.gaze} (away)", C.red
        else:
            gaze, gaze_c = "Forward (road)", C.green
        if ds.blink_rate is None:
            blink, blink_c = "-- /min", C.dim
        else:
            blink = f"{ds.blink_rate:.0f}/min"
            blink_c = C.green if 6 <= ds.blink_rate <= 30 else C.amber
        pose = ds.head_pose if face else "--"
        pose_c = C.dim if pose == "--" else C.green if pose == "Forward" else C.red
        emotion = ds.emotion if face else "--"
        rows = [("Eyes Open", eyes, eyes_c), ("Gaze Direction", gaze, gaze_c), ("Blink Rate", blink, blink_c),
                ("Head Pose", pose, pose_c), ("Emotion", emotion, C.white if face else C.dim)]
        for i, (label, value, color) in enumerate(rows):
            self._row(p, x1, x2, r.top() + 104 + i * 50, label, value, color)

    def _activity(self, p: QPainter, r: QRectF, ds) -> None:
        face = ds.face_found
        icon_c = QColor("#d6dde4")

        def yn(flag, on_text="Detected", off_text="Not Detected"):
            if not face:
                return "--", C.dim
            return (on_text, C.red) if flag else (off_text, C.green)

        hand_color = {"On Phone": C.red, "Near Ear": C.red, "Near Face": C.amber, "On Wheel": C.green,
                      "Free": C.white, "Not Visible": C.dim}.get(ds.hand_position, C.dim)
        belt_color = {"On": C.green, "Off": C.red}.get(ds.seatbelt, C.dim)
        rows = [
            (icons.phone, "Using Phone", *yn(ds.phone)),
            (icons.hand, "Hand Position", ds.hand_position if face else "--", hand_color if face else C.dim),
            (icons.seatbelt, "Seatbelt", ds.seatbelt, belt_color),
            (icons.cigarette, "Smoking", *yn(ds.smoking)),
            (icons.cup, "Drinking", *yn(ds.drinking)),
        ]
        for i, (icon, label, value, color) in enumerate(rows):
            y = r.top() + 110 + i * 66
            icon(p, QRectF(r.left() + 40, y - 30, 34, 38), icon_c)
            self._text(p, r.left() + 101, y, label, C.label, self.f_row)
            self._text(p, r.right() - 38, y, value, color, self.f_value, align="right")

    def _audio(self, p: QPainter, r: QRectF) -> None:
        au = self.shared.audio
        x1, x2 = r.left() + 33, r.right() - 36
        if not au.enabled:
            self._text(p, x1, r.top() + 105, "Audio monitoring off", C.dim, self.f_row)
            return
        # level meter
        bar = QRectF(x1, r.top() + 78, x2 - x1 - 120, 16)
        level = float(np.clip((au.level_dbfs + 70.0) / 70.0, 0.0, 1.0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(C.track)
        p.drawRect(bar)
        grad = QLinearGradient(bar.topLeft(), bar.topRight())
        grad.setColorAt(0.0, C.green_bar)
        grad.setColorAt(0.7, C.amber)
        grad.setColorAt(1.0, C.red_box)
        p.setBrush(QBrush(grad))
        p.drawRect(QRectF(bar.left(), bar.top(), bar.width() * level, bar.height()))
        self._text(p, x2, bar.bottom() + 1, f"{au.level_dbfs:.0f} dBFS", C.label, self.f_small, "right")
        recent = au.last_event and time.time() - au.last_event_ts < 15
        event = au.last_event.replace("_", " ").title() if au.last_event else "None"
        when = datetime.fromtimestamp(au.last_event_ts).strftime("%H:%M:%S") if au.last_event else ""
        self._row(p, x1, x2, r.top() + 146, "Last Event", f"{event} {when}".strip(), C.red if recent else C.green)
        top = ", ".join(f"{k} {v:.2f}" for k, v in list(au.tags.items())[:2]) or "--"
        self._row(p, x1, x2, r.top() + 192, "Classifier", au.backend.upper() if au.backend != "cochl" else "Cochl.Sense", C.blue)
        kws = "Vosk KWS" if au.kws == "vosk" else "Off"
        self._text(p, x1, r.top() + 228, f"Top: {top}   ·   Keywords: {kws}", C.dim, self.f_small, max_w=x2 - x1)

    def _vlm(self, p: QPainter, r: QRectF) -> None:
        vlm = self.shared.vlm
        pas = self.shared.passenger
        self._text(p, r.left() + 32, r.top() + 50, "Cabin Scene Understanding", C.title, self.f_title)
        status = vlm.status if vlm.enabled else "off"
        if vlm.enabled and vlm.calls and status in ("ok", "throttled", "running"):
            chip_text, chip_c = ("HAZARD", C.red) if vlm.unsafe else ("CLEAR", C.green)
        else:
            chip_text, chip_c = status.upper(), C.amber if status in ("waiting", "error", "throttled") else C.dim
        fm = QFontMetricsF(self.f_small)
        cw = fm.horizontalAdvance(chip_text) + 34
        chip = QRectF(r.left() + 470, r.top() + 22, cw, 38)
        p.setPen(QPen(chip_c, 2))
        p.setBrush(QColor(chip_c.red(), chip_c.green(), chip_c.blue(), 40))
        p.drawRoundedRect(chip, 19, 19)
        self._text(p, chip.center().x(), chip.top() + 27, chip_text, chip_c, self.f_small, "center")
        # Round trip of the last /v1/chat/completions call (upload, vision encode, decode).
        if vlm.latency_s:
            lat = f"Latency {vlm.latency_s:.2f} s"
            if vlm.latency_avg_s and vlm.calls > 1:
                lat += f"  avg {vlm.latency_avg_s:.2f} s"
            lat_color = C.blue
        else:
            lat, lat_color = "Latency --", C.amber
        lat_w = self._text(p, r.right() - 32, r.top() + 50, lat, lat_color, self.f_small, "right")
        meta = []
        if vlm.model:
            meta.append(vlm.model)
        if vlm.interval_s:
            meta.append(f"every {vlm.interval_s:.1f} s")
        if vlm.confidence:
            meta.append(f"conf {vlm.confidence:.2f}")
        meta.append(f"{pas.occupants} in cabin" if pas.connected else "cabin cam off")
        meta_end = r.right() - 48 - lat_w
        self._text(p, meta_end, r.top() + 50, "  ·  ".join(meta), C.dim, self.f_small, "right",
                   max_w=max(40.0, meta_end - chip.right() - 18))
        dms_line = getattr(vlm, "dms_scene", "") or ""
        cms_line = getattr(vlm, "cms_scene", "") or ""
        dms_desc = getattr(vlm, "dms_description", "") or ""
        cms_desc = getattr(vlm, "cms_description", "") or ""
        scene_c = C.red if vlm.unsafe else C.label
        line_w = r.width() - 64
        if dms_line or cms_line:
            scene = " · ".join(part for part in (f"DMS: {dms_line}" if dms_line else "", f"CMS: {cms_line}" if cms_line else "") if part)
            self._text(p, r.left() + 32, r.top() + 100, scene, scene_c, self.f_row, max_w=line_w)
            policy_y = r.top() + 148
            if dms_desc or cms_desc:
                self._text(p, r.left() + 32, r.top() + 138, f"DMS: {dms_desc or '—'}", C.dim, self.f_small, max_w=line_w)
                self._text(p, r.left() + 32, r.top() + 168, f"CMS: {cms_desc or '—'}", C.dim, self.f_small, max_w=line_w)
                policy_y = r.top() + 200
        else:
            scene = vlm.scene or vlm.text or ("Waiting for first scene description..." if vlm.enabled else "VLM disabled")
            self._text(p, r.left() + 32, r.top() + 108, scene, scene_c, self.f_row, max_w=line_w)
            policy_y = r.top() + 158
        if vlm.policy:
            policy, policy_c = vlm.policy, C.red if vlm.unsafe else C.amber if vlm.suppressed else C.dim
        elif vlm.enabled and not vlm.calls:
            policy, policy_c = "Safety policy runs after the first scene answer.", C.dim
        else:
            policy, policy_c = "", C.dim
        if policy:
            self._text(p, r.left() + 32, policy_y, policy, policy_c, self.f_small, max_w=line_w)

    # ------------------------------------------------------------------ right column
    def _analysis(self, p: QPainter, r: QRectF, ds) -> None:
        thumb = QRectF(r.left() + 32, r.top() + 78, 348, 252)
        pas = self.shared.passenger
        src, stamp, boxes, fw, fh = None, None, [], 640, 480
        if pas.frame is not None:
            src, stamp, boxes = pas.frame, pas.frame_ts, pas.faces
        elif ds.frame is not None:
            src, stamp = ds.frame, ds.seq
            boxes = [ds.face_bbox] if ds.face_bbox else []
        if src is not None:
            fh, fw = src.shape[:2]
        img = self._qimage("thumb", src, stamp)
        p.save()
        clip = QPainterPath()
        clip.addRoundedRect(thumb, 4, 4)
        p.setClipPath(clip)
        if img is not None:
            s = max(thumb.width() / fw, thumb.height() / fh)
            ox = (fw * s - thumb.width()) / 2
            oy = (fh * s - thumb.height()) / 2
            p.drawImage(QRectF(thumb.left() - ox, thumb.top() - oy, fw * s, fh * s), img)
            p.setPen(QPen(C.green_box, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            for b in boxes:
                p.drawRect(QRectF(thumb.left() - ox + b[0] * s, thumb.top() - oy + b[1] * s,
                                  (b[2] - b[0]) * s, (b[3] - b[1]) * s))
            fade = QLinearGradient(QPointF(thumb.left() + 140, 0), QPointF(thumb.left() + 250, 0))
            fade.setColorAt(0, QColor(9, 17, 26, 0))
            fade.setColorAt(1, QColor(9, 17, 26, 225))
            p.fillRect(thumb, QBrush(fade))
        else:
            p.fillRect(thumb, QColor("#121c26"))
        p.restore()
        label = "Passenger cam" if pas.frame is not None else "Driver cam"
        self._text(p, thumb.left() + 10, thumb.bottom() - 10, label, QColor(255, 255, 255, 170), self.f_small)
        x1, x2 = r.left() + 258, r.right() - 34
        for i, (name, status) in enumerate(ds.modules.items()):
            y = r.top() + 125 + i * 51
            self._text(p, x1, y, name, C.label, self.f_row)
            self._text(p, x2, y, status, STATUS_COLOR.get(status, C.dim), self.f_value, "right")

    def _gaze(self, p: QPainter, r: QRectF, ds) -> None:
        eye = QRectF(r.left() + 32, r.top() + 74, 222, 150)
        img = self._qimage("eye", ds.eye_crop, ds.seq)
        p.save()
        clip = QPainterPath()
        clip.addRoundedRect(eye, 4, 4)
        p.setClipPath(clip)
        if img is not None:
            p.drawImage(eye, img)
        else:
            p.fillRect(eye, QColor("#121c26"))
            self._text(p, eye.center().x(), eye.center().y() + 8, "no eye lock", C.dim, self.f_small, "center")
        p.restore()
        if img is not None and ds.gaze not in ("--", "Forward"):
            dx, dy = ds.gaze_vector
            c = QPointF(eye.center().x() + 8, eye.center().y() + 2)
            end = QPointF(c.x() + dx * 92, c.y() + dy * 52)
            icons.arrow(p, c, end, QColor("#ff3b30"), 4.5, 20)
            p.setBrush(QColor("#ff3b30"))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(c, 5, 5)
        x = r.left() + 284
        if not ds.face_found or ds.gaze == "--":
            self._text(p, x, r.top() + 136, "Gaze: --", C.dim, self.f_row)
            self._text(p, x, r.top() + 172, "(no face)", C.dim, self.f_row)
        else:
            w = self._text(p, x, r.top() + 136, "Gaze: ", C.label, self.f_row)
            self._text(p, x + w, r.top() + 136, ds.gaze, C.white, self.f_row_m)
            sub = "(away from road)" if ds.gaze_away else "(on road)"
            self._text(p, x, r.top() + 172, sub, C.red if ds.gaze_away else C.green, self.f_row)
            if ds.gaze_angle is not None:
                self._text(p, x, r.top() + 206, f"{ds.gaze_angle:+.0f}° yaw+eyes", C.dim, self.f_small)

    def _inference(self, p: QPainter, r: QRectF, ds) -> None:
        hw = self.shared.hardware or "--"
        x1, x2 = r.left() + 34, r.left() + 222
        rows = [("Model:", ds.model_name), ("Framework:", ds.framework),
                ("Inference Time:", f"{ds.model_ms:.0f} ms  (pipeline {ds.pipeline_ms:.0f} ms)" if ds.model_ms else "--"),
                ("Hardware:", hw)]
        for i, (label, value) in enumerate(rows):
            y = r.top() + 106 + i * 51
            self._text(p, x1, y, label, C.label, self.f_row)
            self._text(p, x2, y, value, C.blue, self.f_row, max_w=r.right() - x2 - 24)

    def _alert(self, p: QPainter, r: QRectF, now: float, channel: str | None = None) -> None:
        alert = self.bus.latest(channel) if self.bus else None
        icon_box = QRectF(r.left() + 32, r.top() + 82, 100, 100)
        under = QRectF(r.left() + 32, r.top() + 204, 100, 62)
        active = alert is not None and now - alert.timestamp < 30
        if active:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(70, 20, 26, 230))
            p.drawRoundedRect(icon_box, 6, 6)
            icons.warning(p, icon_box.adjusted(22, 24, -22, -24), QColor("#ff4a4a"), QColor(70, 20, 26))
            p.setBrush(QColor(255, 255, 255, 10))
            p.drawRoundedRect(under, 6, 6)
            icons.clock(p, QRectF(under.center().x() - 15, under.center().y() - 15, 30, 30), QColor(255, 255, 255, 120))
            x = r.left() + 152
            self._text(p, x, r.top() + 128, alert.title, C.salmon, self.f_alert, max_w=r.right() - x - 20)
            desc = alert.description or alert.type
            self._text(p, x, r.top() + 178, desc, C.white, self.f_row, max_w=r.right() - x - 20)
            stamp = datetime.fromtimestamp(alert.timestamp).strftime("%H:%M:%S")
            cat = "Vision" if alert.category == A.VISION else "Audio"
            self._text(p, x, r.top() + 245, f"{stamp}", C.white, self.f_row)
            self._text(p, r.right() - 34, r.top() + 245, f"{cat} · {alert.confidence:.0%}", C.dim, self.f_small, "right")
        else:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(16, 60, 38, 200))
            p.drawRoundedRect(icon_box, 6, 6)
            icons.check(p, icon_box.adjusted(24, 24, -24, -24), C.green, 5)
            x = r.left() + 152
            quiet = "No Active Cabin Alerts" if channel == "cms" else "No Active Alerts"
            watch = "Monitoring occupants" if channel == "cms" else "Monitoring driver and cabin"
            self._text(p, x, r.top() + 128, quiet, C.green, self.f_alert)
            self._text(p, x, r.top() + 178, watch, C.white, self.f_row)
            if alert is not None:
                last = datetime.fromtimestamp(alert.timestamp).strftime("%H:%M:%S")
                self._text(p, x, r.top() + 245, f"Last: {alert.title} {last}", C.dim, self.f_small, max_w=r.right() - x - 20)
            else:
                self._text(p, x, r.top() + 245, datetime.now().strftime("%H:%M:%S"), C.white, self.f_row)

    def _system(self, p: QPainter, r: QRectF, ds, extra: str = "") -> None:
        sysd = self.shared.system
        mq = self.shared.mqtt
        parts = [f"{ds.fps:.0f} FPS"]
        if extra:
            parts.append(extra)
        if sysd is not None:
            if sysd.gpu_temp_c is not None:
                parts.append(f"GPU {sysd.gpu_temp_c:.0f}°C")
            elif sysd.max_temp_c is not None:
                parts.append(f"SoC {sysd.max_temp_c:.0f}°C")
            if sysd.gpu_load_pct is not None:
                parts.append(f"GPU {sysd.gpu_load_pct:.0f}%")
            parts.append(f"RAM {sysd.mem_used_mb / 1024:.1f}/{sysd.mem_total_mb / 1024:.1f} GB")
        self._text(p, r.left() + 30, r.top() + 37, "   ·   ".join(parts), C.label, self.f_small, max_w=r.width() - 60)
        if mq.enabled:
            dot = C.green if mq.connected else C.red
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(dot)
            p.drawEllipse(QPointF(r.left() + 38, r.top() + 64), 7, 7)
            state = "MQTT connected" if mq.connected else "MQTT offline"
            text = f"{state}   ·   sent {mq.acked}   ·   queued {mq.queued}   ·   {mq.success_rate:.1f}%"
        else:
            text = "MQTT uplink disabled"
        self._text(p, r.left() + 56, r.top() + 72, text, C.label if mq.enabled else C.dim, self.f_small, max_w=r.width() - 80)

    # ------------------------------------------------------------------ modes
    def _modes(self, p: QPainter) -> None:
        for rect, mode, text in self._mode_rects():
            active = mode == self.mode
            color = C.green if active else C.label
            p.setPen(QPen(color, 1.6))
            fill = QColor(color.red(), color.green(), color.blue(), 48 if active else 18)
            p.setBrush(fill)
            p.drawRoundedRect(rect, 8, 8)
            self._text(p, rect.center().x(), rect.top() + 28, text, color, self.f_small, "center")
        self._text(p, W - 28, 34, "Click a view", C.dim, self.f_small, "right")

    def _render_cms(self, p: QPainter) -> None:
        cabin = self.shared.passenger
        now = time.time()
        (_, r_occ), (_, r_map), (_, r_obj), (_, r_audio), (_, r_vlm), (_, r_cam), (_, r_safe), \
            (_, r_alert), (_, r_sys) = CMS_LAYOUT
        self._cabin_video(p, VIDEO_RECT, cabin)
        self._occupancy(p, r_occ, cabin)
        self._seat_map(p, r_map, cabin)
        self._objects(p, r_obj, cabin)
        self._audio(p, r_audio)
        self._vlm(p, r_vlm)
        self._cameras(p, r_cam, cabin)
        self._safety(p, r_safe, cabin)
        self._alert(p, r_alert, now, channel="cms")
        self._system(p, r_sys, self.shared.driver, extra=f"cabin {cabin.fps:.0f} FPS")

    def _render_split(self, p: QPainter) -> None:
        cabin = self.shared.passenger
        now = time.time()
        (_, r_drv), (_, r_cab), (_, r_da), (_, r_ca), (_, r_audio), (_, r_sys) = SPLIT_LAYOUT
        self._video(p, r_drv, self.shared.driver)
        self._cabin_video(p, r_cab, cabin)
        self._alert(p, r_da, now, channel="dms")
        self._alert(p, r_ca, now, channel="cms")
        self._audio(p, r_audio)
        self._system(p, r_sys, self.shared.driver, extra=f"cabin {cabin.fps:.0f} FPS")

    def _cover(self, p: QPainter, key: str, frame, stamp, rect: QRectF):
        if frame is None:
            return None
        fh, fw = frame.shape[:2]
        img = self._qimage(key, frame, stamp)
        if img is None:
            return None
        scale = max(rect.width() / fw, rect.height() / fh)
        ox = (fw * scale - rect.width()) / 2
        oy = (fh * scale - rect.height()) / 2
        p.drawImage(QRectF(rect.left() - ox, rect.top() - oy, fw * scale, fh * scale), img)

        def mapped(x, y):
            return QPointF(rect.left() + x * scale - ox, rect.top() + y * scale - oy)

        return mapped, fw, fh

    def _cabin_video(self, p: QPainter, rect: QRectF, cabin) -> None:
        p.save()
        clip = QPainterPath()
        clip.addRoundedRect(rect, 6, 6)
        p.setClipPath(clip)
        if cabin.frame is None:
            p.fillRect(rect, QColor("#05090d"))
            self._text(p, rect.center().x(), rect.center().y(), "Waiting for cabin camera...", C.dim,
                       self.f_alert, "center")
            p.restore()
            self._frame_border(p, rect)
            return
        covered = self._cover(p, "cabin", cabin.frame, cabin.frame_ts, rect)
        p.restore()
        self._frame_border(p, rect)
        if covered is None:
            return
        mapped, fw, fh = covered
        p.save()
        clip = QPainterPath()
        clip.addRoundedRect(rect, 6, 6)
        p.setClipPath(clip)
        kinds = list(cabin.face_kinds) + ["unknown"] * max(0, len(cabin.faces) - len(cabin.face_kinds))
        for box, kind in zip(cabin.faces, kinds):
            color = {"adult": C.green_box, "child": C.amber}.get(kind, C.blue)
            face = QRectF(mapped(box[0], box[1]), mapped(box[2], box[3]))
            p.setPen(QPen(color, 4))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(face)
            self._text(p, face.left() + 8, face.top() + 28, kind, color, self.f_small)
        for obj in cabin.objects:
            box = obj.bbox
            rect_o = QRectF(mapped(box[0], box[1]), mapped(box[2], box[3]))
            p.setPen(QPen(C.red_box, 3))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(rect_o)
            self._text(p, rect_o.left() + 6, max(rect.top() + 24, rect_o.top() - 8), obj.label, C.red, self.f_small)
        if cabin.child_present:
            banner = QRectF(rect.center().x() - 190, rect.top() + 16, 380, 52)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(C.red_tag)
            p.drawRoundedRect(banner, 4, 4)
            self._text(p, banner.center().x(), banner.top() + 36, "CHILD PRESENT", C.white, self.f_callout, "center")
        p.restore()

    def _occupancy(self, p: QPainter, r: QRectF, cabin) -> None:
        x1, x2 = r.left() + 34, r.right() - 34
        if not cabin.connected and cabin.frame is None:
            self._text(p, x1, r.top() + 110, "Cabin camera off", C.dim, self.f_row)
            return
        y = r.top() + 100
        if not cabin.seats:
            self._row(p, x1, x2, y, "Occupants", str(cabin.occupants), C.white)
            return
        for seat in cabin.seats:
            color = {"adult": C.green, "child": C.amber, "unknown": C.blue, "empty": C.dim}.get(seat.occupant, C.dim)
            self._row(p, x1, x2, y, seat.name.replace("_", " ").title(), seat.occupant.title(), color)
            y += 48

    def _seat_map(self, p: QPainter, r: QRectF, cabin) -> None:
        by_name = {s.name: s for s in cabin.seats}
        cells = (
            ("rear_left", QRectF(r.left() + 32, r.top() + 90, 150, 130)),
            ("rear_center", QRectF(r.left() + 196, r.top() + 90, 150, 130)),
            ("rear_right", QRectF(r.left() + 360, r.top() + 90, 150, 130)),
            ("front_passenger", QRectF(r.left() + 32, r.top() + 240, 478, 120)),
        )
        for name, cell in cells:
            seat = by_name.get(name)
            occupant = seat.occupant if seat is not None else "empty"
            color = {"adult": C.green, "child": C.amber, "unknown": C.blue}.get(occupant, C.dim)
            p.setPen(QPen(color, 2))
            p.setBrush(QColor(color.red(), color.green(), color.blue(), 36))
            p.drawRoundedRect(cell, 6, 6)
            self._text(p, cell.left() + 12, cell.top() + 32, name.replace("_", " ").title(), C.label, self.f_small,
                       max_w=cell.width() - 20)
            self._text(p, cell.left() + 12, cell.top() + 68, occupant.title(), color, self.f_row)
            self._text(p, cell.left() + 12, cell.bottom() - 16, "Belt Unknown", C.dim, self.f_small)

    def _objects(self, p: QPainter, r: QRectF, cabin) -> None:
        x1, x2 = r.left() + 34, r.right() - 34
        shown = [obj for obj in cabin.objects if obj.seat] or list(cabin.objects)
        if not shown:
            self._text(p, x1, r.top() + 110, "No objects in seat zones", C.green, self.f_row)
            return
        y = r.top() + 100
        for obj in shown[:5]:
            where = obj.seat.replace("_", " ") if obj.seat else "unassigned"
            self._row(p, x1, x2, y, obj.label.title(), where, C.amber if obj.seat else C.dim)
            y += 46

    def _cameras(self, p: QPainter, r: QRectF, cabin) -> None:
        x1, x2 = r.left() + 34, r.right() - 34
        driver = self.shared.driver
        driver_on = driver.frame is not None
        self._row(p, x1, x2, r.top() + 110, "Driver", "Active" if driver_on else "Off",
                  C.green if driver_on else C.dim)
        self._row(p, x1, x2, r.top() + 168, "Cabin", "Active" if cabin.connected else "Off",
                  C.green if cabin.connected else C.dim)
        self._row(p, x1, x2, r.top() + 226, "Cabin FPS", f"{cabin.fps:.0f}" if cabin.connected else "--", C.blue)

    def _safety(self, p: QPainter, r: QRectF, cabin) -> None:
        x1, x2 = r.left() + 34, r.right() - 34
        self._row(p, x1, x2, r.top() + 110, "Child present", "Yes" if cabin.child_present else "No",
                  C.amber if cabin.child_present else C.green)
        self._row(p, x1, x2, r.top() + 168, "Adult in cabin", "Yes" if cabin.adult_present else "No",
                  C.green if cabin.adult_present else C.dim)
        self._row(p, x1, x2, r.top() + 226, "Passenger belts", "Unknown", C.dim)
        self._text(p, x1, r.top() + 284, "No ignition or door signal, so this is not a left-behind alarm.",
                   C.dim, self.f_small, max_w=r.width() - 68)
        vlm = self.shared.vlm
        if vlm.policy:
            color = C.red if vlm.unsafe else C.amber if vlm.suppressed else C.green
            if vlm.categories:
                short = "Scene kept " + ", ".join(name.replace("_", " ") for name in vlm.categories)
            elif vlm.suppressed:
                short = "Scene clear, hazard suppressed"
            else:
                short = "Scene clear"
            if vlm.suppressed and vlm.categories:
                short += f" · {len(vlm.suppressed)} suppressed"
            self._text(p, x1, r.top() + 332, short, color, self.f_small, max_w=r.width() - 68)


def render_png(shared, bus, path: str, width: int = 2576, height: int = 1438, mode: str = "dms") -> None:
    """Render one frame of the dashboard to a PNG (used by tests and --snapshot)."""
    from PySide6.QtGui import QImage as _QImage

    from PySide6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])
    widget = Dashboard(shared, bus, fps=1)
    widget.timer.stop()
    widget.mode = mode if mode in MODES else "dms"
    image = _QImage(width, height, _QImage.Format.Format_ARGB32)
    image.fill(C.bg)
    painter = QPainter(image)
    painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing |
                           QPainter.RenderHint.SmoothPixmapTransform)
    s = min(width / W, height / H)
    painter.scale(s, s)
    widget._pan_x = None
    widget.render_canvas(painter)
    painter.end()
    image.save(path)
