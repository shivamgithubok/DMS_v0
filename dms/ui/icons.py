"""Vector icons drawn with QPainter (no image assets, crisp at any scale)."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen


def _pen(color: QColor, width: float) -> QPen:
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def warning(p: QPainter, rect: QRectF, fill: QColor, mark: QColor) -> None:
    w, h = rect.width(), rect.height()
    path = QPainterPath()
    r = min(w, h) * 0.08
    top = QPointF(rect.center().x(), rect.top())
    left = QPointF(rect.left(), rect.bottom())
    right = QPointF(rect.right(), rect.bottom())
    path.moveTo(top)
    path.lineTo(right)
    path.lineTo(left)
    path.closeSubpath()
    p.save()
    p.setPen(_pen(fill, r * 2))
    p.setBrush(QBrush(fill))
    p.drawPath(path)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(mark))
    cx = rect.center().x()
    bar_w = w * 0.12
    p.drawRoundedRect(QRectF(cx - bar_w / 2, rect.top() + h * 0.30, bar_w, h * 0.38), bar_w / 2, bar_w / 2)
    p.drawEllipse(QPointF(cx, rect.top() + h * 0.82), bar_w * 0.62, bar_w * 0.62)
    p.restore()


def phone(p: QPainter, rect: QRectF, color: QColor, width: float = 2.5) -> None:
    w, h = rect.width(), rect.height()
    body = QRectF(rect.left() + w * 0.22, rect.top(), w * 0.56, h)
    p.save()
    p.setPen(_pen(color, width))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(body, w * 0.1, w * 0.1)
    p.drawLine(QPointF(body.left() + body.width() * 0.3, body.bottom() - h * 0.14),
               QPointF(body.right() - body.width() * 0.3, body.bottom() - h * 0.14))
    p.restore()


def hand(p: QPainter, rect: QRectF, color: QColor, width: float = 2.2) -> None:
    """Open palm in a circle (matches the 'Hand Position' glyph style)."""
    p.save()
    p.setPen(_pen(color, width))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(rect)
    c = rect.center()
    s = rect.width() / 2
    path = QPainterPath()
    path.moveTo(c.x() - s * 0.32, c.y() + s * 0.45)
    path.lineTo(c.x() - s * 0.32, c.y() - s * 0.05)
    for i, top in enumerate((-0.42, -0.52, -0.45, -0.3)):
        x = c.x() - s * 0.22 + i * s * 0.16
        path.moveTo(x, c.y() + s * 0.05)
        path.lineTo(x, c.y() + top * s)
    path.moveTo(c.x() - s * 0.32, c.y() + s * 0.45)
    path.lineTo(c.x() + s * 0.30, c.y() + s * 0.45)
    path.lineTo(c.x() + s * 0.30, c.y() - s * 0.05)
    p.drawPath(path)
    p.restore()


def seatbelt(p: QPainter, rect: QRectF, color: QColor, width: float = 2.2) -> None:
    w, h = rect.width(), rect.height()
    p.save()
    p.setPen(_pen(color, width))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QPointF(rect.left() + w * 0.55, rect.top() + h * 0.14), w * 0.13, w * 0.13)
    torso = QPainterPath()
    torso.moveTo(rect.left() + w * 0.2, rect.bottom())
    torso.lineTo(rect.left() + w * 0.35, rect.top() + h * 0.42)
    torso.quadTo(rect.left() + w * 0.55, rect.top() + h * 0.3, rect.left() + w * 0.75, rect.top() + h * 0.42)
    torso.lineTo(rect.left() + w * 0.9, rect.bottom())
    p.drawPath(torso)
    p.drawLine(QPointF(rect.left() + w * 0.72, rect.top() + h * 0.38), QPointF(rect.left() + w * 0.3, rect.bottom() - h * 0.05))
    p.restore()


def cigarette(p: QPainter, rect: QRectF, color: QColor, width: float = 2.2) -> None:
    w, h = rect.width(), rect.height()
    p.save()
    p.setPen(_pen(color, width))
    p.setBrush(Qt.BrushStyle.NoBrush)
    # hand holding a stick, plus smoke
    p.drawRoundedRect(QRectF(rect.left(), rect.top() + h * 0.62, w * 0.82, h * 0.16), 2, 2)
    p.drawLine(QPointF(rect.left() + w * 0.66, rect.top() + h * 0.62), QPointF(rect.left() + w * 0.66, rect.top() + h * 0.78))
    smoke = QPainterPath()
    smoke.moveTo(rect.left() + w * 0.88, rect.top() + h * 0.55)
    smoke.cubicTo(rect.left() + w * 0.70, rect.top() + h * 0.40, rect.left() + w * 1.0, rect.top() + h * 0.28,
                  rect.left() + w * 0.82, rect.top() + h * 0.05)
    p.drawPath(smoke)
    base = QPainterPath()
    base.moveTo(rect.left() + w * 0.1, rect.bottom())
    base.quadTo(rect.left() + w * 0.3, rect.top() + h * 0.80, rect.left() + w * 0.55, rect.bottom())
    p.drawPath(base)
    p.restore()


def cup(p: QPainter, rect: QRectF, color: QColor, width: float = 2.2) -> None:
    w, h = rect.width(), rect.height()
    p.save()
    p.setPen(_pen(color, width))
    p.setBrush(Qt.BrushStyle.NoBrush)
    body = QPainterPath()
    body.moveTo(rect.left() + w * 0.15, rect.top() + h * 0.3)
    body.lineTo(rect.left() + w * 0.25, rect.bottom() - h * 0.12)
    body.lineTo(rect.left() + w * 0.65, rect.bottom() - h * 0.12)
    body.lineTo(rect.left() + w * 0.75, rect.top() + h * 0.3)
    body.closeSubpath()
    p.drawPath(body)
    p.drawArc(QRectF(rect.left() + w * 0.62, rect.top() + h * 0.42, w * 0.3, h * 0.28), -90 * 16, 180 * 16)
    p.drawLine(QPointF(rect.left() + w * 0.45, rect.top() + h * 0.3), QPointF(rect.left() + w * 0.6, rect.top()))
    p.drawLine(QPointF(rect.left() + w * 0.05, rect.bottom()), QPointF(rect.left() + w * 0.85, rect.bottom()))
    p.restore()


def check(p: QPainter, rect: QRectF, color: QColor, width: float = 4.0) -> None:
    p.save()
    p.setPen(_pen(color, width))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(rect)
    path = QPainterPath()
    path.moveTo(rect.left() + rect.width() * 0.28, rect.top() + rect.height() * 0.52)
    path.lineTo(rect.left() + rect.width() * 0.44, rect.top() + rect.height() * 0.68)
    path.lineTo(rect.left() + rect.width() * 0.74, rect.top() + rect.height() * 0.36)
    p.drawPath(path)
    p.restore()


def clock(p: QPainter, rect: QRectF, color: QColor, width: float = 2.0) -> None:
    p.save()
    p.setPen(_pen(color, width))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(rect)
    c = rect.center()
    p.drawLine(c, QPointF(c.x(), rect.top() + rect.height() * 0.25))
    p.drawLine(c, QPointF(rect.left() + rect.width() * 0.72, c.y()))
    p.restore()


def arrow(p: QPainter, start: QPointF, end: QPointF, color: QColor, width: float = 4.0, head: float = 16.0) -> None:
    import math

    p.save()
    p.setPen(_pen(color, width))
    p.drawLine(start, end)
    angle = math.atan2(end.y() - start.y(), end.x() - start.x())
    path = QPainterPath()
    path.moveTo(end)
    path.lineTo(end.x() - head * math.cos(angle - 0.45), end.y() - head * math.sin(angle - 0.45))
    path.lineTo(end.x() - head * math.cos(angle + 0.45), end.y() - head * math.sin(angle + 0.45))
    path.closeSubpath()
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(color))
    p.drawPath(path)
    p.restore()
