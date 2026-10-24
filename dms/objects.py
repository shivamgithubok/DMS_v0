"""Lightweight object detectors for the driver channel.

* PhoneDetector  - NanoDet-Plus (COCO, 416 px, ~1 GFLOP): cell phone / cup / bottle boxes.
* HandDetector   - MediaPipe palm detector (192 px SSD) exported to ONNX: hand boxes.

Both run through ``inference.create_session`` (TensorRT EP FP16 on the Orin) and are
scheduled every Nth frame by the real-time detector; between runs the last boxes are
held, which is plenty for objects that move at hand speed.
"""

from __future__ import annotations

import cv2
import numpy as np

from .inference import create_session

COCO_PHONE = 67
COCO_CUP = 41
COCO_BOTTLE = 39
COCO_WINE_GLASS = 40
DRINK_CLASSES = {COCO_CUP, COCO_BOTTLE, COCO_WINE_GLASS}


def _letterbox(image: np.ndarray, size: int):
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    nh, nw = int(round(h * ratio)), int(round(w * ratio))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, ratio, left, top


def _nms(boxes: np.ndarray, scores: np.ndarray, score_thr: float, iou_thr: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)
    xywh = boxes.copy()
    xywh[:, 2:] -= xywh[:, :2]
    keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), score_thr, iou_thr)
    return np.array(keep, dtype=np.int64).reshape(-1)


class PhoneDetector:
    size = 416
    strides = (8, 16, 32)
    reg_max = 7
    mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    std = np.array([57.375, 57.12, 58.395], dtype=np.float32)

    def __init__(self, onnx_path, engine_path=None, score_threshold: float = 0.35,
                 allow_trt: bool = True, allow_cuda: bool = True) -> None:
        # The cabin monitor stays off TensorRT and off the CUDA execution provider.
        # This machine's ONNX Runtime CUDA build has no kernel for the Orin GPU
        # (cudaErrorNoKernelImageForDevice), and a second TensorRT context is not
        # worth it at 2 Hz.
        self.session = create_session(onnx_path, engine_path, threads=1, allow_trt=allow_trt,
                                      allow_cuda=allow_cuda)
        self.score_threshold = score_threshold
        self.input = self.session.inputs[0]
        self.cls_names: list[str] = []
        self.reg_names: list[str] = []
        self.centers = []
        for stride in self.strides:
            n = self.size // stride
            ys, xs = np.mgrid[0:n, 0:n]
            cx = xs.reshape(-1) * stride + 0.5 * (stride - 1)
            cy = ys.reshape(-1) * stride + 0.5 * (stride - 1)
            self.centers.append(np.stack([cx, cy], axis=1).astype(np.float32))
        self.project = np.arange(self.reg_max + 1, dtype=np.float32)

    def detect(self, frame: np.ndarray, classes=(COCO_PHONE,) + tuple(DRINK_CLASSES)):
        canvas, ratio, left, top = _letterbox(frame, self.size)
        blob = ((canvas.astype(np.float32) - self.mean) / self.std).transpose(2, 0, 1)[None]
        out = self.session.run({self.input: np.ascontiguousarray(blob)})
        if not self.cls_names:
            # Map outputs by shape (engines may reorder them): class maps (N, 80) and box
            # distributions (N, 32), each sorted from the finest level (most anchors) down.
            shaped = {name: arr.reshape(-1, arr.shape[-1]) for name, arr in out.items()}
            self.cls_names = sorted((n for n, a in shaped.items() if a.shape[1] == 80), key=lambda n: -shaped[n].shape[0])
            self.reg_names = sorted((n for n, a in shaped.items() if a.shape[1] == 4 * (self.reg_max + 1)),
                                    key=lambda n: -shaped[n].shape[0])
        boxes, scores, labels = [], [], []
        for stride, centers, cname, rname in zip(self.strides, self.centers, self.cls_names, self.reg_names):
            cls = out[cname].reshape(-1, 80)
            reg = out[rname].reshape(-1, 4, self.reg_max + 1)
            cls = cls[:, list(classes)]
            best = cls.max(axis=1)
            mask = best >= self.score_threshold
            if not mask.any():
                continue
            reg = reg[mask]
            reg = np.exp(reg - reg.max(axis=2, keepdims=True))
            reg /= reg.sum(axis=2, keepdims=True)
            dist = (reg @ self.project) * stride
            c = centers[mask]
            box = np.stack([c[:, 0] - dist[:, 0], c[:, 1] - dist[:, 1], c[:, 0] + dist[:, 2], c[:, 1] + dist[:, 3]], 1)
            boxes.append(box)
            scores.append(best[mask])
            labels.append(np.array(classes)[cls[mask].argmax(axis=1)])
        if not boxes:
            return []
        boxes = np.concatenate(boxes)
        scores = np.concatenate(scores)
        labels = np.concatenate(labels)
        keep = _nms(boxes, scores, self.score_threshold, 0.5)
        result = []
        h, w = frame.shape[:2]
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            x1 = float(np.clip((x1 - left) / ratio, 0, w - 1))
            x2 = float(np.clip((x2 - left) / ratio, 0, w - 1))
            y1 = float(np.clip((y1 - top) / ratio, 0, h - 1))
            y2 = float(np.clip((y2 - top) / ratio, 0, h - 1))
            result.append(((x1, y1, x2, y2), float(scores[i]), int(labels[i])))
        return result


def palm_anchors() -> np.ndarray:
    """SSD anchors for the 192x192 palm model: 24x24x2 (stride 8) + 12x12x6 (stride 16)."""
    out = []
    for grid, per_cell in ((24, 2), (12, 6)):
        for y in range(grid):
            for x in range(grid):
                out.extend([((x + 0.5) / grid, (y + 0.5) / grid)] * per_cell)
    return np.array(out, dtype=np.float32)


class HandDetector:
    size = 192

    def __init__(self, onnx_path, engine_path=None, score_threshold: float = 0.5) -> None:
        self.session = create_session(onnx_path, engine_path, threads=1)
        self.score_threshold = score_threshold
        self.anchors = palm_anchors()
        self.input = self.session.inputs[0]
        self.reg_name = self.score_name = None

    def detect(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        canvas, ratio, left, top = _letterbox(frame, self.size)
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)[None] / 255.0
        out = self.session.run({self.input: np.ascontiguousarray(blob)})
        if self.reg_name is None:  # (1, 2016, 18) regressors vs (1, 2016, 1) scores
            self.reg_name = max(out, key=lambda n: out[n].shape[-1])
            self.score_name = min(out, key=lambda n: out[n].shape[-1])
        reg = out[self.reg_name][0]
        logits = out[self.score_name][0, :, 0].astype(np.float64)
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
        mask = scores >= self.score_threshold
        if not mask.any():
            return []
        reg, scores, anchors = reg[mask], scores[mask], self.anchors[mask]
        cxy = reg[:, :2] / self.size + anchors
        wh = reg[:, 2:4] / self.size
        boxes = np.concatenate([cxy - wh / 2, cxy + wh / 2], axis=1) * self.size
        keep = _nms(boxes, scores.astype(np.float32), self.score_threshold, 0.3)
        result = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            # palms are detected tight; grow to cover the fingers
            cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * 1.6, (y2 - y1) * 1.6
            x1, x2 = (cx - bw / 2 - left) / ratio, (cx + bw / 2 - left) / ratio
            y1, y2 = (cy - bh / 2 - top) / ratio, (cy + bh / 2 - top) / ratio
            result.append((float(np.clip(x1, 0, w - 1)), float(np.clip(y1, 0, h - 1)),
                           float(np.clip(x2, 0, w - 1)), float(np.clip(y2, 0, h - 1)), float(scores[i])))
        return result


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def overlap(a, b) -> float:
    """Fraction of box a covered by box b."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area
