#!/usr/bin/env python3
"""Enroll a driver into data/drivers/roster.json.

    python tools/enroll_driver.py --id 4827 --name "Emily Carter" --age 28 --image face.jpg
    python tools/enroll_driver.py --id 4827 --name "Emily Carter" --age 28 --camera csi:0 --frames 20

With --camera, several frames are averaged into one embedding (more robust to pose).
The largest face in each image is used. The cropped face is stored as the ID photo
unless --photo is given.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dms.bootstrap import ensure_venv  # noqa: E402

ensure_venv()

from dms.capture import Camera  # noqa: E402
from dms.config import data_path, load_config, model_path  # noqa: E402
from dms.driver_id import DriverIdentifier, DriverRoster  # noqa: E402
from dms.face import FaceDetector  # noqa: E402


def portrait(image: np.ndarray, bbox, size=(276, 346)) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 + 0.05 * (y2 - y1)
    half_h = (y2 - y1) * 0.85
    half_w = half_h * size[0] / size[1]
    h, w = image.shape[:2]
    pad = int(max(half_w, half_h)) + 2
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    xa, ya = int(cx - half_w) + pad, int(cy - half_h) + pad
    crop = padded[ya:ya + int(2 * half_h), xa:xa + int(2 * half_w)]
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--age", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--image", action="append", default=[], help="face image (repeatable)")
    ap.add_argument("--camera", help="camera source: auto:0 (CSI, else USB), csi:0, usb:0, v4l2:0")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--photo", help="ID photo to show on the dashboard (default: crop of the first face)")
    ap.add_argument("--config", default="config/dms.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config if Path(ROOT / args.config).exists() else None)
    detector = FaceDetector(model_path(cfg.realtime.face_detector), 320)
    roster = DriverRoster(data_path("drivers", "roster.json"))
    identifier = DriverIdentifier(model_path(cfg.realtime.face_recognizer), roster)

    images = [cv2.imread(p) for p in args.image]
    if any(img is None for img in images):
        print("could not read one of the images", file=sys.stderr)
        return 2
    if args.camera:
        cam = Camera("enroll", args.camera, 640, 480, 30).start()
        seq, deadline = 0, time.monotonic() + 20
        print(f"look at the camera; capturing {args.frames} frames...")
        while len(images) < args.frames + len(args.image) and time.monotonic() < deadline:
            frame = cam.wait_next(seq, 2.0)
            if frame:
                seq = frame.seq
                images.append(frame.image.copy())
                time.sleep(0.1)
        cam.stop()
    if not images:
        print("no input; use --image or --camera", file=sys.stderr)
        return 2

    vectors, first = [], None
    for image in images:
        faces = detector.detect(image)
        if not faces:
            continue
        vectors.append(identifier.embed(image, faces[0].row))
        if first is None:
            first = (image, faces[0].bbox)
    if not vectors:
        print("no face found in the input", file=sys.stderr)
        return 3
    embedding = np.mean(vectors, axis=0)
    embedding /= np.linalg.norm(embedding)

    photo_name = f"driver_{args.id}.jpg"
    photo_path = roster.path.parent / photo_name
    if args.photo:
        photo = cv2.imread(args.photo)
        if photo is None:
            print(f"cannot read photo {args.photo}", file=sys.stderr)
            return 2
    else:
        photo = portrait(*first)
    cv2.imwrite(str(photo_path), photo, [cv2.IMWRITE_JPEG_QUALITY, 92])

    roster.upsert({"driver_id": str(args.id), "name": args.name, "age": args.age, "notes": args.notes,
                   "photo": photo_name, "samples": len(vectors),
                   "embedding": [round(float(v), 6) for v in embedding]})
    print(f"enrolled {args.name} (ID {args.id}) from {len(vectors)} face sample(s) -> {roster.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
