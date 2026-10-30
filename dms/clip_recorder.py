"""Keep a few seconds of JPEG frames and upload them when an alert fires.

The guardian trip page plays this list. It is not the live Watch stream.
Each alert gets its own clip id on the current ride.
"""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from . import alerts as A

log = logging.getLogger("dms.clips")

BEFORE_MS = 4000
AFTER_MS = 4000
MAX_FRAMES = 24


def frames_for_alert(samples: list[tuple[int, bytes]], alert_ms: int, before_ms: int = BEFORE_MS, after_ms: int = AFTER_MS, limit: int = MAX_FRAMES) -> list[tuple[int, bytes]]:
    """Frames whose capture time falls in the window around the alert."""
    lo, hi = alert_ms - before_ms, alert_ms + after_ms
    chosen = [item for item in samples if lo <= item[0] <= hi]
    if len(chosen) <= limit:
        return chosen
    step = len(chosen) / limit
    return [chosen[int(i * step)] for i in range(limit)]


def encode_clip_mp4(frames: list[tuple[int, bytes]]) -> bytes:
    """Pack the alert window into one H.264 MP4 the guardian page can play."""
    if not frames:
        raise ValueError("clip needs frames")
    ordered = list(frames)
    if len(ordered) == 1:
        ordered = [ordered[0], (ordered[0][0] + 500, ordered[0][1])]
    span_ms = max(500, ordered[-1][0] - ordered[0][0])
    fps = max(1.0, min(8.0, (len(ordered) - 1) * 1000.0 / span_ms))
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        for index, (_, jpeg) in enumerate(ordered):
            (root / f"{index:03d}.jpg").write_bytes(jpeg)
        out = root / "clip.mp4"
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", f"{fps:.3f}",
                "-i", str(root / "%03d.jpg"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(out),
            ],
            capture_output=True,
        )
        if proc.returncode != 0 or not out.is_file():
            raise RuntimeError(proc.stderr.decode("utf-8", "replace")[:300] or "ffmpeg failed")
        return out.read_bytes()


def clip_channel(alert: A.Alert) -> str:
    if alert.channel in {"dms", "cms"}:
        return alert.channel
    if alert.type in A.CMS_TYPES:
        return "cms"
    return "dms"


class ClipRecorder(threading.Thread):
    def __init__(self, cfg, capture, bus: A.AlertBus, before_ms: int = BEFORE_MS, after_ms: int = AFTER_MS, sample_s: float = 0.5) -> None:
        super().__init__(name="clip-recorder", daemon=True)
        self.cfg = cfg.webrtc
        self.device_id = cfg.device_id
        self.capture = capture
        self.bus = bus
        self.before_ms = before_ms
        self.after_ms = after_ms
        self.sample_s = sample_s
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self._rings: dict[str, deque] = {"dms": deque(), "cms": deque()}
        self.uploaded = 0
        self.errors = 0
        bus.subscribe(self._on_alert)

    def stop(self) -> None:
        self._halt.set()

    def _on_alert(self, alert: A.Alert) -> None:
        threading.Thread(target=self._upload, args=(alert,), name="clip-upload", daemon=True).start()

    def _grab(self, channel: str) -> np.ndarray | None:
        grabber = self.capture.driver if channel == "dms" else getattr(self.capture, "passenger", None)
        if grabber is None:
            grabber = self.capture.driver
        latest = grabber.latest() if grabber is not None else None
        image = getattr(latest, "image", None)
        if image is None:
            return None
        h, w = image.shape[:2]
        scale = min(1.0, 320 / max(w, 1))
        if scale < 1.0:
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        return image

    def _sample(self) -> None:
        now = int(time.time() * 1000)
        keep_after = now - (self.before_ms + self.after_ms + 1000)
        for channel in ("dms", "cms"):
            image = self._grab(channel)
            if image is None:
                continue
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if not ok:
                continue
            with self._lock:
                ring = self._rings[channel]
                ring.append((now, buf.tobytes()))
                while ring and ring[0][0] < keep_after:
                    ring.popleft()

    def run(self) -> None:
        log.info("clip recorder → %s (%.1fs before, %.1fs after)", self.cfg.api_url, self.before_ms / 1000, self.after_ms / 1000)
        while not self._halt.is_set():
            try:
                self._sample()
            except Exception:
                log.exception("clip sample failed")
            self._halt.wait(self.sample_s)

    def _upload(self, alert: A.Alert) -> None:
        if self.after_ms > 0:
            if self._halt.wait(self.after_ms / 1000):
                return
        channel = clip_channel(alert)
        alert_ms = int(alert.timestamp * 1000)
        with self._lock:
            samples = list(self._rings.get(channel) or [])
        frames = frames_for_alert(samples, alert_ms, self.before_ms, self.after_ms)
        if not frames:
            image = self._grab(channel)
            if image is None:
                log.warning("alert %s has no frames for a clip", alert.id)
                return
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if not ok:
                return
            frames = [(alert_ms, buf.tobytes())]
        body = {
            "key": self.cfg.device_key,
            "channel": channel,
            "alert_id": alert.id,
            "start_at": frames[0][0],
            "end_at": frames[-1][0],
            "frames": [{"at": at, "image_b64": base64.b64encode(raw).decode()} for at, raw in frames],
        }
        try:
            body["video_b64"] = base64.b64encode(encode_clip_mp4(frames)).decode()
        except (OSError, RuntimeError, ValueError) as exc:
            log.warning("clip mp4 encode failed for %s: %s", alert.id, exc)
        url = self.cfg.api_url.rstrip("/") + f"/v1/devices/{self.device_id}/clips"
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=float(self.cfg.timeout_s)) as resp:
                payload = json.loads(resp.read().decode() or "{}")
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
            self.errors += 1
            log.warning("clip upload failed for %s: %s", alert.id, exc)
            return
        self.uploaded += 1
        log.info("clip %s for alert %s (%d frames)", payload.get("id"), alert.id, len(frames))
