"""Record cabin video and microphone only while a guardian has Record on.

Alert clips stay in dms.clip_recorder. This file is the guardian's own take.
"""

from __future__ import annotations

import base64
import json
import logging
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
from pathlib import Path

import cv2
import numpy as np

from .cloud_api import post_json
from .webrtc_publisher import stitch_cabin

log = logging.getLogger("dms.guardian_media")

AUDIO_RATE = 16000
MAX_MS = 10 * 60 * 1000


def encode_recording(root: Path, frames: int, sample_rate: int = AUDIO_RATE) -> bytes:
    """H.264 video plus AAC audio from JPEGs and PCM already on disk.

    The take is written as it is captured. Holding every JPEG and the whole
    microphone buffer in RAM grew without a break for up to 10 minutes.
    """
    if frames <= 0:
        raise ValueError("recording needs frames")
    pcm_path = root / "cabin.pcm"
    pcm_size = pcm_path.stat().st_size if pcm_path.is_file() else 0
    span_s = max(0.5, pcm_size / 2 / sample_rate) if pcm_size else max(0.5, frames / 2)
    fps = max(1.0, min(8.0, (frames - 1) / span_s)) if frames > 1 else 2.0
    out = root / "clip.mp4"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", f"{fps:.3f}",
        "-i", str(root / "%05d.jpg"),
    ]
    if pcm_size:
        cmd.extend([
            "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", str(pcm_path),
            "-c:a", "aac", "-shortest",
        ])
    cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)])
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not out.is_file():
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[:300] or "ffmpeg failed")
    return out.read_bytes()


class GuardianRecorder(threading.Thread):
    def __init__(self, cfg, capture, audio=None, sample_s: float = 0.5) -> None:
        super().__init__(name="guardian-recorder", daemon=True)
        self.cfg = cfg.webrtc
        self.device_id = cfg.device_id
        self.capture = capture
        self.audio = audio
        self.sample_s = sample_s
        self._halt = threading.Event()
        self._rec_id = ""
        self._started_at = 0
        self._root: Path | None = None
        self._pcm_fh = None
        self._count = 0

    def stop(self) -> None:
        self._halt.set()

    def _post(self, path: str, body: dict) -> dict:
        return post_json(self.cfg.api_url.rstrip("/") + path, body, float(self.cfg.timeout_s))

    def _cabin_jpeg(self) -> bytes | None:
        def image(track: str):
            grabber = self.capture.driver if track == "driver" else getattr(self.capture, "passenger", None)
            latest = grabber.latest() if grabber is not None else None
            return getattr(latest, "image", None)

        frame = stitch_cabin(image("driver"), image("passenger"))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return buf.tobytes() if ok else None

    def _pcm_chunk(self) -> bytes:
        n = max(320, int(AUDIO_RATE * self.sample_s))
        ring = None
        if self.audio is not None:
            ring = getattr(self.audio, "ring16", None) or getattr(self.audio, "ring", None)
        if ring is None or getattr(ring, "total", 0) <= 0:
            return np.zeros(n, dtype=np.int16).tobytes()
        samples, _ = ring.latest(n)
        pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
        return (pcm * 32767.0).astype(np.int16).tobytes()

    def _open_take(self, rec_id: str) -> None:
        self._discard()
        self._rec_id = rec_id
        self._started_at = int(time.time() * 1000)
        self._root = Path(tempfile.mkdtemp(prefix="guardian-rec-"))
        self._pcm_fh = open(self._root / "cabin.pcm", "wb")
        self._count = 0

    def _discard(self) -> None:
        if self._pcm_fh is not None:
            self._pcm_fh.close()
            self._pcm_fh = None
        root = self._root
        self._root = None
        self._count = 0
        self._rec_id = ""
        self._started_at = 0
        if root is not None:
            shutil.rmtree(root, ignore_errors=True)

    def _sample(self) -> None:
        if self._root is None or self._pcm_fh is None:
            return
        jpeg = self._cabin_jpeg()
        if jpeg:
            (self._root / f"{self._count:05d}.jpg").write_bytes(jpeg)
            self._count += 1
        self._pcm_fh.write(self._pcm_chunk())

    def _finish(self) -> None:
        if self._pcm_fh is not None:
            self._pcm_fh.close()
            self._pcm_fh = None
        rec_id, root, frames = self._rec_id, self._root, self._count
        self._rec_id = ""
        self._root = None
        self._count = 0
        self._started_at = 0
        if not rec_id or root is None or frames <= 0:
            if root is not None:
                shutil.rmtree(root, ignore_errors=True)
            return
        try:
            video = encode_recording(root, frames)
            self._post(
                f"/v1/devices/{self.device_id}/recordings/{rec_id}",
                {"key": self.cfg.device_key, "video_b64": base64.b64encode(video).decode()},
            )
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            log.warning("guardian recording upload failed for %s: %s", rec_id, exc)
            return
        finally:
            shutil.rmtree(root, ignore_errors=True)
        log.info("guardian recording %s uploaded (%d frames)", rec_id, frames)

    def run(self) -> None:
        log.info("guardian recorder → %s", self.cfg.api_url)
        try:
            while not self._halt.is_set():
                try:
                    token = self._post(f"/v1/devices/{self.device_id}/media-token", {"key": self.cfg.device_key})
                    current = token.get("recording") or None
                    rec_id = str(current.get("id") or "") if isinstance(current, dict) else ""
                    if rec_id and rec_id != self._rec_id:
                        if self._rec_id:
                            self._finish()
                        self._open_take(rec_id)
                        log.info("guardian recording started %s", rec_id)
                    if self._rec_id and not rec_id:
                        self._finish()
                    elif self._rec_id:
                        self._sample()
                        if int(time.time() * 1000) - self._started_at >= MAX_MS:
                            log.info("guardian recording %s reached 10 minutes", self._rec_id)
                            self._finish()
                except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
                    log.warning("guardian recorder poll failed: %s", exc)
                    self._halt.wait(max(2.0, self.sample_s))
                    continue
                except Exception:
                    log.exception("guardian recorder failed")
                self._halt.wait(self.sample_s)
        finally:
            self._discard()
