"""Record cabin video and microphone only while a guardian has Record on.

Alert clips stay in dms.clip_recorder. This file is the guardian's own take.
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
import wave
from pathlib import Path

import cv2
import numpy as np

from .webrtc_publisher import stitch_cabin

log = logging.getLogger("dms.guardian_media")

AUDIO_RATE = 16000
MAX_MS = 10 * 60 * 1000


def encode_recording(frames: list[bytes], pcm: bytes, sample_rate: int = AUDIO_RATE) -> bytes:
    """H.264 video plus AAC audio from the cabin microphone."""
    if not frames:
        raise ValueError("recording needs frames")
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        for index, jpeg in enumerate(frames):
            (root / f"{index:05d}.jpg").write_bytes(jpeg)
        span_s = max(0.5, len(pcm) / 2 / sample_rate) if pcm else max(0.5, len(frames) / 2)
        fps = max(1.0, min(8.0, (len(frames) - 1) / span_s)) if len(frames) > 1 else 2.0
        out = root / "clip.mp4"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", f"{fps:.3f}",
            "-i", str(root / "%05d.jpg"),
        ]
        if pcm:
            wav_path = root / "cabin.wav"
            with wave.open(str(wav_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(sample_rate)
                handle.writeframes(pcm)
            cmd.extend(["-i", str(wav_path), "-c:a", "aac", "-shortest"])
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
        self._frames: list[bytes] = []
        self._pcm = bytearray()

    def stop(self) -> None:
        self._halt.set()

    def _post(self, path: str, body: dict) -> dict:
        url = self.cfg.api_url.rstrip("/") + path
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=float(self.cfg.timeout_s)) as resp:
            return json.loads(resp.read().decode() or "{}")

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

    def _sample(self) -> None:
        jpeg = self._cabin_jpeg()
        if jpeg:
            self._frames.append(jpeg)
        self._pcm.extend(self._pcm_chunk())

    def _finish(self) -> None:
        rec_id, frames, pcm = self._rec_id, self._frames, bytes(self._pcm)
        self._rec_id = ""
        self._frames = []
        self._pcm = bytearray()
        self._started_at = 0
        if not rec_id or not frames:
            return
        try:
            video = encode_recording(frames, pcm)
        except (OSError, RuntimeError, ValueError) as exc:
            log.warning("guardian recording encode failed for %s: %s", rec_id, exc)
            return
        try:
            self._post(
                f"/v1/devices/{self.device_id}/recordings/{rec_id}",
                {"key": self.cfg.device_key, "video_b64": base64.b64encode(video).decode()},
            )
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
            log.warning("guardian recording upload failed for %s: %s", rec_id, exc)
            return
        log.info("guardian recording %s uploaded (%d frames, audio %d bytes)", rec_id, len(frames), len(pcm))

    def run(self) -> None:
        log.info("guardian recorder → %s", self.cfg.api_url)
        while not self._halt.is_set():
            try:
                token = self._post(f"/v1/devices/{self.device_id}/media-token", {"key": self.cfg.device_key})
                current = token.get("recording") or None
                rec_id = str(current.get("id") or "") if isinstance(current, dict) else ""
                if rec_id and rec_id != self._rec_id:
                    if self._rec_id:
                        self._finish()
                    self._rec_id = rec_id
                    self._started_at = int(time.time() * 1000)
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
            except Exception:
                log.exception("guardian recorder failed")
            self._halt.wait(self.sample_s)
