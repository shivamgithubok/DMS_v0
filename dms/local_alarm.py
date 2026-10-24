"""In-cabin audible alarm for high-severity alerts (the "local" half of local + MQTT).

Plays a short two-tone chime through ALSA ``aplay`` (always present on JetPack) without
blocking the caller. Rate-limited so an alert storm does not turn into a siren.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
import wave

import numpy as np

from . import alerts as A
from .config import data_path

log = logging.getLogger("dms.alarm")


def _chime(path, rate: int = 22050) -> None:
    t = np.arange(int(rate * 0.18)) / rate
    env = np.minimum(1.0, np.minimum(t / 0.01, (t[-1] - t) / 0.03))
    tone = lambda f: np.sin(2 * np.pi * f * t) * env  # noqa: E731
    gap = np.zeros(int(rate * 0.06))
    sig = np.concatenate([tone(988), gap, tone(1319), gap, tone(988), gap, tone(1319)]) * 0.6
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes((sig * 32767).astype(np.int16).tobytes())


class LocalAlarm:
    def __init__(self, min_interval_s: float = 4.0, device: str | None = None) -> None:
        self.path = data_path("chime.wav")
        if not self.path.exists():
            _chime(self.path)
        self.player = shutil.which("aplay")
        self.device = device
        self.min_interval_s = min_interval_s
        self._last = 0.0
        if not self.player:
            log.info("aplay not found; audible alarm disabled (visual alerts unaffected)")

    def __call__(self, alert: A.Alert) -> None:
        if alert.severity != "high" or not self.player:
            return
        now = time.monotonic()
        if now - self._last < self.min_interval_s:
            return
        self._last = now
        cmd = [self.player, "-q"] + (["-D", self.device] if self.device else []) + [str(self.path)]
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:
            log.debug("alarm playback failed: %s", exc)
