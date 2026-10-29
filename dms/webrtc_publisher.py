"""Publish driver/cabin stills (dummy) or LiveKit tracks (when LIVEKIT_URL is wss).

v1 dummy path: POST JPEG frames to the guardian API. Guardian fetches
``GET /v1/rides/{id}/video/preview``. A real LiveKit SFU uses the same media-token.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

from .state import WebRtcState

log = logging.getLogger("dms.webrtc")


def dummy_frame(track: str, width: int = 320, height: int = 180, seq: int = 0) -> np.ndarray:
    """Synthetic BGR frame so video can be tested without cameras."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    color = (40, 90, 180) if track == "driver" else (40, 140, 90)
    image[:] = color
    cv2.putText(image, track.upper(), (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(image, f"seq {seq}", (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
    cv2.putText(image, time.strftime("%H:%M:%S"), (12, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return image


def jpeg_b64(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        raise ValueError("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode()


class WebRtcPublisher(threading.Thread):
    def __init__(self, cfg, capture, shared) -> None:
        super().__init__(name="webrtc-publisher", daemon=True)
        self.cfg = cfg.webrtc
        self.capture = capture
        self.shared = shared
        self.device_id = cfg.device_id
        self._halt = threading.Event()
        self.seq = 0
        self.errors = 0
        self.last_token = ""

    def stop(self) -> None:
        self._halt.set()

    def _publish_state(self, **changes) -> None:
        current = getattr(self.shared, "webrtc", WebRtcState())
        values = {k: getattr(current, k) for k in current.__dataclass_fields__}
        values.update(changes)
        values.update(enabled=True, seq=self.seq, errors=self.errors)
        self.shared.webrtc = WebRtcState(**values)

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

    def _frame_for(self, track: str) -> np.ndarray:
        if self.cfg.dummy:
            return dummy_frame(track, seq=self.seq)
        grabber = self.capture.driver if track == "driver" else self.capture.passenger
        latest = grabber.latest() if grabber is not None else None
        if latest is None or getattr(latest, "image", None) is None:
            return dummy_frame(track, seq=self.seq)
        image = latest.image
        h, w = image.shape[:2]
        scale = min(1.0, 320 / max(w, 1))
        if scale < 1.0:
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        return image

    def run(self) -> None:
        self._publish_state(status="starting", api=self.cfg.api_url)
        log.info("WebRTC publisher → %s", self.cfg.api_url)
        while not self._halt.is_set():
            try:
                token_body = self._post(
                    f"/v1/devices/{self.device_id}/media-token",
                    {"key": self.cfg.device_key},
                )
                self.last_token = token_body.get("token") or ""
                url = str(token_body.get("url") or "")
                mode = token_body.get("mode") or "dummy"
                self._publish_state(status="online", mode=mode, room=token_body.get("room") or "", url=url)
                if mode == "livekit" and url.startswith(("ws://", "wss://")):
                    try:
                        asyncio.run(self._livekit(token_body))
                    except Exception as exc:
                        self.errors += 1
                        self._publish_state(status="error", last_error=str(exc))
                        log.warning("LiveKit publish failed: %s", exc)
                        self._halt.wait(float(self.cfg.interval_s))
                else:
                    self._post_stills()
                    self._halt.wait(float(self.cfg.interval_s))
            except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
                self.errors += 1
                self._publish_state(status="error", last_error=str(exc))
                log.warning("WebRTC publish failed: %s", exc)
                self._halt.wait(float(self.cfg.interval_s))

    def _post_stills(self) -> None:
        now = int(time.time() * 1000)
        for track in ("driver", "cabin"):
            self._post(
                f"/v1/devices/{self.device_id}/video/frame",
                {"key": self.cfg.device_key, "track": track, "at": now, "image_b64": jpeg_b64(self._frame_for(track))},
            )
        self.seq += 1

    async def _livekit(self, info: dict) -> None:
        from livekit import rtc

        room = rtc.Room()
        await room.connect(info["url"], info["token"])
        log.info("LiveKit connected room=%s", info.get("room"))
        sources = {}
        for name in ("driver", "cabin"):
            source = rtc.VideoSource(320, 180)
            track = rtc.LocalVideoTrack.create_video_track(name, source)
            await room.local_participant.publish_track(
                track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
            )
            sources[name] = source
        started = time.monotonic()
        last_still = 0.0
        try:
            while not self._halt.is_set() and time.monotonic() - started < 1200:
                for name, source in sources.items():
                    rgb = cv2.cvtColor(self._frame_for(name), cv2.COLOR_BGR2RGB)
                    rgb = cv2.resize(rgb, (320, 180))
                    source.capture_frame(rtc.VideoFrame(320, 180, rtc.VideoBufferType.RGB24, rgb.tobytes()))
                self.seq += 1
                self._publish_state(status="online", mode="livekit", room=info.get("room") or "", url=info.get("url") or "")
                if time.monotonic() - last_still > 2.0:
                    last_still = time.monotonic()
                    try:
                        self._post_stills()
                    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                        log.warning("still upload failed: %s", exc)
                await asyncio.sleep(max(0.1, float(self.cfg.interval_s)))
        finally:
            await room.disconnect()
