"""Publish driver/cabin stills, and a cabin video+audio stream when a guardian is watching.

Stills: POST JPEG frames to the guardian API (preview tiles). Those always run.

Stream: the media-token response includes ``publish``. When that is true and the
mode is ``local``, one stitched cabin frame plus a PCM chunk is posted to
``/video/live`` so the path can be tested without a LiveKit server. When the
mode is ``livekit`` and the URL is ``ws``/``wss``, the same cabin picture and
microphone audio are published as one WebRTC video track and one audio track.
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

CABIN_W = 640
CABIN_H = 360
AUDIO_RATE = 16000


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


def stitch_cabin(driver: np.ndarray | None, passenger: np.ndarray | None, width: int = CABIN_W, height: int = CABIN_H) -> np.ndarray:
    """One BGR video frame: driver on the left, passenger on the right."""
    frames = [frame for frame in (driver, passenger) if frame is not None]
    if not frames:
        return dummy_frame("cabin", width, height)
    tile_w = width // len(frames)
    tiles = [cv2.resize(frame, (tile_w, height), interpolation=cv2.INTER_AREA) for frame in frames]
    return np.hstack(tiles)


def wants_stream(token: dict) -> bool:
    return bool(token.get("publish")) and token.get("mode") in {"local", "livekit"}


class WebRtcPublisher(threading.Thread):
    def __init__(self, cfg, capture, shared, audio=None) -> None:
        super().__init__(name="webrtc-publisher", daemon=True)
        self.cfg = cfg.webrtc
        self.capture = capture
        self.shared = shared
        self.audio = audio
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

    def _camera_image(self, track: str) -> np.ndarray:
        if self.cfg.dummy:
            return dummy_frame("driver" if track == "driver" else "cabin", seq=self.seq)
        grabber = self.capture.driver if track == "driver" else self.capture.passenger
        latest = grabber.latest() if grabber is not None else None
        if latest is None or getattr(latest, "image", None) is None:
            return dummy_frame("driver" if track == "driver" else "cabin", seq=self.seq)
        return latest.image

    def _frame_for(self, track: str) -> np.ndarray:
        image = self._camera_image("driver" if track == "driver" else "passenger")
        h, w = image.shape[:2]
        scale = min(1.0, 320 / max(w, 1))
        if scale < 1.0:
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        return image

    def _cabin_frame(self) -> np.ndarray:
        passenger = None if self.capture.passenger is None and not self.cfg.dummy else self._camera_image("passenger")
        return stitch_cabin(self._camera_image("driver"), passenger)

    def _pcm(self) -> tuple[bytes, int]:
        n = min(AUDIO_RATE, max(320, int(AUDIO_RATE * float(self.cfg.interval_s))))
        ring = None
        if self.audio is not None:
            ring = getattr(self.audio, "ring16", None) or getattr(self.audio, "ring", None)
        if ring is None or getattr(ring, "total", 0) <= 0:
            return np.zeros(n, dtype=np.int16).tobytes(), n
        samples, _ = ring.latest(n)
        pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
        out = (pcm * 32767.0).astype(np.int16)
        return out.tobytes(), int(out.shape[0])

    def run(self) -> None:
        self._publish_state(status="starting", api=self.cfg.api_url, publishing=False)
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
                streaming = wants_stream(token_body)
                self._publish_state(
                    status="online",
                    mode=mode,
                    room=token_body.get("room") or "",
                    url=url,
                    publishing=streaming,
                )
                if streaming and mode == "livekit" and url.startswith(("ws://", "wss://")):
                    try:
                        asyncio.run(self._livekit(token_body))
                    except Exception as exc:
                        self.errors += 1
                        self._publish_state(status="error", last_error=str(exc), publishing=False)
                        log.warning("LiveKit publish failed: %s", exc)
                        self._halt.wait(float(self.cfg.interval_s))
                else:
                    self._post_stills()
                    if streaming and mode == "local":
                        self._post_live()
                    elif not streaming:
                        self._publish_state(publishing=False)
                    self._halt.wait(float(self.cfg.interval_s))
            except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
                self.errors += 1
                self._publish_state(status="error", last_error=str(exc), publishing=False)
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

    def _post_live(self) -> None:
        pcm, samples = self._pcm()
        frame = self._cabin_frame()
        self._post(
            f"/v1/devices/{self.device_id}/video/live",
            {
                "key": self.cfg.device_key,
                "at": int(time.time() * 1000),
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
                "samples": samples,
                "rate": AUDIO_RATE,
                "image_b64": jpeg_b64(frame),
                "audio_b64": base64.b64encode(pcm).decode(),
            },
        )
        self._publish_state(publishing=True, mode="local")

    async def _livekit(self, info: dict) -> None:
        from livekit import rtc

        room = rtc.Room()
        await room.connect(info["url"], info["token"])
        log.info("LiveKit connected room=%s", info.get("room"))
        source = rtc.VideoSource(CABIN_W, CABIN_H)
        video = rtc.LocalVideoTrack.create_video_track("cabin", source)
        await room.local_participant.publish_track(video, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA))
        audio_source = rtc.AudioSource(AUDIO_RATE, 1)
        audio = rtc.LocalAudioTrack.create_audio_track("cabin", audio_source)
        await room.local_participant.publish_track(audio, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
        started = time.monotonic()
        last_still = 0.0
        last_token = started
        try:
            while not self._halt.is_set() and time.monotonic() - started < 1200:
                if time.monotonic() - last_token > 2.0:
                    last_token = time.monotonic()
                    try:
                        fresh = await asyncio.to_thread(
                            self._post,
                            f"/v1/devices/{self.device_id}/media-token",
                            {"key": self.cfg.device_key},
                        )
                    except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
                        log.warning("token refresh failed: %s", exc)
                    else:
                        if not wants_stream(fresh):
                            log.info("guardian stopped watching, leaving LiveKit")
                            break
                rgb = cv2.cvtColor(self._cabin_frame(), cv2.COLOR_BGR2RGB)
                source.capture_frame(rtc.VideoFrame(CABIN_W, CABIN_H, rtc.VideoBufferType.RGB24, rgb.tobytes()))
                pcm, samples = self._pcm()
                await audio_source.capture_frame(rtc.AudioFrame(pcm, AUDIO_RATE, 1, samples))
                self.seq += 1
                self._publish_state(status="online", mode="livekit", publishing=True, room=info.get("room") or "", url=info.get("url") or "")
                if time.monotonic() - last_still > 2.0:
                    last_still = time.monotonic()
                    try:
                        await asyncio.to_thread(self._post_stills)
                    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                        log.warning("still upload failed: %s", exc)
                await asyncio.sleep(max(0.1, float(self.cfg.interval_s)))
        finally:
            self._publish_state(publishing=False)
            await room.disconnect()
