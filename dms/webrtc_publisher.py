"""Publish a live cabin video+audio stream when a guardian is watching.

Live pictures are two LiveKit video tracks, ``driver`` and ``cabin``. JPEG
stills on ``/video/frame`` are the idle preview: driver and passenger, one
pair every 2 seconds. Those stills stop while the live tracks are publishing.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
import time
import urllib.error

import cv2
import numpy as np

from .cloud_api import post_json
from .state import ConsentState, WebRtcState

log = logging.getLogger("dms.webrtc")

CABIN_W = 640
CABIN_H = 360
AUDIO_RATE = 16000
STREAM_FPS = 12
STREAM_BITRATE = 800_000
# Idle preview on /video/frame. One driver still and one passenger still, then wait.
STILL_INTERVAL_S = 2.0


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
    """One BGR picture: driver on the left, passenger on the right.

    Kept for a later two-camera stream. The live track does not use this,
    because a stitched still is what the guardian app was showing.
    """
    frames = [frame for frame in (driver, passenger) if frame is not None]
    if not frames:
        return dummy_frame("cabin", width, height)
    tile_w = width // len(frames)
    tiles = [cv2.resize(frame, (tile_w, height), interpolation=cv2.INTER_AREA) for frame in frames]
    return np.hstack(tiles)


def fit_cover(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale to fill ``width`` x ``height`` and center-crop. Returns a new contiguous BGR array."""
    h, w = image.shape[:2]
    if w == width and h == height:
        return np.ascontiguousarray(image)
    scale = max(width / max(w, 1), height / max(h, 1))
    nw = max(width, int(round(w * scale)))
    nh = max(height, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x0 = max(0, (resized.shape[1] - width) // 2)
    y0 = max(0, (resized.shape[0] - height) // 2)
    crop = resized[y0:y0 + height, x0:x0 + width]
    if crop.shape[0] != height or crop.shape[1] != width:
        crop = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(crop)


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

    def _note_consent(self, token: dict) -> None:
        watch = token.get("watchRequest") or {}
        talk = token.get("talkRequest") or {}
        self.shared.consent = ConsentState(
            watch_request=str(watch.get("name") or ""),
            talk_request=str(talk.get("name") or ""),
            live=bool(token.get("publish")),
            talking=bool(token.get("talk")),
        )

    def _publish_state(self, **changes) -> None:
        current = getattr(self.shared, "webrtc", WebRtcState())
        values = {k: getattr(current, k) for k in current.__dataclass_fields__}
        values.update(changes)
        values.update(enabled=True, seq=self.seq, errors=self.errors)
        self.shared.webrtc = WebRtcState(**values)

    def _post(self, path: str, body: dict, timeout: float | None = None) -> dict:
        url = self.cfg.api_url.rstrip("/") + path
        return post_json(url, body, float(self.cfg.timeout_s if timeout is None else timeout))

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

    def _track_frame(self, track: str) -> np.ndarray:
        """One camera, fitted to the LiveKit track size. ``cabin`` is the passenger source."""
        return fit_cover(self._camera_image(track), CABIN_W, CABIN_H)

    def _pcm(self, samples: int | None = None) -> tuple[bytes, int]:
        n = samples if samples is not None else min(AUDIO_RATE, max(320, int(AUDIO_RATE * float(self.cfg.interval_s))))
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
                self._note_consent(token_body)
                self._publish_state(
                    status="online",
                    mode=mode,
                    room=token_body.get("room") or "",
                    url=url,
                    publishing=streaming,
                )
                if streaming and mode == "livekit" and url.startswith(("ws://", "wss://")):
                    self._post_stills_safe()
                    try:
                        asyncio.run(self._livekit(token_body))
                    except Exception as exc:
                        self.errors += 1
                        self._publish_state(status="error", last_error=str(exc), publishing=False)
                        log.warning("LiveKit publish failed: %s", exc)
                        self._halt.wait(2.0)
                else:
                    self._post_stills()
                    if streaming and mode == "local":
                        self._post_live()
                    elif not streaming:
                        self._publish_state(publishing=False)
                    self._halt.wait(STILL_INTERVAL_S)
            except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
                self.errors += 1
                self._publish_state(status="error", last_error=str(exc), publishing=False)
                log.warning("WebRTC publish failed: %s", exc)
                self._halt.wait(max(2.0, float(self.cfg.interval_s)))

    def _post_stills_safe(self) -> None:
        try:
            self._post_stills()
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
            log.warning("live still failed: %s", exc)

    def _post_stills(self) -> None:
        """Driver (DMS) and passenger stills for the preview. Not a video stream."""
        now = int(time.time() * 1000)
        failed = []
        for track in ("driver", "cabin"):
            try:
                self._post(
                    f"/v1/devices/{self.device_id}/video/frame",
                    {"key": self.cfg.device_key, "track": track, "at": now, "image_b64": jpeg_b64(self._frame_for(track))},
                )
            except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
                failed.append(f"{track}: {exc}")
        self.seq += 1
        if failed:
            raise OSError("frame " + "; ".join(failed))

    def _post_live(self) -> None:
        pcm, samples = self._pcm()
        frame = self._track_frame("driver")
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

    def _push_video(self, source, frame, timestamp_us: int) -> None:
        """Hand one raw frame to LiveKit. ``timestamp_us`` is wall-clock microseconds.

        A timestamp counted from zero is in the past, so the SFU drops the frame
        and the phone keeps the last picture.
        """
        kind = getattr(self, "_video_push", None)
        if kind is None:
            try:
                source.capture_frame(frame, timestamp_us=timestamp_us)
                self._video_push = "timestamp_us"
                return
            except TypeError:
                pass
            try:
                source.capture_frame(frame, timestamp=timestamp_us)
                self._video_push = "timestamp"
                return
            except TypeError:
                source.capture_frame(frame)
                self._video_push = "plain"
                return
        if kind == "timestamp_us":
            source.capture_frame(frame, timestamp_us=timestamp_us)
        elif kind == "timestamp":
            source.capture_frame(frame, timestamp=timestamp_us)
        else:
            source.capture_frame(frame)

    def _camera_publish_options(self, rtc):
        """Ask LiveKit to encode one camera video, not a stack of stills."""
        kwargs = {"source": rtc.TrackSource.SOURCE_CAMERA}
        encoding = getattr(rtc, "VideoEncoding", None)
        if encoding is not None:
            try:
                kwargs["video_encoding"] = encoding(max_bitrate=STREAM_BITRATE, max_framerate=STREAM_FPS)
            except TypeError:
                log.info("VideoEncoding unsupported; publishing raw frames anyway")
        opts = rtc.TrackPublishOptions(**kwargs)
        if hasattr(opts, "simulcast"):
            opts.simulcast = False
        return opts

    async def _session_sidework(self, stopped: asyncio.Event) -> None:
        """Refresh the publisher token. JPEG stills stay off while the tracks are live."""
        while not self._halt.is_set() and not stopped.is_set():
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
                    stopped.set()
                    return
            try:
                await asyncio.wait_for(stopped.wait(), timeout=10.0)
                return
            except asyncio.TimeoutError:
                continue

    async def _livekit(self, info: dict) -> None:
        from livekit import rtc

        room = rtc.Room()
        stopped = asyncio.Event()
        side = None
        try:
            await room.connect(info["url"], info["token"])
            log.info("LiveKit video room=%s driver+cabin %dx%d @ %dfps", info.get("room"), CABIN_W, CABIN_H, STREAM_FPS)
            sources = {}
            for name in ("driver", "cabin"):
                source = rtc.VideoSource(CABIN_W, CABIN_H)
                video = rtc.LocalVideoTrack.create_video_track(name, source)
                await room.local_participant.publish_track(video, self._camera_publish_options(rtc))
                sources[name] = source
            audio_source = rtc.AudioSource(AUDIO_RATE, 1)
            audio = rtc.LocalAudioTrack.create_audio_track("cabin", audio_source)
            await room.local_participant.publish_track(audio, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
            side = asyncio.create_task(self._session_sidework(stopped))
            started = time.monotonic()
            next_at = started
            frame_s = 1.0 / STREAM_FPS
            samples_per = max(1, AUDIO_RATE // STREAM_FPS)
            index = 0
            while not self._halt.is_set() and not stopped.is_set() and time.monotonic() - started < 1200:
                stamp = time.time_ns() // 1000
                for name, source in sources.items():
                    i420 = cv2.cvtColor(self._track_frame(name), cv2.COLOR_BGR2YUV_I420)
                    frame = rtc.VideoFrame(CABIN_W, CABIN_H, rtc.VideoBufferType.I420, i420.tobytes())
                    self._push_video(source, frame, stamp)
                pcm, samples = self._pcm(samples_per)
                await audio_source.capture_frame(rtc.AudioFrame(pcm, AUDIO_RATE, 1, samples))
                self.seq += 1
                index += 1
                if index % STREAM_FPS == 0:
                    self._publish_state(
                        status="online",
                        mode="livekit",
                        publishing=True,
                        room=info.get("room") or "",
                        url=info.get("url") or "",
                    )
                next_at += frame_s
                delay = next_at - time.monotonic()
                if delay < -frame_s:
                    next_at = time.monotonic()
                    delay = 0.0
                if delay > 0:
                    await asyncio.sleep(delay)
        finally:
            stopped.set()
            if side is not None:
                side.cancel()
                try:
                    await side
                except asyncio.CancelledError:
                    pass
            self._publish_state(publishing=False)
            try:
                await room.disconnect()
            except Exception:
                log.debug("LiveKit disconnect failed", exc_info=True)
