"""Play the guardian's phone microphone on the cabin speaker while Speak is on."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import threading
import urllib.error

from .cloud_api import post_json

log = logging.getLogger("dms.talk")


class CabinSpeaker:
    """16-bit mono PCM to the default output. sounddevice first, then aplay."""

    def __init__(self, rate: int = 48000) -> None:
        self.rate = rate
        self._stream = None
        self._proc = None
        try:
            import sounddevice as sd

            self._stream = sd.RawOutputStream(samplerate=rate, channels=1, dtype="int16")
            self._stream.start()
        except Exception as exc:
            player = shutil.which("aplay")
            if player is None:
                log.warning("no speaker output (%s)", exc)
                return
            self._proc = subprocess.Popen(
                [player, "-q", "-f", "S16_LE", "-c", "1", "-r", str(rate), "-t", "raw"],
                stdin=subprocess.PIPE,
            )

    def write(self, pcm: bytes) -> None:
        if not pcm:
            return
        if self._stream is not None:
            self._stream.write(pcm)
            return
        if self._proc is not None and self._proc.stdin is not None:
            try:
                self._proc.stdin.write(pcm)
            except (BrokenPipeError, OSError):
                self._proc = None

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._proc is not None:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
            self._proc.terminate()
            self._proc = None


def _guardian_mic(track, publication, participant) -> bool:
    name = str(getattr(publication, "name", "") or "")
    identity = str(getattr(participant, "identity", "") or "")
    if name == "cabin" or identity.startswith("camera-") or identity.startswith("speaker-"):
        return False
    kind = getattr(track, "kind", None)
    from livekit import rtc

    if kind != rtc.TrackKind.KIND_AUDIO:
        return False
    return name in {"guardian-mic", "microphone"} or identity.startswith("guardian")


class TalkPlayer(threading.Thread):
    def __init__(self, cfg) -> None:
        super().__init__(name="talk-player", daemon=True)
        self.cfg = cfg.webrtc
        self.device_id = cfg.device_id
        self._halt = threading.Event()

    def stop(self) -> None:
        self._halt.set()

    def _post(self, path: str, body: dict) -> dict:
        return post_json(self.cfg.api_url.rstrip("/") + path, body, float(self.cfg.timeout_s))

    def _token(self) -> dict:
        return self._post(f"/v1/devices/{self.device_id}/media-token", {"key": self.cfg.device_key})

    async def _play(self, track, speaker: CabinSpeaker, stopped: asyncio.Event) -> None:
        from livekit import rtc

        stream = rtc.AudioStream(track, sample_rate=speaker.rate, num_channels=1)
        try:
            async for event in stream:
                if stopped.is_set() or self._halt.is_set():
                    break
                frame = getattr(event, "frame", event)
                speaker.write(bytes(frame.data))
        except Exception:
            log.exception("guardian mic playback failed")

    async def _session(self, info: dict) -> None:
        from livekit import rtc

        room = rtc.Room()
        speaker = CabinSpeaker()
        stopped = asyncio.Event()
        tasks: list[asyncio.Task] = []

        @room.on("track_subscribed")
        def _on_track(track, publication, participant):
            if not _guardian_mic(track, publication, participant):
                return
            log.info("playing guardian mic from %s", getattr(participant, "identity", ""))
            tasks.append(asyncio.create_task(self._play(track, speaker, stopped)))

        try:
            await room.connect(info["url"], info["speakerToken"])
            log.info("Speak listening in %s", info.get("room"))
            while not self._halt.is_set():
                fresh = await asyncio.to_thread(self._token)
                if not fresh.get("talk"):
                    break
                await asyncio.sleep(1.0)
        finally:
            stopped.set()
            for task in tasks:
                task.cancel()
            speaker.close()
            await room.disconnect()
            log.info("Speak stopped")

    def run(self) -> None:
        log.info("talk player → %s", self.cfg.api_url)
        while not self._halt.is_set():
            try:
                token = self._token()
                if token.get("talk") and token.get("speakerToken") and token.get("mode") == "livekit":
                    asyncio.run(self._session(token))
            except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
                log.warning("talk poll failed: %s", exc)
                self._halt.wait(2.0)
                continue
            except Exception:
                log.exception("talk player failed")
            self._halt.wait(1.0)
