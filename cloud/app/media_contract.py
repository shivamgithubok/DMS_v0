"""Watch, idle stills, and alert clips. The phone and the Jetson share this shape.

Live watch is a WebRTC track named ``cabin``. That track is the driver camera.
JPEG uploads are the idle preview and the stills inside an alert clip, not the
watch picture.
"""

from __future__ import annotations

CABIN_TRACK = "cabin"
STILL_TRACKS = ["driver", "cabin"]

VIDEO = {
    "track": CABIN_TRACK,
    "camera": "driver",
    "width": 640,
    "height": 360,
    "fps": 12,
    "bitrate": 800_000,
}
AUDIO = {"track": CABIN_TRACK, "rate": 16000, "channels": 1}


def describe(mode: str, publish: bool) -> dict:
    """How media should move for this token mode and watch flag."""
    if mode == "livekit" and publish:
        return {
            "stream": "webrtc",
            "publishTo": "livekit",
            "tracks": [CABIN_TRACK],
            "video": dict(VIDEO),
            "audio": dict(AUDIO),
        }
    if mode == "local" and publish:
        return {
            "stream": "jpeg",
            "publishTo": "POST /v1/devices/{deviceId}/video/live",
            "tracks": [CABIN_TRACK],
            "video": {**VIDEO, "mime": "image/jpeg"},
            "audio": dict(AUDIO),
        }
    return {
        "stream": "jpeg",
        "publishTo": "POST /v1/devices/{deviceId}/video/frame",
        "tracks": list(STILL_TRACKS),
        "video": None,
        "audio": None,
    }
