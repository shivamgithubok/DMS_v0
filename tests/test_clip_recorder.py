"""Alert clips: a few seconds of JPEGs uploaded onto the current ride."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from dms.alerts import Alert, AlertBus
from dms.clip_recorder import ClipRecorder, frames_for_alert
from dms.config import Config


def test_window_keeps_frames_around_the_alert():
    samples = [(1000, b"a"), (5000, b"b"), (9000, b"c"), (20000, b"d")]
    chosen = frames_for_alert(samples, alert_ms=8000, before_ms=4000, after_ms=4000)
    assert [item[0] for item in chosen] == [5000, 9000]


class _Grab:
    def __init__(self) -> None:
        self.image = np.zeros((48, 64, 3), dtype=np.uint8)
        self.image[:] = (30, 40, 200)

    def latest(self):
        return self


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        size = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(size).decode())
        self.server.bodies.append((self.path, body))
        raw = json.dumps({"id": "clip-test", "frames": body["frames"]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt, *args):
        return


def test_alert_uploads_a_clip_for_that_id(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.bodies = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        capture = type("Cap", (), {})()
        capture.driver = _Grab()
        capture.passenger = _Grab()
        cfg = Config(
            {
                "device_id": "orin-nano-01",
                "webrtc": {
                    "api_url": f"http://127.0.0.1:{port}",
                    "device_key": "dev-media-key",
                    "timeout_s": 2.0,
                },
            }
        )
        bus = AlertBus("orin-nano-01", cooldown_s=0)
        recorder = ClipRecorder(
            cfg, capture, bus, before_ms=4000, after_ms=0, sample_s=10,
            queue_path=tmp_path / "clips.sqlite3",
        )
        recorder._sample()
        alert = Alert(type="DROWSINESS", category="vision", confidence=0.9, channel="dms", description="drowsy")
        bus.emit(alert)
        deadline = time.time() + 15
        while time.time() < deadline and not server.bodies:
            time.sleep(0.02)
        assert server.bodies, "clip was not uploaded"
        path, body = server.bodies[0]
        assert path == "/v1/devices/orin-nano-01/clips"
        assert body["alert_id"] == alert.id
        assert body["channel"] == "dms"
        assert body["key"] == "dev-media-key"
        assert len(body["frames"]) >= 1
        assert body["frames"][0]["image_b64"]
        video = __import__("base64").b64decode(body["video_b64"])
        assert video[4:8] == b"ftyp"
    finally:
        server.shutdown()
        thread.join(timeout=2)
