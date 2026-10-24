"""Store-and-forward against a real broker: alerts raised while the broker is down must
all arrive (in order) after it comes up."""

import json
import shutil
import socket
import subprocess
import threading
import time

import paho.mqtt.client as mqtt
import pytest

from dms import alerts as A
from dms.config import load_config
from dms.mqtt_publisher import MqttPublisher
from dms.state import SharedState

pytestmark = pytest.mark.skipif(shutil.which("mosquitto") is None, reason="mosquitto not installed")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _broker(port):
    proc = subprocess.Popen(["mosquitto", "-p", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", port), 0.2).close()
            return proc
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("broker did not start")


def test_offline_queue_drains_on_reconnect(tmp_path):
    port = _free_port()
    cfg = load_config(None, {"device_id": "t1", "data_dir": str(tmp_path),
                             "mqtt": {"port": port, "heartbeat_s": 1}})
    shared, bus = SharedState(), A.AlertBus("t1", cooldown_s=0)
    pub = MqttPublisher(cfg, bus, shared)
    pub.start()
    try:
        for i in range(30):  # broker is down
            bus.emit(A.Alert(type=A.DISTRACTION, category=A.VISION, confidence=0.9, reason=f"r{i}"))
        bus.emit(A.Alert(type=A.SCREAM, category=A.AUDIO, confidence=0.8))
        time.sleep(1.0)
        assert shared.mqtt.queued == 31 and not shared.mqtt.connected

        broker = _broker(port)
        got, lock = [], threading.Lock()
        sub = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv5)
        sub.on_message = lambda c, u, m: (lock.acquire(), got.append((m.topic, json.loads(m.payload))), lock.release())
        sub.connect("127.0.0.1", port)
        sub.subscribe("dms/t1/#", qos=1)
        sub.loop_start()
        deadline = time.time() + 30
        while time.time() < deadline and shared.mqtt.queued:
            time.sleep(0.2)
        time.sleep(1.5)
        sub.loop_stop()
        broker.terminate()

        vision = [p for t, p in got if t == "dms/t1/alert/vision"]
        audio = [p for t, p in got if t == "dms/t1/alert/audio"]
        status = [p for t, p in got if t == "dms/t1/status"]
        assert shared.mqtt.queued == 0
        assert [p["reason"] for p in vision] == [f"r{i}" for i in range(30)]  # all, in order
        assert len(audio) == 1 and audio[0]["type"] == "SCREAM"
        assert status and status[-1]["state"] == "online"
        assert shared.mqtt.success_rate >= 99.0
    finally:
        pub.stop()
        pub.join(timeout=10)
