#!/usr/bin/env python3
"""Publish Orin-shaped MQTT messages to a local broker."""

from __future__ import annotations

import json
import os
import time
import uuid

import paho.mqtt.client as mqtt

HOST = os.environ.get("MQTT_HOST", "127.0.0.1")
PORT = int(os.environ.get("MQTT_PORT", "1883"))
DEVICE = os.environ.get("DEFAULT_DEVICE_ID", "orin-nano-01")
NOW = int(time.time() * 1000)
SID = uuid.uuid4().hex[:8]


def pub(client: mqtt.Client, topic: str, body: dict) -> None:
    payload = json.dumps(body)
    info = client.publish(topic, payload, qos=1)
    info.wait_for_publish(timeout=5)
    print(f"published {topic} {body.get('type') or body.get('state')}")


def main() -> None:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv311)
    client.connect(HOST, PORT, keepalive=30)
    client.loop_start()
    pub(
        client,
        f"cms/{DEVICE}/status",
        {
            "device_id": DEVICE,
            "state": "online",
            "timestamp": time.time(),
            "occupants": 2,
            "seats": [
                {"name": "front_passenger", "occupant": "empty", "belt": "Unknown"},
                {"name": "rear_left", "occupant": "adult", "belt": "Unknown"},
                {"name": "rear_right", "occupant": "empty", "belt": "Unknown"},
            ],
            "queued_alerts": 0,
            "connected": True,
        },
    )
    pub(
        client,
        f"dms/{DEVICE}/alert/vision",
        {
            "id": f"distract-{SID}",
            "device_id": DEVICE,
            "type": "DISTRACTION",
            "category": "vision",
            "severity": "high",
            "confidence": 0.74,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000+00:00", time.gmtime()),
            "epoch_ms": NOW,
            "description": "Gaze left the road corridor",
            "channel": "dms",
            "reason": "gaze_away",
        },
    )
    pub(
        client,
        f"dms/{DEVICE}/alert/audio",
        {
            "id": f"scream-{SID}",
            "device_id": DEVICE,
            "type": "SCREAM",
            "category": "audio",
            "severity": "high",
            "confidence": 0.82,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000+00:00", time.gmtime()),
            "epoch_ms": NOW + 1,
            "description": "Scream detected",
            "channel": "dms",
            "reason": "scream",
        },
    )
    time.sleep(0.3)
    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    main()
