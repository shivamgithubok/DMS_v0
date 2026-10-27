from __future__ import annotations

import json
import logging
import threading

import paho.mqtt.client as mqtt

from .config import settings
from .db import SessionLocal
from .hub import hub
from .store import alert_public, alert_row, apply_status, ride_for_device, ride_public

log = logging.getLogger("vms.mqtt")


class Ingest:
    def __init__(self) -> None:
        self.connected = False
        self.last_error = ""
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="vms-api",
            protocol=mqtt.MQTTv311,
        )
        if settings.mqtt_username:
            self.client.username_pw_set(settings.mqtt_username, settings.mqtt_password or None)
        if settings.mqtt_tls:
            self.client.tls_set()
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    def start(self) -> None:
        threading.Thread(target=self._run, name="mqtt-ingest", daemon=True).start()

    def _run(self) -> None:
        try:
            self.client.connect_async(settings.mqtt_host, settings.mqtt_port, keepalive=30)
            self.client.loop_forever(retry_first_connection=True)
        except Exception as exc:
            self.last_error = str(exc)
            log.warning("MQTT ingest failed: %s", exc)

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        if getattr(reason_code, "is_failure", False):
            self.last_error = f"connect refused: {reason_code}"
            self.connected = False
            log.warning("MQTT connect refused: %s", reason_code)
            return
        self.connected = True
        self.last_error = ""
        client.subscribe([("dms/#", 1), ("cms/#", 1)])
        log.info("MQTT ingest subscribed on %s:%s", settings.mqtt_host, settings.mqtt_port)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties) -> None:
        self.connected = False
        if reason_code:
            self.last_error = str(reason_code)

    def _on_message(self, client, userdata, message) -> None:
        try:
            body = json.loads(message.payload.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            log.warning("bad JSON on %s", message.topic)
            return
        if not isinstance(body, dict):
            return
        topic = message.topic
        try:
            with SessionLocal() as db:
                if topic.endswith("/status"):
                    ride = apply_status(db, topic, body)
                    db.commit()
                    hub.publish(ride.id, {"type": "status", "ride": ride_public(ride)})
                    return
                if "/alert" in topic:
                    device_id = str(body.get("device_id") or settings.default_device_id)
                    ride = ride_for_device(db, device_id)
                    row = alert_row(db, ride, body, origin="vehicle")
                    db.commit()
                    if row is None:
                        return
                    event = {"type": "alert", "alert": alert_public(row), "ride": ride_public(ride)}
                    hub.publish(ride.id, event)
                    if row.kind == "sos":
                        hub.publish(ride.id, {"type": "sos", "alert": alert_public(row), "ride": ride_public(ride)})
        except Exception:
            log.exception("ingest failed for %s", topic)


ingest = Ingest()
