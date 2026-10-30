"""MQTT uplink with persistent store-and-forward (spec Thread 5).

Every alert is written to a SQLite outbox *before* any network attempt, and a row is
deleted only when the broker's PUBACK for it arrives (QoS 1). So alerts survive broker
outages, network loss and process restarts, and are drained in order on reconnect.

Topics
  dms/{device_id}/alert/vision   QoS 1
  dms/{device_id}/alert/audio    QoS 1
  dms/{device_id}/status         QoS 0 heartbeat every 30 s, retained; LWT "offline"
  dms/{device_id}/trips          QoS 0 retained trip list; kept on the broker when the Jetson is off
  dms/{device_id}/analysis       QoS 0 retained latest driver VLM slice
  cms/{device_id}/alert          QoS 1   cabin alerts (child, object, aggression, occupancy)
  cms/{device_id}/status         QoS 0   retained cabin heartbeat
  cms/{device_id}/analysis       QoS 0 retained latest cabin VLM slice

Uses paho-mqtt 2.x with MQTT v5 (falls back to v3.1.1 if the broker refuses v5).
"""

from __future__ import annotations

import json
import logging
import queue
import socket
import sqlite3
import ssl
import threading
import time
from pathlib import Path

import paho.mqtt.client as mqtt

from . import alerts as A
from .cms import status_body
from .config import data_path
from .state import MqttState

log = logging.getLogger("dms.mqtt")


class Outbox:
    """Durable FIFO of pending messages (SQLite WAL, safe across threads)."""

    def __init__(self, path: Path, max_rows: int = 50000) -> None:
        self.path = Path(path)
        self.max_rows = max_rows
        self.lock = threading.Lock()
        self.db = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS outbox (
            id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT NOT NULL, payload TEXT NOT NULL,
            qos INTEGER NOT NULL, created REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
            last_attempt REAL NOT NULL DEFAULT 0)""")

    def put(self, topic: str, payload: str, qos: int) -> int:
        with self.lock:
            cur = self.db.execute("INSERT INTO outbox (topic, payload, qos, created) VALUES (?, ?, ?, ?)",
                                  (topic, payload, qos, time.time()))
            count = self.db.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
            if count > self.max_rows:  # bounded disk use: drop the oldest beyond the cap
                self.db.execute("DELETE FROM outbox WHERE id IN (SELECT id FROM outbox ORDER BY id LIMIT ?)",
                                (count - self.max_rows,))
                log.warning("outbox over %d rows, dropped %d oldest", self.max_rows, count - self.max_rows)
            return cur.lastrowid

    def due(self, limit: int, retry_after: float, exclude: set) -> list[tuple]:
        now = time.time()
        with self.lock:
            rows = self.db.execute(
                "SELECT id, topic, payload, qos FROM outbox WHERE last_attempt <= ? ORDER BY id LIMIT ?",
                (now - retry_after, limit + len(exclude))).fetchall()
        return [r for r in rows if r[0] not in exclude][:limit]

    def mark_attempt(self, row_id: int) -> None:
        with self.lock:
            self.db.execute("UPDATE outbox SET attempts = attempts + 1, last_attempt = ? WHERE id = ?",
                            (time.time(), row_id))

    def reset_attempts(self) -> None:
        with self.lock:
            self.db.execute("UPDATE outbox SET last_attempt = 0")

    def delete(self, row_id: int) -> None:
        with self.lock:
            self.db.execute("DELETE FROM outbox WHERE id = ?", (row_id,))

    def count(self) -> int:
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]

    def close(self) -> None:
        with self.lock:
            self.db.close()


class MqttPublisher(threading.Thread):
    def __init__(self, cfg, bus: A.AlertBus, shared, status_provider=None) -> None:
        super().__init__(name="mqtt-publisher", daemon=True)
        self.cfg = cfg.mqtt
        self.device_id = cfg.device_id
        self.bus = bus
        self.shared = shared
        self.status_provider = status_provider
        prefix = self.cfg.topic_prefix.strip("/")
        self.topic_vision = f"{prefix}/{self.device_id}/alert/vision"
        self.topic_audio = f"{prefix}/{self.device_id}/alert/audio"
        self.topic_status = f"{prefix}/{self.device_id}/status"
        self.topic_trips = f"{prefix}/{self.device_id}/trips"
        cms_prefix = str(cfg.cms.topic_prefix).strip("/") or "cms"
        self.topic_cms = f"{cms_prefix}/{self.device_id}/alert"
        self.topic_cms_status = f"{cms_prefix}/{self.device_id}/status"
        self.topic_dms_analysis = f"{prefix}/{self.device_id}/analysis"
        self.topic_cms_analysis = f"{cms_prefix}/{self.device_id}/analysis"
        self._latest_analysis: dict[str, dict] = {}
        self.outbox = Outbox(data_path(self.cfg.outbox_db), int(self.cfg.outbox_max_rows))
        self._halt = threading.Event()
        self._wake = threading.Event()
        self._connected = threading.Event()
        self._inflight: dict[int, int] = {}  # mid -> outbox row id
        self._inflight_rows: set[int] = set()
        # PUBACKs that arrive before publish() has returned the mid to us: paho invokes
        # on_publish before MessageInfo.is_published() flips, and we must not hold our lock
        # across client.publish() (paho holds its own mutex while calling on_publish).
        self._early_acks: dict[int, tuple] = {}
        self._ever_connected = False
        self._lock = threading.Lock()
        self.published = 0
        self.acked = 0
        self.failed = 0
        self.last_error = ""
        proto = str(self.cfg.get("protocol", "auto")).lower()
        self._auto_protocol = proto == "auto"
        self._protocol = mqtt.MQTTv311 if proto in ("3", "3.1.1", "311", "v311") else mqtt.MQTTv5
        self.client = self._make_client()

    # ------------------------------------------------------------------ client
    def _make_client(self) -> mqtt.Client:
        client_id = self.cfg.client_id or f"dms-{self.device_id}"
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id, protocol=self._protocol)
        if self.cfg.username:
            client.username_pw_set(self.cfg.username, self.cfg.password or None)
        tls = self.cfg.tls
        if tls.enabled:
            client.tls_set(ca_certs=tls.ca_certs or None, certfile=tls.certfile or None, keyfile=tls.keyfile or None,
                           cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS_CLIENT)
            client.tls_insecure_set(bool(tls.insecure))
        client.max_inflight_messages_set(int(self.cfg.max_inflight))
        client.max_queued_messages_set(0)
        client.reconnect_delay_set(min_delay=1, max_delay=30)
        client.will_set(self.topic_status, json.dumps({"device_id": self.device_id, "state": "offline"}),
                        qos=1, retain=True)
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_publish = self._on_publish
        return client

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        if reason_code.is_failure:
            self.last_error = f"connect refused: {reason_code}"
            log.warning("MQTT connect refused: %s", reason_code)
            return
        log.info("MQTT connected to %s:%s (%s)", self.cfg.host, self.cfg.port,
                 "v5" if self._protocol == mqtt.MQTTv5 else "v3.1.1")
        self.last_error = ""
        self._ever_connected = True
        with self._lock:
            # anything in flight on the old session is re-sent from the outbox
            self._inflight.clear()
            self._inflight_rows.clear()
            self._early_acks.clear()
        self.outbox.reset_attempts()
        self._connected.set()
        self._send_status("online")
        self._flush_analysis(force=True)
        self._wake.set()

    def _on_disconnect(self, client, userdata, flags, reason_code, properties) -> None:
        self._connected.clear()
        if not self._halt.is_set():
            self.last_error = f"disconnected: {reason_code}"
            log.warning("MQTT disconnected (%s); alerts are queued in the outbox", reason_code)

    def _on_publish(self, client, userdata, mid, reason_code, properties) -> None:
        with self._lock:
            row = self._inflight.pop(mid, None)
            if row is not None:
                self._inflight_rows.discard(row)
            else:
                self._early_acks[mid] = (time.monotonic(), reason_code)
                return
        self._settle(row, reason_code)

    def _settle(self, row: int, reason_code) -> None:
        if reason_code is not None and getattr(reason_code, "is_failure", False):
            self.failed += 1
            self.last_error = f"publish rejected: {reason_code}"
            return  # stays in the outbox, retried after retry_after_s
        self.outbox.delete(row)
        self.acked += 1

    # ------------------------------------------------------------------ thread
    def stop(self) -> None:
        self._halt.set()
        self._wake.set()

    def run(self) -> None:
        self.client.connect_async(self.cfg.host, int(self.cfg.port), keepalive=int(self.cfg.keepalive))
        self.client.loop_start()
        last_heartbeat = 0.0
        fallback_at = time.monotonic() + 8.0
        try:
            while not self._halt.is_set():
                self._drain_bus()
                self._drain_analysis()
                if self._connected.is_set():
                    self._flush()
                    if time.monotonic() - last_heartbeat >= float(self.cfg.heartbeat_s):
                        last_heartbeat = time.monotonic()
                        self._send_status("online")
                elif not self._ever_connected and time.monotonic() > fallback_at:
                    fallback_at = time.monotonic() + 8.0
                    if not self._broker_reachable():
                        continue  # nothing listening: a protocol switch would not help
                    if not self._auto_protocol:
                        continue
                    # The broker accepts TCP but did not CONNACK: some brokers (older Mosquitto,
                    # amqtt, several cloud gateways) drop MQTT 5 CONNECTs. Alternate levels.
                    nxt = mqtt.MQTTv311 if self._protocol == mqtt.MQTTv5 else mqtt.MQTTv5
                    log.warning("no CONNACK with %s, retrying with %s",
                                "MQTT 5" if self._protocol == mqtt.MQTTv5 else "MQTT 3.1.1",
                                "MQTT 5" if nxt == mqtt.MQTTv5 else "MQTT 3.1.1")
                    self._switch_protocol(nxt)
                self._publish_state()
                self._wake.wait(0.2)
                self._wake.clear()
        finally:
            self._drain_bus()
            self._drain_analysis()
            if self._connected.is_set():
                self._flush()
                deadline = time.monotonic() + 3.0
                while self._inflight and time.monotonic() < deadline:
                    time.sleep(0.05)
                self._send_status("offline")
                time.sleep(0.2)
            self.client.disconnect()
            self.client.loop_stop()
            self.outbox.close()

    def _broker_reachable(self) -> bool:
        host, port = self.cfg.host, int(self.cfg.port)
        try:
            socket.create_connection((host, port), timeout=1.5).close()
            self._unreachable_logged = 0.0
            return True
        except OSError as exc:
            self.last_error = f"broker {host}:{port} unreachable: {exc.strerror or exc}"
            now = time.monotonic()
            if now - getattr(self, "_unreachable_logged", 0.0) > 300 or not getattr(self, "_unreachable_logged", 0.0):
                self._unreachable_logged = now
                log.warning("MQTT broker %s:%d unreachable (%s). Alerts are stored in the outbox and sent when it "
                            "comes up. Set mqtt.host / --mqtt-host, start a local broker "
                            "(sudo apt install mosquitto), or run with --no-mqtt.", host, port, exc.strerror or exc)
            return False

    def _switch_protocol(self, protocol) -> None:
        try:
            self.client.disconnect()
            self.client.loop_stop()
        except Exception:
            pass
        self._protocol = protocol
        self.client = self._make_client()
        self.client.connect_async(self.cfg.host, int(self.cfg.port), keepalive=int(self.cfg.keepalive))
        self.client.loop_start()

    def _drain_bus(self) -> None:
        while True:
            try:
                alert = self.bus.mqtt_queue.get_nowait()
            except queue.Empty:
                return
            if alert.channel == "cms" or alert.type in A.CMS_TYPES:
                topic = self.topic_cms
            elif alert.category == A.AUDIO:
                topic = self.topic_audio
            else:
                topic = self.topic_vision
            self.outbox.put(topic, json.dumps(self._with_gps(alert.payload()), separators=(",", ":")),
                            int(self.cfg.alert_qos))

    def _with_gps(self, payload: dict) -> dict:
        gps = getattr(self.shared, "gps", None)
        if gps is not None and gps.available and gps.lat is not None and gps.lon is not None:
            payload["lat"] = round(float(gps.lat), 6)
            payload["lon"] = round(float(gps.lon), 6)
            payload["gps_fix"] = gps.fix
            payload["gps_at"] = gps.at or None
        return payload

    def _drain_analysis(self) -> None:
        while True:
            try:
                channel, body = self.bus.analysis_queue.get_nowait()
            except queue.Empty:
                break
            self._latest_analysis[channel] = body
        self._flush_analysis()

    def _flush_analysis(self, force: bool = False) -> None:
        if not self._connected.is_set() and not force:
            return
        if not self._connected.is_set():
            return
        for channel, body in list(self._latest_analysis.items()):
            topic = self.topic_cms_analysis if channel == "cms" else self.topic_dms_analysis
            info = self.client.publish(topic, json.dumps(body, separators=(",", ":")), qos=0, retain=True)
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                self.last_error = f"analysis publish rc={info.rc}"
                continue
            self.published += 1

    def _flush(self) -> None:
        with self._lock:
            room = int(self.cfg.max_inflight) - len(self._inflight)
            exclude = set(self._inflight_rows)
        if room <= 0:
            return
        for row_id, topic, payload, qos in self.outbox.due(room, float(self.cfg.retry_after_s), exclude):
            if not self._connected.is_set():
                return
            info = self.client.publish(topic, payload, qos=qos)
            self.outbox.mark_attempt(row_id)
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                self.failed += 1
                self.last_error = f"publish rc={info.rc}"
                return
            self.published += 1
            if qos == 0:
                self.outbox.delete(row_id)
                continue
            with self._lock:
                early = self._early_acks.pop(info.mid, None)
                if early is None:
                    self._inflight[info.mid] = row_id
                    self._inflight_rows.add(row_id)
            if early is not None:
                self._settle(row_id, early[1])
        with self._lock:  # forget stray acks (QoS 0 status messages also report on_publish)
            cutoff = time.monotonic() - 30.0
            for mid in [m for m, (t, _) in self._early_acks.items() if t < cutoff]:
                del self._early_acks[mid]

    def _send_status(self, state: str) -> None:
        body = {"device_id": self.device_id, "state": state, "timestamp": time.time(),
                "queued_alerts": self.outbox.count()}
        if self.status_provider is not None and state == "online":
            try:
                body.update(self.status_provider())
            except Exception:
                log.exception("status provider failed")
        self.client.publish(self.topic_status, json.dumps(body, separators=(",", ":")),
                            qos=int(self.cfg.heartbeat_qos), retain=True)
        cabin = status_body(self.device_id, self.shared.passenger, state, self.outbox.count())
        self.client.publish(self.topic_cms_status, json.dumps(cabin, separators=(",", ":")),
                            qos=int(self.cfg.heartbeat_qos), retain=True)
        self._send_trips()

    def _send_trips(self) -> None:
        body = getattr(self.shared, "history", None)
        if not isinstance(body, dict) or not body.get("trips"):
            return
        self.client.publish(self.topic_trips, json.dumps(body, separators=(",", ":")),
                            qos=int(self.cfg.heartbeat_qos), retain=True)

    def _publish_state(self) -> None:
        attempts = self.acked + self.failed
        self.shared.mqtt = MqttState(
            enabled=True, connected=self._connected.is_set(), broker=f"{self.cfg.host}:{self.cfg.port}",
            published=self.published, acked=self.acked, failed=self.failed, queued=self.outbox.count(),
            success_rate=100.0 * self.acked / attempts if attempts else 100.0, last_error=self.last_error)
