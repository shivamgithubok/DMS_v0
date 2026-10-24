#!/usr/bin/env python3
"""Driver Monitoring System - orchestrator.

Threads
  1  cam-driver / cam-passenger   CSI capture (dms.capture)          -> latest-frame slots
  2  realtime-detector            face/EAR/pose/8-head model/objects  -> alert bus  (>= 15 FPS)
  3  vlm-worker                   cabin VLM over HTTP, 0.3-0.5 Hz     -> alert bus
  4  audio-classifier (+ kws)     scream / altercation / keywords     -> alert bus
  5  mqtt-publisher               SQLite outbox -> broker, heartbeat
  6  cms-monitor                  cabin seats and objects at 2 Hz     -> alert bus
  +  telemetry, supervisor, and the Qt UI on the main thread.

Usage
  python main.py                                   # CSI cameras, mic, VLM, MQTT, dashboard
  python main.py --headless                        # no UI (service mode)
  python main.py --driver assets/demo/driver_reference.jpg --passenger assets/demo/cabin_reference.jpg \
                 --audio-device tests/data/cabin.wav --no-vlm     # bench run without hardware
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dms.bootstrap import ensure_venv  # noqa: E402

ensure_venv()

from dms import __version__  # noqa: E402
from dms.alerts import AlertBus  # noqa: E402
from dms.capture import CaptureService  # noqa: E402
from dms.config import data_path, load_config  # noqa: E402
from dms.state import SharedState  # noqa: E402
from dms.telemetry import Telemetry, hardware_label  # noqa: E402

log = logging.getLogger("dms")
SHUTDOWN_TIMEOUT_S = 20.0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Driver Monitoring System (Jetson Orin Nano)")
    ap.add_argument("--config", default="config/dms.yaml")
    ap.add_argument("--driver", help="driver camera source (csi:0, v4l2:0, file, image, gst:...)")
    ap.add_argument("--passenger", help="passenger camera source; 'none' to disable")
    ap.add_argument("--camera", choices=["auto", "csi", "usb"],
                    help="both cameras: auto = CSI if present else USB (default), csi = CSI only, usb = USB only")
    ap.add_argument("--audio-device", help="sounddevice name/index, or a .wav file to replay")
    ap.add_argument("--audio-backend", choices=["auto", "cochl", "yamnet", "dsp"])
    ap.add_argument("--vlm-endpoint", help="OpenAI-compatible chat completions URL")
    ap.add_argument("--mqtt-host")
    ap.add_argument("--mqtt-port", type=int)
    ap.add_argument("--device-id")
    ap.add_argument("--headless", action="store_true", help="run without the dashboard")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--no-mqtt", action="store_true")
    ap.add_argument("--duration", type=float, default=0.0, help="exit after N seconds (tests/benchmarks)")
    ap.add_argument("--snapshot", help="write a dashboard PNG at exit (with --duration)")
    ap.add_argument("--log-level")
    return ap.parse_args(argv)


def build_overrides(args) -> dict:
    o: dict = {"cameras": {"driver": {}, "passenger": {}}, "audio": {}, "vlm": {}, "mqtt": {}, "ui": {}}
    if args.camera:
        o["cameras"]["driver"]["source"] = f"{args.camera}:0"
        o["cameras"]["passenger"]["source"] = f"{args.camera}:1"
        if args.camera == "csi":
            o["cameras"]["driver"]["usb_fallback"] = o["cameras"]["passenger"]["usb_fallback"] = False
    if args.driver:
        o["cameras"]["driver"]["source"] = args.driver
    if args.passenger:
        o["cameras"]["passenger"]["source"] = "" if args.passenger.lower() == "none" else args.passenger
    if args.audio_device:
        o["audio"]["device"] = int(args.audio_device) if args.audio_device.isdigit() else args.audio_device
    if args.audio_backend:
        o["audio"]["backend"] = args.audio_backend
    if args.vlm_endpoint:
        o["vlm"]["endpoint"] = args.vlm_endpoint
    if args.mqtt_host:
        o["mqtt"]["host"] = args.mqtt_host
    if args.mqtt_port:
        o["mqtt"]["port"] = args.mqtt_port
    if args.device_id:
        o["device_id"] = args.device_id
    if args.no_vlm:
        o["vlm"]["enabled"] = False
    if args.no_audio:
        o["audio"]["enabled"] = False
    if args.no_mqtt:
        o["mqtt"]["enabled"] = False
    if args.headless:
        o["ui"]["enabled"] = False
    if args.fullscreen:
        o["ui"]["fullscreen"] = True
    if args.log_level:
        o["log_level"] = args.log_level
    return o


class System:
    """Builds, starts and stops every component in a safe order."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.shared = SharedState()
        self.shared.device_id = cfg.device_id
        self.shared.hardware = cfg.ui.hardware_label or hardware_label()
        self.bus = AlertBus(cfg.device_id, cooldown_s=float(cfg.realtime.alert_cooldown_s))
        self.telemetry = Telemetry()
        self.capture = CaptureService(cfg)
        self.stop_event = threading.Event()
        self.workers: list = []
        self.audio = None
        self.mqtt = None

    def start(self) -> None:
        from dms.cms import CabinMonitor
        from dms.realtime_detector import DriverMonitor

        cfg = self.cfg
        self.telemetry.start()
        self.shared.system = self.telemetry.stats
        if cfg.mqtt.enabled:
            from dms.mqtt_publisher import MqttPublisher

            self.mqtt = MqttPublisher(cfg, self.bus, self.shared, status_provider=self.status)
            self.mqtt.start()
        if cfg.ui.local_alarm:
            from dms.local_alarm import LocalAlarm

            self.bus.subscribe(LocalAlarm())
        self.capture.start()
        detector = DriverMonitor(cfg, self.capture.driver, self.shared, self.bus)
        detector.start()
        self.workers.append(detector)
        if self.capture.passenger is not None and cfg.cms.enabled:
            cabin = CabinMonitor(cfg, self.capture.passenger, self.shared, self.bus)
            cabin.start()
            self.workers.append(cabin)
        if cfg.vlm.enabled:
            from dms.vlm_worker import VlmWorker

            vlm = VlmWorker(cfg, self.capture, self.shared, self.bus)
            vlm.start()
            self.workers.append(vlm)
        if cfg.audio.enabled:
            from dms.audio_detector import AudioDetector

            try:
                self.audio = AudioDetector(cfg, self.shared, self.bus).start()
            except Exception:
                log.exception("audio pipeline failed to start; continuing without audio")
        supervisor = threading.Thread(target=self._supervise, name="supervisor", daemon=True)
        supervisor.start()
        log.info("DMS %s running on %s as %s", __version__, self.shared.hardware, cfg.device_id)

    def status(self) -> dict:
        ds = self.shared.driver
        sysd = self.shared.system
        vlm = self.shared.vlm
        return {
            "version": __version__,
            "uptime_s": round(time.time() - self.shared.started),
            "realtime_fps": round(ds.fps, 1),
            "model_ms": round(ds.model_ms, 1),
            "pipeline_ms": round(ds.pipeline_ms, 1),
            "driver_present": ds.face_found,
            "driver_id": ds.identity.driver_id if ds.identity.known else None,
            "cameras": {name: {"connected": cam.stats.connected, "source": cam.source,
                               "requested": cam.requested or cam.source, "backend": cam.stats.backend,
                               "fps": round(cam.stats.fps, 1), "error": cam.stats.error[:200]}
                        for name, cam in (("driver", self.capture.driver), ("passenger", self.capture.passenger))
                        if cam is not None},
            "vlm": {"status": vlm.status, "calls": vlm.calls, "errors": vlm.errors,
                    "latency_s": round(vlm.latency_s, 2)},
            "cabin": {"occupants": self.shared.passenger.occupants,
                      "child_present": self.shared.passenger.child_present,
                      "connected": self.shared.passenger.connected},
            "audio_backend": self.shared.audio.backend,
            "gpu_temp_c": None if sysd is None or sysd.gpu_temp_c is None else round(sysd.gpu_temp_c, 1),
            "mem_used_mb": None if sysd is None else round(sysd.mem_used_mb),
            "alert_counts": dict(self.bus.counts),
        }

    def _supervise(self) -> None:
        last_log = time.monotonic()
        limit = float(self.cfg.vlm.memory_limit_mb)
        health = data_path("health.json")
        while not self.stop_event.wait(1.0):
            stats = self.telemetry.stats
            self.shared.system = stats
            ds = self.shared.driver
            fresh = ds.frame_ts and time.monotonic() - ds.frame_ts < 3.0
            try:
                tmp = health.with_suffix(".tmp")
                tmp.write_text(json.dumps({"ts": time.time(), "fps": round(ds.fps, 1) if fresh else 0.0,
                                           "mqtt": self.shared.mqtt.connected, "queued": self.shared.mqtt.queued}))
                tmp.replace(health)
            except OSError:
                pass
            now = time.monotonic()
            if now - last_log >= 10.0:
                last_log = now
                ds = self.shared.driver
                temp = stats.max_temp_c
                log.info("fps=%.1f model=%.1fms pipeline=%.1fms temp=%s mem=%.0fMB vlm=%s/%d audio=%s mqtt=%s q=%d",
                         ds.fps, ds.model_ms, ds.pipeline_ms, "n/a" if temp is None else f"{temp:.0f}C",
                         stats.mem_used_mb, self.shared.vlm.status, self.shared.vlm.calls, self.shared.audio.backend,
                         "up" if self.shared.mqtt.connected else "down", self.shared.mqtt.queued)
                if stats.mem_used_mb > limit:
                    log.warning("unified memory %.0f MB above the %d MB budget", stats.mem_used_mb, limit)

    def stop(self) -> None:
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        log.info("shutting down...")
        step = ["start"]

        def watchdog() -> None:  # shutdown must always finish, even if a driver call hangs
            if not finished.wait(SHUTDOWN_TIMEOUT_S):
                log.error("shutdown stuck in '%s' for %.0f s; forcing exit", step[0], SHUTDOWN_TIMEOUT_S)
                logging.shutdown()
                os._exit(1)

        finished = threading.Event()
        threading.Thread(target=watchdog, name="shutdown-watchdog", daemon=True).start()
        # stop producers first so nothing new is queued, then the uplink drains what is left
        step[0] = "cameras (signal)"
        for cam in (self.capture.driver, self.capture.passenger):
            if cam is not None:
                cam._halt.set()
        if self.audio:
            step[0] = "audio"
            self.audio.stop()
        for worker in self.workers:
            worker.stop()
        for worker in self.workers:
            step[0] = worker.name
            worker.join(timeout=5)
        step[0] = "cameras"
        self.capture.stop()
        if self.mqtt:
            step[0] = "mqtt"
            self.mqtt.stop()  # flushes the outbox while connected, then sends "offline"
            self.mqtt.join(timeout=8)
        step[0] = "telemetry"
        self.telemetry.stop()
        finished.set()
        log.info("stopped")


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg_path = args.config if (ROOT / args.config).exists() or Path(args.config).exists() else None
    cfg = load_config(cfg_path, build_overrides(args))
    logging.basicConfig(level=getattr(logging, str(cfg.log_level).upper(), logging.INFO),
                        format="%(asctime)s %(levelname).1s %(name)s: %(message)s", datefmt="%H:%M:%S")
    system = System(cfg)
    system.start()

    done = threading.Event()

    def on_signal(*_):
        if done.is_set():  # second Ctrl+C: leave now
            print("\nforced exit", file=sys.stderr, flush=True)
            os._exit(130)
        log.info("stop requested (press Ctrl+C again to force)")
        done.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, on_signal)

    if not cfg.ui.enabled:
        done.wait(args.duration if args.duration > 0 else None)
        if args.snapshot:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            from PySide6.QtWidgets import QApplication

            from dms.ui.dashboard import render_png

            _app = QApplication.instance() or QApplication([])
            render_png(system.shared, system.bus, args.snapshot)
            log.info("dashboard snapshot written to %s", args.snapshot)
        system.stop()
        return 0

    # Dashboard in a separate process (own GIL): painting never stalls the detector.
    from dms.ui.app import start_ui_process

    ui_proc, publisher, ui_stop = start_ui_process(cfg, system.shared, system.bus, args.snapshot)
    deadline = time.monotonic() + args.duration if args.duration > 0 else None
    while not done.is_set() and ui_proc.is_alive():
        if deadline and time.monotonic() >= deadline:
            break
        done.wait(0.5)
    if ui_proc.is_alive():
        ui_stop.set()
        ui_proc.join(timeout=5)
        if ui_proc.is_alive():
            ui_proc.terminate()
    else:
        log.info("dashboard closed")
    publisher.stop()
    system.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
