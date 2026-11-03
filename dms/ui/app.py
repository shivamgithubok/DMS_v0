"""Dashboard in its own process.

The core process (cameras, detectors, VLM, audio, MQTT) publishes immutable state
snapshots at the UI frame rate over a size-1 multiprocessing queue; the HMI process
paints them. Separate processes mean separate GILs: Qt painting can never stall the
real-time detector, and a UI crash cannot stop monitoring or alert upload.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import signal
import threading
import time
from dataclasses import replace

log = logging.getLogger("dms.ui")


class StatePublisher(threading.Thread):
    """Core side: pushes the newest snapshot, dropping frames the UI has not consumed."""

    def __init__(self, shared, bus, out_queue, fps: float) -> None:
        super().__init__(name="ui-publisher", daemon=True)
        self.shared = shared
        self.bus = bus
        self.queue = out_queue
        self.period = 1.0 / max(1.0, float(fps))
        self._halt = threading.Event()
        self._cabin_ts = None
        self._photo_key = None

    def stop(self) -> None:
        self._halt.set()

    def snapshot(self) -> dict:
        """Newest state for the dashboard.

        The cabin picture and the enrollment photo change much slower than the driver
        frame. Sending them on every driver frame pickled a multi-megabyte copy about
        20 times a second, and that heap was never given back.
        """
        s = self.shared
        driver = s.driver
        ident = driver.identity
        photo_key = (ident.known, ident.driver_id, ident.photo is not None)
        if photo_key == self._photo_key:
            ident = replace(ident, photo=None)
        else:
            self._photo_key = photo_key
        # Landmarks are only an input to the mesh. The dashboard draws mesh_points.
        driver = replace(driver, identity=ident, landmarks=None)
        cabin = s.passenger
        if cabin.frame is not None and cabin.frame_ts == self._cabin_ts:
            cabin = replace(cabin, frame=None)
        else:
            self._cabin_ts = cabin.frame_ts
        return {"driver": driver, "passenger": cabin, "vlm": s.vlm, "audio": s.audio, "mqtt": s.mqtt,
                "gps": s.gps, "system": s.system, "hardware": s.hardware, "device_id": s.device_id,
                "alerts": self.bus.recent(12)}

    def run(self) -> None:
        last_seq = None
        while not self._halt.wait(self.period):
            cabin = self.shared.passenger
            gps = self.shared.gps
            seq = (self.shared.driver.seq, self.shared.vlm.calls, self.shared.audio.level_dbfs,
                   len(self.bus.history), self.shared.mqtt.acked, cabin.frame_ts, cabin.occupants,
                   cabin.child_present, len(cabin.objects), gps.fix, gps.lat, gps.lon)
            if seq == last_seq:
                continue
            last_seq = seq
            try:
                self.queue.put_nowait(self.snapshot())
            except queue.Full:
                pass  # UI still busy with the previous one


class _StateView:
    """UI side: same attributes the Dashboard reads from SharedState."""

    def __init__(self) -> None:
        from ..state import AudioState, DriverState, GpsState, MqttState, PassengerState, VlmState

        self.driver = DriverState()
        self.passenger = PassengerState()
        self.vlm = VlmState()
        self.audio = AudioState()
        self.mqtt = MqttState()
        self.gps = GpsState()
        self.system = None
        self.hardware = ""
        self.device_id = ""
        self.alerts: list = []

    def apply(self, snap: dict) -> None:
        prev_ident = self.driver.identity
        prev_cabin_frame = self.passenger.frame
        prev_cabin_ts = self.passenger.frame_ts
        for key, value in snap.items():
            setattr(self, key, value)
        ident = self.driver.identity
        if ident.photo is None and prev_ident.driver_id == ident.driver_id and prev_ident.known == ident.known:
            ident.photo = prev_ident.photo
        cabin = self.passenger
        if cabin.frame is None and cabin.frame_ts == prev_cabin_ts:
            cabin.frame = prev_cabin_frame


class _BusView:
    def __init__(self, view: _StateView) -> None:
        self.view = view

    def latest(self, channel=None):
        alerts = self.view.alerts
        if channel is None:
            return alerts[-1] if alerts else None
        for alert in reversed(alerts):
            if getattr(alert, "channel", "dms") == channel:
                return alert
        return None


def run_ui(in_queue, stop_event, fps: int, fullscreen: bool, snapshot: str | None,
           log_level: str, log_path: str | None = None) -> int:
    """Entry point of the HMI process."""
    from dms.logutil import configure_logging

    configure_logging(log_level, log_path or None)
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the core owns shutdown
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from .dashboard import Dashboard

    app = QApplication(["dms-dashboard"])
    app.setApplicationName("Driver Monitoring System")
    view = _StateView()
    window = Dashboard(view, _BusView(view), fps=fps)

    def pump() -> None:
        latest = None
        while True:
            try:
                latest = in_queue.get_nowait()
            except queue.Empty:
                break
            except (EOFError, OSError):
                app.quit()
                return
        if latest is not None:
            view.apply(latest)
        if stop_event.is_set():
            app.quit()

    timer = QTimer()
    timer.timeout.connect(pump)
    timer.start(max(5, int(1000 / max(1, fps)) // 2))
    window.showFullScreen() if fullscreen else window.show()
    code = app.exec()
    if snapshot:
        window.grab().save(snapshot)
    return code


def start_ui_process(cfg, shared, bus, snapshot: str | None = None):
    """Spawn the HMI process and the core-side publisher. Returns (process, publisher, stop_event)."""
    ctx = mp.get_context("spawn")  # never fork a process that has CUDA/Qt state
    q = ctx.Queue(maxsize=1)
    q.cancel_join_thread()  # never block core shutdown on a UI that has gone away
    stop_event = ctx.Event()
    from dms.logutil import log_file

    proc = ctx.Process(target=run_ui, name="dms-dashboard",
                       args=(q, stop_event, int(cfg.ui.fps), bool(cfg.ui.fullscreen), snapshot,
                             str(cfg.log_level).upper(), str(log_file(cfg, ui=True))))
    proc.start()
    publisher = StatePublisher(shared, bus, q, cfg.ui.fps)
    publisher.start()
    return proc, publisher, stop_event
