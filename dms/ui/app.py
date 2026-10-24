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

    def stop(self) -> None:
        self._halt.set()

    def snapshot(self) -> dict:
        s = self.shared
        return {"driver": s.driver, "passenger": s.passenger, "vlm": s.vlm, "audio": s.audio, "mqtt": s.mqtt,
                "system": s.system, "hardware": s.hardware, "device_id": s.device_id,
                "alerts": self.bus.recent(12)}

    def run(self) -> None:
        last_seq = None
        while not self._halt.wait(self.period):
            cabin = self.shared.passenger
            seq = (self.shared.driver.seq, self.shared.vlm.calls, self.shared.audio.level_dbfs,
                   len(self.bus.history), self.shared.mqtt.acked, cabin.frame_ts, cabin.occupants,
                   cabin.child_present, len(cabin.objects))
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
        from ..state import AudioState, DriverState, MqttState, PassengerState, VlmState

        self.driver = DriverState()
        self.passenger = PassengerState()
        self.vlm = VlmState()
        self.audio = AudioState()
        self.mqtt = MqttState()
        self.system = None
        self.hardware = ""
        self.device_id = ""
        self.alerts: list = []

    def apply(self, snap: dict) -> None:
        for key, value in snap.items():
            setattr(self, key, value)


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


def run_ui(in_queue, stop_event, fps: int, fullscreen: bool, snapshot: str | None, log_level: str) -> int:
    """Entry point of the HMI process."""
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname).1s ui: %(message)s", datefmt="%H:%M:%S")
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
    proc = ctx.Process(target=run_ui, name="dms-dashboard",
                       args=(q, stop_event, int(cfg.ui.fps), bool(cfg.ui.fullscreen), snapshot,
                             str(cfg.log_level).upper()))
    proc.start()
    publisher = StatePublisher(shared, bus, q, cfg.ui.fps)
    publisher.start()
    return proc, publisher, stop_event
