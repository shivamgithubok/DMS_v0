"""Trip history for one Guardian ride.

``ride-1842`` stays the ride the phone already opens. Each time this process
starts, the open trip is closed and kept, and a new row is inserted with a new
``started_at``. Rows are never deleted. GPS is the live fix when one exists,
otherwise the single configured point.

The snapshot on ``shared.history`` is what MQTT retains for Guardian. The
broker keeps that message when the Jetson is off.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path

log = logging.getLogger("dms.history")

PATH_LIMIT = 200
LIST_LIMIT = 50


def _pin(lat, lon, at) -> dict | None:
    if lat is None or lon is None or at is None:
        return None
    return {"lat": round(float(lat), 6), "lon": round(float(lon), 6), "at": int(at)}


class HistoryStore:
    def __init__(self, path: Path, ride_id: str, device_id: str) -> None:
        self.path = Path(path)
        self.ride_id = ride_id
        self.device_id = device_id
        self._lock = threading.Lock()
        self.db = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS trips (
                ride_id TEXT NOT NULL,
                started_at INTEGER NOT NULL,
                ended_at INTEGER,
                end_seen_at INTEGER,
                status TEXT NOT NULL,
                device_id TEXT NOT NULL,
                safety_status TEXT NOT NULL,
                score INTEGER NOT NULL,
                alert_count INTEGER NOT NULL DEFAULT 0,
                start_lat REAL,
                start_lon REAL,
                end_lat REAL,
                end_lon REAL,
                PRIMARY KEY (ride_id, started_at)
            )"""
        )
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS trip_points (
                ride_id TEXT NOT NULL,
                started_at INTEGER NOT NULL,
                at INTEGER NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                fix TEXT NOT NULL,
                PRIMARY KEY (ride_id, started_at, at)
            )"""
        )

    def close(self) -> None:
        with self._lock:
            self.db.close()

    def open_session(self, now_ms: int) -> int:
        """End whatever trip was left open, keep it, and start another."""
        with self._lock:
            self.db.execute(
                """UPDATE trips
                   SET status = 'ended',
                       ended_at = COALESCE(ended_at, end_seen_at, ?)
                   WHERE ride_id = ? AND status = 'active'""",
                (now_ms, self.ride_id),
            )
            self.db.execute(
                """INSERT INTO trips (
                       ride_id, started_at, status, device_id, safety_status, score, alert_count
                   ) VALUES (?, ?, 'active', ?, 'ok', 100, 0)""",
                (self.ride_id, now_ms, self.device_id),
            )
        log.info("trip started ride=%s started_at=%s", self.ride_id, now_ms)
        return now_ms

    def close_trip(self, started_at: int, now_ms: int) -> None:
        with self._lock:
            self.db.execute(
                """UPDATE trips
                   SET status = 'ended', ended_at = COALESCE(ended_at, ?)
                   WHERE ride_id = ? AND started_at = ?""",
                (now_ms, self.ride_id, started_at),
            )
        log.info("trip kept ride=%s started_at=%s ended_at=%s", self.ride_id, started_at, now_ms)

    def add_alert(self, started_at: int) -> None:
        with self._lock:
            self.db.execute(
                """UPDATE trips
                   SET alert_count = alert_count + 1,
                       safety_status = 'caution',
                       score = MAX(60, 100 - (alert_count + 1) * 5)
                   WHERE ride_id = ? AND started_at = ? AND status = 'active'""",
                (self.ride_id, started_at),
            )

    def note_fix(self, started_at: int, lat: float, lon: float, fix: str, at_ms: int, min_interval_ms: int) -> None:
        lat = round(float(lat), 6)
        lon = round(float(lon), 6)
        with self._lock:
            trip = self.db.execute(
                "SELECT start_lat, end_lat, end_lon, end_seen_at FROM trips WHERE ride_id = ? AND started_at = ?",
                (self.ride_id, started_at),
            ).fetchone()
            if trip is None:
                return
            last = self.db.execute(
                """SELECT at, lat, lon FROM trip_points
                   WHERE ride_id = ? AND started_at = ? ORDER BY at DESC LIMIT 1""",
                (self.ride_id, started_at),
            ).fetchone()
            moved = last is None or abs(last["lat"] - lat) > 1e-5 or abs(last["lon"] - lon) > 1e-5
            due = last is None or at_ms - int(last["at"]) >= min_interval_ms
            if moved and due:
                self.db.execute(
                    """INSERT OR IGNORE INTO trip_points (ride_id, started_at, at, lat, lon, fix)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (self.ride_id, started_at, at_ms, lat, lon, fix or "fixed"),
                )
            self.db.execute(
                """UPDATE trips
                   SET start_lat = COALESCE(start_lat, ?),
                       start_lon = COALESCE(start_lon, ?),
                       end_lat = ?,
                       end_lon = ?,
                       end_seen_at = ?
                   WHERE ride_id = ? AND started_at = ?""",
                (lat, lon, lat, lon, at_ms, self.ride_id, started_at),
            )

    def snapshot(self, focus_started_at: int | None = None) -> dict:
        with self._lock:
            rows = self.db.execute(
                """SELECT * FROM trips WHERE ride_id = ? ORDER BY started_at DESC LIMIT ?""",
                (self.ride_id, LIST_LIMIT),
            ).fetchall()
            trips = [_summary(row) for row in rows]
            focus = focus_started_at
            if focus is None and rows:
                ended = next((row for row in rows if row["status"] == "ended"), None)
                focus = int((ended or rows[0])["started_at"])
            last = None
            if focus is not None:
                row = self.db.execute(
                    "SELECT * FROM trips WHERE ride_id = ? AND started_at = ?",
                    (self.ride_id, focus),
                ).fetchone()
                if row is not None:
                    points = self.db.execute(
                        """SELECT at, lat, lon FROM trip_points
                           WHERE ride_id = ? AND started_at = ? ORDER BY at""",
                        (self.ride_id, focus),
                    ).fetchall()
                    last = _summary(row)
                    last["path"] = _thin([{"at": int(p["at"]), "lat": p["lat"], "lon": p["lon"]} for p in points])
        return {
            "rideId": self.ride_id,
            "deviceId": self.device_id,
            "trips": trips,
            "lastTrip": last,
        }


def _summary(row: sqlite3.Row) -> dict:
    started = int(row["started_at"])
    return {
        "startedAt": started,
        "endedAt": None if row["ended_at"] is None else int(row["ended_at"]),
        "status": row["status"],
        "safetyStatus": row["safety_status"],
        "score": int(row["score"]),
        "alertCount": int(row["alert_count"]),
        "start": _pin(row["start_lat"], row["start_lon"], started),
        "end": _pin(row["end_lat"], row["end_lon"], row["end_seen_at"] or row["ended_at"] or started),
    }


def _thin(points: list[dict]) -> list[dict]:
    if len(points) <= PATH_LIMIT:
        return points
    step = (len(points) - 1) / (PATH_LIMIT - 1)
    chosen = [points[int(round(i * step))] for i in range(PATH_LIMIT)]
    chosen[-1] = points[-1]
    return chosen


class TripHistory(threading.Thread):
    def __init__(self, cfg, shared, bus) -> None:
        super().__init__(name="trip-history", daemon=True)
        section = cfg.history
        from .config import data_path

        self.store = HistoryStore(data_path(str(section.db or "history.sqlite3")), str(section.ride_id), cfg.device_id)
        self.shared = shared
        self.bus = bus
        self.interval_ms = max(1000, int(float(section.point_interval_s or 10) * 1000))
        self._halt = threading.Event()
        self._closed = False
        self._started_at = 0
        self.bus.subscribe(self._on_alert)

    def stop(self) -> None:
        self._halt.set()

    def _on_alert(self, _alert) -> None:
        if self._closed or not self._started_at:
            return
        try:
            self.store.add_alert(self._started_at)
            self._publish()
        except sqlite3.ProgrammingError:
            return

    def _sample(self) -> None:
        gps = self.shared.gps
        if not gps.available or gps.lat is None or gps.lon is None:
            return
        at_ms = int(gps.at or time.time() * 1000)
        self.store.note_fix(self._started_at, gps.lat, gps.lon, gps.fix or "fixed", at_ms, self.interval_ms)

    def _publish(self) -> None:
        self.shared.history = self.store.snapshot(self._started_at if self._halt.is_set() else None)

    def run(self) -> None:
        self._started_at = self.store.open_session(int(time.time() * 1000))
        try:
            while not self._halt.is_set():
                self._sample()
                self._publish()
                self._halt.wait(1.0)
        finally:
            self._closed = True
            if self._started_at:
                self.store.close_trip(self._started_at, int(time.time() * 1000))
                self._publish()
            self.store.close()
