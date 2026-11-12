"""Vehicle position for the heartbeat and for alerts.

There is no GPS receiver on the Jetson yet. With ``gps.source: fixed`` the
configured ``lat`` / ``lon`` is the position (one point). A live fix replaces
it as soon as one exists:

* ``GpsFeed.update(lat, lon)`` from any thread
* ``source: nmea`` reading ``$GGA`` from ``gps.device``
* ``source: file`` reading ``{"lat", "lon"}`` from ``gps.file``

If that live fix stops (port missing, file gone, or older than ``stale_s``),
the configured point is used again. With no point configured, GPS stays
unavailable and nothing is attached to alerts.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

from .state import GpsState

log = logging.getLogger("dms.gps")

ROOT = Path(__file__).resolve().parents[1]


def parse_gga(line: str) -> tuple[float, float] | None:
    """Return decimal degrees from one ``$GGA`` sentence, or None when it is not a fix."""
    raw = line.strip()
    if not raw.startswith("$") or "GGA" not in raw[:6]:
        return None
    if not _checksum_ok(raw):
        return None
    body = raw[1:].split("*", 1)[0]
    fields = body.split(",")
    if len(fields) < 7 or fields[6] in ("", "0"):
        return None
    lat = _nmea_deg(fields[2], fields[3], lat=True)
    lon = _nmea_deg(fields[4], fields[5], lat=False)
    if lat is None or lon is None:
        return None
    return lat, lon


def _checksum_ok(line: str) -> bool:
    star = line.rfind("*")
    if star < 0 or star + 3 > len(line):
        return False
    try:
        expect = int(line[star + 1:star + 3], 16)
    except ValueError:
        return False
    got = 0
    for ch in line[1:star]:
        got ^= ord(ch)
    return got == expect


def _nmea_deg(raw: str, hemi: str, lat: bool) -> float | None:
    if not raw or hemi not in (("N", "S") if lat else ("E", "W")):
        return None
    try:
        number = float(raw)
    except ValueError:
        return None
    degrees = int(number // 100)
    minutes = number - degrees * 100
    if minutes >= 60:
        return None
    value = degrees + minutes / 60.0
    if (lat and not -90 <= value <= 90) or (not lat and not -180 <= value <= 180):
        return None
    if hemi in ("S", "W"):
        value = -value
    return value


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in metres between two decimal-degree points."""
    from math import asin, cos, radians, sin, sqrt

    radius = 6_371_000.0
    phi1, phi2 = radians(lat1), radians(lat2)
    d_phi = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(d_lon / 2) ** 2
    return 2 * radius * asin(min(1.0, sqrt(a)))


def route_point(source: str, fix: str, lat, lon) -> tuple[float, float] | None:
    """A point that may be drawn on the trip. A configured bench pin is not a route."""
    if str(source or "") == "fixed" or str(fix or "") != "live":
        return None
    lat_v = _coord(lat, -90, 90)
    lon_v = _coord(lon, -180, 180)
    if lat_v is None or lon_v is None:
        return None
    return lat_v, lon_v


def normalize_places(rows) -> list[dict]:
    places = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        lat = _coord(row.get("lat", row.get("latitude")), -90, 90)
        lon = _coord(row.get("lon", row.get("longitude")), -180, 180)
        if lat is None or lon is None:
            continue
        try:
            radius = float(row.get("radius_m") or row.get("radiusM") or 80)
        except (TypeError, ValueError):
            radius = 80.0
        places.append({
            "id": str(row.get("id") or f"{lat:.4f},{lon:.4f}"),
            "name": str(row.get("name") or "Home"),
            "lat": lat,
            "lon": lon,
            "radius_m": max(20.0, radius),
        })
    return places


class PlaceWatch:
    """One arrival per saved place per trip, and one unexpected stop per hold.

    A fixed point or a stale fix produces nothing.
    The stop has to hold, and the vehicle has to have moved first, so a bad stick
    at the curb does not page the family at the start of the trip.
    """

    def __init__(self, places=None) -> None:
        self.places = normalize_places(places)
        self.arrived: set[str] = set()
        self.moved_m = 0.0
        self._last: tuple[float, float, float] | None = None
        self._anchor: tuple[float, float, float] | None = None
        self.stopped = False

    def consider(
        self,
        *,
        source: str,
        fix: str,
        fresh: bool,
        lat: float,
        lon: float,
        at_s: float,
        hold_s: float = 120.0,
        move_m: float = 80.0,
        gap_s: float = 15.0,
    ) -> list[dict]:
        if not fresh or route_point(source, fix, lat, lon) is None:
            return []
        if self._last is not None and at_s - self._last[2] > gap_s:
            self._anchor = None
            self.stopped = False
        if self._last is not None:
            self.moved_m += haversine_m(self._last[0], self._last[1], lat, lon)
        events = []
        inside = None
        for place in self.places:
            if haversine_m(lat, lon, place["lat"], place["lon"]) > place["radius_m"]:
                continue
            inside = place
            if place["id"] in self.arrived:
                continue
            self.arrived.add(place["id"])
            events.append({
                "type": "ARRIVED",
                "reason": place["id"],
                "description": f"Arrived at {place['name']}",
            })
        if inside is not None:
            self._anchor = None
            self.stopped = False
        else:
            if self._anchor is None or haversine_m(self._anchor[0], self._anchor[1], lat, lon) > 30:
                self._anchor = (lat, lon, at_s)
                self.stopped = False
            elif not self.stopped and self.moved_m >= move_m and at_s - self._anchor[2] >= hold_s:
                self.stopped = True
                events.append({
                    "type": "UNEXPECTED_STOP",
                    "reason": f"stop-{int(self._anchor[2])}",
                    "description": "The vehicle has stopped",
                })
        self._last = (lat, lon, at_s)
        return events


def _coord(value, low: float, high: float) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not low <= number <= high:
        return None
    return number


class GpsFeed(threading.Thread):
    def __init__(self, cfg, shared) -> None:
        super().__init__(name="gps", daemon=True)
        section = cfg.gps
        self.shared = shared
        self.enabled = bool(section.enabled)
        self.source = str(section.source or "fixed").lower()
        if self.source not in ("fixed", "nmea", "file"):
            log.warning("gps.source %r is unknown; using the fixed point", self.source)
            self.source = "fixed"
        self.fixed_lat = _coord(section.lat, -90, 90)
        self.fixed_lon = _coord(section.lon, -180, 180)
        if (self.fixed_lat is None) != (self.fixed_lon is None):
            log.warning("gps.lat and gps.lon must both be set; ignoring the single coordinate")
            self.fixed_lat = self.fixed_lon = None
        self.interval = max(0.2, float(section.interval_s or 1))
        self.stale_s = max(1.0, float(section.stale_s or 15))
        self.device = str(section.device or "")
        self.baud = int(section.baud or 9600)
        self.file = _resolve_file(str(section.file or ""))
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self._live: tuple[float, float, float] | None = None  # lat, lon, monotonic time
        self._fd: int | None = None
        self._buf = b""
        self._port_logged = False

    def stop(self) -> None:
        self._halt.set()

    def update(self, lat: float, lon: float) -> None:
        """Push a live fix. It replaces the configured point until it goes stale."""
        lat_v = _coord(lat, -90, 90)
        lon_v = _coord(lon, -180, 180)
        if lat_v is None or lon_v is None:
            log.warning("ignored GPS update lat=%s lon=%s", lat, lon)
            return
        with self._lock:
            self._live = (lat_v, lon_v, time.monotonic())
        self._publish()

    def run(self) -> None:
        if self.fixed_lat is None:
            log.info("GPS has no fixed point yet (set gps.lat and gps.lon); source=%s", self.source)
        else:
            log.info("GPS fixed point %.6f, %.6f (source=%s)", self.fixed_lat, self.fixed_lon, self.source)
        try:
            while not self._halt.is_set():
                if self.source == "nmea":
                    self._read_nmea()
                elif self.source == "file":
                    self._read_file()
                self._publish()
                self._halt.wait(self.interval)
        finally:
            self._close_port()

    def _publish(self) -> None:
        now_ms = int(time.time() * 1000)
        live = self._fresh_live()
        if live is not None:
            lat, lon = live
            state = GpsState(enabled=True, available=True, fix="live", source=self.source,
                             lat=lat, lon=lon, at=now_ms)
        elif self.fixed_lat is not None and self.fixed_lon is not None:
            state = GpsState(enabled=True, available=True, fix="fixed", source="fixed",
                             lat=self.fixed_lat, lon=self.fixed_lon, at=now_ms)
        else:
            state = GpsState(enabled=True, available=False, fix="none", source=self.source, at=0)
        self.shared.gps = state

    def _fresh_live(self) -> tuple[float, float] | None:
        with self._lock:
            live = self._live
        if live is None:
            return None
        lat, lon, seen = live
        if time.monotonic() - seen > self.stale_s:
            return None
        return lat, lon

    def _read_file(self) -> None:
        if self.file is None:
            return
        try:
            body = json.loads(self.file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            return
        if not isinstance(body, dict):
            return
        lat = _coord(body.get("lat"), -90, 90)
        lon = _coord(body.get("lon"), -180, 180)
        if lat is None or lon is None:
            return
        with self._lock:
            self._live = (lat, lon, time.monotonic())

    def _read_nmea(self) -> None:
        if not self.device:
            if not self._port_logged:
                self._port_logged = True
                log.info("gps.source is nmea but gps.device is empty; using the fixed point")
            return
        if self._fd is None:
            self._open_port()
        if self._fd is None:
            return
        try:
            chunk = os.read(self._fd, 2048)
        except BlockingIOError:
            return
        except OSError as exc:
            log.warning("GPS port %s read failed (%s); using the fixed point", self.device, exc)
            self._close_port()
            return
        if not chunk:
            return
        self._buf = (self._buf + chunk)[-8192:]
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            fix = parse_gga(line.decode("ascii", "ignore"))
            if fix is None:
                continue
            with self._lock:
                self._live = (fix[0], fix[1], time.monotonic())

    def _open_port(self) -> None:
        try:
            fd = os.open(self.device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        except OSError as exc:
            if not self._port_logged:
                self._port_logged = True
                log.warning("GPS port %s is not available (%s); using the fixed point", self.device, exc.strerror or exc)
            return
        try:
            import termios
            attrs = termios.tcgetattr(fd)
            speed = getattr(termios, f"B{self.baud}", termios.B9600)
            attrs[4] = attrs[5] = speed
            attrs[0] = 0
            attrs[1] = 0
            attrs[2] = (attrs[2] & ~termios.CSIZE & ~termios.PARENB & ~termios.CSTOPB) | termios.CS8 | termios.CREAD
            attrs[3] = 0
            attrs[6][termios.VMIN] = 0
            attrs[6][termios.VTIME] = 0
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
        except Exception as exc:
            os.close(fd)
            if not self._port_logged:
                self._port_logged = True
                log.warning("GPS port %s could not be configured (%s); using the fixed point", self.device, exc)
            return
        self._fd = fd
        self._port_logged = False
        log.info("GPS reading NMEA from %s at %d baud", self.device, self.baud)

    def _close_port(self) -> None:
        if self._fd is None:
            return
        try:
            os.close(self._fd)
        except OSError:
            pass
        self._fd = None


def _resolve_file(raw: str) -> Path | None:
    if not raw:
        return None
    path = Path(os.path.expanduser(raw))
    if not path.is_absolute():
        path = ROOT / path
    return path
