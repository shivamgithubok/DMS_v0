from __future__ import annotations

import json
import time
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .mapping import audience_for, is_auto_sos, kind_for, seats_from_cms, severity_for, title_for
from .models import Alert, Guardian, MediaClip, Notice, Ride, Trip, TripPoint, utcnow


def ride_for_device(db: Session, device_id: str) -> Ride:
    ride = db.scalars(select(Ride).where(Ride.device_id == device_id)).first()
    if ride is None:
        ride = db.get(Ride, settings.default_ride_id)
    if ride is None:
        ride = Ride(id=settings.default_ride_id, device_id=device_id or settings.default_device_id)
        db.add(ride)
        db.flush()
    if device_id and not ride.device_id:
        ride.device_id = device_id
    return ride


def alert_row(db: Session, ride: Ride, body: dict, origin: str = "vehicle") -> Alert | None:
    alert_id = str(body.get("id") or uuid.uuid4().hex)
    if db.scalars(select(Alert).where(Alert.id == alert_id)).first():
        return None
    orin_type = str(body.get("type") or "")
    kind = str(body.get("kind") or "") or kind_for(orin_type)
    if is_auto_sos(orin_type) or orin_type.upper() == "SOS":
        kind = "sos"
    at = int(body.get("epoch_ms") or body.get("at") or int(time.time() * 1000))
    coord = body.get("coordinate") if isinstance(body.get("coordinate"), dict) else {}
    lat = _number(coord.get("latitude"), _number(body.get("lat"), ride.lat))
    lon = _number(coord.get("longitude"), _number(body.get("lon"), ride.lon))
    conf_raw = body.get("confidence")
    try:
        confidence = float(conf_raw) if conf_raw is not None else None
    except (TypeError, ValueError):
        confidence = None
    channel = str(body.get("channel") or "").strip().lower()
    if channel not in {"dms", "cms"}:
        channel = "cms" if kind in {"seat_occupancy", "suspicious_activity"} else "dms"
        if kind == "sos":
            channel = "dms"
    row = Alert(
        id=alert_id,
        ride_id=ride.id,
        kind=kind,
        orin_type=orin_type,
        severity=severity_for(kind, body.get("severity")),
        title=title_for(kind, str(body.get("title") or "")),
        detail=str(body.get("description") or body.get("detail") or ""),
        at=at,
        location_label=str(body.get("locationLabel") or body.get("reason") or ""),
        lat=lat,
        lon=lon,
        origin=origin,
        evidence_id=str(body.get("id") or "") if body.get("snapshot_path") or body.get("audio_path") else "",
        channel=channel,
        confidence=confidence,
    )
    active = db.scalars(
        select(Trip).where(Trip.ride_id == ride.id, Trip.status == "active").order_by(Trip.started_at.desc())
    ).first()
    if active is not None:
        row.trip_started_at = active.started_at
    db.add(row)
    if kind == "sos":
        ride.safety_status = "emergency"
        ride.score = max(8, ride.score - 20)
    elif row.severity in {"high", "critical"} and ride.safety_status != "emergency":
        ride.safety_status = "caution"
        ride.score = max(46, ride.score - 8)
    if kind == "drowsiness":
        ride.driver_state = "drowsy"
    elif kind == "distraction":
        ride.driver_state = "distracted"
    ride.updated_at = utcnow()
    clip = db.scalars(
        select(MediaClip).where(MediaClip.alert_id == alert_id, MediaClip.ride_id == ride.id)
    ).first()
    if clip is not None:
        row.clip_id = clip.id
    write_story(row, clip)
    return row


def apply_status(db: Session, topic: str, body: dict) -> Ride:
    device_id = str(body.get("device_id") or settings.default_device_id)
    ride = ride_for_device(db, device_id)
    ride.device_state = str(body.get("state") or "online")
    ride.updated_at = utcnow()
    if topic.endswith("/status") and "cms/" in topic:
        seats = seats_from_cms(body.get("seats"))
        if seats:
            ride.cabin_json = json.dumps(seats)
    if body.get("type") == "DRIVER_ABSENT" or body.get("reason") == "no_face":
        ride.driver_state = "unavailable"
    if topic.startswith("dms/"):
        gps = body.get("gps") if isinstance(body.get("gps"), dict) else None
        if gps and gps.get("available") and gps.get("lat") is not None and gps.get("lon") is not None:
            record_point(db, ride, float(gps["lat"]), float(gps["lon"]), int(gps.get("at") or time.time() * 1000), str(gps.get("fix") or "fixed"))
        if str(body.get("state") or "") == "offline":
            close_open_trips(db, ride.id, int(time.time() * 1000))
    return ride


def apply_trips(db: Session, topic: str, body: dict) -> Ride:
    """Keep each Jetson trip. Same ride id, different started_at. Points are not deleted for other trips."""
    parts = topic.split("/")
    device_id = str(body.get("deviceId") or body.get("device_id") or (parts[1] if len(parts) >= 3 else settings.default_device_id))
    ride = ride_for_device(db, device_id)
    incoming = []
    for item in body.get("trips") or []:
        if isinstance(item, dict) and item.get("startedAt"):
            incoming.append(item)
    last = body.get("lastTrip") if isinstance(body.get("lastTrip"), dict) else None
    if last and last.get("startedAt"):
        merged = {int(item["startedAt"]): item for item in incoming}
        started = int(last["startedAt"])
        merged[started] = {**merged.get(started, {}), **last}
        incoming = list(merged.values())
    active_starts = set()
    for item in incoming:
        trip = _upsert_trip(db, ride, item)
        if trip.status == "active":
            active_starts.add(trip.started_at)
        path = item.get("path")
        if isinstance(path, list) and path:
            _replace_path(db, trip, path)
    if active_starts:
        now = int(time.time() * 1000)
        for other in db.scalars(select(Trip).where(Trip.ride_id == ride.id, Trip.status == "active")).all():
            if other.started_at not in active_starts:
                other.status = "ended"
                other.ended_at = other.ended_at or now
                stamp_link(other, now)
    _pin_ride(ride, last or (incoming[0] if incoming else None))
    return ride


def apply_analysis(db: Session, topic: str, body: dict) -> tuple[Ride, str, dict]:
    device_id = str(body.get("device_id") or settings.default_device_id)
    ride = ride_for_device(db, device_id)
    prefix = str(topic).split("/")[0] if topic else ""
    channel = str(body.get("channel") or prefix).strip().lower()
    if channel not in {"dms", "cms"}:
        channel = "cms" if prefix == "cms" else "dms"
    at = int(body.get("at") or body.get("epoch_ms") or int(time.time() * 1000))
    side = {
        "scene": str(body.get("scene") or ""),
        "description": str(body.get("description") or ""),
        "hazards": list(body.get("hazards") or []),
        "confidence": body.get("confidence") if body.get("confidence") is not None else 0.0,
        "suppressed": list(body.get("suppressed") or []),
        "observations": body.get("observations") if isinstance(body.get("observations"), dict) else {},
        "at": at,
    }
    try:
        data = json.loads(ride.analysis_json or "{}")
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    data[channel] = side
    data["updatedAt"] = at
    ride.analysis_json = json.dumps(data)
    ride.updated_at = utcnow()
    return ride, channel, side


def analysis_public(ride: Ride, channel: str | None = None) -> dict:
    try:
        data = json.loads(ride.analysis_json or "{}")
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    out = {
        "deviceId": ride.device_id,
        "updatedAt": data.get("updatedAt"),
    }
    if channel == "dms":
        out["dms"] = data.get("dms")
        return out
    if channel == "cms":
        out["cms"] = data.get("cms")
        return out
    out["dms"] = data.get("dms")
    out["cms"] = data.get("cms")
    return out


def alert_channel(row: Alert) -> str:
    if row.channel in {"dms", "cms"}:
        return row.channel
    if row.kind in {"seat_occupancy", "suspicious_activity"}:
        return "cms"
    return "dms"


def raise_sos(db: Session, ride_id: str, passenger: str, lat: float | None, lon: float | None) -> tuple[Ride, Alert, list[str]]:
    ride = db.get(Ride, ride_id) or ride_for_device(db, settings.default_device_id)
    ride.passenger = passenger or ride.passenger
    if lat is not None:
        ride.lat = lat
    if lon is not None:
        ride.lon = lon
    body = {
        "id": f"sos-{uuid.uuid4().hex[:12]}",
        "kind": "sos",
        "type": "SOS",
        "severity": "critical",
        "description": f"{ride.passenger} raised SOS.",
        "epoch_ms": int(time.time() * 1000),
        "coordinate": {"latitude": lat, "longitude": lon},
    }
    row = alert_row(db, ride, body, origin="device")
    assert row is not None
    names = [
        g.name
        for g in db.scalars(select(Guardian).where(Guardian.ride_id == ride.id, Guardian.sharing.is_(True)))
    ]
    now = int(time.time() * 1000)
    for guardian in db.scalars(select(Guardian).where(Guardian.ride_id == ride.id, Guardian.sharing.is_(True))):
        guardian.last_notified_at = now
    db.add(
        Notice(
            id=f"notice-{uuid.uuid4().hex[:10]}",
            ride_id=ride.id,
            at=now,
            title="SOS sent",
            body=f"Notified {', '.join(names) or 'nobody sharing'}.",
            guardian_names=",".join(names),
            origin="device",
        )
    )
    return ride, row, names


def alert_public(row: Alert) -> dict:
    return {
        "id": row.id,
        "tripId": row.ride_id,
        "kind": row.kind,
        "orinType": row.orin_type or None,
        "channel": alert_channel(row),
        "severity": row.severity,
        "title": row.title,
        "detail": row.detail,
        "at": row.at,
        "locationLabel": row.location_label,
        "coordinate": {"latitude": row.lat, "longitude": row.lon},
        "acknowledged": row.acknowledged,
        "evidenceId": row.evidence_id or None,
        "origin": row.origin,
        "confidence": row.confidence,
        "clipId": row.clip_id or None,
        "tripStartedAt": row.trip_started_at,
        "audience": audience_for(row.orin_type or "", row.kind),
        "story": row.story or "",
    }


_STORY_LABEL = {
    "DROWSINESS": "Drowsiness",
    "DRIVER_ABSENT": "The driver was not visible",
    "SCREAM": "A scream",
    "ALTERCATION": "An altercation",
    "CHILD_PRESENT": "A child was visible with no adult",
    "SOS": "An SOS",
    "CAMERA_BLOCKED": "The camera was covered",
    "ARRIVED": "Arrival",
    "UNEXPECTED_STOP": "An unexpected stop",
    "PHONE_USAGE": "Phone use",
    "DISTRACTION": "A glance off the road",
    "SMOKING": "Smoking",
    "DRINKING": "Drinking",
    "SEATBELT_OFF": "The seatbelt was off",
    "OCCUPANCY_CHANGE": "The occupancy changed",
    "VLM_ALERT": "A cabin note",
    "ABUSIVE_LANGUAGE": "Abusive language",
    "OBJECT_ON_SEAT": "An object on a seat",
    "PASSENGER_AGGRESSIVE": "An aggressive passenger",
}


def story_text(row: Alert, clip: MediaClip | None = None) -> str:
    """One sentence from this alert only. A missing clip says it is still on the way."""
    label = _STORY_LABEL.get(str(row.orin_type or "").upper(), "An alert")
    parts = [f"{label} was recorded"]
    if clip is not None and clip.end_at and clip.start_at and int(clip.end_at) > int(clip.start_at):
        seconds = max(1, round((int(clip.end_at) - int(clip.start_at)) / 1000))
        parts.append(f"for {seconds} seconds")
    if row.confidence is not None:
        parts.append(f"at {round(float(row.confidence) * 100)}% confidence")
    if row.lat is not None and row.lon is not None:
        parts.append(f"near {float(row.lat):.5f}, {float(row.lon):.5f}")
    sentence = " ".join(parts) + "."
    if not (row.clip_id or ""):
        sentence += " The clip is on the way."
    return sentence


def write_story(row: Alert, clip: MediaClip | None = None) -> None:
    row.story = story_text(row, clip)


def stamp_link(trip: Trip, now_ms: int | None = None) -> None:
    """The shared link dies at ended_at plus the grace period. An open trip stays valid."""
    if trip is None or ((trip.status or "") == "active" and trip.ended_at is None):
        return
    if trip.link_expires_at:
        return
    end = int(trip.ended_at or now_ms or int(time.time() * 1000))
    trip.ended_at = trip.ended_at or end
    trip.link_expires_at = end + int(settings.link_grace_s) * 1000


def link_expired(trip: Trip | None, now_ms: int | None = None) -> bool:
    if trip is None or ((trip.status or "") == "active" and trip.ended_at is None):
        return False
    now = int(now_ms if now_ms is not None else time.time() * 1000)
    if trip.link_expires_at:
        return now >= int(trip.link_expires_at)
    return now >= int(trip.ended_at or 0) + int(settings.link_grace_s) * 1000


def record_point(db: Session, ride: Ride, lat: float, lon: float, at_ms: int, fix: str = "live") -> Trip | None:
    """Append a live fix. A fixed bench pin is not a route and does not open a trip."""
    if str(fix or "") != "live":
        return db.scalars(
            select(Trip).where(Trip.ride_id == ride.id, Trip.status == "active").order_by(Trip.started_at.desc())
        ).first()
    trip = db.scalars(
        select(Trip).where(Trip.ride_id == ride.id, Trip.status == "active").order_by(Trip.started_at.desc())
    ).first()
    if trip is None:
        trip = Trip(
            ride_id=ride.id,
            started_at=at_ms,
            status="active",
            device_id=ride.device_id or settings.default_device_id,
            safety_status=ride.safety_status or "safe",
            score=ride.score if ride.score is not None else 98,
            alert_count=0,
        )
        db.add(trip)
        db.flush()
    last = db.scalars(
        select(TripPoint).where(TripPoint.ride_id == ride.id, TripPoint.started_at == trip.started_at).order_by(TripPoint.at.desc())
    ).first()
    moved = last is None or abs(last.lat - lat) > 1e-5 or abs(last.lon - lon) > 1e-5
    if moved and (last is None or at_ms > last.at):
        db.add(TripPoint(ride_id=ride.id, started_at=trip.started_at, at=at_ms, lat=lat, lon=lon, fix=fix or "live"))
    if trip.start_lat is None:
        trip.start_lat = lat
        trip.start_lon = lon
    trip.end_lat = lat
    trip.end_lon = lon
    ride.lat = lat
    ride.lon = lon
    ride.updated_at = utcnow()
    return trip


def close_open_trips(db: Session, ride_id: str, now_ms: int) -> None:
    for trip in db.scalars(select(Trip).where(Trip.ride_id == ride_id, Trip.status == "active")).all():
        trip.status = "ended"
        trip.ended_at = trip.ended_at or now_ms
        stamp_link(trip, now_ms)


def trip_route(db: Session, ride_id: str, started_at: int | None = None) -> tuple[Trip | None, list[TripPoint]]:
    if started_at:
        trip = db.scalars(select(Trip).where(Trip.ride_id == ride_id, Trip.started_at == started_at)).first()
    else:
        rows = db.scalars(select(Trip).where(Trip.ride_id == ride_id).order_by(Trip.started_at.desc())).all()
        trip = next((row for row in rows if row.end_lat is not None), rows[0] if rows else None)
    if trip is None:
        return None, []
    points = db.scalars(
        select(TripPoint).where(TripPoint.ride_id == ride_id, TripPoint.started_at == trip.started_at).order_by(TripPoint.at)
    ).all()
    return trip, list(points)


def trips_for_device(db: Session, device_id: str = "") -> list[Trip]:
    query = select(Trip)
    if device_id:
        query = query.where(Trip.device_id == device_id)
    rows = db.scalars(query).all()
    rows.sort(key=lambda trip: trip.started_at or 0, reverse=True)
    return rows


def alerts_for_window(db: Session, ride_id: str, started_at: int, ended_at: int | None) -> list[Alert]:
    start = int(started_at)
    end = int(ended_at) if ended_at else None
    rows = db.scalars(select(Alert).where(Alert.ride_id == ride_id)).all()
    kept = []
    for row in rows:
        at = int(row.at or 0)
        if at < start:
            continue
        if end is not None and at > end:
            continue
        kept.append(row)
    kept.sort(key=lambda row: row.at or 0, reverse=True)
    return kept


def trip_public(ride: Ride, trip: Trip, db: Session | None = None) -> dict:
    count = trip.alert_count
    if db is not None:
        count = len(alerts_for_window(db, trip.ride_id, trip.started_at, trip.ended_at))
    return {
        "id": trip.public_id or trip.ride_id,
        "rideId": trip.ride_id,
        "deviceId": trip.device_id or ride.device_id,
        "passenger": ride.passenger,
        "status": trip.status or "ended",
        "startedAt": trip.started_at,
        "endedAt": trip.ended_at,
        "safetyStatus": trip.safety_status or ride.safety_status,
        "score": trip.score,
        "alertCount": count,
        "start": _coord(trip.start_lat, trip.start_lon),
        "end": _coord(trip.end_lat, trip.end_lon),
    }


def _number(value, fallback):
    if value is None or value == "":
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _coord(lat, lon) -> dict | None:
    if lat is None or lon is None:
        return None
    return {"latitude": lat, "longitude": lon}


def _upsert_trip(db: Session, ride: Ride, item: dict) -> Trip:
    started = int(item["startedAt"])
    trip = db.scalars(select(Trip).where(Trip.ride_id == ride.id, Trip.started_at == started)).first()
    if trip is None:
        trip = Trip(ride_id=ride.id, started_at=started, device_id=ride.device_id or settings.default_device_id)
        db.add(trip)
    trip.ended_at = item.get("endedAt")
    trip.status = str(item.get("status") or ("ended" if trip.ended_at else "active"))
    trip.device_id = str(item.get("deviceId") or trip.device_id or ride.device_id or "")
    trip.safety_status = str(item.get("safetyStatus") or trip.safety_status or "safe")
    public_id = str(item.get("id") or item.get("tripId") or "").strip()
    if public_id and public_id != ride.id:
        trip.public_id = public_id
    if item.get("score") is not None:
        trip.score = int(item["score"])
    if item.get("alertCount") is not None:
        trip.alert_count = int(item["alertCount"])
    start = item.get("start") if isinstance(item.get("start"), dict) else {}
    end = item.get("end") if isinstance(item.get("end"), dict) else {}
    if start.get("lat") is not None and start.get("lon") is not None:
        trip.start_lat = float(start["lat"])
        trip.start_lon = float(start["lon"])
    if end.get("lat") is not None and end.get("lon") is not None:
        trip.end_lat = float(end["lat"])
        trip.end_lon = float(end["lon"])
    stamp_link(trip)
    return trip


def _replace_path(db: Session, trip: Trip, path: list) -> None:
    db.query(TripPoint).filter(TripPoint.ride_id == trip.ride_id, TripPoint.started_at == trip.started_at).delete()
    db.flush()
    seen = set()
    for point in path:
        if not isinstance(point, dict) or point.get("lat") is None or point.get("lon") is None:
            continue
        at = int(point.get("at") or trip.started_at)
        if at in seen:
            continue
        seen.add(at)
        db.add(
            TripPoint(
                ride_id=trip.ride_id,
                started_at=trip.started_at,
                at=at,
                lat=float(point["lat"]),
                lon=float(point["lon"]),
                fix="live",
            )
        )


def _pin_ride(ride: Ride, item: dict | None) -> None:
    if not item:
        return
    end = item.get("end") if isinstance(item.get("end"), dict) else {}
    lat = end.get("lat", end.get("latitude"))
    lon = end.get("lon", end.get("longitude"))
    if lat is None or lon is None:
        return
    ride.lat = float(lat)
    ride.lon = float(lon)
    ride.updated_at = utcnow()


def ride_public(ride: Ride, db: Session | None = None, started_at: int | None = None) -> dict:
    try:
        seats = json.loads(ride.cabin_json or "[]")
    except json.JSONDecodeError:
        seats = []
    body = {
        "id": ride.id,
        "deviceId": ride.device_id,
        "passenger": ride.passenger,
        "safetyStatus": ride.safety_status,
        "driverState": ride.driver_state,
        "deviceState": ride.device_state,
        "etaMin": ride.eta_min,
        "speedKmh": ride.speed_kmh,
        "heading": ride.heading,
        "score": ride.score,
        "shareLink": ride.share_link,
        "position": {"latitude": ride.lat, "longitude": ride.lon},
        "seats": seats,
        "updatedAt": ride.updated_at.isoformat() if ride.updated_at else None,
        "publisherOnline": publisher_online(ride),
        "status": ride.status or "active",
        "startedAt": ride.started_at or None,
        "endedAt": ride.ended_at,
        "route": [],
    }
    if db is None:
        return body
    trip, points = trip_route(db, ride.id, started_at)
    if trip is None:
        return body
    body["startedAt"] = trip.started_at
    body["endedAt"] = trip.ended_at
    if trip.public_id:
        body["tripId"] = trip.public_id
    body["status"] = trip.status or body["status"]
    body["score"] = trip.score
    body["safetyStatus"] = trip.safety_status or body["safetyStatus"]
    body["route"] = [{"latitude": point.lat, "longitude": point.lon} for point in points[:200]]
    if trip.end_lat is not None and trip.end_lon is not None:
        body["position"] = {"latitude": trip.end_lat, "longitude": trip.end_lon}
    return body


def history_public(db: Session, ride: Ride) -> dict:
    count = db.scalars(select(Alert).where(Alert.ride_id == ride.id)).all()
    return {
        "id": ride.id,
        "deviceId": ride.device_id,
        "passenger": ride.passenger,
        "status": ride.status or "active",
        "startedAt": ride.started_at or None,
        "endedAt": ride.ended_at,
        "safetyStatus": ride.safety_status,
        "score": ride.score,
        "alertCount": len(count),
    }


def resolve_ride(db: Session, key: str, started_at: int | None = None) -> tuple[Ride | None, int | None]:
    """Open a ride by its id, or by a trip id published for one start.

    ``ride-1842`` stays the alias. A ``trip-...`` id focuses that start.
    """
    ride = db.get(Ride, key)
    if ride is not None:
        return ride, started_at
    trip = db.scalars(select(Trip).where(Trip.public_id == key)).first()
    if trip is None:
        return None, None
    parent = db.get(Ride, trip.ride_id)
    return parent, started_at if started_at is not None else trip.started_at


def sharing_on(db: Session, ride_id: str) -> bool:
    return any(
        db.scalars(select(Guardian).where(Guardian.ride_id == ride_id, Guardian.sharing.is_(True))).all()
    )


def touch_publisher(ride: Ride) -> None:
    ride.publisher_at = int(time.time() * 1000)
    ride.updated_at = utcnow()


def publisher_online(ride: Ride) -> bool:
    if not ride.publisher_at:
        return False
    age_ms = int(time.time() * 1000) - int(ride.publisher_at)
    return 0 <= age_ms <= settings.publisher_online_s * 1000
