from __future__ import annotations

import json
import time
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .mapping import is_auto_sos, kind_for, seats_from_cms, severity_for, title_for
from .models import Alert, Guardian, Notice, Ride, utcnow


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
    coord = body.get("coordinate") or {}
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
        lat=coord.get("latitude", ride.lat),
        lon=coord.get("longitude", ride.lon),
        origin=origin,
        evidence_id=str(body.get("id") or "") if body.get("snapshot_path") or body.get("audio_path") else "",
    )
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
    return ride


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
        "severity": row.severity,
        "title": row.title,
        "detail": row.detail,
        "at": row.at,
        "locationLabel": row.location_label,
        "coordinate": {"latitude": row.lat, "longitude": row.lon},
        "acknowledged": row.acknowledged,
        "evidenceId": row.evidence_id or None,
        "origin": row.origin,
    }


def ride_public(ride: Ride) -> dict:
    try:
        seats = json.loads(ride.cabin_json or "[]")
    except json.JSONDecodeError:
        seats = []
    return {
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
    }
