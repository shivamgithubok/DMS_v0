"""Phone notifications for SOS and high alerts.

An Expo token is sent through the Expo push service. Any other token is sent
through FCM when FCM_SERVER_KEY is set. A missing key skips that token.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .mapping import audience_for
from .models import Guardian, Trip

log = logging.getLogger("vms.push")

EXPO_URL = "https://exp.host/--/api/v2/push/send"
FCM_URL = "https://fcm.googleapis.com/fcm/send"


def should_notify(alert) -> bool:
    """Family pages are distress and SOS. Coaching alerts stay on the vehicle."""
    return audience_for(getattr(alert, "orin_type", ""), getattr(alert, "kind", "")) == "family"


def trip_name(db: Session, ride_id: str, alert) -> str:
    started = getattr(alert, "trip_started_at", None)
    if started:
        trip = db.scalars(
            select(Trip).where(Trip.ride_id == ride_id, Trip.started_at == int(started))
        ).first()
        if trip is not None and trip.public_id:
            return trip.public_id
    return ride_id


def tokens_for(db: Session, ride_id: str) -> list[str]:
    rows = db.scalars(
        select(Guardian).where(Guardian.ride_id == ride_id, Guardian.can_notify.is_(True))
    ).all()
    return [row.push_token.strip() for row in rows if (row.push_token or "").strip()]


def notify(db: Session, ride_id: str, alert) -> int:
    if not should_notify(alert):
        return 0
    targets = tokens_for(db, ride_id)
    if not targets:
        return 0
    body = {
        "title": alert.title or "Cabin alert",
        "detail": alert.detail or alert.title or "",
        "alertId": alert.id,
        "rideId": ride_id,
        "tripId": trip_name(db, ride_id, alert),
        "kind": alert.kind,
        "severity": alert.severity,
        "audience": "family",
    }
    threading.Thread(target=deliver, args=(targets, body), name="push", daemon=True).start()
    return len(targets)


def deliver(targets: list[str], body: dict) -> None:
    expo = [token for token in targets if token.startswith(("ExponentPushToken[", "ExpoPushToken["))]
    fcm = [token for token in targets if token not in expo]
    if expo:
        _post(EXPO_URL, [_expo(token, body) for token in expo], {})
    if fcm and settings.fcm_server_key:
        for token in fcm:
            _post(FCM_URL, _fcm(token, body), {"Authorization": f"key={settings.fcm_server_key}"})
    elif fcm:
        log.info("push skipped for %d FCM token(s); set FCM_SERVER_KEY", len(fcm))


def _expo(token: str, body: dict) -> dict:
    return {
        "to": token,
        "title": body["title"],
        "body": body["detail"],
        "sound": "default",
        "priority": "high",
        "channelId": "alerts",
        "data": {
            "rideId": body["rideId"],
            "tripId": body.get("tripId") or body["rideId"],
            "alertId": body["alertId"],
            "kind": body["kind"],
            "severity": body["severity"],
            "audience": body.get("audience") or "family",
        },
    }


def _fcm(token: str, body: dict) -> dict:
    return {
        "to": token,
        "priority": "high",
        "notification": {"title": body["title"], "body": body["detail"], "sound": "default"},
        "data": {
            "rideId": body["rideId"],
            "tripId": body.get("tripId") or body["rideId"],
            "alertId": body["alertId"],
            "kind": body["kind"],
            "severity": body["severity"],
            "audience": body.get("audience") or "family",
        },
    }


def _post(url: str, payload: dict | list, headers: dict) -> None:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        log.warning("push to %s failed: %s", url, exc)
