"""Orin MQTT alert types → VMS SafetyAlert fields."""

from __future__ import annotations

ORIN_TO_KIND = {
    "DISTRACTION": "distraction",
    "PHONE_USAGE": "distraction",
    "DROWSINESS": "drowsiness",
    "SOS": "sos",
    "SCREAM": "sos",
    "ALTERCATION": "sos",
    "OCCUPANCY_CHANGE": "seat_occupancy",
    "CHILD_PRESENT": "seat_occupancy",
    "OBJECT_ON_SEAT": "suspicious_activity",
    "PASSENGER_AGGRESSIVE": "suspicious_activity",
    "VLM_ALERT": "suspicious_activity",
}

TITLES = {
    "distraction": "Driver distraction",
    "drowsiness": "Driver drowsiness",
    "sos": "SOS raised",
    "seat_occupancy": "Seat occupancy",
    "suspicious_activity": "Suspicious activity",
    "harsh_braking": "Harsh braking",
    "route_deviation": "Route deviation",
}

SEAT_MAP = {
    "front_passenger": "front",
    "rear_left": "rear-left",
    "rear_right": "rear-right",
    "driver": "driver",
}


def kind_for(orin_type: str) -> str:
    return ORIN_TO_KIND.get(str(orin_type or "").upper(), "suspicious_activity")


def title_for(kind: str, fallback: str = "") -> str:
    return TITLES.get(kind) or fallback or kind.replace("_", " ").title()


def severity_for(kind: str, orin_severity: str | None = None) -> str:
    if kind == "sos":
        return "critical"
    if kind == "seat_occupancy":
        return "info"
    mapped = (orin_severity or "").lower()
    if mapped in {"info", "low", "medium", "high", "critical"}:
        return mapped
    if kind == "suspicious_activity":
        return "high"
    return "medium"


def is_auto_sos(orin_type: str) -> bool:
    return str(orin_type or "").upper() in {"SCREAM", "ALTERCATION"}


def seats_from_cms(rows: list | None) -> list[dict]:
    out = []
    for row in rows or []:
        name = str(row.get("name") or "")
        mapped = SEAT_MAP.get(name)
        if not mapped:
            continue
        occupant = str(row.get("occupant") or "empty")
        out.append({"id": mapped, "label": mapped.replace("-", " ").title(), "occupied": occupant != "empty"})
    if not any(s["id"] == "driver" for s in out):
        out.insert(0, {"id": "driver", "label": "Driver", "occupied": True})
    return out
