from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import auth, consent
from .config import settings
from .db import get_db, init_db
from .hub import hub
from .models import AccessAudit, Alert, Guardian, GuardianRecording, LiveFrame, MediaClip, Notice, Place, Ride, Trip
from .mqtt_ingest import ingest
from . import clips, guardian_media, livekit_tokens, preview, push, retention, talk, watch
from .media_contract import describe as describe_media
from .store import (
    alert_channel,
    alert_public,
    alerts_for_window,
    analysis_public,
    history_public,
    link_expired,
    publisher_online,
    raise_sos,
    record_point,
    resolve_ride,
    ride_for_device,
    ride_public,
    trip_public,
    trips_for_device,
    sharing_on,
    touch_publisher,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    init_db()
    hub.attach(asyncio.get_running_loop())
    ingest.start()
    retention.start_retention()
    yield


app = FastAPI(title="VMS guardian API", lifespan=lifespan)


def _phone_route(path: str) -> bool:
    if path in {"/health", "/v1/session"} or path.startswith(("/docs", "/redoc", "/openapi.json")):
        return False
    if path.startswith("/v1/devices/") and path.rstrip("/").endswith("/push-token"):
        return True
    if path.startswith("/v1/devices/"):
        return False
    return path.startswith("/v1/")


@app.middleware("http")
async def guardian_session(request: Request, call_next):
    if settings.auth_required and _phone_route(request.url.path):
        try:
            auth.check_phone_session(request.headers.get("authorization"))
        except HTTPException as exc:
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    return await call_next(request)


_cors_origins = settings.cors_origins if settings.auth_required else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SosIn(BaseModel):
    passenger: str = ""
    latitude: float | None = None
    longitude: float | None = None
    source: str = "phone"


class TelemetryIn(BaseModel):
    latitude: float
    longitude: float
    heading: float | None = None
    speed_kmh: float | None = Field(default=None, alias="speedKmh")
    eta_min: int | None = Field(default=None, alias="etaMin")

    model_config = {"populate_by_name": True}


class GuardianIn(BaseModel):
    name: str
    relationship: str = ""
    phone: str = ""
    sharing: bool = True


class CapabilitiesIn(BaseModel):
    notify: bool | None = None
    watch: bool | None = None
    talk: bool | None = None


class ConsentIn(BaseModel):
    key: str
    kind: str
    decision: str


class PlaceIn(BaseModel):
    name: str = "Home"
    lat: float | None = None
    lon: float | None = None
    latitude: float | None = None
    longitude: float | None = None
    radiusM: float = 80


class PushIn(BaseModel):
    token: str
    ride_id: str = ""


class SessionIn(BaseModel):
    key: str = ""
    guardianId: str = ""


class MediaTokenIn(BaseModel):
    key: str
    ride_id: str = ""


class VideoFrameIn(BaseModel):
    key: str
    track: str = "driver"
    mime: str = "image/jpeg"
    image_b64: str
    at: int | None = None


class VideoLiveIn(BaseModel):
    key: str
    image_b64: str
    audio_b64: str = ""
    samples: int = 0
    rate: int = 16000
    width: int = 640
    height: int = 360
    at: int | None = None


class ClipFrameIn(BaseModel):
    at: int
    image_b64: str


class RecordingIn(BaseModel):
    action: str = "start"
    guardianId: str = ""


class SnapshotIn(BaseModel):
    guardianId: str = ""
    channel: str = "dms"


class RecordingUploadIn(BaseModel):
    key: str
    video_b64: str


class ClipIn(BaseModel):
    key: str
    channel: str = "dms"
    alert_id: str = ""
    frames: list[ClipFrameIn] = []
    video_b64: str = ""
    start_at: int | None = None
    end_at: int | None = None


@app.get("/health")
def health():
    return {"ok": True, "mqtt": {"connected": bool(ingest.connected)}}


@app.post("/v1/session")
def create_session(body: SessionIn):
    """Mint a phone token. Open routes ignore it until GUARDIAN_AUTH_REQUIRED is on."""
    if not settings.auth_required:
        guardian_id = body.guardianId.strip() or "guardian"
        if not settings.session_secret:
            return {"token": "", "expiresAt": 0, "guardianId": guardian_id, "authRequired": False}
        token, expires_at = auth.issue_session(guardian_id)
        return {"token": token, "expiresAt": expires_at, "guardianId": guardian_id, "authRequired": False}
    if not auth.session_configured():
        raise HTTPException(503, "guardian auth is not configured")
    if body.key != settings.guardian_api_key:
        raise HTTPException(401, "sign in required")
    guardian_id = body.guardianId.strip()
    if not guardian_id:
        raise HTTPException(400, "guardianId is required")
    token, expires_at = auth.issue_session(guardian_id)
    return {"token": token, "expiresAt": expires_at, "guardianId": guardian_id, "authRequired": True}


@app.get("/v1/history")
def history(deviceId: str = "", db: Session = Depends(get_db)):
    trips = trips_for_device(db, deviceId)
    if trips:
        items = []
        for trip in trips:
            ride = db.get(Ride, trip.ride_id) or ride_for_device(db, trip.device_id)
            items.append(trip_public(ride, trip, db))
        return {"rides": items}
    query = select(Ride)
    if deviceId:
        query = query.where(Ride.device_id == deviceId)
    rows = db.scalars(query).all()
    rows.sort(key=lambda ride: ride.started_at or 0, reverse=True)
    return {"rides": [history_public(db, ride) for ride in rows]}


@app.get("/v1/rides/{ride_id}")
def get_ride(ride_id: str, startedAt: int | None = None, db: Session = Depends(get_db)):
    ride, focus = resolve_ride(db, ride_id, startedAt)
    if ride is None:
        raise HTTPException(404, "ride not found")
    return ride_public(ride, db, focus)


@app.get("/v1/rides/{ride_id}/analysis")
def get_analysis(ride_id: str, channel: str | None = None, db: Session = Depends(get_db)):
    ride, _focus = resolve_ride(db, ride_id)
    if ride is None:
        raise HTTPException(404, "ride not found")
    wanted = (channel or "").strip().lower() or None
    if wanted and wanted not in {"dms", "cms"}:
        raise HTTPException(400, "channel must be dms or cms")
    return analysis_public(ride, wanted)


@app.get("/v1/rides/{ride_id}/alerts")
def list_alerts(
    ride_id: str,
    status: str = "all",
    channel: str | None = None,
    startedAt: int | None = None,
    db: Session = Depends(get_db),
):
    ride, focus = resolve_ride(db, ride_id, startedAt)
    if ride is None:
        raise HTTPException(404, "ride not found")
    ride_id = ride.id
    if focus is not None:
        trip = db.scalars(select(Trip).where(Trip.ride_id == ride_id, Trip.started_at == int(focus))).first()
        rows = [] if trip is None else alerts_for_window(db, ride_id, trip.started_at, trip.ended_at)
    else:
        rows = db.scalars(select(Alert).where(Alert.ride_id == ride_id).order_by(Alert.at.desc())).all()
    wanted = (channel or "").strip().lower()
    if wanted in {"dms", "cms"}:
        rows = [row for row in rows if alert_channel(row) == wanted]
    items = [alert_public(row) for row in rows]
    if status == "open":
        items = [a for a in items if not a["acknowledged"]]
    elif status == "emergency":
        items = [a for a in items if a["severity"] in {"high", "critical"}]
    elif status == "caution":
        items = [a for a in items if a["severity"] in {"info", "low", "medium"}]
    return {"alerts": items}


@app.post("/v1/alerts/{alert_id}/ack")
def ack_alert(alert_id: str, db: Session = Depends(get_db)):
    row = db.scalars(select(Alert).where(Alert.id == alert_id)).first()
    if row is None:
        raise HTTPException(404, "alert not found")
    row.acknowledged = True
    db.commit()
    return alert_public(row)


@app.post("/v1/rides/{ride_id}/sos")
def sos(ride_id: str, body: SosIn, db: Session = Depends(get_db)):
    ride, row, names = raise_sos(db, ride_id, body.passenger, body.latitude, body.longitude)
    pushed = push.notify(db, ride.id, row)
    db.commit()
    event = {"type": "sos", "alert": alert_public(row), "ride": ride_public(ride), "notified": names}
    hub.publish(ride.id, event)
    return {
        "id": row.id,
        "at": row.at,
        "topic": f"ride/{ride.id}/sos",
        "origin": "device",
        "note": f"Notified {', '.join(names) or 'nobody sharing'}",
        "pushed": pushed,
        "alert": alert_public(row),
    }


@app.post("/v1/rides/{ride_id}/telemetry")
def telemetry(ride_id: str, body: TelemetryIn, db: Session = Depends(get_db)):
    ride = db.get(Ride, ride_id)
    if ride is None:
        raise HTTPException(404, "ride not found")
    ride.lat = body.latitude
    ride.lon = body.longitude
    if body.heading is not None:
        ride.heading = body.heading
    if body.speed_kmh is not None:
        ride.speed_kmh = body.speed_kmh
    if body.eta_min is not None:
        ride.eta_min = body.eta_min
    record_point(db, ride, body.latitude, body.longitude, int(time.time() * 1000), "live")
    db.commit()
    public = ride_public(ride, db)
    hub.publish(ride.id, {"type": "telemetry", "ride": public})
    return public


@app.get("/v1/rides/{ride_id}/cabin")
def cabin(ride_id: str, db: Session = Depends(get_db)):
    ride = db.get(Ride, ride_id)
    if ride is None:
        raise HTTPException(404, "ride not found")
    public = ride_public(ride)
    return {"driverState": public["driverState"], "seats": public["seats"], "safetyStatus": public["safetyStatus"]}


@app.get("/v1/rides/{ride_id}/guardians")
def list_guardians(ride_id: str, db: Session = Depends(get_db)):
    rows = db.scalars(select(Guardian).where(Guardian.ride_id == ride_id)).all()
    return {
        "guardians": [
            {
                "id": g.id,
                "name": g.name,
                "relationship": g.relationship,
                "phone": g.phone,
                "sharing": g.sharing,
                "notify": bool(g.can_notify),
                "watch": bool(g.can_watch),
                "talk": bool(g.can_talk),
                "lastNotifiedAt": g.last_notified_at,
            }
            for g in rows
        ]
    }


@app.post("/v1/rides/{ride_id}/guardians")
def add_guardian(ride_id: str, body: GuardianIn, db: Session = Depends(get_db)):
    if db.get(Ride, ride_id) is None:
        raise HTTPException(404, "ride not found")
    row = Guardian(
        id=f"g-{uuid.uuid4().hex[:8]}",
        ride_id=ride_id,
        name=body.name.strip(),
        relationship=body.relationship.strip(),
        phone=body.phone.strip(),
        sharing=body.sharing,
        can_notify=body.sharing,
        can_watch=False,
        can_talk=False,
    )
    db.add(row)
    db.commit()
    return {"id": row.id}


@app.post("/v1/rides/{ride_id}/guardians/{guardian_id}/sharing")
def toggle_sharing(ride_id: str, guardian_id: str, db: Session = Depends(get_db)):
    row = db.get(Guardian, guardian_id)
    if row is None or row.ride_id != ride_id:
        raise HTTPException(404, "guardian not found")
    row.sharing = not row.sharing
    db.commit()
    return {"id": row.id, "sharing": row.sharing}


@app.post("/v1/rides/{ride_id}/guardians/{guardian_id}/capabilities")
def set_capabilities(ride_id: str, guardian_id: str, body: CapabilitiesIn, db: Session = Depends(get_db)):
    row = db.get(Guardian, guardian_id)
    if row is None or row.ride_id != ride_id:
        raise HTTPException(404, "guardian not found")
    if body.notify is not None:
        row.can_notify = body.notify
    if body.watch is not None:
        row.can_watch = body.watch
    if body.talk is not None:
        row.can_talk = body.talk
    db.commit()
    return {"id": row.id, "notify": row.can_notify, "watch": row.can_watch, "talk": row.can_talk}


def _active_trip(db: Session, ride_id: str) -> Trip | None:
    return db.scalars(
        select(Trip).where(Trip.ride_id == ride_id, Trip.status == "active").order_by(Trip.started_at.desc())
    ).first()


def _ride_for_link(db: Session, key: str) -> Ride:
    """A trip id is the shared link. It stops working after the trip ends and the grace passes."""
    trip = db.scalars(select(Trip).where(Trip.public_id == key)).first()
    if trip is not None:
        if link_expired(trip):
            raise HTTPException(410, "this trip link has ended")
        ride = db.get(Ride, trip.ride_id)
        if ride is None:
            raise HTTPException(404, "ride not found")
        return ride
    ride = db.get(Ride, key)
    if ride is None:
        raise HTTPException(404, "ride not found")
    return ride


def _place_public(row: Place) -> dict:
    return {"id": row.id, "name": row.name, "lat": row.lat, "lon": row.lon, "radiusM": row.radius_m}


@app.get("/v1/rides/{ride_id}/places")
def list_places(ride_id: str, db: Session = Depends(get_db)):
    if db.get(Ride, ride_id) is None:
        raise HTTPException(404, "ride not found")
    rows = db.scalars(select(Place).where(Place.ride_id == ride_id)).all()
    return {"places": [_place_public(row) for row in rows]}


@app.post("/v1/rides/{ride_id}/places")
def add_place(ride_id: str, body: PlaceIn, db: Session = Depends(get_db)):
    if db.get(Ride, ride_id) is None:
        raise HTTPException(404, "ride not found")
    lat = body.lat if body.lat is not None else body.latitude
    lon = body.lon if body.lon is not None else body.longitude
    if lat is None or lon is None:
        raise HTTPException(400, "lat and lon are required")
    row = Place(
        id=f"place-{uuid.uuid4().hex[:8]}",
        ride_id=ride_id,
        name=(body.name or "Home").strip() or "Home",
        lat=float(lat),
        lon=float(lon),
        radius_m=float(body.radiusM or 80),
    )
    db.add(row)
    db.commit()
    return _place_public(row)


@app.get("/v1/devices/{device_id}/places")
def device_places(device_id: str, key: str = "", db: Session = Depends(get_db)):
    _check_device_key(key)
    ride = ride_for_device(db, device_id)
    rows = db.scalars(select(Place).where(Place.ride_id == ride.id)).all()
    return {"places": [_place_public(row) for row in rows]}


@app.post("/v1/rides/{ride_id}/share")
def share(ride_id: str, db: Session = Depends(get_db)):
    ride = db.get(Ride, ride_id)
    if ride is None:
        raise HTTPException(404, "ride not found")
    trip = _active_trip(db, ride_id)
    if trip is None:
        trip = db.scalars(select(Trip).where(Trip.ride_id == ride_id).order_by(Trip.started_at.desc())).first()
    if trip is not None and trip.public_id:
        ride.share_link = f"vms://trip/{trip.public_id}"
        expires = trip.link_expires_at
    else:
        ride.share_link = f"vms://trip/{ride_id}"
        expires = None
    names = [g.name for g in db.scalars(select(Guardian).where(Guardian.ride_id == ride_id, Guardian.sharing.is_(True)))]
    db.add(
        Notice(
            id=f"notice-{uuid.uuid4().hex[:10]}",
            ride_id=ride_id,
            at=int(time.time() * 1000),
            title="Trip link refreshed",
            body=ride.share_link,
            guardian_names=",".join(names),
            origin="device",
        )
    )
    db.commit()
    return {"link": ride.share_link, "expiresAt": expires, "origin": "vehicle"}


@app.get("/v1/rides/{ride_id}/notices")
def notices(ride_id: str, db: Session = Depends(get_db)):
    rows = db.scalars(select(Notice).where(Notice.ride_id == ride_id).order_by(Notice.at.desc())).all()
    return {
        "notices": [
            {
                "id": n.id,
                "at": n.at,
                "title": n.title,
                "body": n.body,
                "guardianNames": [p for p in n.guardian_names.split(",") if p],
                "origin": n.origin,
            }
            for n in rows
        ]
    }


def _check_device_key(key: str) -> None:
    auth.check_device_key(key)


def _video_tracks(track: str | None) -> list[str]:
    wanted = (track or "all").strip().lower()
    if wanted in {"driver", "cabin"}:
        return [wanted]
    return ["driver", "cabin"]


def _stream_on(db: Session, ride_id: str, mode: str) -> bool:
    return mode != "dummy" and consent.live(ride_id, "watch") is not None and sharing_on(db, ride_id)


def _may_ask(db: Session, ride_id: str, guardian_id: str, kind: str) -> Guardian:
    if not sharing_on(db, ride_id):
        raise HTTPException(403, "sharing is off")
    row = db.get(Guardian, guardian_id)
    allowed = row is not None and row.ride_id == ride_id and row.sharing and (
        row.can_watch if kind == "watch" else row.can_talk
    )
    if not allowed:
        raise HTTPException(403, f"{kind} is not allowed")
    return row


def media_mode(url: str) -> str:
    text = str(url or "")
    if text.startswith("dummy"):
        return "dummy"
    if text.startswith("local"):
        return "local"
    return "livekit"


@app.get("/v1/rides/{ride_id}/video")
def get_video(ride_id: str, track: str = "all", guardianId: str = "", db: Session = Depends(get_db)):
    ride = _ride_for_link(db, ride_id)
    ride_id = ride.id
    if not guardianId:
        raise HTTPException(400, "guardianId is required")
    guardian = _may_ask(db, ride_id, guardianId, "watch")
    opened = consent.live(ride.id, "watch")
    if opened and opened["guardianId"] != guardian.id:
        raise HTTPException(403, "cabin is already live")
    if opened is None:
        current = consent.pending(ride.id, "watch")
        if current is None or current["guardianId"] != guardian.id:
            consent.ask(ride.id, "watch", guardian.id, guardian.name)
            consent.audit(db, ride.id, guardian.id, "watch", "ask", guardian.id)
            db.commit()
        raise HTTPException(403, "waiting for the cabin")
    token, expires_at = livekit_tokens.viewer_token(ride.id, uuid.uuid4().hex[:10])
    mode = media_mode(settings.livekit_url)
    watch.note(ride.id)
    publish = _stream_on(db, ride.id, mode)
    media = describe_media(mode, publish)
    tracks = list(media["tracks"])
    if media["stream"] == "jpeg":
        wanted = set(_video_tracks(track))
        tracks = [name for name in tracks if name in wanted] or tracks
    return {
        "url": settings.livekit_url,
        "mode": mode,
        "stream": media["stream"],
        "token": token,
        "room": livekit_tokens.room_name(ride.id),
        "tracks": tracks,
        "video": media["video"],
        "audio": media["audio"],
        "expiresAt": expires_at,
        "publisherOnline": publisher_online(ride),
        "publish": publish,
        "publishTo": media["publishTo"],
        "preview": f"/v1/rides/{ride.id}/video/preview",
    }


@app.post("/v1/rides/{ride_id}/video/cancel")
def cancel_video(ride_id: str, db: Session = Depends(get_db)):
    """Guardian stopped watching. The car drops the cabin stream on its next token poll."""
    ride = db.get(Ride, ride_id)
    if ride is None:
        raise HTTPException(404, "ride not found")
    watch.clear(ride_id)
    consent.clear(ride_id, "watch")
    return {"ok": True, "publish": False, "room": livekit_tokens.room_name(ride.id)}


@app.post("/v1/rides/{ride_id}/video/talk")
def start_talk(ride_id: str, guardianId: str = "", db: Session = Depends(get_db)):
    """Phone publishes one microphone track. The car plays it on the cabin speaker."""
    ride = _ride_for_link(db, ride_id)
    ride_id = ride.id
    mode = media_mode(settings.livekit_url)
    if mode != "livekit":
        raise HTTPException(409, "talk needs a LiveKit room")
    if not guardianId:
        raise HTTPException(400, "guardianId is required")
    guardian = _may_ask(db, ride_id, guardianId, "talk")
    opened = consent.live(ride.id, "talk")
    if opened and opened["guardianId"] != guardian.id:
        raise HTTPException(403, "cabin is already live")
    if opened is None:
        current = consent.pending(ride.id, "talk")
        if current is None or current["guardianId"] != guardian.id:
            consent.ask(ride.id, "talk", guardian.id, guardian.name)
            consent.audit(db, ride.id, guardian.id, "talk", "ask", guardian.id)
            db.commit()
        raise HTTPException(403, "waiting for the cabin")
    talk.note(ride.id)
    token, expires_at = livekit_tokens.talker_token(ride.id)
    return {
        "url": settings.livekit_url,
        "token": token,
        "room": livekit_tokens.room_name(ride.id),
        "expiresAt": expires_at,
        "tracks": ["guardian-mic"],
    }


@app.post("/v1/rides/{ride_id}/video/talk/cancel")
def cancel_talk(ride_id: str, db: Session = Depends(get_db)):
    ride = db.get(Ride, ride_id)
    if ride is None:
        raise HTTPException(404, "ride not found")
    talk.clear(ride_id)
    consent.clear(ride_id, "talk")
    return {"ok": True, "talk": False, "room": livekit_tokens.room_name(ride.id)}


@app.post("/v1/rides/{ride_id}/recordings")
def start_guardian_recording(ride_id: str, body: RecordingIn, db: Session = Depends(get_db)):
    ride = db.get(Ride, ride_id)
    if ride is None:
        raise HTTPException(404, "ride not found")
    if not sharing_on(db, ride_id):
        raise HTTPException(403, "sharing is off")
    if not body.guardianId:
        raise HTTPException(400, "guardianId is required")
    try:
        row = guardian_media.start_recording(db, ride, body.guardianId)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    db.commit()
    return guardian_media.recording_public(row)


@app.post("/v1/rides/{ride_id}/recordings/{rec_id}/cancel")
def cancel_guardian_recording(ride_id: str, rec_id: str, db: Session = Depends(get_db)):
    ride = db.get(Ride, ride_id)
    if ride is None:
        raise HTTPException(404, "ride not found")
    try:
        row = guardian_media.cancel_recording(db, ride_id, rec_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    db.commit()
    return guardian_media.recording_public(row)


@app.get("/v1/rides/{ride_id}/recordings")
def get_guardian_recordings(ride_id: str, guardianId: str = "", db: Session = Depends(get_db)):
    ride = db.get(Ride, ride_id)
    if ride is None:
        raise HTTPException(404, "ride not found")
    if not sharing_on(db, ride_id):
        raise HTTPException(403, "sharing is off")
    rows = guardian_media.list_recordings(db, ride_id, guardianId)
    return {"recordings": [guardian_media.recording_public(row) for row in rows]}


@app.post("/v1/rides/{ride_id}/snapshots")
def take_guardian_snapshot(ride_id: str, body: SnapshotIn, db: Session = Depends(get_db)):
    ride = db.get(Ride, ride_id)
    if ride is None:
        raise HTTPException(404, "ride not found")
    if not sharing_on(db, ride_id):
        raise HTTPException(403, "sharing is off")
    if not body.guardianId:
        raise HTTPException(400, "guardianId is required")
    channel = clips.channel_name(body.channel)
    if not channel:
        raise HTTPException(400, "channel must be dms or cms")
    try:
        row = guardian_media.take_snapshot(db, ride, body.guardianId, channel)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    db.commit()
    return guardian_media.snapshot_public(row)


@app.get("/v1/rides/{ride_id}/snapshots")
def get_guardian_snapshots(ride_id: str, guardianId: str = "", db: Session = Depends(get_db)):
    ride = db.get(Ride, ride_id)
    if ride is None:
        raise HTTPException(404, "ride not found")
    if not sharing_on(db, ride_id):
        raise HTTPException(403, "sharing is off")
    rows = guardian_media.list_snapshots(db, ride_id, guardianId)
    return {"snapshots": [guardian_media.snapshot_public(row) for row in rows]}


@app.get("/v1/rides/{ride_id}/video/preview")
def video_preview(ride_id: str, track: str = "driver", db: Session = Depends(get_db)):
    ride = db.get(Ride, ride_id)
    if ride is None:
        raise HTTPException(404, "ride not found")
    if consent.live(ride_id, "watch") is None or not sharing_on(db, ride_id):
        raise HTTPException(403, "waiting for the cabin")
    wanted = (track or "driver").strip().lower()
    if wanted not in {"driver", "cabin"}:
        raise HTTPException(400, "track must be driver or cabin")
    row = preview.get(ride_id, wanted)
    if row is None:
        raise HTTPException(404, "no frame yet")
    return Response(content=row["image"], media_type=row["mime"])


@app.post("/v1/devices/{device_id}/media-token")
def media_token(device_id: str, body: MediaTokenIn, db: Session = Depends(get_db)):
    _check_device_key(body.key)
    ride = ride_for_device(db, device_id)
    if ride.device_id and ride.device_id != device_id:
        ride = db.scalars(select(Ride).where(Ride.device_id == device_id)).first() or ride
    touch_publisher(ride)
    db.commit()
    watching = consent.live(ride.id, "watch") is not None and sharing_on(db, ride.id)
    token, expires_at = livekit_tokens.publisher_token(ride.id, device_id, can_publish=watching)
    mode = media_mode(settings.livekit_url)
    publish = _stream_on(db, ride.id, mode)
    media = describe_media(mode, publish)
    talking = talk.wanted(ride.id) and consent.live(ride.id, "talk") is not None and sharing_on(db, ride.id) and mode == "livekit"
    speaker_token = ""
    if talking:
        speaker_token, _expires = livekit_tokens.speaker_token(ride.id, device_id)
    return {
        "url": settings.livekit_url,
        "mode": mode,
        "stream": media["stream"],
        "token": token,
        "room": livekit_tokens.room_name(ride.id),
        "identity": f"camera-{device_id}",
        "tracks": media["tracks"],
        "video": media["video"],
        "audio": media["audio"],
        "expiresAt": expires_at,
        "publish": publish,
        "publishTo": media["publishTo"],
        "talk": talking,
        "speakerToken": speaker_token,
        "watchRequest": _request_public(consent.pending(ride.id, "watch")),
        "talkRequest": _request_public(consent.pending(ride.id, "talk")),
        **guardian_media.recording_command(db, ride.id),
    }


def _request_public(row: dict | None) -> dict | None:
    if not row:
        return None
    return {"guardianId": row["guardianId"], "name": row["name"], "kind": row["kind"]}


@app.post("/v1/devices/{device_id}/consent")
def cabin_consent(device_id: str, body: ConsentIn, db: Session = Depends(get_db)):
    _check_device_key(body.key)
    kind = (body.kind or "").strip().lower()
    decision = (body.decision or "").strip().lower()
    if kind not in {"watch", "talk"} or decision not in {"accept", "deny"}:
        raise HTTPException(400, "kind must be watch or talk and decision must be accept or deny")
    ride = ride_for_device(db, device_id)
    if decision == "accept":
        row = consent.accept(ride.id, kind)
    else:
        row = consent.deny(ride.id, kind)
    if row is None:
        raise HTTPException(409, "nothing is waiting")
    consent.audit(db, ride.id, row["guardianId"], kind, decision, "cabin")
    if kind == "watch" and decision == "accept":
        watch.note(ride.id)
    if kind == "talk" and decision == "accept":
        talk.note(ride.id)
    if decision == "deny":
        if kind == "watch":
            watch.clear(ride.id)
        else:
            talk.clear(ride.id)
    db.commit()
    accepted = db.scalars(
        select(AccessAudit).where(AccessAudit.ride_id == ride.id, AccessAudit.action == "accept")
    ).all()
    return {"ok": True, "kind": kind, "decision": decision, "accepted": len(accepted)}


@app.post("/v1/devices/{device_id}/video/frame")
def video_frame(device_id: str, body: VideoFrameIn, db: Session = Depends(get_db)):
    import base64

    _check_device_key(body.key)
    track = (body.track or "driver").strip().lower()
    if track not in {"driver", "cabin"}:
        raise HTTPException(400, "track must be driver or cabin")
    try:
        raw = base64.b64decode(body.image_b64, validate=False)
    except Exception as exc:
        raise HTTPException(400, "invalid image") from exc
    if len(raw) < 32 or len(raw) > 400_000:
        raise HTTPException(400, "image size not allowed")
    ride = ride_for_device(db, device_id)
    touch_publisher(ride)
    channel = clips.channel_name(track)
    live = clips.save_live(db, ride, channel, raw, body.at)
    db.commit()
    preview.put(ride.id, track, raw, body.mime or "image/jpeg")
    return {"ok": True, "track": track, "channel": channel, "at": live.at, "bytes": len(raw), "rideId": ride.id}


@app.post("/v1/devices/{device_id}/video/live")
def video_live(device_id: str, body: VideoLiveIn, db: Session = Depends(get_db)):
    """Local-mode fallback only: one JPEG plus a PCM chunk. Live watch uses LiveKit."""
    import base64

    _check_device_key(body.key)
    try:
        raw = base64.b64decode(body.image_b64, validate=False)
        audio = base64.b64decode(body.audio_b64, validate=False) if body.audio_b64 else b""
    except Exception as exc:
        raise HTTPException(400, "invalid payload") from exc
    if len(raw) < 32 or len(raw) > 800_000:
        raise HTTPException(400, "image size not allowed")
    if len(audio) > 200_000:
        raise HTTPException(400, "audio size not allowed")
    ride = ride_for_device(db, device_id)
    touch_publisher(ride)
    db.commit()
    preview.put_stream(ride.id, raw, audio, body.samples, body.rate, body.width, body.height)
    return {"ok": True, "rideId": ride.id, "width": body.width, "height": body.height, "samples": body.samples}


@app.get("/v1/rides/{ride_id}/video/live")
def video_live_state(ride_id: str, db: Session = Depends(get_db)):
    import base64

    ride = db.get(Ride, ride_id)
    if ride is None:
        raise HTTPException(404, "ride not found")
    if consent.live(ride_id, "watch") is None or not sharing_on(db, ride_id):
        raise HTTPException(403, "waiting for the cabin")
    row = preview.stream(ride_id)
    if row is None:
        raise HTTPException(404, "no live stream yet")
    return {
        "at": row["at"],
        "seq": row["seq"],
        "width": row["width"],
        "height": row["height"],
        "samples": row["samples"],
        "rate": row["rate"],
        "image_b64": base64.b64encode(row["image"]).decode(),
        "audio_b64": base64.b64encode(row["audio"]).decode(),
    }


@app.get("/v1/rides/{ride_id}/frames/live")
def live_frame(ride_id: str, channel: str = "dms", db: Session = Depends(get_db)):
    ride = db.get(Ride, ride_id)
    if ride is None:
        raise HTTPException(404, "ride not found")
    if not sharing_on(db, ride_id):
        raise HTTPException(403, "sharing is off")
    name = clips.channel_name(channel)
    if not name:
        raise HTTPException(400, "channel must be dms or cms")
    row = db.scalars(select(LiveFrame).where(LiveFrame.ride_id == ride_id, LiveFrame.channel == name)).first()
    if row is None:
        raise HTTPException(404, "no frame yet")
    return clips.live_public(row)


@app.get("/v1/rides/{ride_id}/frames/live/image")
def live_frame_image(ride_id: str, channel: str = "dms", db: Session = Depends(get_db)):
    ride = db.get(Ride, ride_id)
    if ride is None:
        raise HTTPException(404, "ride not found")
    if not sharing_on(db, ride_id):
        raise HTTPException(403, "sharing is off")
    name = clips.channel_name(channel)
    if not name:
        raise HTTPException(400, "channel must be dms or cms")
    row = db.scalars(select(LiveFrame).where(LiveFrame.ride_id == ride_id, LiveFrame.channel == name)).first()
    if row is None:
        raise HTTPException(404, "no frame yet")
    return Response(content=clips.read_live(row), media_type="image/jpeg")


@app.post("/v1/devices/{device_id}/clips")
def upload_clip(device_id: str, body: ClipIn, db: Session = Depends(get_db)):
    import base64

    _check_device_key(body.key)
    channel = clips.channel_name(body.channel)
    if not channel:
        raise HTTPException(400, "channel must be dms or cms")
    if len(body.frames) > 24:
        raise HTTPException(400, "send at most 24 frames")
    decoded = []
    for frame in body.frames:
        try:
            raw = base64.b64decode(frame.image_b64, validate=False)
        except Exception as exc:
            raise HTTPException(400, "invalid image") from exc
        if len(raw) < 32 or len(raw) > 400_000:
            raise HTTPException(400, "image size not allowed")
        decoded.append((int(frame.at), raw))
    video = b""
    if body.video_b64:
        try:
            video = base64.b64decode(body.video_b64, validate=False)
        except Exception as exc:
            raise HTTPException(400, "invalid video") from exc
        if len(video) < 32 or len(video) > 8_000_000:
            raise HTTPException(400, "video size not allowed")
    if not decoded and not video:
        raise HTTPException(400, "send a video or 1 to 24 frames")
    ride = ride_for_device(db, device_id)
    try:
        row = clips.save_clip(db, ride, channel, decoded, body.alert_id, video, body.start_at, body.end_at)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    db.commit()
    return clips.clip_public(row)


@app.post("/v1/devices/{device_id}/recordings/{rec_id}")
def upload_guardian_recording(device_id: str, rec_id: str, body: RecordingUploadIn, db: Session = Depends(get_db)):
    import base64

    _check_device_key(body.key)
    try:
        video = base64.b64decode(body.video_b64, validate=False)
    except Exception as exc:
        raise HTTPException(400, "invalid video") from exc
    if len(video) < 32 or len(video) > 32_000_000:
        raise HTTPException(400, "video size not allowed")
    ride = ride_for_device(db, device_id)
    try:
        row = guardian_media.save_recording_video(db, ride, rec_id, video)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    db.commit()
    return guardian_media.recording_public(row)


@app.get("/v1/rides/{ride_id}/clips/{clip_id}")
def get_clip(ride_id: str, clip_id: str, db: Session = Depends(get_db)):
    ride = db.get(Ride, ride_id)
    if ride is None:
        raise HTTPException(404, "ride not found")
    if not sharing_on(db, ride_id):
        raise HTTPException(403, "sharing is off")
    row = db.get(MediaClip, clip_id)
    if row is None or row.ride_id != ride_id or clips.clip_expired(row):
        raise HTTPException(404, "clip not found")
    return clips.clip_public(row)


@app.get("/v1/media/{clip_id}/{name}")
def media_file(clip_id: str, name: str, db: Session = Depends(get_db)):
    row = db.get(MediaClip, clip_id)
    if row is None or clips.clip_expired(row):
        raise HTTPException(404, "clip not found")
    if not sharing_on(db, row.ride_id):
        raise HTTPException(403, "sharing is off")
    path = clips.clip_file(row, name)
    if path is None:
        raise HTTPException(404, "file not found")
    media_type = "video/mp4" if path.suffix.lower() == ".mp4" else "image/jpeg"
    return Response(content=path.read_bytes(), media_type=media_type)


@app.get("/v1/media/recordings/{rec_id}/clip.mp4")
def guardian_recording_file(rec_id: str, db: Session = Depends(get_db)):
    path = guardian_media.recording_file(db, rec_id)
    if path is None:
        raise HTTPException(404, "recording not found")
    row = db.get(GuardianRecording, rec_id)
    if row is None or not sharing_on(db, row.ride_id):
        raise HTTPException(403, "sharing is off")
    return Response(content=path.read_bytes(), media_type="video/mp4")


@app.get("/v1/media/snapshots/{snap_id}/snap.jpg")
def guardian_snapshot_file(snap_id: str, db: Session = Depends(get_db)):
    found = guardian_media.snapshot_file(db, snap_id)
    if found is None:
        raise HTTPException(404, "snapshot not found")
    row, path = found
    if not sharing_on(db, row.ride_id):
        raise HTTPException(403, "sharing is off")
    return Response(content=path.read_bytes(), media_type="image/jpeg")


@app.post("/v1/devices/{device_id}/push-token")
def push_token(device_id: str, body: PushIn, db: Session = Depends(get_db)):
    row = db.get(Guardian, device_id)
    if row is None:
        raise HTTPException(404, "store the token against a guardian id for now")
    row.push_token = body.token
    db.commit()
    return {"ok": True}


@app.websocket("/v1/rides/{ride_id}/live")
async def live(ride_id: str, ws: WebSocket):
    if settings.auth_required:
        token = ws.query_params.get("token") or auth.bearer_token(ws.headers.get("authorization"))
        try:
            if not token or not settings.session_secret:
                raise auth.jwt.InvalidTokenError("missing token")
            auth.guardian_id_from_token(token)
        except auth.jwt.PyJWTError:
            await ws.close(code=1008)
            return
    await hub.join(ride_id, ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        hub.leave(ride_id, ws)
