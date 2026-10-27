from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db, init_db
from .hub import hub
from .models import Alert, Guardian, Notice, Ride
from .mqtt_ingest import ingest
from .store import alert_public, raise_sos, ride_public


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    init_db()
    hub.attach(asyncio.get_running_loop())
    ingest.start()
    yield


app = FastAPI(title="VMS guardian API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


class PushIn(BaseModel):
    token: str
    ride_id: str = ""


@app.get("/health")
def health():
    return {
        "ok": True,
        "mqtt": {"host": settings.mqtt_host, "connected": ingest.connected, "error": ingest.last_error},
        "ride": settings.default_ride_id,
    }


@app.get("/v1/rides/{ride_id}")
def get_ride(ride_id: str, db: Session = Depends(get_db)):
    ride = db.get(Ride, ride_id)
    if ride is None:
        raise HTTPException(404, "ride not found")
    return ride_public(ride)


@app.get("/v1/rides/{ride_id}/alerts")
def list_alerts(ride_id: str, status: str = "all", db: Session = Depends(get_db)):
    rows = db.scalars(select(Alert).where(Alert.ride_id == ride_id).order_by(Alert.at.desc())).all()
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
    db.commit()
    event = {"type": "sos", "alert": alert_public(row), "ride": ride_public(ride), "notified": names}
    hub.publish(ride.id, event)
    return {
        "id": row.id,
        "at": row.at,
        "topic": f"ride/{ride.id}/sos",
        "origin": "device",
        "note": f"Notified {', '.join(names) or 'nobody sharing'}",
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
    db.commit()
    hub.publish(ride.id, {"type": "telemetry", "ride": ride_public(ride)})
    return ride_public(ride)


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


@app.post("/v1/rides/{ride_id}/share")
def share(ride_id: str, db: Session = Depends(get_db)):
    ride = db.get(Ride, ride_id)
    if ride is None:
        raise HTTPException(404, "ride not found")
    ride.share_link = f"vms://trip/{ride_id}"
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
    return {"link": ride.share_link, "origin": "vehicle"}


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
    await hub.join(ride_id, ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        hub.leave(ride_id, ws)
