import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))

from app.models import Base, Ride  # noqa: E402
from app.store import alert_public, alert_row, analysis_public, apply_analysis  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402


def _db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = Session(engine)
    db.add(Ride(id="ride-1842", device_id="orin-nano-01", analysis_json="{}"))
    db.commit()
    return db


def test_analysis_sides_are_stored_and_fetched_independently():
    db = _db()
    ride = db.get(Ride, "ride-1842")
    apply_analysis(db, "dms/orin-nano-01/analysis", {
        "device_id": "orin-nano-01",
        "channel": "dms",
        "scene": "Driver facing forward.",
        "description": "Both hands on the wheel, looking ahead.",
        "hazards": [],
        "confidence": 0.82,
        "suppressed": ["driver distracted: detector owns driver state"],
        "observations": {"handsOnWheel": True, "gaze": "forward"},
        "at": 100,
    })
    apply_analysis(db, "cms/orin-nano-01/analysis", {
        "device_id": "orin-nano-01",
        "channel": "cms",
        "scene": "Child in rear-left.",
        "hazards": ["child_unattended"],
        "confidence": 0.84,
        "observations": {"childUnattended": True},
        "at": 200,
    })
    db.commit()
    ride = db.get(Ride, "ride-1842")
    both = analysis_public(ride)
    assert both["dms"]["scene"].startswith("Driver")
    assert both["dms"]["description"].startswith("Both hands")
    assert both["cms"]["hazards"] == ["child_unattended"]
    assert both["updatedAt"] == 200
    assert "cms" not in analysis_public(ride, "dms")
    assert analysis_public(ride, "cms")["cms"]["scene"].startswith("Child")


def test_alerts_keep_channel_and_orin_type():
    db = _db()
    ride = db.get(Ride, "ride-1842")
    dms = alert_row(db, ride, {
        "id": "a-dms", "device_id": "orin-nano-01", "type": "DISTRACTION",
        "channel": "dms", "confidence": 0.9, "description": "head turned",
    })
    cms = alert_row(db, ride, {
        "id": "a-cms", "device_id": "orin-nano-01", "type": "CHILD_PRESENT",
        "channel": "cms", "confidence": 0.81, "description": "vlm child",
    })
    db.commit()
    pub_d, pub_c = alert_public(dms), alert_public(cms)
    assert pub_d["channel"] == "dms" and pub_d["orinType"] == "DISTRACTION"
    assert pub_c["channel"] == "cms" and pub_c["kind"] == "seat_occupancy"
    assert json.dumps(pub_c["confidence"])


def test_http_analysis_and_alert_filters(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/vms.db")
    from app.config import settings
    settings.database_url = f"sqlite:///{tmp_path}/vms.db"
    from app import db as cloud_db
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    cloud_db.engine = engine
    cloud_db.SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    with cloud_db.SessionLocal() as db:
        db.add(Ride(id="ride-1842", device_id="orin-nano-01", analysis_json="{}", share_link="vms://trip/ride-1842"))
        db.commit()
        ride = db.get(Ride, "ride-1842")
        apply_analysis(db, "dms/orin-nano-01/analysis", {
            "device_id": "orin-nano-01", "channel": "dms", "scene": "Hands on wheel.", "hazards": [], "at": 1,
        })
        apply_analysis(db, "cms/orin-nano-01/analysis", {
            "device_id": "orin-nano-01", "channel": "cms", "scene": "Cabin calm.", "hazards": [], "at": 2,
        })
        alert_row(db, ride, {"id": "d1", "type": "DISTRACTION", "channel": "dms"})
        alert_row(db, ride, {"id": "c1", "type": "PASSENGER_AGGRESSIVE", "channel": "cms"})
        db.commit()

    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from app.db import get_db
    from app.main import app

    def _override():
        session = cloud_db.SessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _override
    with patch("app.mqtt_ingest.ingest.start"):
        with TestClient(app) as client:
            both = client.get("/v1/rides/ride-1842/analysis")
            assert both.status_code == 200
            assert both.json()["dms"]["scene"].startswith("Hands")
            assert both.json()["cms"]["scene"].startswith("Cabin")
            dms = client.get("/v1/rides/ride-1842/analysis", params={"channel": "dms"})
            assert "cms" not in dms.json() and dms.json()["dms"]["scene"]
            cms_alerts = client.get("/v1/rides/ride-1842/alerts", params={"channel": "cms"})
            assert [a["channel"] for a in cms_alerts.json()["alerts"]] == ["cms"]
            dms_alerts = client.get("/v1/rides/ride-1842/alerts", params={"channel": "dms"})
            assert [a["channel"] for a in dms_alerts.json()["alerts"]] == ["dms"]
            assert client.get("/v1/rides/missing/analysis").status_code == 404
    app.dependency_overrides.clear()
