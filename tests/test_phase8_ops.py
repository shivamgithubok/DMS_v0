"""Phase 8: reboot uses the cached engine, old clips expire, a backup restores a trip."""

import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
sys.path.insert(0, str(ROOT))

ENGINES = (
    "edge_driver_model_distilled_fp16.engine",
    "object_detection_nanodet_2022nov.engine",
    "palm_detection_mediapipe_2023feb.engine",
)


def _engines(folder: Path, stamp: str) -> None:
    folder.mkdir()
    for name in ENGINES:
        (folder / name).write_bytes(b"cached-engine")
    (folder / ".tensorrt-stamp").write_text(stamp + "\n")


def test_reboot_does_not_call_trtexec(tmp_path):
    service = (ROOT / "deploy" / "dms.service").read_text(encoding="utf-8")
    assert "ExecStartPre" not in service
    assert "trtexec" not in service
    assert "build_engines" not in service
    bench = (ROOT / "deploy" / "dms.cloud.env.example").read_text(encoding="utf-8")
    assert "DMS_MQTT_PORT=1883" in bench
    assert "DMS_MQTT_TLS=0" in bench
    production = (ROOT / "cloud" / "render.yaml").read_text(encoding="utf-8")
    assert 'value: "8883"' in production
    assert 'value: "true"' in production

    models = tmp_path / "models"
    _engines(models, "10.0")
    poison = tmp_path / "trtexec"
    poison.write_text("#!/bin/sh\nexit 99\n")
    poison.chmod(poison.stat().st_mode | stat.S_IEXEC)
    result = subprocess.run(
        ["bash", str(ROOT / "tools" / "build_engines.sh")],
        cwd=ROOT,
        env={
            **os.environ,
            "MODELS_DIR": str(models),
            "TRT_VERSION": "10.0",
            "TRTEXEC": str(poison),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "skip rebuild" in result.stdout

    calls = tmp_path / "calls"
    fake = tmp_path / "rebuild-trtexec"
    fake.write_text('#!/bin/sh\necho "$@" >> "$TRT_CALLS"\nexit 0\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    rebuilt = tmp_path / "rebuilt"
    _engines(rebuilt, "9.0")
    again = subprocess.run(
        ["bash", str(ROOT / "tools" / "build_engines.sh")],
        cwd=ROOT,
        env={
            **os.environ,
            "MODELS_DIR": str(rebuilt),
            "TRT_VERSION": "10.0",
            "TRTEXEC": str(fake),
            "TRT_CALLS": str(calls),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert again.returncode == 0, again.stderr
    assert "--onnx=" in calls.read_text(encoding="utf-8")
    assert (rebuilt / ".tensorrt-stamp").read_text(encoding="utf-8").strip() == "10.0"


def test_old_clip_expires_and_a_fresh_one_stays(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import sessionmaker

    from app.config import settings
    from app.models import Base, Guardian, MediaClip, Ride
    from app.retention import retain_clips

    monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path}/vms.db")
    monkeypatch.setattr(settings, "media_dir", str(tmp_path / "media"))
    monkeypatch.setattr(settings, "auth_required", False)
    monkeypatch.setattr(settings, "media_device_key", "dev-media-key")
    from app import consent, db as cloud_db
    from app.db import get_db
    from app.main import app

    consent.clear()
    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
    previous_engine, previous_session = cloud_db.engine, cloud_db.SessionLocal
    cloud_db.engine = engine
    cloud_db.SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    Base.metadata.create_all(engine)
    now = int(time.time() * 1000)
    old_file = tmp_path / "media" / "ride-1842" / "clip-old" / "0.jpg"
    fresh_file = tmp_path / "media" / "ride-1842" / "clip-new" / "0.jpg"
    old_file.parent.mkdir(parents=True)
    fresh_file.parent.mkdir(parents=True)
    old_file.write_bytes(b"old-clip")
    fresh_file.write_bytes(b"fresh-clip")
    with cloud_db.SessionLocal() as db:
        db.add(Ride(id="ride-1842", device_id="orin-nano-01", analysis_json="{}"))
        db.add(Guardian(id="g-priya", ride_id="ride-1842", name="Priya Shah", sharing=True, can_notify=True))
        db.add(MediaClip(
            id="clip-old", ride_id="ride-1842", channel="dms", start_at=1, end_at=2,
            frames_json='[{"at": 1, "file": "0.jpg"}]', expires_at=now - 1000,
        ))
        db.add(MediaClip(
            id="clip-new", ride_id="ride-1842", channel="dms", start_at=now, end_at=now,
            frames_json='[{"at": 1, "file": "0.jpg"}]', expires_at=now + 86_400_000,
        ))
        db.commit()

    def _override():
        session = cloud_db.SessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _override
    try:
        from unittest.mock import patch

        with patch("app.mqtt_ingest.ingest.start"), patch("app.retention.start_retention"):
            with TestClient(app) as client:
                expired = client.get("/v1/media/clip-old/0.jpg")
                fresh = client.get("/v1/media/clip-new/0.jpg")
                assert expired.status_code == 404
                assert fresh.status_code == 200
                assert fresh.content == b"fresh-clip"
        with cloud_db.SessionLocal() as db:
            removed = retain_clips(db, now)
            db.commit()
            left = db.scalars(select(MediaClip.id)).all()
        assert removed == 1
        assert left == ["clip-new"]
        assert not old_file.exists()
        assert fresh_file.read_bytes() == b"fresh-clip"
    finally:
        app.dependency_overrides.clear()
        consent.clear()
        cloud_db.engine = previous_engine
        cloud_db.SessionLocal = previous_session


def test_restore_drill_returns_the_trip(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "cloud" / "scripts" / "restore_drill.py")],
        cwd=ROOT,
        env={**os.environ, "DRILL_DIR": str(tmp_path / "drill")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "trip-drill is back" in result.stdout
