"""Phase 4: a serious alert has one sentence and one clip, and the sentence stays on that type."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
sys.path.insert(0, str(ROOT))

_OTHER = ("phone", "drowsiness", "smoke", "drink", "belt", "child", "camera", "altercation", "glance")


def test_two_distress_alerts_each_get_a_clip_and_a_story(tmp_path, monkeypatch):
    pytest.importorskip("sqlalchemy")
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import sessionmaker

    from app.clips import save_clip
    from app.config import settings
    from app.models import Alert, Base, MediaClip, Ride
    from app.store import alert_public, alert_row

    monkeypatch.setattr(settings, "media_dir", str(tmp_path / "media"))
    engine = create_engine(f"sqlite:///{tmp_path}/story.db")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as db:
        ride = Ride(id="ride-1842", device_id="orin-nano-01", lat=12.91, lon=77.61)
        db.add(ride)
        db.flush()
        first = alert_row(db, ride, {
            "id": "scream-a", "type": "SCREAM", "confidence": 0.9,
            "epoch_ms": 1000, "lat": 12.91, "lon": 77.61,
        })
        second = alert_row(db, ride, {
            "id": "scream-b", "type": "SCREAM", "confidence": 0.8,
            "epoch_ms": 4000, "lat": 12.92, "lon": 77.62,
        })
        assert first.story and second.story
        assert "scream" in first.story.lower() and "scream" in second.story.lower()
        assert "on the way" in first.story
        for story in (first.story, second.story):
            lowered = story.lower()
            for word in _OTHER:
                assert word not in lowered

        frame = b"\xff\xd8\xff" + b"x" * 64
        save_clip(db, ride, "dms", [(1000, frame), (5000, frame)], "scream-a")
        save_clip(db, ride, "dms", [(4000, frame), (9000, frame)], "scream-b")
        db.commit()
        rows = db.scalars(select(Alert).where(Alert.ride_id == "ride-1842").order_by(Alert.at)).all()
        clips = db.scalars(select(MediaClip).order_by(MediaClip.start_at)).all()
        assert len(clips) == 2
        assert [row.clip_id for row in rows] == [clips[0].id, clips[1].id]
        shown = [alert_public(row) for row in rows]
        assert all(item["story"] and item["clipId"] for item in shown)
        assert all("on the way" not in item["story"] for item in shown)
        assert "for 4 seconds" in shown[0]["story"]
        assert "for 5 seconds" in shown[1]["story"]
        for item in shown:
            lowered = item["story"].lower()
            assert "scream" in lowered
            for word in _OTHER:
                assert word not in lowered
