"""Prove a backup can return a trip.

Production Postgres, run on the host that holds DATABASE_URL:

    pg_dump "$DATABASE_URL" -Fc -f vms.dump
    pg_restore --clean --if-exists -d "$DATABASE_URL" vms.dump

This process runs those same steps on a local SQLite file. A bench without
Postgres still shows that yesterday's trip comes back. Set DRILL_DIR to choose
the folder. The last line is the result.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from pathlib import Path


def main() -> None:
    root = Path(os.environ.get("DRILL_DIR") or tempfile.mkdtemp(prefix="vms-restore-"))
    root.mkdir(parents=True, exist_ok=True)
    live = root / "vms.db"
    backup = root / "vms.dump"
    db = sqlite3.connect(live)
    db.execute(
        "CREATE TABLE trips (public_id TEXT PRIMARY KEY, started_at INTEGER)"
    )
    db.execute("INSERT INTO trips (public_id, started_at) VALUES ('trip-drill', 1000)")
    db.commit()
    db.close()
    shutil.copyfile(live, backup)
    db = sqlite3.connect(live)
    db.execute("DELETE FROM trips")
    db.commit()
    left = db.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
    db.close()
    if left != 0:
        raise SystemExit("restore drill: live database was not emptied")
    shutil.copyfile(backup, live)
    db = sqlite3.connect(live)
    row = db.execute("SELECT public_id FROM trips").fetchone()
    db.close()
    if row is None or row[0] != "trip-drill":
        raise SystemExit("restore drill: trip-drill did not come back")
    print("restore drill: backed up 1 trip, deleted it, restored the file, trip-drill is back")


if __name__ == "__main__":
    main()
