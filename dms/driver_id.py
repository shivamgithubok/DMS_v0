"""Driver identification: SFace embeddings (128-d) matched against an enrolled roster.

Roster file ``data/drivers/roster.json``::

    [{"driver_id": "4827", "name": "Emily Carter", "age": 28,
      "photo": "emily_carter.jpg", "embedding": [...128 floats...]}]

Enroll with ``python tools/enroll_driver.py --name ... --id ... --age ... --image photo.jpg``.
Identification runs every ``identify_every_s`` seconds (not per frame) and keeps an
exponential average of the cosine similarity so one bad frame cannot flip identity.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np

from .state import Identity

log = logging.getLogger("dms.driver_id")

SFACE_COSINE_THRESHOLD = 0.363  # OpenCV SFace reference threshold (LFW)


def match_percent(cosine: float) -> float:
    """Map cosine similarity to a display confidence: 50 % at the decision threshold,
    ~99 % at 0.6 and above."""
    return float(100.0 / (1.0 + math.exp(-(cosine - SFACE_COSINE_THRESHOLD) * 20.0)))


class DriverRoster:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.entries: list[dict] = []
        self.matrix = np.zeros((0, 128), dtype=np.float32)
        self.photos: dict[str, np.ndarray] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.entries = []
            return
        self.entries = json.loads(self.path.read_text())
        vectors = []
        for entry in self.entries:
            vec = np.asarray(entry["embedding"], dtype=np.float32)
            vectors.append(vec / (np.linalg.norm(vec) + 1e-9))
            photo = entry.get("photo")
            if photo:
                photo_path = (self.path.parent / photo) if not Path(photo).is_absolute() else Path(photo)
                image = cv2.imread(str(photo_path))
                if image is not None:
                    self.photos[str(entry["driver_id"])] = image
        self.matrix = np.stack(vectors) if vectors else np.zeros((0, 128), dtype=np.float32)
        log.info("roster: %d enrolled driver(s)", len(self.entries))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=2))

    def upsert(self, entry: dict) -> None:
        self.entries = [e for e in self.entries if str(e["driver_id"]) != str(entry["driver_id"])]
        self.entries.append(entry)
        self.save()
        self.load()


class DriverIdentifier:
    def __init__(self, model_path, roster: DriverRoster, ema: float = 0.5) -> None:
        self.recognizer = cv2.FaceRecognizerSF.create(str(model_path), "")
        self.roster = roster
        self.ema = ema
        self.scores: dict[str, float] = {}
        self.identity = Identity()

    def embed(self, frame: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        aligned = self.recognizer.alignCrop(frame, face_row)
        feature = self.recognizer.feature(aligned).reshape(-1).astype(np.float32)
        return feature / (np.linalg.norm(feature) + 1e-9)

    def reset(self) -> None:
        self.scores.clear()
        self.identity = Identity()

    def identify(self, frame: np.ndarray, face_row: np.ndarray, age_group: str = "--") -> Identity:
        if not len(self.roster.entries):
            self.identity = Identity(known=False, name="Unregistered", age=age_group)
            return self.identity
        vec = self.embed(frame, face_row)
        sims = self.roster.matrix @ vec
        for entry, sim in zip(self.roster.entries, sims):
            key = str(entry["driver_id"])
            prev = self.scores.get(key)
            self.scores[key] = float(sim) if prev is None else self.ema * prev + (1 - self.ema) * float(sim)
        best_key = max(self.scores, key=self.scores.get)
        best = self.scores[best_key]
        entry = next(e for e in self.roster.entries if str(e["driver_id"]) == best_key)
        if best >= SFACE_COSINE_THRESHOLD:
            self.identity = Identity(
                known=True, name=entry.get("name", "Driver"), driver_id=best_key,
                age=str(entry.get("age", age_group)), match_pct=match_percent(best),
                photo=self.roster.photos.get(best_key), notes=entry.get("notes", ""))
        else:
            self.identity = Identity(known=False, name="Unknown Driver", driver_id="----", age=age_group,
                                     match_pct=match_percent(best), photo=None)
        return self.identity
