#!/usr/bin/env python3
"""Download every model the DMS uses into ./models (no Docker, standard library only).

    python3 tools/download_models.py              # required models (~115 MB)
    python3 tools/download_models.py --optional   # + MediaPipe face landmarker, YAMNet, Vosk (~60 MB)
    python3 tools/download_models.py --vlm        # + Qwen3.5-0.8B Q4_K_M GGUF + vision projector (~0.7 GB)
    python3 tools/download_models.py --all        # everything
    python3 tools/download_models.py --list       # show what is present / missing
    python3 tools/download_models.py --verify     # re-check SHA-256 of downloaded files

Features: resumable downloads (.part + HTTP Range), retries with back-off, mirror fallback,
SHA-256 verification for pinned files, zip extraction, honours HTTPS_PROXY, and an
HF_TOKEN / HF_ENDPOINT for Hugging Face. Files that already exist and verify are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
ZOO_GH = "https://github.com/opencv/opencv_zoo/raw/main/models"
MP = "https://storage.googleapis.com/mediapipe-models"
HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


@dataclass
class Item:
    name: str  # file (or directory for archives) inside models/
    urls: list
    group: str  # core | optional
    sha256: str | None = None  # pinned hash (verified on the build machine)
    size_mb: float = 0.0
    extract: bool = False  # zip archive -> directory `name`
    purpose: str = ""
    extra: dict = field(default_factory=dict)


MANIFEST = [
    # ---------------------------------------------------------------- core (required)
    Item("edge_driver_model_distilled.onnx",
         ["https://raw.githubusercontent.com/shivamgithubok/DMS-CMS/main/models/edge_driver_model_distilled.onnx"],
         "core", "db2ab2fa57068afc5a6b55224cc3d41316d897dcbd73a33aa872f03e7cf3afc1", 13.4,
         purpose="8-head MobileNetV3 driver-state model (drowsiness, gaze, eyes, activity, ...)"),
    Item("face_detection_yunet_2023mar.onnx",
         [f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
          f"{ZOO_GH}/face_detection_yunet/face_detection_yunet_2023mar.onnx"],
         "core", "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4", 0.2,
         purpose="YuNet face detector + 5 face keypoints"),
    Item("face_recognition_sface_2021dec.onnx",
         [f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
          f"{ZOO_GH}/face_recognition_sface/face_recognition_sface_2021dec.onnx"],
         "core", "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79", 38.7,
         purpose="SFace embeddings for driver identification"),
    Item("object_detection_nanodet_2022nov.onnx",
         [f"{ZOO}/object_detection_nanodet/object_detection_nanodet_2022nov.onnx",
          f"{ZOO_GH}/object_detection_nanodet/object_detection_nanodet_2022nov.onnx"],
         "core", "4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186", 3.8,
         purpose="NanoDet-Plus COCO detector (phone, cup, bottle)"),
    Item("palm_detection_mediapipe_2023feb.onnx",
         [f"{ZOO}/palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx",
          f"{ZOO_GH}/palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx"],
         "core", "78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c", 3.9,
         purpose="MediaPipe palm detector (hand near ear / on phone)"),
    Item("lbfmodel.yaml",
         ["https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"],
         "core", "70dd8b1657c42d1595d6bd13d97d932877b3bed54a95d3c4733a0f740d1fd66b", 56.4,
         purpose="Facemark LBF 68-point landmarks (fallback when MediaPipe is not installed)"),
    Item("yamnet_class_map.csv",
         ["https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"],
         "core", "cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2", 0.01,
         purpose="AudioSet labels for YAMNet"),
    # ---------------------------------------------------------------- optional back-ends
    Item("face_landmarker.task",
         [f"{MP}/face_landmarker/face_landmarker/float16/1/face_landmarker.task"],
         "optional", None, 3.6, purpose="MediaPipe Tasks 478-point landmarks with iris"),
    Item("yamnet.tflite",
         [f"{MP}/audio_classifier/yamnet/float32/latest/yamnet.tflite"],
         "optional", None, 4.0, purpose="YAMNet audio events (scream / shout / speech)"),
    Item("vosk-model-small-en-us-0.15",
         ["https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"],
         "optional", None, 40.0, extract=True, purpose="Vosk keyword spotting (abusive language)"),
]

# Qwen3.5-0.8B is natively multimodal (GGUF + mmproj). Q4_K_M is about 533 MB
# plus a 205 MB F16 projector, small enough to sit beside the detector on an 8 GB Orin.
# Qwen2.5-VL-3B Q4 was about 2.6 GB and crowded the same memory budget.
VLM_REPO = "unsloth/Qwen3.5-0.8B-GGUF"
VLM_QUANT = "Q4_K_M"


# --------------------------------------------------------------------------- helpers
def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _request(url: str, start: int = 0) -> urllib.request.Request:
    headers = {"User-Agent": "orin-dms-model-downloader/1.0"}
    if start:
        headers["Range"] = f"bytes={start}-"
    if "huggingface" in url or url.startswith(HF):
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def fetch(url: str, dest: Path, retries: int = 4) -> None:
    """Resumable download to dest (via dest.part)."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        start = part.stat().st_size if part.exists() else 0
        try:
            with urllib.request.urlopen(_request(url, start), timeout=60) as resp:
                if start and resp.status != 206:  # server ignored Range: restart
                    start = 0
                total = resp.headers.get("Content-Length")
                total = int(total) + start if total else None
                done = start
                t0, last = time.monotonic(), 0.0
                with open(part, "ab" if start else "wb") as out:
                    while True:
                        chunk = resp.read(1 << 18)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        now = time.monotonic()
                        if now - last > 0.5 and sys.stdout.isatty():
                            last = now
                            rate = (done - start) / max(1e-3, now - t0)
                            pct = f"{100 * done / total:5.1f}%" if total else "     "
                            print(f"\r      {pct} {human(done):>9} {human(rate)}/s   ", end="", flush=True)
                if sys.stdout.isatty():
                    print("\r" + " " * 50 + "\r", end="")
                if total and done < total:
                    raise IOError(f"incomplete: {done}/{total} bytes")
            part.replace(dest)
            return
        except (urllib.error.URLError, IOError, TimeoutError, ConnectionError) as exc:
            code = getattr(exc, "code", None)
            if code in (401, 403, 404):
                raise
            if attempt == retries:
                raise
            wait = 2 ** attempt
            print(f"      retry {attempt}/{retries - 1} in {wait}s ({exc})")
            time.sleep(wait)


def install(item: Item, force: bool = False) -> bool:
    target = MODELS / item.name
    if target.exists() and not force:
        if item.sha256 and target.is_file() and sha256(target) != item.sha256:
            print(f"bad   {item.name}: checksum mismatch, downloading again")
        else:
            print(f"ok    {item.name}")
            return True
    print(f"get   {item.name}  (~{item.size_mb:g} MB)  {item.purpose}")
    download = target.with_name(target.name + ".zip") if item.extract else target
    errors = []
    for url in item.urls:
        try:
            fetch(url, download)
            break
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    else:
        print(f"FAIL  {item.name}\n      " + "\n      ".join(errors))
        return False
    if item.sha256 and not item.extract:
        digest = sha256(download)
        if digest != item.sha256:
            download.unlink()
            print(f"FAIL  {item.name}: sha256 {digest} != expected {item.sha256}")
            return False
    if item.extract:
        tmp = MODELS / (item.name + ".extract")
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(download) as zf:
            zf.extractall(tmp)
        inner = [p for p in tmp.iterdir()]
        src = inner[0] if len(inner) == 1 and inner[0].is_dir() else tmp
        shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(src), str(target))
        shutil.rmtree(tmp, ignore_errors=True)
        download.unlink()
    print(f"done  {item.name}  ({human(_size(target))})")
    return True


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


# --------------------------------------------------------------------------- VLM (Hugging Face GGUF)
def _quant_file(name: str, quant: str) -> bool:
    """True when the filename is that quant, not a longer name that merely contains it."""
    return re.search(rf"(^|[-_.]){re.escape(quant.lower())}\.gguf$", name.lower()) is not None


def _projector_rank(name: str) -> int:
    low = name.lower()
    if "bf16" in low:
        return 1
    if re.search(r"(^|[^a-z0-9])f16([^a-z0-9]|$)", low):
        return 0  # F16 projector; "f16" must not match inside "bf16"
    if "f32" in low:
        return 3
    return 2


def vlm_files(repo: str, quant: str) -> list[tuple[str, str]]:
    """Pick the model GGUF for `quant` and the F16 vision projector from the repo listing."""
    with urllib.request.urlopen(_request(f"{HF}/api/models/{repo}"), timeout=30) as resp:
        siblings = [s["rfilename"] for s in json.load(resp).get("siblings", [])]
    ggufs = [f for f in siblings if f.lower().endswith(".gguf")]
    weights = [f for f in ggufs if "mmproj" not in f.lower()]
    model = [f for f in weights if _quant_file(f, quant)]
    proj = [f for f in ggufs if "mmproj" in f.lower()]
    if not model:
        raise RuntimeError(f"no {quant} GGUF in {repo}: {ggufs}")
    if not proj:
        raise RuntimeError(f"no mmproj (vision projector) GGUF in {repo}")
    model.sort(key=lambda f: (0 if f.lower().endswith(f"-{quant.lower()}.gguf") else 1, len(f), f))
    return [(model[0], "model"), (sorted(proj, key=_projector_rank)[0], "projector")]


def install_vlm(repo: str, quant: str, force: bool = False) -> bool:
    out = MODELS / "vlm"
    try:
        files = vlm_files(repo, quant)
    except Exception as exc:
        print(f"FAIL  VLM listing for {repo}: {exc}")
        return False
    ok = True
    for name, kind in files:
        dest = out / Path(name).name
        if dest.exists() and not force:
            print(f"ok    vlm/{dest.name}")
            continue
        print(f"get   vlm/{dest.name}  ({kind})")
        try:
            fetch(f"{HF}/{repo}/resolve/main/{name}", dest)
            print(f"done  vlm/{dest.name}  ({human(dest.stat().st_size)})")
        except Exception as exc:
            print(f"FAIL  vlm/{dest.name}: {exc}")
            ok = False
    if ok:
        (out / "vlm.json").write_text(json.dumps({"repo": repo, "quant": quant,
                                                  "model": Path(files[0][0]).name,
                                                  "mmproj": Path(files[1][0]).name}, indent=2))
    return ok


# --------------------------------------------------------------------------- CLI
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--optional", action="store_true", help="also MediaPipe face landmarker, YAMNet, Vosk")
    ap.add_argument("--vlm", action="store_true", help="also the VLM GGUF + vision projector")
    ap.add_argument("--all", action="store_true", help="core + optional + VLM")
    ap.add_argument("--vlm-repo", default=VLM_REPO)
    ap.add_argument("--vlm-quant", default=VLM_QUANT)
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--list", action="store_true", help="show status and exit")
    ap.add_argument("--verify", action="store_true", help="check pinned SHA-256 of present files and exit")
    args = ap.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)

    if args.list or args.verify:
        bad = 0
        for item in MANIFEST:
            path = MODELS / item.name
            if not path.exists():
                state = "missing" + (" (required)" if item.group == "core" else "")
                bad += item.group == "core"
            elif args.verify and item.sha256:
                good = sha256(path) == item.sha256
                state = "verified" if good else "CHECKSUM MISMATCH"
                bad += not good
            else:
                state = f"present {human(_size(path))}"
            print(f"{item.group:8s} {item.name:42s} {state}")
        vlm = MODELS / "vlm" / "vlm.json"
        print(f"vlm      {'vlm/':42s} {'present ' + json.loads(vlm.read_text())['model'] if vlm.exists() else 'missing'}")
        engines = sorted(p.name for p in MODELS.glob("*.engine"))
        print(f"engines  {', '.join(engines) if engines else 'none (run tools/build_engines.sh on the Jetson)'}")
        return 1 if bad else 0

    groups = {"core"} | ({"optional"} if args.optional or args.all else set())
    failed = [item.name for item in MANIFEST if item.group in groups and not install(item, args.force)]
    if args.vlm or args.all:
        if not install_vlm(args.vlm_repo, args.vlm_quant, args.force):
            failed.append("vlm")
    print()
    if failed:
        core_missing = [n for n in failed if any(i.name == n and i.group == "core" for i in MANIFEST)]
        print(f"{len(failed)} download(s) failed: {', '.join(failed)}")
        if core_missing:
            print("required models are missing; the DMS will not start until they download")
            return 1
        if "vlm" in failed:
            print("without models/vlm the DMS runs with scene understanding off (use --no-vlm)")
        print("optional back-ends that failed are skipped automatically at run time")
        return 0
    print(f"all requested models are in {MODELS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
