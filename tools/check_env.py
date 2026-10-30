#!/usr/bin/env python3
"""Check the Python environment, cameras' GStreamer access and models. Exit 1 if a
required module is missing.  Usage: python3 tools/check_env.py"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dms.bootstrap import ensure_venv  # noqa: E402

ensure_venv()

REQUIRED = ("numpy", "cv2", "yaml", "paho.mqtt.client", "livekit", "sounddevice", "PySide6.QtWidgets", "onnxruntime")
OPTIONAL = ("tensorrt", "gi", "mediapipe", "vosk", "ai_edge_litert", "sense")


def main() -> int:
    print(f"python {sys.version.split()[0]} at {sys.executable}")
    missing = []
    for mod in REQUIRED:
        try:
            m = importlib.import_module(mod)
            print(f"  ok       {mod} {getattr(m, '__version__', '')}")
        except Exception as exc:
            missing.append(mod)
            hint = {"sounddevice": "  -> sudo apt install libportaudio2",
                    "livekit": "  -> .venv/bin/pip install 'livekit>=0.17.6'",
                    "PySide6.QtWidgets": "  -> sudo apt install libxkbcommon-x11-0 libxcb-cursor0 libegl1"}.get(mod, "")
            print(f"  MISSING  {mod}: {exc}{hint}")
    for mod in OPTIONAL:
        try:
            m = importlib.import_module(mod)
            print(f"  ok       {mod} {getattr(m, '__version__', '')} (optional)")
        except Exception as exc:
            print(f"  -        {mod} not available ({type(exc).__name__}) (optional)")
    try:
        import cv2

        gst = any("GStreamer" in line and "YES" in line for line in cv2.getBuildInformation().splitlines())
        print(f"opencv {cv2.__version__}: contrib={hasattr(cv2, 'face')} gstreamer={gst} from {cv2.__file__}")
    except Exception:
        pass
    try:
        import onnxruntime as ort

        print(f"onnxruntime providers: {ort.get_available_providers()}")
    except Exception:
        pass
    try:
        from dms.capture import _gi_available, _opencv_has_gstreamer

        if not (_gi_available() or _opencv_has_gstreamer()):
            print("WARNING: no GStreamer access from Python; CSI cameras will not open "
                  "(sudo apt install python3-gi gir1.2-gst-plugins-base-1.0)")
    except Exception:
        pass
    try:
        from dms.capture import SYSFS_V4L, csi_sensors, usb_cameras

        csi, usb = csi_sensors(), usb_cameras()

        def describe(nodes):
            return ", ".join("/dev/video%d %s" % (n["index"], n["name"]) for n in nodes) or "none"

        def pick(n):  # same rule as the auto:<n> source
            if csi and n < len(csi):
                return "CSI sensor %d" % n
            if n < len(usb):
                return "USB /dev/video%d (%s)" % (usb[n]["index"], usb[n]["name"])
            return "nothing"

        if csi is None:
            print("cameras: /sys/class/video4linux not present (no V4L2 devices on this system)")
        else:
            print("cameras: CSI sensors: " + describe(csi))
            print("         USB cameras: " + describe(usb))
            print("         auto:0 (driver) -> %s   auto:1 (passenger) -> %s" % (pick(0), pick(1)))
            if not csi and not usb:
                print("         no camera found: connect a USB camera, or enable the CSI overlay "
                      "(sudo /opt/nvidia/jetson-io/jetson-io.py) and reboot")
    except Exception as exc:
        print(f"cameras: could not enumerate ({exc})")
    models = ROOT / "models"
    need = ["edge_driver_model_distilled.onnx", "face_detection_yunet_2023mar.onnx",
            "face_recognition_sface_2021dec.onnx", "object_detection_nanodet_2022nov.onnx",
            "palm_detection_mediapipe_2023feb.onnx", "lbfmodel.yaml"]
    absent = [n for n in need if not (models / n).exists()]
    print("models: " + ("all required present" if not absent else f"missing {absent} -> python3 tools/download_models.py"))
    if missing:
        print(f"\nmissing required modules: {missing}\n  fix: source .venv/bin/activate && "
              f"pip install -c constraints.txt -r requirements.txt")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
