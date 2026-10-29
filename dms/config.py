"""Configuration: built-in defaults, deep-merged with a YAML file and CLI overrides.

Every tunable used anywhere in the system lives here so a deployment can be
reviewed from one file (config/dms.yaml).
"""

from __future__ import annotations

import copy
import os
import socket
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

DEFAULTS: dict[str, Any] = {
    "device_id": "",  # empty -> hostname
    "data_dir": "data",
    "models_dir": "models",
    "log_level": "INFO",
    "cameras": {
        # Source forms: csi:<sensor-id>, v4l2:<index|/dev/videoN>, gst:<pipeline>,
        # rtsp://..., a video file path (loops), or an image path (static stream).
        "driver": {"source": "auto:0", "width": 640, "height": 480, "fps": 30, "flip": 0},
        "passenger": {"source": "auto:1", "width": 640, "height": 480, "fps": 15, "flip": 0},
    },
    "realtime": {
        "target_fps": 30,
        "driver_model": "edge_driver_model_distilled_fp16.engine",
        "driver_model_onnx": "edge_driver_model_distilled.onnx",
        "face_detector": "face_detection_yunet_2023mar.onnx",
        "face_detector_input": 320,
        "landmarks": "auto",  # auto | mediapipe | lbf
        "lbf_model": "lbfmodel.yaml",
        "face_landmarker_task": "face_landmarker.task",
        "phone_detector": "object_detection_nanodet_2022nov.onnx",
        "hand_detector": "palm_detection_mediapipe_2023feb.onnx",
        "object_detector_hz": 15.0,  # phone/cup + hand detectors run async, capped at this rate
        "object_hold_s": 0.6,  # boxes older than this are dropped
        "mirror": False,  # True if the driver camera image is mirrored (selfie view)
        "gaze_away_deg": 25.0,  # |head yaw + eye angle| above this = eyes off road
        "face_recognizer": "face_recognition_sface_2021dec.onnx",
        "identify_every_s": 2.0,
        "smooth_window": 5,
        # Alert rules (spec): EAR < threshold for > 1.5 s, |yaw| > 30 deg for > 2 s.
        "ear_threshold": 0.21,
        "ear_auto_calibrate": True,  # per-driver baseline; threshold = min(ear_threshold, 0.72 * baseline)
        "eyes_closed_s": 2.5,
        "yaw_threshold_deg": 30.0,
        "head_turn_s": 3.0,
        "pitch_down_threshold_deg": 25.0,
        "phone_s": 2.0,
        "phone_model_threshold": 0.50,
        "phone_detector_threshold": 0.35,
        "activity_threshold": 0.55,
        "smoking_s": 2.0,
        "drinking_s": 2.0,
        "yawn_threshold": 0.70,
        "drowsy_threshold": 0.65,
        "drowsy_s": 2.0,
        "seatbelt_check": True,
        "seatbelt_off_s": 10.0,
        "no_face_s": 5.0,
        "alert_cooldown_s": 15.0,
        "snapshot_alerts": True,
    },
    "passenger": {
        "occupancy_every_s": 0.5,
    },
    "cms": {
        "enabled": True,
        "interval_s": 0.5,  # 2 Hz, off the real-time detector thread
        "adult_face_ratio": 0.18,  # face height / frame height at or above this is an adult
        "child_hold_s": 3.0,
        "object_hold_s": 2.0,
        "occupancy_hold_s": 1.0,
        "topic_prefix": "cms",
        # Normalized polygons (0-1 of the cabin frame). Rear band is the far half of a
        # camera looking back from the front of the cabin; front passenger is the near band.
        "seats": [
            {"name": "front_passenger", "polygon": [[0.0, 0.55], [1.0, 0.55], [1.0, 1.0], [0.0, 1.0]]},
            {"name": "rear_left", "polygon": [[0.0, 0.0], [0.33, 0.0], [0.33, 0.55], [0.0, 0.55]]},
            {"name": "rear_center", "polygon": [[0.33, 0.0], [0.67, 0.0], [0.67, 0.55], [0.33, 0.55]]},
            {"name": "rear_right", "polygon": [[0.67, 0.0], [1.0, 0.0], [1.0, 0.55], [0.67, 0.55]]},
        ],
    },
    "vlm": {
        "enabled": True,
        # OpenAI-compatible chat endpoint: llama.cpp `llama-server` or vLLM.
        "endpoint": "http://127.0.0.1:8080/v1/chat/completions",
        "model": "vlm",
        "api_key": "",
        "interval_s": 2.5,  # 0.4 Hz, inside the 0.3-0.5 Hz window
        "timeout_s": 20.0,  # vision encode on the Orin often exceeds the old 3.5 s cap
        "max_tokens": 320,
        "temperature": 0.1,
        "image_width": 448,
        "jpeg_quality": 85,
        "composite": True,  # driver + passenger side by side in one image
        "confidence_min": 0.7,  # JSON answers below this do not become alerts
        "hold_cycles": 2,  # same CMS hazard must appear this many cycles in a row
        "prompt": (
            "You analyse the cabin. Measurements follow this instruction; trust them and do "
            "not contradict them. Reply with one JSON object and no other text: "
            '{"dms":{"scene":"short","description":"","hazards":[],"confidence":0.0,'
            '"observations":{"handsOnWheel":true,"gaze":"forward","viewBlocked":false,'
            '"lighting":"ok","cameraBlocked":false}},'
            '"cms":{"scene":"short","description":"","hazards":[],"confidence":0.0,'
            '"observations":{"occupants":{"adults":0,"children":0},"childUnattended":false,'
            '"aggression":"calm","viewBlocked":false,"weaponOrSmoke":false}}}. '
            "Keep scene as the short label. description is one new sentence of what that person "
            "is doing in this image: hands, gaze, phone, posture, or that the seat is empty. "
            "Never leave description empty and never copy the word short into description. "
            "dms.hazards must stay empty. cms.hazards may contain only passenger_aggressive, "
            "child_unattended, objects_blocking_driver. aggression is calm, argument, or panic. "
            "Leave cms.hazards empty when the cabin adds nothing unsafe. confidence is 0 to 1."
        ),
        "format_hint": (
            "Left half of the image is the driver camera. Right half is the passenger camera. "
            "Keep dms.scene and cms.scene short. Write what each person is doing in description."
        ),
        "thermal_limit_c": 85.0,
        "thermal_backoff": 3.0,  # interval multiplier while over the limit
        "min_realtime_fps": 15.0,  # back off when the real-time loop falls below this
        "memory_limit_mb": 7500,
    },
    "audio": {
        "enabled": True,
        "device": None,  # sounddevice name/index; or a .wav path for replay
        "sample_rate": 16000,
        "block_ms": 50,
        "backend": "auto",  # auto | cochl | yamnet | dsp
        "hop_s": 0.25,
        "window_s": 0.96,
        "yamnet_model": "yamnet.tflite",
        "yamnet_class_map": "yamnet_class_map.csv",
        "scream_threshold": 0.35,
        "shout_threshold": 0.35,
        "altercation_window_s": 3.0,
        "altercation_min_s": 1.5,
        "altercation_loud_dbfs": -30.0,
        "cochl": {
            "project_key": "",
            "config_file": "config/cochl_config.json",
            "scream_tags": ["Scream", "Screaming", "Shout", "Yell"],
            "speech_tags": ["Speech", "Shout", "Yell", "Conversation"],
            "threshold": 0.5,
        },
        "kws": {
            "enabled": True,
            "engine": "vosk",
            "model_dir": "vosk-model-small-en-us-0.15",
            "keywords_file": "config/abusive_keywords.txt",
            "min_confidence": 0.6,
        },
        "save_snippets": True,
        "alert_cooldown_s": 5.0,
    },
    "mqtt": {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 1883,
        "client_id": "",
        "protocol": "auto",  # auto (try 5, fall back to 3.1.1) | 5 | 3.1.1
        "username": "",
        "password": "",
        "tls": {"enabled": False, "ca_certs": "", "certfile": "", "keyfile": "", "insecure": False},
        "keepalive": 30,
        "topic_prefix": "dms",
        "heartbeat_s": 30,
        "alert_qos": 1,
        "heartbeat_qos": 0,
        "max_inflight": 20,
        "outbox_db": "outbox.sqlite3",
        "outbox_max_rows": 50000,
        "retry_after_s": 15.0,
    },
    "webrtc": {
        "enabled": False,
        "dummy": False,
        "api_url": "http://127.0.0.1:8000",
        "device_key": "dev-media-key",
        "interval_s": 0.2,
        "timeout_s": 5.0,
    },
    "ui": {
        "enabled": True,
        "fullscreen": False,
        "fps": 20,  # dashboard repaint rate; the detector runs independently
        "hardware_label": "",  # empty -> autodetect
        "local_alarm": True,
    },
}


class Config(dict):
    """dict with attribute access for nested sections."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def path(self, key: str) -> Path:
        """Resolve a file setting against the models dir (or root for relative config/ paths)."""
        raw = self.get(key)
        return resolve_path(raw) if raw else Path()


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


_MODELS_DIR = ROOT / "models"
_DATA_DIR = ROOT / "data"


def resolve_path(value: str | os.PathLike, base: str = "models") -> Path:
    path = Path(os.path.expanduser(str(value)))
    if path.is_absolute():
        return path
    if path.exists():  # relative to the current directory (CLI arguments)
        return path.resolve()
    if (ROOT / path).exists():
        return ROOT / path
    if path.parts and path.parts[0] in {"config", "models", "data", "tools", "assets"}:
        return ROOT / path
    return (_MODELS_DIR if base == "models" else _DATA_DIR) / path


def model_path(name: str) -> Path:
    return resolve_path(name, "models")


def data_path(*parts: str) -> Path:
    path = _DATA_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> Config:
    global _MODELS_DIR, _DATA_DIR
    merged = copy.deepcopy(DEFAULTS)
    if path:
        file_path = Path(path)
        if not file_path.is_absolute() and not file_path.exists():
            file_path = ROOT / file_path
        with open(file_path, encoding="utf-8") as handle:
            merged = _deep_merge(merged, yaml.safe_load(handle) or {})
    if overrides:
        merged = _deep_merge(merged, overrides)
    if not merged.get("device_id"):
        merged["device_id"] = os.environ.get("DMS_DEVICE_ID") or socket.gethostname()
    _MODELS_DIR = Path(merged["models_dir"]) if Path(merged["models_dir"]).is_absolute() else ROOT / merged["models_dir"]
    _DATA_DIR = Path(merged["data_dir"]) if Path(merged["data_dir"]).is_absolute() else ROOT / merged["data_dir"]
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Environment overrides for secrets so they stay out of YAML.
    mqtt = merged["mqtt"]
    mqtt["host"] = os.environ.get("DMS_MQTT_HOST", mqtt["host"])
    mqtt["port"] = int(os.environ.get("DMS_MQTT_PORT", mqtt["port"]))
    mqtt["username"] = os.environ.get("DMS_MQTT_USERNAME", mqtt["username"])
    mqtt["password"] = os.environ.get("DMS_MQTT_PASSWORD", mqtt["password"])
    merged["audio"]["cochl"]["project_key"] = os.environ.get(
        "COCHL_PROJECT_KEY", merged["audio"]["cochl"]["project_key"]
    )
    merged["vlm"]["endpoint"] = os.environ.get("DMS_VLM_ENDPOINT", merged["vlm"]["endpoint"])
    webrtc = merged["webrtc"]
    webrtc["api_url"] = os.environ.get("DMS_WEBRTC_API_URL", webrtc["api_url"])
    webrtc["device_key"] = os.environ.get("DMS_MEDIA_DEVICE_KEY", webrtc["device_key"]) or os.environ.get(
        "MEDIA_DEVICE_KEY", webrtc["device_key"]
    )
    return Config(merged)
