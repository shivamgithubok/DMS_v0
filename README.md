# Driver Monitoring System — Jetson Orin Nano 8GB

A complete in-cabin monitoring prototype: real-time driver-state detection (eyes, head pose, gaze, phone, hands, seatbelt, identity), low-rate VLM scene understanding over both cameras, cabin audio events (scream, altercation, abusive language), MQTT store-and-forward uplink, and a PySide6 dashboard laid out like the reference HMI.

## Run on another Jetson

Share this git repo. The other device only needs the files below; cloud, tests, docs, videos, and local `config/dms.yaml` are gitignored.

| Path | Why |
|---|---|
| `setup_jetson.sh` | apt, `.venv`, models, TensorRT, llama.cpp |
| `run.sh` | starts VLM (if present) + DMS |
| `main.py` `dms/` | detector, CMS, VLM client, MQTT, dashboard |
| `tools/` | download models, engines, enroll driver, camera check |
| `requirements.txt` `requirements-optional.txt` `constraints.txt` | Python deps |
| `config/dms.example.yaml` | copy to `config/dms.yaml` (gitignored) |
| `config/abusive_keywords.txt` `config/cochl_config.json` | audio |
| `deploy/dms.cloud.env.example` | MQTT user + `DMS_DEVICE_ID` (copy, do not commit) |
| `deploy/dms.service` `deploy/dms-vlm.service` | optional autostart |

```bash
git clone <this-repo> ~/orin-dms && cd ~/orin-dms
cp config/dms.example.yaml config/dms.yaml   # setup_jetson.sh also does this if missing
# edit config/dms.yaml: mqtt.host = VPS_IP, cameras if not auto:0 / auto:1
export DMS_DEVICE_ID=orin-nano-01            # same id as the Guardian ride, or a new unique id
export DMS_MQTT_USERNAME=dms-orin
export DMS_MQTT_PASSWORD='<emqx password>'
./setup_jetson.sh
./run.sh --mqtt-host VPS_IP --device-id "$DMS_DEVICE_ID"
```

Guardian app: `EXPO_PUBLIC_TELEMETRY_URL=http://VPS_IP:8000`, leave `EXPO_PUBLIC_MQTT_URL` empty, ride id `ride-1842`. Do not give them the EMQX dashboard password.

## Architecture

```
 CSI cam 0 (driver) ──► cam-driver ──┐ latest-frame slot (drop-old)
 CSI cam 1 (cabin)  ──► cam-passenger┤
                                     ├─► realtime-detector  ≥15 FPS ─────────────┐
                                     │     YuNet face ─► landmarks ─► EAR + PnP pose
                                     │     8-head MobileNetV3 (TensorRT FP16)     │
                                     │     ↳ object-detector thread: NanoDet phone/cup + palm
                                     │     seatbelt line check · SFace driver ID  │
                                     ├─► cms-monitor (seats, objects, 2 Hz)       │
                                     └─► vlm-worker 0.4 Hz ──HTTP──► VLM server ──┤   AlertBus
 USB/I2S mic ──► audio capture ──► audio-classifier (Cochl│YAMNet│DSP) + Vosk KWS ─┤ (debounce,
                                                                                 │  fan-out)
                         mqtt-publisher ◄── SQLite outbox ◄──────────────────────┤
                         dashboard process (own GIL) ◄── state snapshots ◄───────┘
```

Threads map to the spec: **T1** capture (one grabber per sensor), **T2** real-time detector, **T3** VLM worker, **T4** audio, **T5** MQTT, **T6** cabin monitor. Design decisions that matter for the 8 GB / 15 W budget:

- **VLM in a separate server process** (llama.cpp `llama-server` or vLLM, OpenAI API). It gets its own CUDA context, so its decode never queues ahead of the detector, and it never holds the DMS's Python GIL. The detector's TensorRT stream is created at the highest CUDA stream priority. The VLM server runs at `nice 10` with `--threads 2`, leaving the CPU to the detector.
- **Dashboard in its own process.** Qt painting in the same interpreter halved the detector rate on a 2-core test machine (GIL contention). As a separate process it costs no detector time, and a UI crash cannot stop monitoring or uploads.
- **Object detectors on their own thread.** The spec's timing rules (EAR, head yaw) never wait for the phone/hand detectors. Their boxes are held for 0.6 s.
- **VLM back-off.** The interval is ×3 above 85 °C (any SoC zone; GPU temperature is logged), ×2 while the detector is below 15 FPS, and requests are skipped above 7.5 GB unified memory.
- **Alerts are durable.** Every alert is written to SQLite before any network attempt and deleted only on PUBACK. Outages and reboots lose nothing, and the outbox drains in order on reconnect.

## Detection logic

| Alert | Rule (defaults in `config/dms.yaml`) |
|---|---|
| `DISTRACTION` / `eyes_closed` | EAR < threshold for > 1.5 s (threshold auto-calibrates per driver, capped at 0.21) |
| `DISTRACTION` / `head_turned` | \|head yaw\| > 30° for > 2 s |
| `DISTRACTION` / `gaze_away`, `looking_down` | \|head yaw + eye-in-head angle\| > 25°, or pitch < −25°, for > 2 s |
| `DISTRACTION` / `phone_usage` | phone box in the driver's reach with a hand on it, hand at the ear, or the model's *Phone Use* head, for > 1 s |
| `DROWSINESS`, `SMOKING`, `DRINKING` | 8-head model (drowsy / activity heads) + cup/bottle near the mouth, held 2 s |
| `SEATBELT_OFF`, `DRIVER_ABSENT` | no diagonal belt edge for 10 s; no face for 3 s |
| `VLM_ALERT` | VLM answer parsed into driver categories only: driver distracted, or an unsafe answer with no specific category (negation-aware) |
| `CHILD_PRESENT` | Rear-seat face smaller than `cms.adult_face_ratio` and no adult face, held `cms.child_hold_s`. Also the VLM `child_unattended` category. Not a left-behind alarm |
| `OBJECT_ON_SEAT` | NanoDet bag, bottle, cup, or phone whose center sits in a seat polygon for `cms.object_hold_s`, or the VLM blocking category |
| `PASSENGER_AGGRESSIVE` | VLM category only |
| `OCCUPANCY_CHANGE` | A seat label changes and holds for `cms.occupancy_hold_s` |
| `SCREAM` | Screaming score ≥ 0.35 in a 0.96 s window, evaluated every 0.25 s |
| `ALTERCATION` | ≥ 1.5 s of loud voice in a 3 s window with repeated shouting or two concurrent voices |
| `ABUSIVE_LANGUAGE` | Vosk keyword spotting, closed grammar from `config/abusive_keywords.txt` (masked in payloads) |

Left/right labels are from the driver's point of view, as in the reference: a non-mirrored camera facing the driver shows the driver's left on the image right. Set `realtime.mirror: true` for a mirrored feed.

## Models

| Role | Model | Runtime |
|---|---|---|
| Driver state (8 heads) | `edge_driver_model_distilled` — MobileNetV3, 224 px, from the reference repo | TensorRT FP16 engine |
| Face box + eye points | YuNet 2023mar | OpenCV DNN |
| Landmarks | MediaPipe Face Mesh 478 pts with iris (preferred), else Facemark LBF 68 pts | CPU |
| Phone / cup | NanoDet-Plus 416 (COCO) | TensorRT FP16 / ORT |
| Hands | MediaPipe palm detector 192 (ONNX) | TensorRT FP16 / ORT |
| Driver ID | SFace 128-d + enrolled roster | OpenCV DNN |
| Scene VLM | **Qwen3.5-0.8B Q4_K_M** plus its F16 vision projector, via llama.cpp. Any OpenAI-compatible VLM can replace it | GPU, separate process |
| Audio events | Cochl.Sense Edge SDK → YAMNet TFLite → built-in DSP classifier (first available) | CPU |
| Keywords | Vosk small-en (40 MB) | CPU |

**Where this departs from the brief, and why:**
- **VLM model.** The default is Qwen3.5-0.8B at Q4_K_M (about 533 MB) with the F16 vision projector (about 205 MB). It is the same family as the 0.8B model used for short scene answers, and it leaves room on the 8 GB Orin beside the detector. Qwen2.5-VL-3B still works by passing `--vlm-repo` and `--vlm-quant` to `tools/download_models.py`. Another OpenAI-compatible server works by changing `vlm.endpoint`.
- **No torch.** Nothing in the DMS needs it: every model runs through TensorRT or ONNX Runtime, using JetPack's own TensorRT.
- **Seatbelt is a heuristic.** It looks for a long diagonal edge across the torso; the face-crop model cannot see the belt. It reports *Unknown* rather than *Off* when the chest is out of frame.

## Install and run on the Jetson (no Docker)

```bash
git clone <this repo> ~/orin-dms && cd ~/orin-dms
./setup_jetson.sh --full        # apt packages, .venv, Python deps, all models, TRT engines, llama.cpp
sudo nvpmodel -m 0              # 15 W profile
python3 tools/enroll_driver.py --id 4827 --name "Emily Carter" --age 28 --camera csi:0
./run.sh                        # starts the VLM server (if downloaded) and the dashboard
```

`./setup_jetson.sh` installs `requirements.txt` with `constraints.txt` (numpy 1.26 / protobuf 4.x, so mediapipe keeps working). It then installs `onnxruntime-gpu` from the Jetson AI Lab index (whatever version it carries for JP6/cu126), falling back to CPU `onnxruntime==1.23.2` from PyPI. Optional packages are installed one at a time, so a missing wheel never blocks the rest. Entry-point scripts switch into `.venv` by themselves, so a plain `python3 main.py` works.

Troubleshooting:
- `python3 tools/check_env.py` lists required and optional modules, the ONNX Runtime providers, GStreamer access and models.
- `.venv/bin/python -m dms.capture csi:0` opens one CSI camera for 5 s and prints the FPS and the capture path used.

`./setup_jetson.sh` without `--full` installs only the required models and skips the VLM. The detector, audio and MQTT still run; start with `--no-vlm`.

### Cameras: CSI or USB

| Source | Meaning |
|---|---|
| `auto:N` (default) | CSI sensor N if the kernel has it registered (a `vi-output` node in `/sys/class/video4linux`), otherwise the Nth USB camera. Also falls back when a registered CSI sensor delivers no frames. |
| `csi:N` | CSI only. Falls back to USB unless `cameras.*.usb_fallback: false`. |
| `usb:N` | Nth USB (UVC) camera, found by name in sysfs (UVC creates two `/dev/video` nodes per camera). |
| `v4l2:N` | A specific `/dev/videoN`. |

`--camera auto|csi|usb` switches both cameras (driver = 0, passenger = 1); `--driver/--passenger` override them individually. `python3 tools/check_env.py` prints the cameras it finds and what `auto` will use. A source that isn't present logs one diagnosis (including the `jetson-io` steps for CSI) and is re-checked every 30 s, so a camera plugged in later is picked up.

### Models

All models are downloaded into `models/`; none are committed to the repo.

```bash
python3 tools/download_models.py            # required, ~115 MB, SHA-256 pinned
python3 tools/download_models.py --optional # MediaPipe face landmarker, YAMNet, Vosk (~50 MB)
python3 tools/download_models.py --vlm      # Qwen3.5-0.8B Q4_K_M GGUF + vision projector (~0.7 GB); setup_jetson.sh does this automatically
python3 tools/download_models.py --list     # status;  --verify re-checks checksums
tools/build_engines.sh                      # TensorRT FP16 engines (on the Jetson, ~5 min)
```

Downloads resume after interruption, retry, and use `HTTPS_PROXY`; `HF_TOKEN` and `HF_ENDPOINT` are honoured for Hugging Face. Use `--vlm-repo` / `--vlm-quant` to choose another GGUF VLM.

| File | Source | Size |
|---|---|---|
| `edge_driver_model_distilled.onnx` | DMS-CMS reference repo | 13 MB |
| `face_detection_yunet_2023mar.onnx`, `face_recognition_sface_2021dec.onnx`, `object_detection_nanodet_2022nov.onnx`, `palm_detection_mediapipe_2023feb.onnx` | OpenCV Zoo | 47 MB |
| `lbfmodel.yaml` | OpenCV GSoC 2017 | 54 MB |
| `yamnet_class_map.csv`, `yamnet.tflite` | TensorFlow / MediaPipe | 4 MB |
| `face_landmarker.task` | MediaPipe | 4 MB |
| `vosk-model-small-en-us-0.15/` | alphacephei | 40 MB |
| `vlm/*.gguf` | `unsloth/Qwen3.5-0.8B-GGUF` (Q4_K_M + mmproj-F16) | ~0.7 GB |

### VLM server

`tools/vlm_server.sh` builds llama.cpp with CUDA for Orin (sm_87) into `third_party/llama.cpp` on first run. It then serves `models/vlm` on `127.0.0.1:8080` through the OpenAI API, and `run.sh` starts it automatically. To use vLLM instead (with the spec's `max_model_len=512, gpu_memory_utilization=0.35`), run `VLM_BACKEND=vllm tools/vlm_server.sh`.

### Autostart

```bash
sudo cp -r . /opt/dms && sudo cp deploy/*.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now dms-vlm dms
```

### Flags and secrets

Useful flags: `--headless`, `--fullscreen`, `--no-vlm`, `--no-audio`, `--no-mqtt`, `--driver/--passenger <csi:N|v4l2:N|file|image>`, `--audio-device <name|index|file.wav>`, `--mqtt-host`, `--device-id`. Secrets come from the environment (or `/etc/default/dms` for systemd): `DMS_MQTT_USERNAME`, `DMS_MQTT_PASSWORD`, `COCHL_PROJECT_KEY`. Dashboard keys: `1` driver view, `2` cabin view, `3` split view, Tab cycles them, F11 toggles fullscreen, Ctrl+Q quits. Seat zones live in `cms.seats` in `config/dms.yaml`. Passenger belts stay Unknown until a cabin belt check exists. `CHILD_PRESENT` means a child is visible with no adult face; it is not a left-behind alarm.

### Bench run on a workstation (no Jetson, CPU only)

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt
python3 tools/download_models.py
python3 tools/mock_vlm_server.py --latency 1.3 --script "SAFE: driver attentive." \
    --script "UNSAFE: the driver is holding a phone while driving." &
mosquitto -p 1883 &
python3 main.py --driver assets/demo/driver_reference.jpg --passenger assets/demo/cabin_reference.jpg \
    --audio-device tests/data/cabin_events.wav
```

## MQTT

| Topic | QoS | Payload |
|---|---|---|
| `dms/{device_id}/alert/vision` | 1 | `id, device_id, type, category, severity, confidence, timestamp (ISO-8601 UTC), epoch_ms, description, reason, snapshot_path, details{yaw, pitch, ear, gaze, driver_id}` |
| `dms/{device_id}/alert/audio` | 1 | the same fields with `audio_path` (1 s WAV) and `details{detection_latency_ms, backend}` |
| `dms/{device_id}/status` | 0, retained | heartbeat every 30 s: fps, model/pipeline ms, cameras, VLM status, GPU temperature, memory, alert counts, queued alerts. The Last Will publishes `offline`. |
| `cms/{device_id}/alert` | 1 | cabin alerts (`CHILD_PRESENT`, `OBJECT_ON_SEAT`, `PASSENGER_AGGRESSIVE`, `OCCUPANCY_CHANGE`) with `details.seats` and `details.objects` |
| `cms/{device_id}/status` | 0, retained | occupants, per-seat labels (belt is `Unknown`), cabin camera fps |

It uses MQTT 5 (paho 2.x) and falls back to 3.1.1 automatically for brokers that reject v5. TLS and username/password are supported.

## Validation

`tools/validate.py` runs the whole system headless and checks the spec's criteria (FPS, VLM latency, audio latency, MQTT success, peak memory), then writes `data/validation_report.json`. Run it on the Orin for the numbers that count:

```bash
python3 tools/validate.py --duration 120
```

The figures below come from the development machine — a 2-vCPU x86 VM, CPU-only ONNX Runtime, no GPU — with the reference frame as the driver feed, the mock VLM (fixed 1.3 s latency) and synthetic cabin audio. **They have not been measured on an Orin.**

| Criterion | Target | Dev VM result |
|---|---|---|
| Real-time detector FPS (p5), headless | ≥ 15 | 23.5 |
| Real-time FPS with the dashboard process also running | ≥ 15 | 14–17 (both processes share 2 vCPUs) |
| VLM latency | ≤ 3 s | not measured (mock server) |
| Audio alert latency (scream onset → alert) | < 500 ms | ~200 ms |
| MQTT success / outbox drained after 31 alerts queued offline | ≥ 99 % | 100 % / 0, in order |
| Peak memory (DMS only, no VLM) | < 7.5 GB | ~1.0 GB |

`pytest -q tests` runs 31 tests: head-pose sign conventions on synthetic projections, EAR, VLM answer parsing, audio scream latency and altercation detection, offline store-and-forward against a real Mosquitto broker, and the full driver pipeline on the reference frame (phone, gaze away, seatbelt, known driver, alerts).

## Not yet exercised end to end

- **Jetson-only paths.** CSI capture through `nvarguscamerasrc`, the native TensorRT engine runner, the ORT TensorRT provider and the Jetson sysfs telemetry are written for JetPack 6 but have not been run on hardware. The same is true of `setup_jetson.sh` and the llama.cpp CUDA build.
- **Optional back-ends.** The MediaPipe Face Mesh path, the YAMNet TFLite path and Vosk could not be tested because their model hosts were blocked on the build machine. The tested paths are LBF landmarks and the DSP audio classifier. The Cochl.Sense adapter follows the Edge SDK's documented Python API (`sense.session`, `create_stream_processor`, `ResultListener`), but tag names vary by project; map them in `audio.cochl.scream_tags`.
- **The 8-head model** comes from the reference repository and its accuracy has not been re-validated here. On the reference frame it labels the emotion *Happy* and gives a weak *Cigarette* score (below the alert threshold).
# DMS_v0
# DMS_v0
# DMS_v0
# DMS_v0
# DMS_v0
