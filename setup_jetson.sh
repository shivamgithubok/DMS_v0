#!/usr/bin/env bash
# One-time bare-metal setup on a Jetson Orin Nano (JetPack 6.x, L4T r36). No Docker.
#
#   ./setup_jetson.sh            # system packages, .venv, Python deps, required models, Qwen3.5-0.8B VLM, TRT engines
#   ./setup_jetson.sh --full     # + optional models (MediaPipe landmarker, YAMNet, Vosk). A Jetson also builds llama.cpp.
#
# Re-running is safe: every step skips work that is already done.
# JETSON_PIP_INDEX overrides the Jetson AI Lab wheel index used for onnxruntime-gpu.
set -euo pipefail
cd "$(dirname "$0")"
FULL=0; [[ "${1:-}" == "--full" ]] && FULL=1
JETSON_INDEX=${JETSON_PIP_INDEX:-https://pypi.jetson-ai-lab.io/jp6/cu126}
log() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*"; }
IS_JETSON=0; [[ -f /etc/nv_tegra_release ]] && IS_JETSON=1

# ------------------------------------------------------------------ platform checks
if [[ $IS_JETSON -eq 1 ]]; then
  log "Jetson: $(head -1 /etc/nv_tegra_release)"
  grep -qE "R3[6-9]" /etc/nv_tegra_release || warn "written for JetPack 6+ (L4T R36 or newer); this board reports otherwise"
else
  warn "not a Jetson: continuing in workstation mode (ONNX Runtime CPU, no CSI cameras)"
fi
python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python >= 3.10 required"'

# ------------------------------------------------------------------ system packages
log "system packages (sudo)"
sudo apt-get update -q
sudo apt-get install -y --no-install-recommends \
  python3-venv python3-pip python3-dev python3-gi gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  libportaudio2 alsa-utils curl ca-certificates git cmake build-essential libcurl4-openssl-dev \
  libgl1 libegl1 libfontconfig1 libdbus-1-3 libxkbcommon0 libxkbcommon-x11-0 \
  libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
  libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 mosquitto-clients

# TensorRT python bindings ship with JetPack (apt), not pip
if [[ $IS_JETSON -eq 1 ]] && ! python3 -c "import tensorrt" 2>/dev/null; then
  sudo apt-get install -y python3-libnvinfer || warn "python3-libnvinfer not installed; ONNX Runtime will be used instead of native engines"
fi

# ------------------------------------------------------------------ virtualenv
# --system-site-packages keeps JetPack's tensorrt and python3-gi (CSI capture) visible.
log "python virtualenv (.venv)"
[[ -d .venv ]] || python3 -m venv --system-site-packages .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip wheel

# ------------------------------------------------------------------ python packages
log "python dependencies"
if [[ $IS_JETSON -eq 1 ]]; then
  pip install -c constraints.txt -r requirements.txt

  # ONNX Runtime: GPU build from the Jetson AI Lab index (whichever version it carries for
  # JP6/cu126), else the CPU build from PyPI. Never fails the setup.
  if ! python -c "import onnxruntime as o, sys; sys.exit('CUDAExecutionProvider' not in o.get_available_providers())" 2>/dev/null; then
    pip uninstall -y -q onnxruntime onnxruntime-gpu 2>/dev/null || true
    if pip install -c constraints.txt --extra-index-url "$JETSON_INDEX" onnxruntime-gpu; then
      echo "onnxruntime-gpu installed from $JETSON_INDEX"
    else
      warn "onnxruntime-gpu unavailable from $JETSON_INDEX; installing CPU onnxruntime (native TensorRT engines are still used)"
      pip install -c constraints.txt onnxruntime==1.23.2
    fi
  fi

  # optional packages one at a time, so one missing wheel cannot block the rest
  while read -r pkg; do
    [[ -z "$pkg" ]] && continue
    pip install -c constraints.txt "$pkg" || warn "optional package $pkg not installed"
  done < <(sed -e 's/#.*//' -e 's/[[:space:]]*$//' requirements-optional.txt | grep -v '^$' || true)

  # mediapipe depends on the GUI opencv build; install it without deps so the single
  # headless opencv-contrib build stays the only cv2.
  if pip install --no-deps mediapipe==0.10.18 && \
     pip install -c constraints.txt "absl-py>=1.4" "attrs>=19.1" "matplotlib<3.9" sentencepiece; then
    echo "mediapipe installed"
  else
    warn "mediapipe not installed: landmarks fall back to Facemark LBF"
  fi
else
  pip install -r requirements-dev.txt
fi

if [[ ! -f config/dms.yaml ]]; then
  cp config/dms.example.yaml config/dms.yaml
  log "wrote config/dms.yaml from example — set mqtt.host (VPS IP) and camera sources"
fi

log "checking the environment"
python tools/check_env.py

# ------------------------------------------------------------------ models
log "models (required detectors + Qwen3.5-0.8B VLM)"
if [[ $FULL -eq 1 ]]; then
  python tools/download_models.py --all || warn "some optional downloads failed (see above)"
else
  python tools/download_models.py --vlm || warn "a model download failed (see above)"
fi
if [[ ! -f models/vlm/vlm.json ]]; then
  warn "VLM weights are missing. Cabin scene understanding stays off until: python3 tools/download_models.py --vlm"
fi

if [[ $IS_JETSON -eq 1 ]]; then
  log "TensorRT FP16 engines"
  tools/build_engines.sh || warn "engine build failed; ONNX Runtime will be used"
  log "llama.cpp (CUDA) for the VLM server"
  tools/vlm_server.sh --build-only || warn "llama.cpp build failed; run with --no-vlm"
fi

# ------------------------------------------------------------------ permissions + power
groups | grep -qw audio || { sudo usermod -aG audio "$USER"; warn "added $USER to 'audio' (log out/in)"; }
groups | grep -qw video || { sudo usermod -aG video "$USER"; warn "added $USER to 'video' (log out/in)"; }
if command -v nvpmodel >/dev/null; then
  echo "power mode: $(sudo nvpmodel -q 2>/dev/null | head -1)  (15 W profile: sudo nvpmodel -m 0)"
fi

log "done"
cat <<EOF
Next (scripts use .venv automatically):
  .venv/bin/python -m dms.capture csi:0        # 5 s camera check, prints fps + backend
  python3 tools/enroll_driver.py --id 4827 --name "Emily Carter" --age 28 --camera csi:0
  ./run.sh                                     # VLM server (if downloaded) + DMS dashboard
  ./run.sh --headless                          # service mode
  python3 tools/validate.py --duration 120     # spec validation report
EOF
