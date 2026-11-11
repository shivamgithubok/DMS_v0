#!/usr/bin/env bash
# One-time bare-metal setup. JetPack, CUDA, and Python are read from this board.
# A different Orin image (JetPack 6 or newer, Python 3.10+) uses its own wheels
# and its own TensorRT. No Docker.
#
#   ./setup_jetson.sh            # apt, .venv, Python deps, models, VLM, TRT engines
#   ./setup_jetson.sh --full     # also MediaPipe landmarker, YAMNet, Vosk
#
# Re-running is safe: every step skips work that is already done.
# JETSON_PIP_INDEX, if set, is the only Jetson AI Lab index tried for onnxruntime-gpu.
set -euo pipefail
cd "$(dirname "$0")"
FULL=0; [[ "${1:-}" == "--full" ]] && FULL=1
log() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*"; }
IS_JETSON=0; [[ -f /etc/nv_tegra_release ]] && IS_JETSON=1

apt_one() {
  sudo apt-get install -y --no-install-recommends "$1" || warn "apt package not on this image: $1"
}

# ------------------------------------------------------------------ platform checks
L4T_MAJOR=""
CUDA_VER=""
if [[ $IS_JETSON -eq 1 ]]; then
  log "Jetson: $(head -1 /etc/nv_tegra_release)"
  L4T_MAJOR=$(sed -n 's/.*[[:space:]]R\([0-9][0-9]*\).*/\1/p' /etc/nv_tegra_release | head -1)
  if [[ -n "$L4T_MAJOR" && "$L4T_MAJOR" -lt 36 ]]; then
    warn "L4T R${L4T_MAJOR} is older than R36 (JetPack 6). Setup continues; some wheels may be missing."
  fi
else
  warn "not a Jetson: continuing in workstation mode (ONNX Runtime CPU, no CSI cameras)"
fi
if [[ -x /usr/local/cuda/bin/nvcc ]]; then
  CUDA_VER=$(/usr/local/cuda/bin/nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
fi
python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python >= 3.10 required"'
log "python $(python3 -c 'import sys; print("%d.%d"%sys.version_info[:2])')  L4T R${L4T_MAJOR:-n/a}  CUDA ${CUDA_VER:-not found}"

# ------------------------------------------------------------------ system packages
log "system packages (sudo)"
sudo apt-get update -q || warn "apt-get update failed; installing from the caches already on this board"
APT_PKGS=(
  python3-venv python3-pip python3-dev python3-gi gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good
  libportaudio2 alsa-utils curl ca-certificates git cmake build-essential libcurl4-openssl-dev
  libgl1 libegl1 libfontconfig1 libdbus-1-3 libxkbcommon0 libxkbcommon-x11-0
  libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0
  libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 mosquitto-clients
)
if ! sudo apt-get install -y --no-install-recommends "${APT_PKGS[@]}"; then
  warn "this image is missing some apt names; installing the ones it has"
  for pkg in "${APT_PKGS[@]}"; do apt_one "$pkg"; done
fi

# TensorRT python bindings ship with JetPack (apt), not pip. The package name
# is stable; the Python it binds to is the image's python3.
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

# pip_flex SPEC FALLBACK — SPEC is preferred; FALLBACK is used when this Python
# has no wheel for SPEC. A missing package is a warning, not a failed setup.
pip_flex() {
  local spec="$1" fallback="${2:-}"
  if pip install -c constraints.txt "$spec"; then
    return 0
  fi
  if [[ -n "$fallback" ]]; then
    warn "no wheel for $spec on this Python; trying $fallback"
    if pip install -c constraints.txt "$fallback"; then
      return 0
    fi
  fi
  warn "not installed: $spec"
  return 1
}

# Jetson AI Lab indexes are per JetPack and CUDA. Try this board's pair first.
# There is often no wheel for a newer L4T (R39 / CUDA 13); native TensorRT
# engines still run the detectors.
ort_indexes() {
  if [[ -n "${JETSON_PIP_INDEX:-}" ]]; then
    printf '%s\n' "$JETSON_PIP_INDEX"
    return
  fi
  local jp="" cu=""
  case "$L4T_MAJOR" in
    35) jp=jp5 ;;
    36) jp=jp6 ;;
    38|39) jp=jp7 ;;
  esac
  if [[ -n "$CUDA_VER" ]]; then
    cu=$(awk -F. '{printf "cu%d%d", $1, $2}' <<<"$CUDA_VER")
  fi
  [[ -n "$jp" && -n "$cu" ]] && printf 'https://pypi.jetson-ai-lab.io/%s/%s\n' "$jp" "$cu"
  # Same JetPack, neighbouring CUDA tags. pip skips an index with no matching wheel.
  if [[ "$jp" == jp6 ]]; then
    printf '%s\n' \
      https://pypi.jetson-ai-lab.io/jp6/cu126 \
      https://pypi.jetson-ai-lab.io/jp6/cu128 \
      https://pypi.jetson-ai-lab.io/jp6/cu129
  fi
}

install_onnxruntime() {
  if python -c "import onnxruntime as o, sys; sys.exit(0 if ('CUDAExecutionProvider' in o.get_available_providers() or 'TensorrtExecutionProvider' in o.get_available_providers()) else 1)" 2>/dev/null; then
    python -c "import onnxruntime as o; print('onnxruntime', o.__version__, o.get_available_providers())"
    return 0
  fi
  pip uninstall -y -q onnxruntime onnxruntime-gpu 2>/dev/null || true
  local idx
  while read -r idx; do
    [[ -z "$idx" ]] && continue
    echo "trying onnxruntime-gpu from $idx"
    if pip install -c constraints.txt --extra-index-url "$idx" onnxruntime-gpu \
       && python -c "import onnxruntime" 2>/dev/null; then
      echo "onnxruntime-gpu installed from $idx"
      return 0
    fi
    pip uninstall -y -q onnxruntime-gpu 2>/dev/null || true
  done < <(ort_indexes | awk '!seen[$0]++')
  warn "no onnxruntime-gpu wheel for this JetPack/Python. Installing the CPU build from PyPI."
  warn "Detectors still use TensorRT engines built on this board when trtexec succeeds."
  pip_flex 'onnxruntime>=1.18' || true
}

# ------------------------------------------------------------------ python packages
log "python dependencies"
if [[ $IS_JETSON -eq 1 ]]; then
  if ! pip install -c constraints.txt -r requirements.txt; then
    warn "a pinned package has no wheel for this Python; installing the closest match for each"
    pip_flex 'numpy==1.26.4' 'numpy>=1.26,<2' || true
    pip_flex 'opencv-contrib-python-headless==4.10.0.84' 'opencv-contrib-python-headless>=4.10' || true
    pip_flex 'PyYAML==6.0.2' 'PyYAML>=6' || true
    pip_flex 'paho-mqtt==2.1.0' 'paho-mqtt>=2.1' || true
    pip_flex 'livekit>=0.17.6' || true
    pip_flex 'sounddevice==0.5.1' 'sounddevice>=0.4.7' || true
    pip_flex 'PySide6-Essentials==6.8.0.2' 'PySide6-Essentials>=6.6' || true
    pip_flex 'protobuf>=4.25.8,<5' 'protobuf>=4.25' || true
    pip_flex 'flatbuffers>=2.0' || true
  fi

  install_onnxruntime

  # optional packages one at a time, so one missing wheel cannot block the rest
  while read -r pkg; do
    [[ -z "$pkg" ]] && continue
    name=${pkg%%[<>=!]*}
    pip_flex "$pkg" "$name" || true
  done < <(sed -e 's/#.*//' -e 's/[[:space:]]*$//' requirements-optional.txt | grep -v '^$' || true)

  # mediapipe depends on the GUI opencv build; install it without deps so the single
  # headless opencv-contrib build stays the only cv2. The pin is dropped if this
  # Python has no 0.10.18 wheel (landmarks then still try the newer release).
  if { pip install --no-deps 'mediapipe==0.10.18' || pip install --no-deps 'mediapipe'; } && \
     pip install -c constraints.txt "absl-py>=1.4" "attrs>=19.1" "matplotlib<3.9" sentencepiece; then
    echo "mediapipe installed"
  else
    warn "mediapipe not installed: landmarks fall back to Facemark LBF"
  fi
else
  pip install -r requirements-dev.txt
fi

# requirements.txt already pins this. Install it again so a .venv created before
# that line still gets the SDK (`from livekit import rtc`).
log "LiveKit SDK"
pip install -c constraints.txt 'livekit>=0.17.6'

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
  echo "power mode: $(sudo nvpmodel -q 2>/dev/null | head -1)"
  echo "Left unchanged. Mode numbers differ by kit (15W, 25W, MAXN). sudo nvpmodel -q lists them."
fi

log "done"
cat <<EOF
Next (scripts use .venv automatically):
  python3 tools/check_env.py                   # modules, cameras, models on this board
  .venv/bin/python -m dms.capture auto:0       # 5 s camera check, prints fps + backend
  python3 tools/enroll_driver.py --id <id> --name "<name>" --camera auto:0
  ./run.sh                                     # VLM server (if downloaded) + DMS dashboard
  ./run.sh --headless                          # service mode
EOF
