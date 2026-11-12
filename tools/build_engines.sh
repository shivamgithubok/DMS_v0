#!/usr/bin/env bash
# Build TensorRT FP16 engines on the target Jetson (engines are tied to the exact
# TensorRT version + GPU, so they are never shipped prebuilt).
#   edge_driver_model_distilled.onnx    -> edge_driver_model_distilled_fp16.engine (8-head MobileNetV3)
#   object_detection_nanodet_2022nov.onnx -> .engine  (phone / cup detector)
#   palm_detection_mediapipe_2023feb.onnx -> .engine  (hand detector)
# Any model without an engine still runs through ONNX Runtime's TensorRT/CUDA provider.
# A matching .tensorrt-stamp skips trtexec. Setup rebuilds when the JetPack version changes.
# systemd does not call this script on reboot.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$ROOT/models}"
cd "$MODELS_DIR"
STAMP_FILE=".tensorrt-stamp"

trt_version() {
  if [[ -n "${TRT_VERSION:-}" ]]; then
    printf '%s\n' "$TRT_VERSION"
    return
  fi
  local bin="${TRTEXEC:-}"
  if [[ -z "$bin" && -x /usr/src/tensorrt/bin/trtexec ]]; then
    bin=/usr/src/tensorrt/bin/trtexec
  fi
  if [[ -z "$bin" ]]; then
    bin=$(command -v trtexec || true)
  fi
  if [[ -n "$bin" && -x "$bin" ]]; then
    "$bin" --version 2>/dev/null | head -1 || true
    return
  fi
  dpkg-query -W -f '${Version}\n' tensorrt 2>/dev/null || true
}

engines_current() {
  local pairs=(
    edge_driver_model_distilled.onnx edge_driver_model_distilled_fp16.engine
    object_detection_nanodet_2022nov.onnx object_detection_nanodet_2022nov.engine
    palm_detection_mediapipe_2023feb.onnx palm_detection_mediapipe_2023feb.engine
  )
  local i=0 onnx engine
  while [[ $i -lt ${#pairs[@]} ]]; do
    onnx="${pairs[$i]}"
    engine="${pairs[$((i + 1))]}"
    i=$((i + 2))
    [[ -s "$engine" ]] || return 1
    if [[ -f "$onnx" && "$onnx" -nt "$engine" ]]; then
      return 1
    fi
  done
}

VERSION="$(trt_version | tr -d '\r' | head -1)"
STAMP=""
if [[ -f "$STAMP_FILE" ]]; then
  STAMP="$(tr -d '\r' < "$STAMP_FILE" | head -1)"
fi
if [[ -n "$VERSION" && "$STAMP" == "$VERSION" ]] && engines_current; then
  echo "engines match ${VERSION}; skip rebuild"
  exit 0
fi

TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
[[ -x "$TRTEXEC" ]] || TRTEXEC=$(command -v trtexec || true)
[[ -n "$TRTEXEC" ]] || { echo "trtexec not found (JetPack: /usr/src/tensorrt/bin/trtexec)"; exit 1; }
FORCE=0
if [[ -z "$VERSION" || "$STAMP" != "$VERSION" ]]; then
  FORCE=1
fi

build() {  # onnx engine
  if [[ $FORCE -eq 0 && -s "$2" && "$2" -nt "$1" ]]; then echo "ok    $2"; return; fi
  echo "build $2"
  "$TRTEXEC" --onnx="$1" --saveEngine="$2" --fp16 --memPoolSize=workspace:512M \
             --skipInference --verbose=false > "$2.log" 2>&1 || { tail -20 "$2.log"; exit 1; }
}

build edge_driver_model_distilled.onnx edge_driver_model_distilled_fp16.engine
build object_detection_nanodet_2022nov.onnx object_detection_nanodet_2022nov.engine
build palm_detection_mediapipe_2023feb.onnx palm_detection_mediapipe_2023feb.engine

for e in *.engine; do
  "$TRTEXEC" --loadEngine="$e" --warmUp=200 --duration=3 2>/dev/null | grep -E "GPU Compute Time: min" | sed "s/^/$e: /" || true
done
if [[ -z "$VERSION" ]]; then
  VERSION="$("$TRTEXEC" --version 2>/dev/null | tr -d '\r' | head -1 || true)"
fi
if [[ -n "$VERSION" ]]; then
  printf '%s\n' "$VERSION" > "$STAMP_FILE"
fi
