#!/usr/bin/env bash
# Build TensorRT FP16 engines on the target Jetson (engines are tied to the exact
# TensorRT version + GPU, so they are never shipped prebuilt).
#   edge_driver_model_distilled.onnx    -> edge_driver_model_distilled_fp16.engine (8-head MobileNetV3)
#   object_detection_nanodet_2022nov.onnx -> .engine  (phone / cup detector)
#   palm_detection_mediapipe_2023feb.onnx -> .engine  (hand detector)
# Any model without an engine still runs through ONNX Runtime's TensorRT/CUDA provider.
set -euo pipefail
cd "$(dirname "$0")/../models"
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
[[ -x "$TRTEXEC" ]] || TRTEXEC=$(command -v trtexec || true)
[[ -n "$TRTEXEC" ]] || { echo "trtexec not found (JetPack: /usr/src/tensorrt/bin/trtexec)"; exit 1; }

build() {  # onnx engine
  if [[ -s "$2" && "$2" -nt "$1" ]]; then echo "ok    $2"; return; fi
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
