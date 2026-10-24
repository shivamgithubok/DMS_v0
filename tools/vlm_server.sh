#!/usr/bin/env bash
# Native VLM server for the DMS (no Docker): llama.cpp `llama-server` built with CUDA for
# Orin (sm_87), serving the GGUF + vision projector from models/vlm/ on 127.0.0.1:8080
# with the OpenAI chat-completions API that dms/vlm_worker.py calls.
#
#   tools/vlm_server.sh               # build if needed, then run in the foreground
#   tools/vlm_server.sh --build-only  # just build third_party/llama.cpp
#   VLM_PORT=8081 VLM_CTX=2048 tools/vlm_server.sh
#
# Alternative: vLLM. If `vllm` is importable (Jetson AI Lab wheel), VLM_BACKEND=vllm runs
#   vllm serve Qwen/Qwen2.5-VL-3B-Instruct-AWQ --max-model-len 512 --gpu-memory-utilization 0.35
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
SRC="$ROOT/third_party/llama.cpp"
BIN="$SRC/build/bin/llama-server"
PORT=${VLM_PORT:-8080}
CTX=${VLM_CTX:-2048}
BACKEND=${VLM_BACKEND:-llamacpp}
REF=${LLAMA_CPP_REF:-master}

build() {
  if [[ -x "$BIN" ]]; then echo "llama-server present: $BIN"; return; fi
  mkdir -p third_party data
  # A second ./run.sh must not start another nvcc against the same build tree.
  exec 9>"$ROOT/data/vlm_build.lock"
  if ! flock -n 9; then
    echo "llama.cpp build already running; waiting for it to finish"
    flock 9
    if [[ -x "$BIN" ]]; then echo "llama-server present: $BIN"; return; fi
  fi
  while pgrep -f "cmake --build ${SRC}/build" >/dev/null 2>&1; do
    echo "llama.cpp build already running; waiting"
    sleep 20
    if [[ -x "$BIN" ]]; then echo "llama-server present: $BIN"; return; fi
  done
  [[ -d "$SRC/.git" ]] || git clone --depth 1 --branch "$REF" https://github.com/ggml-org/llama.cpp "$SRC"
  local cuda=OFF arch=""
  if command -v nvcc >/dev/null || [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export PATH=/usr/local/cuda/bin:$PATH
    cuda=ON; arch="-DCMAKE_CUDA_ARCHITECTURES=87"
  else
    echo "nvcc not found: building CPU-only llama.cpp (too slow for 0.4 Hz on the Orin)"
  fi
  cmake -S "$SRC" -B "$SRC/build" -DGGML_CUDA=$cuda $arch -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
  local jobs
  jobs=$(nproc)
  # nvcc needs about 1 GB per job. An 8 GB Orin with the dashboard up cannot spare six of them.
  local avail_kb
  avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  if [[ ${avail_kb:-0} -lt 2000000 ]]; then
    jobs=1
  elif [[ ${avail_kb:-0} -lt 4000000 && "$jobs" -gt 2 ]]; then
    jobs=2
  fi
  if [[ "$jobs" -lt $(nproc) ]]; then
    echo "low memory (${avail_kb} kB available); building llama-server with -j${jobs}"
  fi
  cmake --build "$SRC/build" --config Release -j"$jobs" --target llama-server
  echo "built $BIN"
}

if [[ "${1:-}" == "--build-only" ]]; then build; exit 0; fi

if [[ "$BACKEND" == "vllm" ]]; then
  exec vllm serve "${VLLM_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct-AWQ}" --host 127.0.0.1 --port "$PORT" \
       --max-model-len 512 --gpu-memory-utilization 0.35 --max-num-seqs 1 \
       --limit-mm-per-prompt image=1 --enforce-eager
fi

build
CFG="$ROOT/models/vlm/vlm.json"
if [[ ! -f "$CFG" ]]; then
  echo "no VLM in models/vlm - run: python3 tools/download_models.py --vlm" >&2
  exit 1
fi
MODEL=$(python3 -c "import json;print(json.load(open('$CFG'))['model'])")
MMPROJ=$(python3 -c "import json;print(json.load(open('$CFG'))['mmproj'])")
echo "serving $MODEL (+ $MMPROJ) on 127.0.0.1:$PORT"
# -ngl 99: all layers on the GPU; -np 1: one request at a time (0.4 Hz needs no batching);
# --threads 2: leave the CPU cores to the real-time detector.
exec nice -n 10 "$BIN" -m "$ROOT/models/vlm/$MODEL" --mmproj "$ROOT/models/vlm/$MMPROJ" \
     --host 127.0.0.1 --port "$PORT" -ngl 99 -c "$CTX" -np 1 --threads 2 --alias vlm \
     --jinja --chat-template-kwargs '{"enable_thinking":false}'
