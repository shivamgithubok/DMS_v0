#!/usr/bin/env bash
# Start the DMS (and the local VLM server when its model is downloaded). No Docker.
#   ./run.sh                      # dashboard
#   ./run.sh --headless           # no UI
#   ./run.sh --no-vlm ...         # any main.py flag is passed through
set -euo pipefail
cd "$(dirname "$0")"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
mkdir -p data
export DISPLAY=${DISPLAY:-:0}

VLM_PID=""
want_vlm=1
for a in "$@"; do [[ "$a" == "--no-vlm" ]] && want_vlm=0; done
if [[ $want_vlm -eq 1 && -f models/vlm/vlm.json ]] && ! curl -sf http://127.0.0.1:${VLM_PORT:-8080}/v1/models >/dev/null; then
  echo "starting VLM server (log: data/vlm_server.log)"
  tools/vlm_server.sh > data/vlm_server.log 2>&1 &
  VLM_PID=$!
fi
cleanup() { [[ -n "$VLM_PID" ]] && kill "$VLM_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

python3 main.py "$@"
