#!/usr/bin/env bash
# Shell wrapper around tools/download_models.py (kept for scripts that call it).
#   tools/fetch_models.sh                 # required models
#   tools/fetch_models.sh --with-optional # + MediaPipe landmarker, YAMNet, Vosk
#   tools/fetch_models.sh --all           # + VLM
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in
  --with-optional) exec python3 tools/download_models.py --optional ;;
  "") exec python3 tools/download_models.py ;;
  *) exec python3 tools/download_models.py "$@" ;;
esac
