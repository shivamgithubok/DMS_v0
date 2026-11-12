#!/usr/bin/env bash
# Publish Orin-shaped MQTT messages, then hit the local FastAPI.
# Usage: API=http://127.0.0.1:8000 ./scripts/e2e_local.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
API="${API:-http://127.0.0.1:8000}"
BROKER="${MQTT_HOST:-127.0.0.1}"
PORT="${MQTT_PORT:-1883}"

need() { command -v "$1" >/dev/null || { echo "missing $1"; exit 1; }; }
need curl
need python3
PY="${ROOT}/.venv/bin/python"
if [ ! -x "$PY" ]; then PY="python3"; fi

echo "== health"
curl -sf "$API/health" | python3 -m json.tool

echo "== MQTT: cms status + distraction + scream"
MQTT_HOST="$BROKER" MQTT_PORT="$PORT" "$PY" "$ROOT/scripts/publish_samples.py"

sleep 1

echo "== ride after MQTT"
curl -sf "$API/v1/rides/ride-1842" | python3 -m json.tool

echo "== alerts after MQTT"
curl -sf "$API/v1/rides/ride-1842/alerts" | python3 -m json.tool

echo "== phone SOS POST"
curl -sf -X POST "$API/v1/rides/ride-1842/sos" \
  -H "content-type: application/json" \
  -d '{"passenger":"Meera Shah","latitude":12.9716,"longitude":77.5946,"source":"phone"}' \
  | python3 -m json.tool

echo "== cabin"
curl -sf "$API/v1/rides/ride-1842/cabin" | python3 -m json.tool

echo
echo "Local path OK: MQTT alert -> API DB, and phone SOS -> API."
