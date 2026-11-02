#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p runtime/logs
"$ROOT/scripts/run-worker.sh" >runtime/logs/worker.log 2>&1 &
WORKER_PID=$!
API_PID=''
cleanup() {
  kill "$WORKER_PID" 2>/dev/null || true
  if [[ -n "$API_PID" ]]; then kill "$API_PID" 2>/dev/null || true; fi
  wait "$WORKER_PID" 2>/dev/null || true
  if [[ -n "$API_PID" ]]; then wait "$API_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo "AlphaLab API: http://127.0.0.1:${PORT:-8000}/"
"$ROOT/scripts/run-api.sh" &
API_PID=$!
wait "$API_PID"
