#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p runtime/logs
"$ROOT/scripts/run-worker.sh" >runtime/logs/worker.log 2>&1 &
WORKER_PID=$!
cleanup() {
  kill "$WORKER_PID" 2>/dev/null || true
  wait "$WORKER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
echo "AlphaLab API: http://127.0.0.1:${PORT:-8000}/"
exec "$ROOT/scripts/run-api.sh"
