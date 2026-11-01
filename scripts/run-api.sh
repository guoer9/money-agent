#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "未找到 $PYTHON，请先按 README.md 创建虚拟环境并安装 requirements.txt" >&2
  exit 1
fi
exec "$PYTHON" -m thesis.api --port "${PORT:-8000}"
