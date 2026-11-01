#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
"${PYTHON:-$ROOT/.venv/bin/python}" -m pytest -q
if [[ -d "$ROOT/frontend/node_modules" ]]; then
  if ! command -v node >/dev/null 2>&1; then
    echo "未找到 node；请先安装 Node.js 20.19+" >&2
    exit 1
  fi
  (cd frontend && CI=1 ./node_modules/.bin/vue-tsc --noEmit --skipLibCheck)
  (cd frontend && CI=1 ./node_modules/.bin/vp build --mode prod)
else
  echo "未安装 frontend/node_modules；请先运行 pnpm install --frozen-lockfile" >&2
  exit 1
fi
