#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  echo "No .venv found. Run scripts/setup_linux.sh first." >&2
  exit 1
fi
exec .venv/bin/python -m rlai.cli serve --port "${PORT:-8000}"
