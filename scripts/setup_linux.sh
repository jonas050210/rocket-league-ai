#!/usr/bin/env bash
# Rocket League AI - Linux/macOS setup
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
python -m pytest -q

echo
echo "Setup complete. Start with:"
echo "  . .venv/bin/activate"
echo "  python -m rlai.cli serve --port 8000"
