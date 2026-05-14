#!/usr/bin/env bash
# Quick-start script
set -e

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "[1/3] Creating virtual environment..."
  python3 -m venv .venv
fi

echo "[2/3] Installing dependencies..."
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

echo "[3/3] Starting server at http://localhost:8001"
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001
