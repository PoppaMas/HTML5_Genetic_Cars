#!/usr/bin/env bash
# Phase 3-B1 replay page in one command (see tools/build_b1_page.py --help):
#   tools/build_b1_page.sh <run_id> [--replay-proof] [--fd-dir DIR] [--proof-gens traj|ends|0,4] [--max-mb 24]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
PY="${PY:-python}"
command -v "$PY" >/dev/null 2>&1 || PY=python3
export PY
exec "$PY" tools/build_b1_page.py "$@"
