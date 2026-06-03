#!/usr/bin/env bash
# EXP 3 — bootstrap stability study on 25nM_endophilin (100 calibrations).
#
# Usage:
#   ./run.sh [n_workers]          # n_workers positional, OR
#   N_WORKERS=32 ./run.sh         # via env var, OR
#   ./run.sh                      # default = os.cpu_count()
#
# This launches 100 independent 200-trial calibrations — set n_workers to the
# VM core count. Override the interpreter with PYTHON=... (e.g. "uv run python").
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
CONFIG="experiments/2026-06-03_bootstrap-endophilin/config_snapshot/study.yaml"
ANALYZE="experiments/2026-06-03_bootstrap-endophilin/analyze.py"

NW="${1:-${N_WORKERS:-}}"

echo "=================================================================="
echo "[exp3] bootstrap stability — 25nM_endophilin (d=25, 100 repeats)"
echo "[exp3] repo root : $REPO_ROOT"
echo "[exp3] config    : $CONFIG"
echo "[exp3] n_workers : ${NW:-default(os.cpu_count())}"
echo "=================================================================="

if [ -n "$NW" ]; then
  "$PYTHON" -m src.calibration.study --config "$CONFIG" --n-workers "$NW"
else
  "$PYTHON" -m src.calibration.study --config "$CONFIG"
fi

echo "[exp3] calibrations done — running analysis ..."
"$PYTHON" "$ANALYZE"
echo "[exp3] complete."
