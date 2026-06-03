#!/usr/bin/env bash
# EXP 1 — discrepancy weight-sweep on 25nM_endophilin (5 configs).
#
# Usage:
#   ./run.sh [n_workers]          # n_workers positional, OR
#   N_WORKERS=32 ./run.sh         # via env var, OR
#   ./run.sh                      # default = os.cpu_count()
#
# Override the interpreter with PYTHON=... (e.g. PYTHON="uv run python").
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
CONFIG="experiments/2026-06-03_weight-sweep-endophilin/config_snapshot/study.yaml"
ANALYZE="experiments/2026-06-03_weight-sweep-endophilin/analyze.py"

NW="${1:-${N_WORKERS:-}}"

echo "=================================================================="
echo "[exp1] discrepancy weight-sweep — 25nM_endophilin (5 configs)"
echo "[exp1] repo root : $REPO_ROOT"
echo "[exp1] config    : $CONFIG"
echo "[exp1] n_workers : ${NW:-default(os.cpu_count())}"
echo "=================================================================="

if [ -n "$NW" ]; then
  "$PYTHON" -m src.calibration.study --config "$CONFIG" --n-workers "$NW"
else
  "$PYTHON" -m src.calibration.study --config "$CONFIG"
fi

echo "[exp1] calibrations done — running analysis ..."
"$PYTHON" "$ANALYZE"
echo "[exp1] complete."
