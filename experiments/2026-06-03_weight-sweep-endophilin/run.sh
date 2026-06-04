#!/usr/bin/env bash
# EXP 1 — discrepancy weight-sweep on 25nM_endophilin (5 configs).
#
# Usage:
#   ./run.sh [n_workers]          # n_workers positional, OR
#   N_WORKERS=32 ./run.sh         # via env var, OR
#   ./run.sh                      # default = os.cpu_count()
#
# Interpreter resolved by scripts/_resolve_python.sh ($PYTHON if set, else
# `uv run python`, else python3/python). Set DRY_RUN=1 to echo the commands only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Resolve $PYTHON + run_py() (works with or without the venv activated).
source "$REPO_ROOT/scripts/_resolve_python.sh"

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
  run_py -m src.calibration.study --config "$CONFIG" --n-workers "$NW"
else
  run_py -m src.calibration.study --config "$CONFIG"
fi

echo "[exp1] calibrations done — running analysis ..."
run_py "$ANALYZE"
echo "[exp1] complete."
