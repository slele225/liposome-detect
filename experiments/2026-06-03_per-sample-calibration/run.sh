#!/usr/bin/env bash
# EXP 2 — per-sample independent calibrations across all six samples.
#
# Usage:
#   ./run.sh [n_workers]          # n_workers positional, OR
#   N_WORKERS=32 ./run.sh         # via env var, OR
#   ./run.sh                      # default = os.cpu_count()
#
# Interpreter is resolved by scripts/_resolve_python.sh: $PYTHON if set, else
# `uv run python` (if uv is installed), else python3/python — so it works with
# or without the venv activated. Set DRY_RUN=1 to echo the commands only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Resolve $PYTHON + run_py() (works with or without the venv activated).
source "$REPO_ROOT/scripts/_resolve_python.sh"

CONFIG="experiments/2026-06-03_per-sample-calibration/config_snapshot/study.yaml"
ANALYZE="experiments/2026-06-03_per-sample-calibration/analyze.py"

# n_workers: positional arg, else $N_WORKERS, else let Python default to cpu_count.
NW="${1:-${N_WORKERS:-}}"

echo "=================================================================="
echo "[exp2] per-sample independent calibration"
echo "[exp2] repo root : $REPO_ROOT"
echo "[exp2] config    : $CONFIG"
echo "[exp2] n_workers : ${NW:-default(os.cpu_count())}"
echo "=================================================================="

if [ -n "$NW" ]; then
  run_py -m src.calibration.study --config "$CONFIG" --n-workers "$NW"
else
  run_py -m src.calibration.study --config "$CONFIG"
fi

echo "[exp2] calibrations done — running analysis ..."
run_py "$ANALYZE"
echo "[exp2] complete."
