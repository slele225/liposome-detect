#!/usr/bin/env bash
# EXP 2 — per-sample independent calibrations across all six samples.
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
  "$PYTHON" -m src.calibration.study --config "$CONFIG" --n-workers "$NW"
else
  "$PYTHON" -m src.calibration.study --config "$CONFIG"
fi

echo "[exp2] calibrations done — running analysis ..."
"$PYTHON" "$ANALYZE"
echo "[exp2] complete."
