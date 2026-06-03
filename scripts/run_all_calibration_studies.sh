#!/usr/bin/env bash
# Run all calibration studies sequentially on the VM, one command for everything.
#
# Order: EXP 2 (per-sample) -> EXP 3 (bootstrap) -> EXP 1 (weight-sweep).
#
# Usage:
#   ./scripts/run_all_calibration_studies.sh [n_workers]   # positional, OR
#   N_WORKERS=32 ./scripts/run_all_calibration_studies.sh  # via env var, OR
#   ./scripts/run_all_calibration_studies.sh               # default os.cpu_count()
#
# The worker count is passed through to each experiment's run.sh. Override the
# interpreter for all studies with PYTHON=... (e.g. PYTHON="uv run python").
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# n_workers: positional arg, else $N_WORKERS, else empty (each run.sh then lets
# Python default to os.cpu_count()). Export so the child run.sh scripts see it.
NW="${1:-${N_WORKERS:-}}"
export N_WORKERS="$NW"

EXPERIMENTS=(
  "experiments/2026-06-03_per-sample-calibration/run.sh"
  "experiments/2026-06-03_bootstrap-endophilin/run.sh"
  "experiments/2026-06-03_weight-sweep-endophilin/run.sh"
)

echo "######################################################################"
echo "# Running ALL calibration studies"
echo "#   repo root : $REPO_ROOT"
echo "#   n_workers : ${NW:-default(os.cpu_count())}"
echo "#   order     : per-sample -> bootstrap -> weight-sweep"
echo "######################################################################"

START_ALL=$(date +%s)
i=0
for exp in "${EXPERIMENTS[@]}"; do
  i=$((i + 1))
  echo ""
  echo "----------------------------------------------------------------------"
  echo ">>> [${i}/${#EXPERIMENTS[@]}] STARTING: $exp"
  echo "----------------------------------------------------------------------"
  START=$(date +%s)
  bash "$exp"
  echo ">>> [${i}/${#EXPERIMENTS[@]}] FINISHED: $exp  ($(( $(date +%s) - START ))s)"
done

echo ""
echo "######################################################################"
echo "# ALL calibration studies complete  (total $(( $(date +%s) - START_ALL ))s)"
echo "######################################################################"
