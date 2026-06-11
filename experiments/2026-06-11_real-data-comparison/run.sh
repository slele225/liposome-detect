#!/usr/bin/env bash
# EXP — real-data corrected-vs-standard alpha comparison.
#
# Runs the trained Stage-2 detector on the REAL EGFP + endophilin images and
# measures the curvature parameter alpha via the STANDARD (OLS) vs CORRECTED
# (errors-in-variables + calibration) pipeline, anchored on EGFP=2.0 and validated
# against DLS.
#
# Order (go/no-go early):
#   1. smoke_check.py     — scaling guard. The human EYEBALLS the PASS/WARN verdict
#                           before trusting anything downstream.
#   2. firm_calibration.py— build the recovered->true calibration curve (8 alphas).
#   3. real_alpha.py      — the EGFP-anchored STANDARD vs CORRECTED table (go/no-go).
#   4. dls_consistency.py — the second real-data anchor.
#
# Usage:
#   ./run.sh [n_workers]          # n_workers positional, OR
#   N_WORKERS=32 ./run.sh         # via env var (32-core VM), OR
#   ./run.sh                      # default = os.cpu_count()
#
# Uses `uv run python`, GPU for inference, all workers for generation. Outputs
# (tables .csv/.txt, plots .png) land in this experiment folder.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export MPLCONFIGDIR=/tmp/mpl
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PY="uv run python"
NW="${1:-${N_WORKERS:-}}"
NW_ARG=()
[ -n "$NW" ] && NW_ARG=(--n-workers "$NW")

echo "=================================================================="
echo "[real-cmp] repo root : $REPO_ROOT"
echo "[real-cmp] n_workers : ${NW:-default(os.cpu_count())}"
echo "=================================================================="

echo "[real-cmp] STEP 1/4 — smoke check (SCALING GUARD; inspect the verdict) ..."
$PY "$SCRIPT_DIR/smoke_check.py"

echo "[real-cmp] STEP 2/4 — firm up the calibration curve ..."
$PY "$SCRIPT_DIR/firm_calibration.py" "${NW_ARG[@]}"

echo "[real-cmp] STEP 3/4 — real-image alpha (EGFP-anchored go/no-go) ..."
$PY "$SCRIPT_DIR/real_alpha.py"

echo "[real-cmp] STEP 4/4 — DLS consistency (second anchor) ..."
$PY "$SCRIPT_DIR/dls_consistency.py"

echo "[real-cmp] complete. Tables + plots are in $SCRIPT_DIR"
