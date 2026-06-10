#!/usr/bin/env bash
# Short DIAGNOSTIC training run — the GATE before a full multi-hour H100 job.
#
# Run this on the H100. It walks the six steps below; read the VERDICT printed by
# step 6, then configure the SEPARATE full convergence run from what it shows
# (term weights confirmed/adjusted, early-stop metric chosen, epochs set). The
# diagnostic is the gate, NOT the final run.
#
# Usage:
#   ./scripts/run_diagnostic.sh [n_workers]
#   N_WORKERS=32 ./scripts/run_diagnostic.sh
#
# Set DRY_RUN=1 to echo every command instead of running it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

NW="${1:-${N_WORKERS:-}}"
PER_SAMPLE="experiments/2026-06-03_per-sample-calibration"
TRAIN_CFG="configs/generator/diag_train.yaml"
VAL_CFG="configs/generator/diag_val.yaml"
TRAIN_SET="datasets/diag_train"
VAL_SET="datasets/diag_val"
TRAIN_YAML="configs/train/hrnet_diagnostic.yaml"
STATS_YAML="configs/train/_diag_stats.yaml"
RUN_DIR="runs/hrnet_diagnostic"

run() { echo "+ $*"; [ "${DRY_RUN:-0}" = "1" ] || "$@"; }

echo "######################################################################"
echo "# DIAGNOSTIC run procedure (gate before the full H100 job)"
echo "#   repo root : $REPO_ROOT"
echo "#   n_workers : ${NW:-default(os.cpu_count())}"
echo "######################################################################"

echo ""
echo ">>> [1/6] uv sync"
echo "    NOTE: on the H100, install the CUDA torch wheel matching the GPU/driver"
echo "    (e.g. a cu12x build) so training runs on the GPU, not CPU."
run uv sync

echo ""
echo ">>> [2/6] ensure per-sample calibrations exist (gitignored — regenerate if absent)"
if [ -f "$PER_SAMPLE/runs/20nM_EGFP/calibration_results.json" ]; then
  echo "    found $PER_SAMPLE/runs/ — skipping regeneration"
else
  echo "    absent — regenerating the per-sample calibration study on this VM"
  run bash "$PER_SAMPLE/run.sh" "$NW"
fi

echo ""
echo ">>> [3/6] generate REAL datasets: a train set + a SEPARATE val set (different seed)"
run uv run python -m src.generator.generate --config "$TRAIN_CFG" --n-workers "$NW"
run uv run python -m src.generator.generate --config "$VAL_CFG" --n-workers "$NW"

echo ""
echo ">>> [4/6] compute norm/eps on the REAL train set; paste into $TRAIN_YAML"
run uv run python -m src.train.compute_stats --dataset "$TRAIN_SET" --out "$STATS_YAML"
echo "    ACTION: copy norm_mean/norm_std + eps_lipid/eps_protein from $STATS_YAML"
echo "    into $TRAIN_YAML (its data.norm_* and loss.eps_* are null placeholders)."

echo ""
echo ">>> [5/6] run the short diagnostic training"
run uv run python -m src.train.train --config "$TRAIN_YAML" --n-workers "$NW"

echo ""
echo ">>> [6/6] read the VERDICT"
run uv run python -m src.train.diagnostic --run "$RUN_DIR"

echo ""
echo "######################################################################"
echo "# Diagnostic complete. The FULL convergence run is a SEPARATE later step,"
echo "# configured from the verdict above (weights, early_stop_metric, epochs)."
echo "######################################################################"
