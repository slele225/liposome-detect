#!/usr/bin/env bash
# Resolve a Python interpreter into $PYTHON and define run_py().
#
# Sourced by the experiment run.sh scripts and the orchestration script so the
# studies work whether or not the uv venv is activated (we hit
# "python: command not found" on the VM when it was not).
#
# Resolution order (first match wins):
#   1. an explicit $PYTHON already set in the environment (override)
#   2. `uv run python`   — if `uv` is on PATH (uses the project's venv)
#   3. `python3`
#   4. `python`
#
# Usage (after sourcing):
#   run_py -m src.calibration.study --config ...   # runs "$PYTHON ..."
# If DRY_RUN is non-empty, run_py only echoes the command instead of running it.
#
# This file is meant to be SOURCED, not executed; it does not set shell options
# (it inherits the caller's `set -euo pipefail`).

if [ -z "${PYTHON:-}" ]; then
  if command -v uv >/dev/null 2>&1; then
    PYTHON="uv run python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
  else
    echo "[resolve_python] ERROR: no interpreter found (tried uv, python3, python)." >&2
    return 1 2>/dev/null || exit 1
  fi
fi

echo "[resolve_python] interpreter: ${PYTHON}${DRY_RUN:+   (DRY_RUN: commands will be echoed, not run)}"

# Run the resolved interpreter. $PYTHON is intentionally UNQUOTED so a multi-word
# command like "uv run python" splits into its three words. Honors DRY_RUN.
run_py() {
  if [ -n "${DRY_RUN:-}" ]; then
    echo "[dry-run] ${PYTHON} $*"
  else
    # shellcheck disable=SC2086
    ${PYTHON} "$@"
  fi
}
