"""Joint multi-sample calibration: fit simulator parameters to real images.

Split into:
  - statistics : summary statistics for moment matching
  - discrepancy: config-driven real-vs-sim discrepancy (per-term weights)
  - optimize   : Optuna joint optimization over shared + per-sample params
  - run        : full pipeline orchestration + comparison plots
"""
