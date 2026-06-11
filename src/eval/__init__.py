"""Evaluation / benchmarking utilities for the trained detector.

The canonical alpha (curvature-sensing exponent) estimator lives in
``src.eval.alpha_fit`` — every benchmark adapter's sorting-curve step calls it so
there is a single source of truth for how a recovered alpha is computed and
calibrated. The ad-hoc analysis scripts in this package (``alpha_recovery``,
``detection_bias``, ``recall_vs_diameter``, ``uncertainty_calibration``,
``york_test``) all import their line fits from there.
"""
