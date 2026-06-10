"""Stage-2 synthetic training-data generator.

A config-driven orchestration + parameter-sampling + serialization layer over the
EXISTING forward model (``src.simulator.forward_model``). It does NOT reimplement
spot rendering or the noise model — it samples microscope/biology parameters per
image, drives ``simulate_image`` (or, for per-spot curvature alpha, reuses the
simulator's spot renderer + PMT-noise path), and serializes two-channel images
with centroid + per-spot property ground truth.

Design / "why": see the decision records
  - docs/decisions/2026-06-04_synthetic-generation-strategy.md
  - docs/decisions/2026-06-04_protein-channel-parameterization.md
  - docs/decisions/2026-06-10_detector-loss-design.md
Exact numeric knobs live in configs/generator/<name>.yaml, never in code.

NOTE: the CLI driver (``src.generator.generate``) intentionally imports nothing
heavy at module top (no numpy) so the worker-pool BLAS pinning takes effect, the
same discipline as ``src.calibration.study``. The numpy-bearing logic lives in the
sibling modules (calibration_io, size_distribution, sampling, protein_channel,
core) which are imported lazily inside the worker.
"""
