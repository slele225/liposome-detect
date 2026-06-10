# `src/generator/` — synthetic training-data generator

Config-driven orchestration layer over the existing forward model
(`src/simulator/forward_model.py`). It samples microscope/biology parameters per
image, drives `simulate_image`, and serializes two-channel images with
centroid + per-spot property ground truth. It does **not** reimplement spot
rendering or the noise model — it reuses the simulator's.

## Why (decision records — read these for the reasoning)
- [Synthetic-generation strategy](../../docs/decisions/2026-06-04_synthetic-generation-strategy.md)
  — structured sampling, alpha modes, size emphasis, reproducibility.
- [Protein-channel parameterization](../../docs/decisions/2026-06-04_protein-channel-parameterization.md)
  — measured-vs-randomized split and the alpha-agnostic invariant.
- [Detector loss design](../../docs/decisions/2026-06-10_detector-loss-design.md)
  — how a later detector consumes this data (context).

Exact numeric knobs live in `configs/generator/<name>.yaml`, never in the docs.

## What it does
- Reads per-sample `calibration_results.json` files; builds the **union** of
  their fitted ranges (`lipid_brightness`, `spot_density`, PSF `sigma`,
  `noise_scale = gain*enf`), **widened ±30%**.
- Per image: samples those independently; PSF as one near-circular width
  (`sigma` × eccentricity × free rotation); gain/enf as **product × log-uniform
  split**; small `optical_bg_lipid`; randomized `protein_brightness`. Protein PSF
  and dark floors are pinned from a per-image **regime** (one of the named
  calibrations → `sample_regime_id`).
- **Alpha modes** (`alpha_mode`): `per_spot_random` (default — independent
  curvature exponent per spot, built without editing the simulator by reusing its
  spot-renderer + PMT-noise path; see `protein_channel.py`), `global_coherent`
  (one alpha per image, straight through `simulate_image`), and `mixed`.
- **Size modes** (`size_mode`): `emphasis` (small-size/curvature-weighted PMF for
  training) or `dls` (real DLS via `io.parse_dls`, for test/Phase-3 sets).
- Determinism: each image is fully reproducible from
  `(config + base_seed + index)` via `SeedSequence([base_seed, index])`.

## CLI
```bash
python -m src.generator.generate --config configs/generator/train_v1.yaml --n-workers 32
python -m src.generator.generate --config configs/generator/smoke.yaml --smoke
```
`--smoke` writes a tiny batch plus `smoke/overlay.png` (GT centroids on the lipid
image) and `smoke/comparison.png` (synthetic protein+lipid, and a real lipid crop
if `smoke.real_tiff` is configured). The worker pool reuses
`src.calibration.study`'s spawn + single-thread-per-worker BLAS pinning.

## Output
`datasets/<name>/` (gitignored):
- `images/img_NNNNNN.npy` — float32, shape `(2, H, W)`: channel 0 = protein, 1 = lipid.
- `labels/img_NNNNNN.json` — per-spot `{x, y, diameter_nm, lipid_intensity,
  protein_intensity, alpha_used, sample_regime_id}` + per-image `meta` (all
  sampled params, `noise_scale`/split, alpha/size mode, rng seed, config hash).
- `dataset_manifest.json`, `provenance.json` (via `src.provenance.write_provenance`).

## Modules
`calibration_io` (ranges/regimes) · `size_distribution` (PMFs) · `sampling`
(per-image params) · `protein_channel` (per-spot alpha) · `core` (per-image
generate + serialize) · `generate` (CLI/pool, intentionally numpy-free at import
so BLAS pinning takes effect) · `smoke` (plots).
