# Benchmark design (2026-06-10)

Decision record for how the detector is benchmarked against established methods,
and — importantly — what can and cannot be CLAIMED. Companion records:
`2026-06-04_synthetic-generation-strategy.md` (the data), `2026-06-10_detector-
loss-design.md` (the model). Numeric knobs (sweep values, match radius, dataset
sizes) live in the benchmark experiment's `config_snapshot/`.

## The claim (read this first — it constrains the whole design)

We do NOT claim "best general-purpose spot detector" or "best detection+
quantification framework." Those titles are held by SOTA generalist tools
(Spotiflow, Nature Methods 2025; SpotMAX, a generalist detect+quantify framework
that already does detect→local-integrate→per-spot-intensity and uses reference
channels). We will likely NOT beat these on raw detection F1, and the benchmark is
designed so our claim never depends on doing so.

What we DO claim, IF the data supports it:
> For the single-liposome curvature-sorting assay, a TWO-CHANNEL, UNCERTAINTY-AWARE
> detector recovers the curvature-sensing parameter (alpha) more accurately —
> particularly in the SMALL-LIPOSOME / high-curvature regime that dominates the
> biology — than applying SOTA general detectors followed by standard local-mask
> integration photometry.

Why this is defensible and not already done: the generalist tools detect on one
channel and quantify as a channel-agnostic downstream step; none JOINTLY detect
using cross-channel evidence, and none propagate calibrated per-spot measurement
uncertainty into a downstream biophysical fit. The curvature signal lives in the
two-channel relationship across size, and the critical regime is the small-liposome
tail where the lipid channel is faintest — exactly where two-channel corroboration
helps. That is the gap. The benchmark TESTS whether the advantage is real and
sufficient; it does not assume it. If we do not beat SpotMAX+photometry on
small-liposome alpha recovery, we do not have the paper — and the synthetic sweep
is how we learn that early and honestly.

## Methods benchmarked (tiers by priority)

Tier 1 (must have):
- **Ours** — two-channel HRNet detector with per-spot uncertainty (the loss-design
  doc). Reported in two configs: (i) lipid-detect + shared photometry (apples-to-
  apples with single-channel methods), (ii) native two-channel + learned intensity
  + uncertainty-weighted fit (the full method).
- **Spotiflow** — the Nature Methods DL detector ("does spot detection very well,
  in Nature" — the PI's reference point). RETRAIN on our data for fairness; do not
  rely on a pretrained model. Detection-only → shared photometry.
- **SpotMAX** — the generalist detect+quantify framework; the real competitor for
  the FULL pipeline. Use its detection + its own integrated-intensity quantification
  (this is the PI's "detect then local-mask integrate" pipeline, already built).
  This is the strongest baseline; treat beating it on small-liposome alpha recovery
  as the bar for the paper.

Tier 2 (strong baselines):
- **DECODE** — SMLM DL fitter with native uncertainty output; the fair comparison
  for OUR uncertainty contribution and strong in dense/crowded regimes. Configure/
  retrain for our PSF + density regime (do not run stock). Lipid detection → shared
  photometry.
- **cme-analysis** — classical Gaussian-fitting CME pipeline; established non-DL
  baseline. Detects on lipid → shared photometry.

Tier 3 (cheap floor, expected by reviewers):
- **Classical LoG/DoG** blob detector → shared photometry.

## The shared pipeline (this is what makes the comparison fair)

ONE pipeline, swap the detector. Defined once, applied identically:

  detect spots  →  local-mask protein integration (PI's method)  →
  fit log-log sorting curve (protein intensity vs diameter)  →  recovered alpha

- Every method's detector produces centroids (on the LIPID channel — one liposome,
  one location, lipid present regardless of protein binding; detecting on protein
  would systematically miss low-binding spots and bias everything).
- The PROTEIN intensity for the sorting curve is read by a SINGLE shared
  photometry module (local mask + integrate) at each method's centroids — EXACTLY
  the PI's prescription. This isolates DETECTION quality; everyone gets the same
  photometry so we are not comparing photometry methods.
- EXCEPTION = the contribution: OUR full config (Tier 1.ii) replaces the shared
  photometry with its own learned per-spot intensity + uncertainty, and weights the
  sorting-curve fit by that uncertainty. Reporting BOTH our configs isolates how
  much advantage comes from (a) better/cross-channel detection vs (b) learned
  uncertainty-weighted intensity.
- SpotMAX is the one method allowed to use its OWN quantification too (since that
  is its published pipeline), reported alongside the shared-photometry version.

Method-neutral matching: matched-detection F1 with a FIXED match radius, identical
for all methods, against synthetic ground truth.

## Synthetic test grid (the core experiments)

All on synthetic data where ground truth (positions, diameters, true alpha) is
known. Two axes, run as a grid:

### Axis 1 — SPOT-DENSITY sweep
Several batches stepping local spot density from very disperse → moderate →
saturated (via the generator's `fixed_spot_density` override). Run the density
sweep TWICE:
- **alpha disabled** (random per-spot protein intensity, no curvature law): isolates
  pure spatial-overlap DETECTION performance — the clean cross-method detection
  comparison.
- **high curvature sensing** (alpha in the strong-sensing range): tests whether the
  curvature coupling helps or hurts detection in crowding, and whether crowding
  biases alpha recovery.
Report per density level, per method: matched-F1, localization error, lipid- and
protein-intensity recovery error, and (alpha-on) recovered alpha.

### Axis 2 — ALPHA sweep (the headline)
Test sets at several KNOWN FIXED global alphas (e.g. 0.5, 0.75, 1.0, 1.5, 2.0),
each on the DLS-realistic size distribution. For every method: run the shared
pipeline → recovered alpha vs true alpha. Plot recovered-vs-true with y=x.

### The headline metric: curvature-sensing recovery, BINNED BY LIPOSOME SIZE
The paper's central plot: recovered-vs-true alpha, AND alpha-recovery error as a
function of liposome size bin — because the claim is specifically about the
SMALL-LIPOSOME tail. Show that our method's recovered alpha is less biased /
lower-variance than the SOTA-detector + photometry pipelines, concentrated in the
small-diameter bins. Detection F1 is reported HONESTLY as secondary; we may lose it
and the claim does not depend on it.

## Real-data check (no annotation — consistency only)
Hand annotation is not possible. Real data is used to confirm the synthetic winner
transfers, not to rank:
- **DLS-consistency:** detected lipid size distribution (efficiency-corrected from
  the synthetic detection-vs-radius curve) must match measured DLS. Disqualifies a
  method that fails it.
- **Classical-agreement floor** on bright easy spots.

## Cross-machine reproducibility (this benchmark spans two machines)
HRNet trained on H100 (synthetic training images discarded; regenerated locally for
test). DECODE/cme-analysis/SpotMAX/Spotiflow run locally. Record in the EXPERIMENT.md
which model checkpoint + which dataset (config+seed+commit) produced each figure.
Every artifact gets provenance.json.

## Experiment layout
`experiments/<date>_curvature-benchmark/`: EXPERIMENT.md (question / method →
links to THIS doc / reproduce / findings), run.sh (or per-machine scripts),
config_snapshot/ (the density + alpha grid configs, match radius, per-method
settings), per-method adapters normalizing each tool's output to the common
detection format, the shared photometry + sorting-curve-fit module, analyze.py
(the recovered-vs-true and error-vs-size-bin plots), and at run time
runs/ figures/ results.json.

## Honest-framing guardrail
Design the grid and metrics BEFORE running, fix them, and report whatever comes
out. Do NOT tune the benchmark until our method wins (motivated methodology;
reviewers detect it). A clean comparison where we happen to win on small-liposome
alpha recovery is publishable; a comparison reverse-engineered to win is not. If we
lose, that is a real finding about the assay and reframes the paper — it is not a
reason to adjust the benchmark.
