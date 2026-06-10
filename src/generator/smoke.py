"""Smoke-check plots: visually confirm correctness before scaling a dataset.

Produces, into ``<output_dir>/smoke/``:
  - ``overlay.png``    : the rendered LIPID image with ground-truth centroids
                         overlaid (so spot placement / counts are checkable).
  - ``comparison.png`` : synthetic protein + lipid side by side, and — if a real
                         TIFF is configured and present — a real lipid crop loaded
                         via ``io.load_tiff_stack`` for a sanity comparison.

Imports matplotlib lazily with the Agg backend (no display needed).
"""

import json
from pathlib import Path

import numpy as np


def make_smoke_plots(output_dir, records, config):
    """Render smoke plots from the first generated image. Returns paths written."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    smoke_dir = out / 'smoke'
    smoke_dir.mkdir(parents=True, exist_ok=True)

    if not records:
        return []
    rec = records[0]
    image = np.load(rec['image'])          # (2, H, W): 0=protein, 1=lipid
    label = json.loads(Path(rec['label']).read_text())
    protein, lipid = image[0], image[1]
    xs = [s['x'] for s in label['spots']]
    ys = [s['y'] for s in label['spots']]

    written = []

    # --- overlay: GT centroids on the lipid image ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(lipid, cmap='gray', origin='upper')
    ax.scatter(xs, ys, s=60, facecolors='none', edgecolors='red', linewidths=0.8)
    ax.set_title(f"lipid + GT centroids (n={label['n_spots']}, "
                 f"mode={label['meta']['alpha_mode']})")
    ax.set_axis_off()
    overlay_path = smoke_dir / 'overlay.png'
    fig.savefig(overlay_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    written.append(str(overlay_path))

    # --- comparison: synthetic protein + lipid (+ optional real lipid crop) ---
    real_lipid = _maybe_load_real_lipid(config)
    ncols = 3 if real_lipid is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    axes[0].imshow(protein, cmap='magma', origin='upper')
    axes[0].set_title('synthetic protein (ch0)')
    axes[1].imshow(lipid, cmap='gray', origin='upper')
    axes[1].set_title('synthetic lipid (ch1)')
    if real_lipid is not None:
        axes[2].imshow(real_lipid, cmap='gray', origin='upper')
        axes[2].set_title('real lipid crop')
    for a in axes:
        a.set_axis_off()
    comparison_path = smoke_dir / 'comparison.png'
    fig.savefig(comparison_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    written.append(str(comparison_path))

    return written


def _maybe_load_real_lipid(config):
    """Load a real lipid crop if ``smoke.real_tiff`` is configured and present."""
    smoke_cfg = config.get('smoke', {}) or {}
    real_tiff = smoke_cfg.get('real_tiff')
    if not real_tiff or not Path(real_tiff).exists():
        return None
    try:
        from src.simulator.io import load_tiff_stack
        return load_tiff_stack(real_tiff)['lipid']
    except Exception as e:  # best-effort: smoke must not fail on a bad real path
        print(f"  smoke: could not load real TIFF {real_tiff}: {e}")
        return None
