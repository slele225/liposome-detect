"""Compute per-channel normalization + intensity ``eps`` floors for a REAL dataset.

    python -m src.train.compute_stats --dataset datasets/<name> [--out <yaml>]

Two quantities a training config needs, recomputed on the ACTUAL training set
(the hrnet_v1 defaults were measured on a representative sample — this matters
most in the dim small-spot regime where the log-space ``eps`` floor bites):

  * ``norm_mean`` / ``norm_std`` — per-channel ``[protein, lipid]`` pixel mean and
    std over every image (channel order matches the (2,H,W) images: 0=protein,
    1=lipid). Streamed (sum / sum-of-squares), so a few-thousand-image set never
    has to be held in memory at once.
  * ``eps_lipid`` / ``eps_protein`` — the log-space intensity floor, set RELATIVE
    TO THE DIMMEST REAL FLUX. From the GT label fluxes (``lipid_intensity`` /
    ``protein_intensity``) we take a low percentile per channel (``--floor-pct``,
    default 1st) and set ``eps = floor_frac * that_pct`` (``--floor-frac``, default
    0.1). The floor sits an order of magnitude BELOW the dim-spot flux but stays
    strictly > 0, so it stabilizes the log without swamping the dim signal.

Prints a short summary and emits a YAML snippet (with the percentiles documented)
ready to paste/include into a train config.
"""

import argparse
from pathlib import Path

import numpy as np


def compute_norm_stats(image_paths):
    """Streamed per-channel ``(mean, std)`` over (2,H,W) images -> ``[protein, lipid]``.

    Accumulates per-channel pixel sum and sum-of-squares so the whole image set is
    never resident at once. Returns two length-2 lists in image channel order
    (0=protein, 1=lipid).
    """
    n_pix = 0
    s = np.zeros(2, dtype=np.float64)
    ss = np.zeros(2, dtype=np.float64)
    for p in image_paths:
        img = np.load(p).astype(np.float64)            # (2, H, W)
        if img.ndim != 3 or img.shape[0] != 2:
            raise ValueError(f"{p}: expected (2,H,W), got {img.shape}")
        s += img.sum(axis=(1, 2))
        ss += (img * img).sum(axis=(1, 2))
        n_pix += img.shape[1] * img.shape[2]
    if n_pix == 0:
        raise ValueError("no image pixels to compute norm stats over")
    mean = s / n_pix
    var = np.maximum(ss / n_pix - mean * mean, 0.0)    # guard tiny negative drift
    std = np.sqrt(var)
    return mean.tolist(), std.tolist()


def _gather_fluxes(label_paths):
    import json
    lip, pro = [], []
    for lp in label_paths:
        lbl = json.loads(Path(lp).read_text())
        for spot in lbl.get('spots', []):
            lip.append(float(spot['lipid_intensity']))
            pro.append(float(spot['protein_intensity']))
    return np.asarray(lip, dtype=np.float64), np.asarray(pro, dtype=np.float64)


def compute_eps(label_paths, floor_pct=1.0, floor_frac=0.1):
    """Per-channel log-space ``eps`` from the GT flux distribution.

    For each channel, take the ``floor_pct``-th percentile of the GT fluxes (the
    "dimmest real flux" proxy) and set ``eps = floor_frac * percentile``. Returns a
    dict with the eps values AND the percentiles used (for documentation/tests).
    """
    lip, pro = _gather_fluxes(label_paths)
    if lip.size == 0 or pro.size == 0:
        raise ValueError("no GT spots found to compute intensity eps from")
    lip_pct = float(np.percentile(lip, floor_pct))
    pro_pct = float(np.percentile(pro, floor_pct))
    return {
        'eps_lipid': float(floor_frac * lip_pct),
        'eps_protein': float(floor_frac * pro_pct),
        'lipid_floor_pct': lip_pct,
        'protein_floor_pct': pro_pct,
        'floor_pct': float(floor_pct),
        'floor_frac': float(floor_frac),
        'n_spots': int(lip.size),
    }


def compute_dataset_stats(dataset_dir, floor_pct=1.0, floor_frac=0.1):
    """Compute norm stats + eps floors for a generator dataset directory."""
    root = Path(dataset_dir)
    image_paths = sorted((root / 'images').glob('img_*.npy'))
    label_paths = sorted((root / 'labels').glob('img_*.json'))
    if not image_paths:
        raise FileNotFoundError(f"no img_*.npy under {root/'images'}")
    if not label_paths:
        raise FileNotFoundError(f"no img_*.json under {root/'labels'}")
    mean, std = compute_norm_stats(image_paths)
    eps = compute_eps(label_paths, floor_pct=floor_pct, floor_frac=floor_frac)
    return {
        'dataset': str(root),
        'n_images': len(image_paths),
        'norm_mean': mean,           # [protein, lipid]
        'norm_std': std,             # [protein, lipid]
        **eps,
    }


def render_yaml_snippet(stats):
    """A paste-ready YAML snippet (with the percentile choice documented)."""
    m, s = stats['norm_mean'], stats['norm_std']
    return (
        "# Computed by src.train.compute_stats on "
        f"{stats['dataset']} ({stats['n_images']} images, "
        f"{stats['n_spots']} spots).\n"
        f"# eps = floor_frac({stats['floor_frac']}) * "
        f"{stats['floor_pct']:g}th-pct GT flux "
        f"(lipid {stats['lipid_floor_pct']:.1f}, "
        f"protein {stats['protein_floor_pct']:.1f}).\n"
        "data:\n"
        f"  norm_mean: [{m[0]:.2f}, {m[1]:.2f}]   # [protein, lipid]\n"
        f"  norm_std: [{s[0]:.2f}, {s[1]:.2f}]    # [protein, lipid]\n"
        "loss:\n"
        f"  eps_lipid: {stats['eps_lipid']:.4g}\n"
        f"  eps_protein: {stats['eps_protein']:.4g}\n"
    )


def main():
    ap = argparse.ArgumentParser(
        description='Compute per-channel norm + intensity eps for a dataset.')
    ap.add_argument('--dataset', required=True, help='Generator dataset directory.')
    ap.add_argument('--out', default=None,
                    help='Write the YAML snippet here (also printed either way).')
    ap.add_argument('--floor-pct', type=float, default=1.0,
                    help='Percentile of GT flux used as the dim-flux floor (default 1).')
    ap.add_argument('--floor-frac', type=float, default=0.1,
                    help='eps = floor_frac * floor-pct percentile (default 0.1).')
    args = ap.parse_args()

    stats = compute_dataset_stats(args.dataset, floor_pct=args.floor_pct,
                                  floor_frac=args.floor_frac)
    snippet = render_yaml_snippet(stats)

    print('=' * 64)
    print(f"[compute_stats] dataset={stats['dataset']} images={stats['n_images']} "
          f"spots={stats['n_spots']}")
    print(f"[compute_stats] norm_mean (protein,lipid) = "
          f"({stats['norm_mean'][0]:.2f}, {stats['norm_mean'][1]:.2f})")
    print(f"[compute_stats] norm_std  (protein,lipid) = "
          f"({stats['norm_std'][0]:.2f}, {stats['norm_std'][1]:.2f})")
    print(f"[compute_stats] {stats['floor_pct']:g}th-pct flux "
          f"lipid={stats['lipid_floor_pct']:.1f} "
          f"protein={stats['protein_floor_pct']:.1f} "
          f"(floor_frac={stats['floor_frac']})")
    print(f"[compute_stats] eps_lipid={stats['eps_lipid']:.4g} "
          f"eps_protein={stats['eps_protein']:.4g}")
    print('=' * 64)
    print(snippet)
    if args.out:
        Path(args.out).write_text(snippet)
        print(f"[compute_stats] wrote {args.out}")


if __name__ == '__main__':
    main()
