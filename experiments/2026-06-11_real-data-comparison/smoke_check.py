"""SMOKE CHECK (run FIRST) — guard the real-image scaling before any result.

Loads ONE 20nM_EGFP image, runs the detector, and prints the detection count and
the distribution (median, 5/95 pct) of predicted lipid_intensity and
protein_intensity. These predicted *fluxes* (total ADU) must land in a plausible
hundreds-to-thousands range, like the synthetic training data the model expects.

The scaling convention (raw, offset included — matches the generator) is set in
``src/eval/real_data.py``. If predicted intensities are ORDERS OF MAGNITUDE off
the synthetic scale, that signals a normalization/offset mismatch and the
downstream numbers cannot be trusted -> the verdict is WARN. A human eyeballs
this before run.sh proceeds to the real result.

    uv run python experiments/2026-06-11_real-data-comparison/smoke_check.py
"""

import argparse

import numpy as np

from _common import (PLAUSIBLE_FLUX_BAND, SYNTH_EPS, SYNTH_NORM_MEAN,
                     add_model_args, sample_dir)
from src.eval.matching import decode_image_array, load_model
from src.eval.real_data import (dark_offsets, list_sample_images,
                                load_real_image)


def _stats(vals):
    a = np.asarray(vals, np.float64)
    if a.size == 0:
        return (float('nan'),) * 3
    return float(np.median(a)), float(np.percentile(a, 5)), float(np.percentile(a, 95))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    add_model_args(ap)
    ap.add_argument('--sample', default='20nM_EGFP')
    args = ap.parse_args()

    sdir = sample_dir(args.data_root, args.sample)
    img_path = list_sample_images(sdir)[0]
    offsets = dark_offsets(sdir) if args.subtract_dark == 'on' else None
    arr = load_real_image(img_path, subtract_dark=args.subtract_dark, offsets=offsets)

    model, cfg, device = load_model(args.config, args.ckpt)
    dets = decode_image_array(model, cfg, device, arr)

    lip = [d['lipid_intensity'] for d in dets]
    pro = [d['protein_intensity'] for d in dets]
    lip_med, lip_lo, lip_hi = _stats(lip)
    pro_med, pro_lo, pro_hi = _stats(pro)

    print('=' * 66)
    print(f"SMOKE CHECK — scaling guard ({args.sample})")
    print('=' * 66)
    print(f"image           : {img_path}")
    print(f"subtract_dark   : {args.subtract_dark} "
          f"(auto==off==matches generator; raw ADU, offset included)")
    print(f"input arr range : [{arr.min():.1f}, {arr.max():.1f}]  "
          f"mean[protein,lipid]=[{arr[0].mean():.1f}, {arr[1].mean():.1f}]")
    print(f"                  (synthetic norm_mean protein~{SYNTH_NORM_MEAN['protein']}, "
          f"lipid~{SYNTH_NORM_MEAN['lipid']})")
    print(f"n_detections    : {len(dets)}")
    print('-' * 66)
    print(f"predicted lipid_intensity   median={lip_med:9.1f}  "
          f"[5pct={lip_lo:8.1f}, 95pct={lip_hi:9.1f}]")
    print(f"predicted protein_intensity median={pro_med:9.1f}  "
          f"[5pct={pro_lo:8.1f}, 95pct={pro_hi:9.1f}]")
    print(f"synthetic eps floors        lipid~{SYNTH_EPS['lipid']}, "
          f"protein~{SYNTH_EPS['protein']}  (dimmest plausible flux)")
    print('-' * 66)

    lo, hi = PLAUSIBLE_FLUX_BAND
    ok = (len(dets) > 0
          and lo <= lip_med <= hi
          and lo <= pro_med <= hi)
    verdict = 'PASS' if ok else 'WARN'
    print(f"plausible flux band (median): [{lo:.0f}, {hi:.0f}] ADU")
    print(f"VERDICT: {verdict}")
    if not ok:
        print("  WARN: predicted fluxes look orders of magnitude off the "
              "synthetic scale (or no detections).")
        print("  -> Suspect a normalization/offset mismatch. Do NOT trust the "
              "downstream alpha table until this is resolved.")
    else:
        print("  Predicted fluxes are on the synthetic scale; scaling looks "
              "consistent. Safe to proceed.")
    print('=' * 66)


if __name__ == '__main__':
    main()
