"""Full calibration pipeline orchestration + comparison plots.

  - _normalize_samples_config: accept single- or multi-sample configs
  - save_comparison_plots    : real-vs-sim image and statistic plots
  - run_full_pipeline        : load -> measure -> optimize -> validate -> save

Ported from the archive's pipeline.py (Modules 10-11). Changes vs archive:
  * output goes wherever ``config['output_dir']`` points (the ``calibrations/``
    convention) instead of a hardcoded ``runs/`` path;
  * a ``discrepancy`` config block is threaded into the optimizer;
  * ``save_comparison_plots`` is wired into the pipeline so each run emits
    real-vs-sim plots (the archive defined it but the joint pipeline never
    called it);
  * provenance is imported directly from ``src.provenance`` rather than via a
    sys.path hack.
All numerical model logic is otherwise unchanged.
"""

import json
import os
from pathlib import Path

import numpy as np

from src.calibration.discrepancy import resolve_discrepancy_config
from src.calibration.optimize import _eval_joint_discrepancy, run_optimization_joint
from src.calibration.statistics import compute_image_statistics
from src.provenance import write_provenance
from src.simulator.estimation import estimate_gain, estimate_psf, extract_backgrounds
from src.simulator.forward_model import _gather_nonpuncta_protein, simulate_batch_dual_bg
from src.simulator.io import analyze_dark_frames, load_all_images, parse_dls


def save_comparison_plots(real_images, sim_protein, sim_lipid,
                         real_stats, sim_stats, output_dir, channel='lipid'):
    """
    Generate side-by-side comparison plots of real vs simulated images.
    Saves as PNG files.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # 1. Example images side by side
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i in range(3):
        if channel == 'lipid':
            real_img = real_images[i]['lipid']
            sim_img = sim_lipid[i]
        else:
            real_img = real_images[i]['protein']
            sim_img = sim_protein[i]

        vmin = np.percentile(real_img, 1)
        vmax = np.percentile(real_img, 99.5)

        axes[0, i].imshow(real_img, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f'Real {i+1}')
        axes[0, i].axis('off')

        axes[1, i].imshow(sim_img, cmap='gray', vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f'Simulated {i+1}')
        axes[1, i].axis('off')

    plt.suptitle(f'{channel.capitalize()} Channel: Real vs Simulated', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_images_{channel}.png'), dpi=150)
    plt.close()

    # 2. Pixel intensity histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(real_stats['pixel_hist_bins'][:-1], real_stats['pixel_hist'],
                 label='Real', alpha=0.7)
    axes[0].plot(sim_stats['pixel_hist_bins'][:-1], sim_stats['pixel_hist'],
                 label='Simulated', alpha=0.7)
    axes[0].set_xlabel('Pixel Intensity')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Pixel Intensity Distribution')
    axes[0].legend()
    axes[0].set_xlim(0, 500)

    # 3. Spot intensity distribution
    if len(real_stats['spot_intensities']) > 0 and len(sim_stats['spot_intensities']) > 0:
        axes[1].hist(real_stats['spot_intensities'], bins=50, density=True,
                     alpha=0.5, label='Real')
        axes[1].hist(sim_stats['spot_intensities'], bins=50, density=True,
                     alpha=0.5, label='Simulated')
        axes[1].set_xlabel('Peak Spot Intensity')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Spot Intensity Distribution')
        axes[1].legend()

    # 4. Power spectral density
    min_len = min(len(real_stats['radial_psd']), len(sim_stats['radial_psd']))
    axes[2].loglog(real_stats['radial_psd'][:min_len], label='Real', alpha=0.7)
    axes[2].loglog(sim_stats['radial_psd'][:min_len], label='Simulated', alpha=0.7)
    axes[2].set_xlabel('Spatial Frequency')
    axes[2].set_ylabel('Power')
    axes[2].set_title('Radial Power Spectral Density')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_stats_{channel}.png'), dpi=150)
    plt.close()

    print(f"  Saved comparison plots to {output_dir}")


def _normalize_samples_config(config):
    """Accept either a single-sample config (legacy) or a multi-sample one
    with `samples: [...]`. Always return a list of sample dicts."""
    if 'samples' in config:
        return list(config['samples'])
    # Legacy single-sample wrap
    return [{
        'name': config.get('name', 'sample0'),
        'image_dir': config['image_dir'],
        'dark_dir': config['dark_dir'],
        'dls_path': config['dls_path'],
        'image_pattern': config.get('image_pattern', '*.tif*'),
        'dark_pattern': config.get('dark_pattern', '*.tif*'),
    }]


def run_full_pipeline(config):
    """
    Joint multi-sample calibration and simulation pipeline.

    `config` keys:
        samples: list of {name, image_dir, dark_dir, dls_path,
                          image_pattern?, dark_pattern?}
        output_dir: results dir
        n_trials, n_sim_per_trial: Optuna controls
        val_fraction: held-out fraction (default 0.2)
        seed: RNG seed for the train/val split (default 0)
        discrepancy: optional per-term {enabled, weight} overrides for the
                     discrepancy loss (defaults reproduce the archive behavior)
    """
    print("=" * 60)
    print("LIPOSOME PUNCTUM DETECTION PIPELINE  (joint multi-sample)")
    print("=" * 60)

    sample_cfgs = _normalize_samples_config(config)
    output_dir = config.get('output_dir', './pipeline_output')
    os.makedirs(output_dir, exist_ok=True)
    val_fraction = float(config.get('val_fraction', 0.2))
    split_seed = int(config.get('seed', 0))
    rng_split = np.random.default_rng(split_seed)
    discrepancy_config = config.get('discrepancy')  # None -> defaults

    samples = []
    measurement_pool = []  # pool images for shared PSF/gain estimation
    for sc in sample_cfgs:
        name = sc['name']
        print(f"\n--- Loading sample: {name} ---")
        dls_d, dls_p, _ = parse_dls(sc['dls_path'], weighting='number',
                                    max_diameter_nm=500)
        peak_d = float(dls_d[np.argmax(dls_p)])
        images = load_all_images(sc['image_dir'],
                                 sc.get('image_pattern', '*.tif*'))
        # load_all_images returns frames in sorted filename order
        # (deterministic). If the per-sample config caps the frame count,
        # take the first N to balance contributions across samples.
        n_frames_cap = sc.get('n_frames_for_calibration', None)
        if n_frames_cap is not None:
            n_frames_cap = int(n_frames_cap)
            if len(images) > n_frames_cap:
                print(f"  capping frames: {len(images)} -> {n_frames_cap}")
                images = images[:n_frames_cap]
        if len(images) < 2:
            raise RuntimeError(f"Sample '{name}' has too few images "
                               f"({len(images)}) for a train/val split.")
        # Train/val split
        n = len(images)
        idx = np.arange(n)
        rng_split.shuffle(idx)
        n_val = max(1, int(round(n * val_fraction)))
        val_idx = set(idx[:n_val].tolist())
        train_imgs = [images[i] for i in range(n) if i not in val_idx]
        val_imgs = [images[i] for i in range(n) if i in val_idx]
        print(f"  split: {len(train_imgs)} train / {len(val_imgs)} val")

        dark_res = analyze_dark_frames(sc['dark_dir'],
                                       sc.get('dark_pattern', '*.tif*'))
        bg_lipid, bg_stats_lipid = extract_backgrounds(train_imgs,
                                                       channel='lipid')
        bg_protein, bg_stats_protein = extract_backgrounds(train_imgs,
                                                           channel='protein')

        rng_stats = np.random.default_rng(split_seed + 1)
        train_stats = compute_image_statistics(train_imgs, channel='lipid',
                                               is_simulated=False)
        train_stats['protein_nonpuncta'] = _gather_nonpuncta_protein(
            train_imgs, rng=rng_stats)
        val_stats = compute_image_statistics(val_imgs, channel='lipid',
                                             is_simulated=False)
        val_stats['protein_nonpuncta'] = _gather_nonpuncta_protein(
            val_imgs, rng=rng_stats)

        samples.append({
            'name': name,
            'images': train_imgs,
            'val_images': val_imgs,
            'dls_diameters': dls_d,
            'dls_probs': dls_p,
            'dls_peak_diameter_nm': peak_d,
            'dark_results': dark_res,
            'bg_patches_lipid': bg_lipid,
            'bg_patches_protein': bg_protein,
            'bg_stats_lipid': bg_stats_lipid,
            'bg_stats_protein': bg_stats_protein,
            'train_stats': train_stats,
            'val_stats': val_stats,
            'n_frames_used': int(len(images)),
        })
        measurement_pool.extend(train_imgs)

    # Shared PSF / gain measurement on pooled training images
    print("\n--- Shared microscope measurements (pooled) ---")
    gain = estimate_gain(measurement_pool, samples[0]['dark_results'],
                         channel='lipid')
    psf_sx, psf_sy, _ = estimate_psf(measurement_pool, channel='lipid')
    psf_sx_p, psf_sy_p, _ = estimate_psf(measurement_pool, channel='protein')
    measured_params = {
        'psf_sigma_x': psf_sx, 'psf_sigma_y': psf_sy,
        'psf_sigma_x_protein': psf_sx_p, 'psf_sigma_y_protein': psf_sy_p,
        'gain': gain,
    }

    # Joint optimization
    shared_best, per_sample_best, study, train_loss = run_optimization_joint(
        samples, measured_params,
        n_trials=int(config.get('n_trials', 150)),
        n_sim_per_trial=int(config.get('n_sim_per_trial', 30)),
        discrepancy_config=discrepancy_config,
    )

    # Validation discrepancy on held-out 20% (using same params)
    print("\n--- Validation discrepancy on held-out split ---")
    # Build a samples-shaped dict that uses val_stats
    val_loss = _eval_joint_discrepancy(
        shared_best, per_sample_best, samples,
        n_sim_per_trial=int(config.get('n_sim_per_trial', 30)),
        seed_base=99, image_stats_key='val',
        discrepancy_config=discrepancy_config,
    )
    print(f"  training_discrepancy   = {train_loss:.4f}")
    print(f"  validation_discrepancy = {val_loss:.4f}")

    # Persist results
    save_results = {
        'best_params': {k: float(v) if isinstance(v, (np.floating, float))
                        else v for k, v in shared_best.items()},
        'per_sample_params': {
            n: {k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in p.items()}
            for n, p in per_sample_best.items()
        },
        'training_discrepancy': float(train_loss),
        'validation_discrepancy': float(val_loss),
        'measured_params': {k: float(v) for k, v in measured_params.items()},
        'discrepancy_config': resolve_discrepancy_config(discrepancy_config),
        'dark_results': {
            s['name']: {
                ch: {'offset': float(d['offset']),
                     'read_noise_var': float(d['read_noise_var'])}
                for ch, d in s['dark_results'].items()
            }
            for s in samples
        },
        'dls_peak_diameter_nm': {s['name']: s['dls_peak_diameter_nm']
                                 for s in samples},
        'bg_stats': {
            s['name']: {'lipid': s['bg_stats_lipid'],
                        'protein': s['bg_stats_protein']}
            for s in samples
        },
        'sample_names': [s['name'] for s in samples],
        'per_sample_n_frames_used': {s['name']: int(s['n_frames_used'])
                                     for s in samples},
        'val_fraction': val_fraction,
    }
    results_path = os.path.join(output_dir, 'calibration_results.json')
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Saved calibration results to {results_path}")

    # Comparison plots per sample (lipid channel) using the best-fit params.
    # Uses the existing simulator + plotting code (no new physics); the
    # archive defined save_comparison_plots but never wired it into the joint
    # pipeline.
    n_sim_plot = max(3, int(config.get('n_sim_per_trial', 30)))
    for k, sample in enumerate(samples):
        sp = per_sample_best[sample['name']]
        sim_params = dict(shared_best)
        sim_params['offset_lipid'] = sample['dark_results']['lipid']['offset']
        sim_params['offset_protein'] = sample['dark_results']['protein']['offset']
        sim_params['read_noise_var_lipid'] = sample['dark_results']['lipid']['read_noise_var']
        sim_params['read_noise_var_protein'] = sample['dark_results']['protein']['read_noise_var']
        sim_params['spot_density'] = sp['spot_density']
        sim_params['bg_amplitude_protein'] = sp['bg_amplitude_protein']
        sim_params['autofl_protein'] = sp['autofl_protein']
        sim_params.setdefault('curvature_alpha', 1.0)
        sim_protein, sim_lipid, _ = simulate_batch_dual_bg(
            sim_params, sample['dls_diameters'], sample['dls_probs'],
            sample['bg_patches_lipid'], sample['bg_patches_protein'],
            n_images=n_sim_plot, seed=12345 + k,
        )
        sim_stats_lipid = compute_image_statistics(sim_lipid, is_simulated=True)
        plot_dir = os.path.join(output_dir, 'plots', sample['name'])
        try:
            save_comparison_plots(sample['images'], sim_protein, sim_lipid,
                                  sample['train_stats'], sim_stats_lipid,
                                  plot_dir, channel='lipid')
        except Exception as e:
            print(f"  warning: comparison plot failed for {sample['name']}: {e}")

    try:
        write_provenance(
            output_dir,
            config_path=config.get('_config_path', '<inline>'),
            n_trials=int(config.get('n_trials', 0)),
            n_sim_per_trial=int(config.get('n_sim_per_trial', 0)),
            samples_used=[s['name'] for s in samples],
            val_fraction=val_fraction,
        )
    except Exception as e:
        print(f"  warning: provenance write failed: {e}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    return shared_best, per_sample_best, measured_params
