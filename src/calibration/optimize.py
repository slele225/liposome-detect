"""Joint multi-sample moment-matching optimization (Optuna).

  - _build_sample_real_stats: real-image stats for one sample's joint loss
  - _eval_joint_discrepancy : summed per-sample discrepancy for one assignment
  - run_optimization_joint  : Optuna search over shared + per-sample params

Ported from the archive's pipeline.py (Module 9). The only behavioral change
is that a ``discrepancy_config`` is threaded down to ``compute_discrepancy`` so
per-term weights are configurable (defaults reproduce the old behavior).
"""

import numpy as np

from src.calibration.discrepancy import compute_discrepancy
from src.calibration.statistics import compute_image_statistics
from src.simulator.forward_model import (
    _gather_nonpuncta_protein,
    simulate_batch_dual_bg,
    simulate_protein_calibration_batch,
)


def _build_sample_real_stats(sample, rng):
    """
    Compute the real-image statistics needed for the joint loss for one sample,
    given a sample dict that has been augmented with 'images' (list of dicts).
    Returns (lipid_stats_with_nonpuncta, n_images).
    """
    real_stats_lipid = compute_image_statistics(sample['images'], channel='lipid',
                                                is_simulated=False)
    real_stats_lipid['protein_nonpuncta'] = _gather_nonpuncta_protein(
        sample['images'], rng=rng)
    return real_stats_lipid


def _eval_joint_discrepancy(shared_params, per_sample_params, samples,
                            n_sim_per_trial, seed_base, image_stats_key='train',
                            discrepancy_config=None):
    """
    Sum the per-sample discrepancy for one parameter assignment over all samples.
    `samples[i]` must have keys: dls_diameters, dls_probs, bg_patches_lipid,
    bg_patches_protein, dark_results, train_stats / val_stats.
    """
    total = 0.0
    rng = np.random.default_rng(seed_base)
    for k, sample in enumerate(samples):
        sample_params = per_sample_params[sample['name']]
        # Lipid sim uses shared params
        sim_params = dict(shared_params)
        sim_params['offset_lipid'] = sample['dark_results']['lipid']['offset']
        sim_params['offset_protein'] = sample['dark_results']['protein']['offset']
        sim_params['read_noise_var_lipid'] = sample['dark_results']['lipid']['read_noise_var']
        sim_params['read_noise_var_protein'] = sample['dark_results']['protein']['read_noise_var']
        sim_params['spot_density'] = sample_params['spot_density']
        # alpha is not fitted in calibration; pin to 1.0. The protein channel
        # output of simulate_image is overwritten below by
        # simulate_protein_calibration_batch (no-spot sim), so the spot-bound
        # protein intensities here are discarded.
        sim_params.setdefault('curvature_alpha', 1.0)

        sim_protein_full, sim_lipid, _ = simulate_batch_dual_bg(
            sim_params, sample['dls_diameters'], sample['dls_probs'],
            sample['bg_patches_lipid'], sample['bg_patches_protein'],
            n_images=n_sim_per_trial, seed=seed_base + 1000 * k,
        )
        # Replace protein channel with the calibration-only (no-spot) sim
        sim_protein = simulate_protein_calibration_batch(
            sim_params, sample_params, sample['dark_results'],
            sample['bg_patches_protein'], n_images=n_sim_per_trial,
            seed=seed_base + 1000 * k + 7,
        )
        sim_stats_lipid = compute_image_statistics(sim_lipid, is_simulated=True)
        sim_stats_lipid['protein_nonpuncta'] = _gather_nonpuncta_protein(
            None, sim_protein_arrays=sim_protein, sim_lipid_arrays=sim_lipid,
            rng=rng,
        )
        real_stats = sample[f'{image_stats_key}_stats']
        total += compute_discrepancy(real_stats, sim_stats_lipid,
                                     discrepancy_config=discrepancy_config)
    return total


def run_optimization_joint(samples, measured_params,
                           n_trials=150, n_sim_per_trial=30,
                           discrepancy_config=None):
    """
    Joint multi-sample Optuna calibration.

    Shared (microscope) parameters across all samples:
        gain, enf, psf_sigma_x, psf_sigma_y, bg_amplitude, haze_level,
        labeling_eff (lipid brightness coefficient).
    Per-sample free parameters:
        spot_density, bg_amplitude_protein, autofl_protein, voltage_scale_protein.
    Pinned per-sample: dark frame offset / read_noise_var.

    Returns:
        shared_best, per_sample_best, study, training_loss
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("=== Joint Multi-Sample Calibration ===")
    print(f"  Samples ({len(samples)}): " + ", ".join(s['name'] for s in samples))

    def objective(trial):
        shared = {
            'labeling_eff': trial.suggest_float('labeling_eff', 50, 3000, log=True),
            'psf_sigma_x': trial.suggest_float('psf_sigma_x',
                measured_params['psf_sigma_x'] * 0.8,
                measured_params['psf_sigma_x'] * 1.2),
            'psf_sigma_y': trial.suggest_float('psf_sigma_y',
                measured_params['psf_sigma_y'] * 0.8,
                measured_params['psf_sigma_y'] * 1.2),
            'gain': trial.suggest_float('gain', 1.0, 100.0, log=True),
            'enf': trial.suggest_float('enf', 1.0, 2.5),
            'bg_amplitude': trial.suggest_float('bg_amplitude', 0.0, 2.0),
            'haze_level': trial.suggest_float('haze_level', 0.0, 50.0),
            'n_frame_avg': 3,
        }
        per_sample = {}
        for s in samples:
            ref_spots = max(1.0, s['train_stats']['mean_spot_count'])
            name = s['name']
            per_sample[name] = {
                'spot_density': trial.suggest_float(
                    f'spot_density__{name}', ref_spots * 0.5, ref_spots * 2.0),
                'bg_amplitude_protein': trial.suggest_float(
                    f'bg_amplitude_protein__{name}', 0.0, 2.0),
                'autofl_protein': trial.suggest_float(
                    f'autofl_protein__{name}', 0.0, 200.0),
                'voltage_scale_protein': trial.suggest_float(
                    f'voltage_scale_protein__{name}', 0.1, 5.0, log=True),
            }
        return _eval_joint_discrepancy(shared, per_sample, samples,
                                       n_sim_per_trial, seed_base=42,
                                       image_stats_key='train',
                                       discrepancy_config=discrepancy_config)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=1)

    print(f"\n  Best joint training loss: {study.best_value:.4f}")
    bp = study.best_params
    shared_best = {k: bp[k] for k in (
        'labeling_eff', 'psf_sigma_x', 'psf_sigma_y', 'gain', 'enf',
        'bg_amplitude', 'haze_level',
    )}
    shared_best['n_frame_avg'] = 3
    per_sample_best = {}
    for s in samples:
        n = s['name']
        per_sample_best[n] = {
            'spot_density': bp[f'spot_density__{n}'],
            'bg_amplitude_protein': bp[f'bg_amplitude_protein__{n}'],
            'autofl_protein': bp[f'autofl_protein__{n}'],
            'voltage_scale_protein': bp[f'voltage_scale_protein__{n}'],
            'offset_lipid': s['dark_results']['lipid']['offset'],
            'offset_protein': s['dark_results']['protein']['offset'],
            'read_noise_var_lipid': s['dark_results']['lipid']['read_noise_var'],
            'read_noise_var_protein': s['dark_results']['protein']['read_noise_var'],
        }
    return shared_best, per_sample_best, study, study.best_value
