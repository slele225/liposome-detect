"""Joint multi-sample moment-matching optimization (Optuna), lipid-only.

  - _eval_joint_discrepancy : summed per-sample lipid discrepancy for one assignment
  - run_optimization_joint  : Optuna search over shared + per-sample params

The objective uses ONLY the lipid channel (detection-free moment matching); the
protein channel is not simulated or scored during calibration.

Per-trial objective logging
---------------------------
``run_optimization_joint`` records (trial_number, objective_value, params) for
every Optuna trial via a callback and, when given an ``output_dir``, writes
``trials.csv`` and a ``convergence.png`` (objective vs trial with the
running-best overlaid) into it. This is lightweight and applies to every
calibration (single ``calibrate.py`` runs and every run inside a study).
"""

import csv
import os

import numpy as np

from src.calibration.discrepancy import compute_discrepancy
from src.calibration.statistics import compute_image_statistics
from src.simulator.forward_model import simulate_batch


def _eval_joint_discrepancy(shared_params, per_sample_params, samples,
                            n_sim_per_trial, seed_base, image_stats_key='train',
                            discrepancy_config=None):
    """
    Sum the per-sample lipid discrepancy for one parameter assignment over all
    samples. `samples[i]` must have keys: dls_diameters, dls_probs,
    dark_results, train_stats / val_stats.
    """
    total = 0.0
    for k, sample in enumerate(samples):
        sample_params = per_sample_params[sample['name']]
        # Lipid sim uses shared params + this sample's pinned dark-frame floor.
        sim_params = dict(shared_params)
        sim_params['offset_lipid'] = sample['dark_results']['lipid']['offset']
        sim_params['read_noise_var_lipid'] = sample['dark_results']['lipid']['read_noise_var']
        sim_params['spot_density'] = sample_params['spot_density']

        _, sim_lipid, _ = simulate_batch(
            sim_params, sample['dls_diameters'], sample['dls_probs'],
            n_images=n_sim_per_trial, seed=seed_base + 1000 * k, lipid_only=True,
        )
        sim_stats_lipid = compute_image_statistics(sim_lipid, is_simulated=True)
        real_stats = sample[f'{image_stats_key}_stats']
        total += compute_discrepancy(real_stats, sim_stats_lipid,
                                     discrepancy_config=discrepancy_config)
    return total


def _save_trial_log(trial_records, output_dir):
    """Write per-trial objective logs to ``output_dir``.

    Saves ``trials.csv`` (trial_number, objective_value, one column per Optuna
    param) and ``convergence.png`` (objective vs trial with the running-best
    minimum overlaid). Best-effort: never raises into the optimizer.
    """
    if not trial_records:
        return
    os.makedirs(output_dir, exist_ok=True)

    # Stable column order: take param names from the first record (they are the
    # same set every trial), then union in any stragglers for safety.
    param_keys = list(trial_records[0].get('params', {}).keys())
    for rec in trial_records:
        for k in rec.get('params', {}):
            if k not in param_keys:
                param_keys.append(k)

    csv_path = os.path.join(output_dir, 'trials.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['trial_number', 'objective_value'] + param_keys)
        for rec in trial_records:
            params = rec.get('params', {})
            writer.writerow(
                [rec['trial_number'], rec['objective_value']]
                + [params.get(k, '') for k in param_keys])

    # Convergence plot: per-trial objective + running best (cumulative min).
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        trials = np.array([r['trial_number'] for r in trial_records], dtype=float)
        values = np.array(
            [r['objective_value'] if r['objective_value'] is not None
             else np.nan for r in trial_records], dtype=float)
        # np.fmin.accumulate ignores NaNs (failed/pruned trials).
        running_best = np.fmin.accumulate(values)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(trials, values, s=18, alpha=0.5, color='C0',
                   label='trial objective')
        ax.plot(trials, running_best, color='C3', lw=2.0,
                label='running best (min)')
        ax.set_xlabel('Trial number')
        ax.set_ylabel('Objective (discrepancy)')
        ax.set_title('Calibration convergence')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'convergence.png'), dpi=150)
        plt.close(fig)
    except Exception as e:  # pragma: no cover - plotting is non-critical
        print(f"  warning: convergence plot failed: {e}")


def run_optimization_joint(samples, measured_params,
                           n_trials=150, n_sim_per_trial=30,
                           discrepancy_config=None, output_dir=None,
                           show_progress_bar=True):
    """
    Joint multi-sample Optuna calibration (lipid channel only).

    Fitted shared (microscope) parameters across all samples:
        lipid_brightness, psf_sigma_x, psf_sigma_y, psf_theta (lipid PSF),
        gain, enf, optical_bg_lipid.
    Fitted per-sample free parameter:
        spot_density.
    Pinned per-sample (measured from dark frames): offset / read_noise_var
        (lipid + protein; protein values recorded for downstream generation).

    Args:
        output_dir: if given, write ``trials.csv`` and ``convergence.png`` here.
        show_progress_bar: forwarded to Optuna; set False inside parallel
            studies to avoid many interleaved tqdm bars.

    Returns:
        shared_best, per_sample_best, study, training_loss
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("=== Joint Multi-Sample Calibration (lipid-only) ===")
    print(f"  Samples ({len(samples)}): " + ", ".join(s['name'] for s in samples))

    def objective(trial):
        shared = {
            # Total integrated flux (ADU) of a d=100nm lipid spot. The PSF is
            # sum-normalized, so this is total flux, not a peak; the range
            # brackets the ~5k-15k implied by the real bright-spot peaks.
            'lipid_brightness': trial.suggest_float('lipid_brightness', 100.0, 50000.0, log=True),
            # Lipid PSF: rotated 2D Gaussian (sigma_x, sigma_y, theta).
            'psf_sigma_x': trial.suggest_float('psf_sigma_x',
                measured_params['psf_sigma_x'] * 0.8,
                measured_params['psf_sigma_x'] * 1.2),
            'psf_sigma_y': trial.suggest_float('psf_sigma_y',
                measured_params['psf_sigma_y'] * 0.8,
                measured_params['psf_sigma_y'] * 1.2),
            'psf_theta': trial.suggest_float('psf_theta', -45.0, 45.0),
            'gain': trial.suggest_float('gain', 1.0, 100.0, log=True),
            'enf': trial.suggest_float('enf', 1.0, 2.5),
            # Optical background in PHOTONS (injected at the photon stage).
            'optical_bg_lipid': trial.suggest_float('optical_bg_lipid', 0.0, 20.0),
            'n_frame_avg': 3,
        }
        per_sample = {}
        for s in samples:
            ref_spots = max(1.0, s['train_stats']['mean_spot_count'])
            name = s['name']
            per_sample[name] = {
                'spot_density': trial.suggest_float(
                    f'spot_density__{name}', ref_spots * 0.5, ref_spots * 2.0),
            }
        return _eval_joint_discrepancy(shared, per_sample, samples,
                                       n_sim_per_trial, seed_base=42,
                                       image_stats_key='train',
                                       discrepancy_config=discrepancy_config)

    # Per-trial objective log: a callback records every trial's number, value
    # and suggested params; written to trials.csv / convergence.png below.
    trial_records = []

    def _logging_callback(study, trial):
        trial_records.append({
            'trial_number': trial.number,
            'objective_value': (float(trial.value)
                                if trial.value is not None else None),
            'params': dict(trial.params),
        })

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials,
                   show_progress_bar=show_progress_bar, n_jobs=1,
                   callbacks=[_logging_callback])

    if output_dir is not None:
        _save_trial_log(trial_records, output_dir)

    print(f"\n  Best joint training loss: {study.best_value:.4f}")
    bp = study.best_params
    shared_best = {k: bp[k] for k in (
        'lipid_brightness', 'psf_sigma_x', 'psf_sigma_y', 'psf_theta',
        'gain', 'enf', 'optical_bg_lipid',
    )}
    shared_best['n_frame_avg'] = 3
    per_sample_best = {}
    for s in samples:
        n = s['name']
        per_sample_best[n] = {
            'spot_density': bp[f'spot_density__{n}'],
            'offset_lipid': s['dark_results']['lipid']['offset'],
            'offset_protein': s['dark_results']['protein']['offset'],
            'read_noise_var_lipid': s['dark_results']['lipid']['read_noise_var'],
            'read_noise_var_protein': s['dark_results']['protein']['read_noise_var'],
        }
    return shared_best, per_sample_best, study, study.best_value
