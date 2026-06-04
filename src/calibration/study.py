"""Reusable parallel calibration-study runner.

A *study* runs many independent single-sample calibrations (each the standard
``src.calibration.run.run_full_pipeline``) in parallel and aggregates their
fitted parameters into the experiment folder. There are three modes:

  per_sample   : calibrate each of several samples independently.
  bootstrap    : one sample, ``n_repeats`` repeats, each on a random
                 ``d``-image subset (distinct seed per repeat).
  weight_sweep : one sample, calibrated once per named discrepancy-weight
                 config.

Parallelism and single-threaded numpy
-------------------------------------
Calibrations are CPU-bound and independent, so they run in a
``multiprocessing.Pool`` of ``n_workers`` processes (CLI ``--n-workers`` /
function arg; default ``os.cpu_count()``). Each calibration's numpy/BLAS would
otherwise spawn its own thread pool, oversubscribing the cores. To prevent that,
every worker pins OMP/MKL/OpenBLAS thread counts to 1 in a worker *initializer*
(:func:`_init_worker`) that runs BEFORE numpy is imported in that worker.

This is why **this module imports nothing heavy at module top** — no numpy, no
``run_full_pipeline``. Those are imported lazily inside :func:`_run_one_calibration`,
which executes only after the initializer has pinned the thread counts. The pool
also uses the ``spawn`` start method so every worker is a fresh interpreter that
runs the initializer before importing numpy, identically on Windows and Linux.

CLI
---
    python -m src.calibration.study --config <study.yaml> --n-workers N

Outputs (under the study config's ``output_dir`` = the experiment folder):
    runs/<run_id>/            one standard calibration each (results, plots,
                              provenance, trials.csv, convergence.png)
    results.json              list of {run_id, fitted_params, status}
    aggregated_params.csv     readable fitted-parameter table
    run_manifest.json         per-run diagnostics (pid, threads, timing, errors)
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import yaml

# Shared (microscope) parameters fitted by every calibration, in display order.
SHARED_PARAM_KEYS = (
    'lipid_brightness', 'psf_sigma_x', 'psf_sigma_y', 'psf_theta',
    'gain', 'enf', 'optical_bg_lipid',
)
# Full fitted-parameter set per run (shared + the single per-sample free param).
FITTED_PARAM_KEYS = SHARED_PARAM_KEYS + ('spot_density',)

# Thread-count env vars pinned to 1 in every worker to avoid oversubscription.
# The first three are the ones that matter for numpy's common BLAS backends; the
# last two are harmless extras (numexpr / macOS Accelerate) set for good measure.
_THREAD_ENV_VARS = (
    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
)


def _init_worker():
    """Pool worker initializer: pin BLAS/OpenMP thread counts to 1.

    Runs once per worker process, BEFORE that worker imports numpy (the heavy
    imports live inside :func:`_run_one_calibration`). With one thread per
    calibration, ``n_workers`` calibrations use about ``n_workers`` cores in
    total rather than ``n_workers * cores`` threads.
    """
    for var in _THREAD_ENV_VARS:
        os.environ[var] = '1'


# --------------------------------------------------------------------------- #
# Reading a finished calibration                                              #
# --------------------------------------------------------------------------- #
def _read_results(output_dir):
    """Load a finished calibration's ``calibration_results.json``."""
    with open(Path(output_dir) / 'calibration_results.json') as f:
        return json.load(f)


def _fitted_params_from_results(results):
    """Flatten a calibration's fitted params (shared + the one sample's free
    ``spot_density``) into a single dict. Each study run is single-sample."""
    fitted = {k: results.get('best_params', {}).get(k) for k in SHARED_PARAM_KEYS}
    per_sample = results.get('per_sample_params', {})
    if per_sample:
        only = next(iter(per_sample.values()))  # single sample per run
        fitted['spot_density'] = only.get('spot_density')
    else:
        fitted['spot_density'] = None
    return fitted


# --------------------------------------------------------------------------- #
# The unit of work (runs in a worker process)                                 #
# --------------------------------------------------------------------------- #
def _run_one_calibration(job):
    """Run one calibration in a worker. Returns a record; never raises.

    ``job`` is ``(run_id, run_config)`` where ``run_config`` is a complete
    ``run_full_pipeline`` config (single-sample). Heavy imports (numpy, via
    ``run_full_pipeline``) happen HERE, after :func:`_init_worker` has pinned the
    thread counts for this process.
    """
    run_id, run_config = job
    rec = {
        'run_id': run_id,
        'output_dir': run_config.get('output_dir'),
        'pid': os.getpid(),
        # Captured to verify the single-threaded-per-worker setup actually took.
        'omp_num_threads': os.environ.get('OMP_NUM_THREADS'),
        'mkl_num_threads': os.environ.get('MKL_NUM_THREADS'),
        'openblas_num_threads': os.environ.get('OPENBLAS_NUM_THREADS'),
        'start_time': time.time(),
        'status': 'failed',
        'error': None,
        'fitted_params': None,
        'training_discrepancy': None,
        'validation_discrepancy': None,
        'measured_gain': None,
    }
    try:
        from src.calibration.run import run_full_pipeline  # imports numpy now
        run_full_pipeline(run_config)
        results = _read_results(run_config['output_dir'])
        rec['fitted_params'] = _fitted_params_from_results(results)
        rec['training_discrepancy'] = results.get('training_discrepancy')
        rec['validation_discrepancy'] = results.get('validation_discrepancy')
        rec['measured_gain'] = results.get('measured_params', {}).get('gain')
        rec['status'] = 'ok'
    except Exception as e:
        rec['error'] = f'{type(e).__name__}: {e}'
        rec['traceback'] = traceback.format_exc()
    rec['end_time'] = time.time()
    rec['duration_sec'] = rec['end_time'] - rec['start_time']
    return rec


# --------------------------------------------------------------------------- #
# Building the job list for each mode                                         #
# --------------------------------------------------------------------------- #
def _base_run_config(study_config, run_id):
    """Per-run config fields common to all modes."""
    exp_dir = study_config['output_dir']
    cfg = {
        'output_dir': str(Path(exp_dir) / 'runs' / run_id),
        'n_trials': int(study_config.get('n_trials', 200)),
        'n_sim_per_trial': int(study_config.get('n_sim_per_trial', 30)),
        'val_fraction': float(study_config.get('val_fraction', 0.2)),
        'seed': int(study_config.get('seed', 0)),
        # Parallel study: suppress per-trial tqdm bars (dozens would interleave).
        'show_progress_bar': False,
        '_config_path': study_config.get('_config_path', '<study>'),
    }
    return cfg


def _jobs_per_sample(study_config):
    """per_sample: one independent calibration per sample."""
    jobs = []
    for sample in study_config['samples']:
        run_id = sample['name']
        cfg = _base_run_config(study_config, run_id)
        cfg['samples'] = [dict(sample)]
        if study_config.get('discrepancy') is not None:
            cfg['discrepancy'] = study_config['discrepancy']
        jobs.append((run_id, cfg))
    return jobs


def _jobs_bootstrap(study_config):
    """bootstrap: one sample, n_repeats subsets of d images (distinct seed)."""
    sample = study_config['sample']
    bs = study_config['bootstrap']
    d = int(bs['d'])
    n_repeats = int(bs['n_repeats'])
    replace = bool(bs.get('replacement', False))
    base_seed = int(bs.get('base_seed', 0))
    jobs = []
    pad = max(3, len(str(n_repeats - 1)))
    for i in range(n_repeats):
        run_id = f'repeat_{i:0{pad}d}'
        cfg = _base_run_config(study_config, run_id)
        s = dict(sample)
        s['frame_sample_size'] = d
        s['frame_sample_seed'] = base_seed + i      # distinct subset per repeat
        s['frame_sample_replacement'] = replace
        cfg['samples'] = [s]
        cfg['seed'] = base_seed + i                  # vary the train/val split too
        if study_config.get('discrepancy') is not None:
            cfg['discrepancy'] = study_config['discrepancy']
        jobs.append((run_id, cfg))
    return jobs


def _jobs_weight_sweep(study_config):
    """weight_sweep: one sample, one calibration per named weight config."""
    sample = study_config['sample']
    jobs = []
    for wc in study_config['weight_configs']:
        run_id = wc['name']
        cfg = _base_run_config(study_config, run_id)
        cfg['samples'] = [dict(sample)]
        cfg['discrepancy'] = wc['discrepancy']
        jobs.append((run_id, cfg))
    return jobs


_MODE_BUILDERS = {
    'per_sample': _jobs_per_sample,
    'bootstrap': _jobs_bootstrap,
    'weight_sweep': _jobs_weight_sweep,
}


def build_jobs(study_config):
    """Return ``[(run_id, run_config), ...]`` for the study's mode."""
    mode = study_config['mode']
    if mode not in _MODE_BUILDERS:
        raise ValueError(
            f"Unknown study mode '{mode}'. Valid: {sorted(_MODE_BUILDERS)}")
    return _MODE_BUILDERS[mode](study_config)


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #
def _write_aggregates(records, exp_dir, write_manifest=True):
    """Write results.json + aggregated_params.csv (and optionally run_manifest).

    ``exp_dir`` is the experiment folder. ``write_manifest`` is False when
    re-aggregating from disk (the per-worker diagnostics can't be reconstructed,
    so the original run_manifest.json is left untouched).
    """
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # results.json: the canonical aggregated parameter table.
    results = [{'run_id': r['run_id'],
                'fitted_params': r['fitted_params'],
                'status': r['status']} for r in records]
    (exp_dir / 'results.json').write_text(json.dumps(results, indent=2))

    # Readable CSV table.
    with open(exp_dir / 'aggregated_params.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run_id', 'status', *FITTED_PARAM_KEYS,
                    'training_discrepancy', 'validation_discrepancy',
                    'duration_sec', 'pid'])
        for r in records:
            fp = r['fitted_params'] or {}
            dur = r.get('duration_sec')
            w.writerow([
                r['run_id'], r['status'],
                *[fp.get(k, '') for k in FITTED_PARAM_KEYS],
                r['training_discrepancy'] if r['training_discrepancy'] is not None else '',
                r['validation_discrepancy'] if r['validation_discrepancy'] is not None else '',
                '' if dur is None else round(dur, 2),
                r.get('pid') if r.get('pid') is not None else '',
            ])

    if write_manifest:
        # Full manifest with worker diagnostics (supports the parallel/threads check).
        (exp_dir / 'run_manifest.json').write_text(json.dumps(records, indent=2))
    msg = "results.json, aggregated_params.csv"
    if write_manifest:
        msg += ", run_manifest.json"
    print(f"[study] wrote {msg} to {exp_dir}")


def _record_from_results_dir(run_dir):
    """Build a run record from an existing ``runs/<run_id>/`` folder.

    Used by :func:`aggregate_from_runs` (no calibration). Run-diagnostic fields
    (pid / threads / duration) are not recoverable from disk and are None.
    """
    run_dir = Path(run_dir)
    rec = {
        'run_id': run_dir.name, 'status': 'failed', 'pid': None,
        'omp_num_threads': None, 'mkl_num_threads': None,
        'openblas_num_threads': None, 'duration_sec': None,
        'fitted_params': None, 'training_discrepancy': None,
        'validation_discrepancy': None, 'measured_gain': None, 'error': None,
    }
    try:
        results = _read_results(run_dir)
        rec['fitted_params'] = _fitted_params_from_results(results)
        rec['training_discrepancy'] = results.get('training_discrepancy')
        rec['validation_discrepancy'] = results.get('validation_discrepancy')
        rec['measured_gain'] = results.get('measured_params', {}).get('gain')
        rec['status'] = 'ok'
    except Exception as e:
        rec['error'] = f'{type(e).__name__}: {e}'
    return rec


def aggregate_from_runs(exp_dir):
    """Re-aggregate results.json + aggregated_params.csv from existing
    ``runs/<run_id>/calibration_results.json`` WITHOUT recalibrating.

    Lets the analysis be re-run (or a plot fixed) on existing outputs without
    redoing the calibrations. run_manifest.json is left untouched (its
    per-worker diagnostics can't be reconstructed from disk). Returns the
    rebuilt records.
    """
    exp_dir = Path(exp_dir)
    runs_dir = exp_dir / 'runs'
    run_dirs = sorted(d for d in runs_dir.glob('*') if d.is_dir())
    if not run_dirs:
        raise FileNotFoundError(
            f"No runs/<run_id>/ subdirectories under {runs_dir} to aggregate.")
    records = [_record_from_results_dir(d) for d in run_dirs]
    _write_aggregates(records, exp_dir, write_manifest=False)
    n_ok = sum(r['status'] == 'ok' for r in records)
    print(f"[study] re-aggregated {n_ok}/{len(records)} runs from {runs_dir}")
    return records


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def run_study(study_config, n_workers=None):
    """Run all calibrations for ``study_config`` in parallel and aggregate.

    Args:
        study_config: parsed study YAML (see module docstring / experiment
            ``config_snapshot/study.yaml`` files).
        n_workers: worker-process count; ``None`` -> ``os.cpu_count()``.

    Returns the list of per-run records.
    """
    mode = study_config['mode']
    exp_dir = Path(study_config['output_dir'])
    (exp_dir / 'runs').mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(study_config)
    if not jobs:
        raise RuntimeError('Study produced no calibration jobs.')

    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, min(int(n_workers), len(jobs)))

    print('=' * 60)
    print(f"[study] mode={mode}  runs={len(jobs)}  n_workers={n_workers}")
    print(f"[study] output_dir={exp_dir}")
    print('=' * 60)

    order = {rid: idx for idx, (rid, _) in enumerate(jobs)}
    records = []
    t0 = time.time()

    if n_workers == 1:
        # Serial path (still pin threads for parity with the parallel path).
        _init_worker()
        for i, job in enumerate(jobs, 1):
            rec = _run_one_calibration(job)
            _print_progress(i, len(jobs), rec)
            records.append(rec)
    else:
        # Force 'spawn' so every worker is a fresh interpreter that runs
        # _init_worker (pinning threads) BEFORE importing numpy, on every OS.
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=n_workers, initializer=_init_worker) as pool:
            for i, rec in enumerate(
                    pool.imap_unordered(_run_one_calibration, jobs), 1):
                _print_progress(i, len(jobs), rec)
                records.append(rec)

    records.sort(key=lambda r: order.get(r['run_id'], 1 << 30))
    _write_aggregates(records, exp_dir)

    n_ok = sum(r['status'] == 'ok' for r in records)
    n_fail = len(records) - n_ok
    pids = sorted({r['pid'] for r in records})
    print('=' * 60)
    print(f"[study] DONE in {time.time() - t0:.1f}s  "
          f"ok={n_ok} failed={n_fail}  distinct worker pids={len(pids)}")
    if n_fail:
        for r in records:
            if r['status'] != 'ok':
                print(f"[study]   FAILED {r['run_id']}: {r['error']}")
    print('=' * 60)
    return records


def _print_progress(i, total, rec):
    msg = (f"[study] ({i}/{total}) {rec['run_id']}: {rec['status']} "
           f"({rec.get('duration_sec', 0.0):.1f}s, pid={rec['pid']}, "
           f"OMP={rec['omp_num_threads']})")
    print(msg)
    if rec['status'] != 'ok':
        print(f"[study]   error: {rec['error']}")


def main():
    parser = argparse.ArgumentParser(
        description='Parallel calibration-study runner (per_sample / bootstrap '
                    '/ weight_sweep).')
    parser.add_argument('--config',
                        help='Path to a study YAML config (required unless '
                             '--aggregate-only is given).')
    parser.add_argument('--n-workers', type=int, default=None,
                        help='Worker processes (default: os.cpu_count()).')
    parser.add_argument('--aggregate-only', metavar='EXP_DIR', default=None,
                        help='Skip calibration entirely; re-aggregate '
                             'results.json + aggregated_params.csv from an '
                             "experiment folder's existing runs/. Use to "
                             're-aggregate or fix a plot without recalibrating.')
    args = parser.parse_args()

    # Re-aggregate-only mode: no study, no calibration.
    if args.aggregate_only:
        aggregate_from_runs(args.aggregate_only)
        return

    if not args.config:
        parser.error('--config is required unless --aggregate-only is given.')

    with open(args.config) as f:
        study_config = yaml.safe_load(f)
    study_config.setdefault(
        'output_dir', str(Path('experiments') / Path(args.config).stem))
    study_config['_config_path'] = args.config

    run_study(study_config, n_workers=args.n_workers)


if __name__ == '__main__':
    main()
