"""Entry point for the joint multi-sample calibration pipeline.

Usage:
    python calibrate.py --config configs/calibration/joint_smoke.yaml

Loads a YAML config and runs ``src.calibration.run.run_full_pipeline``. Output
goes to the config's ``output_dir`` (the ``calibrations/`` convention); if the
config omits it, a default of ``calibrations/<config-stem>`` is used.
"""

import argparse
from pathlib import Path

import yaml

from src.calibration.run import run_full_pipeline


def main():
    parser = argparse.ArgumentParser(
        description='Liposome detection calibration pipeline.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to a YAML config file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Honor the config's output_dir (calibrations/...); fall back to a
    # default derived from the config filename if it is absent.
    config.setdefault('output_dir',
                      str(Path('calibrations') / Path(args.config).stem))
    config['_config_path'] = args.config

    run_full_pipeline(config)


if __name__ == '__main__':
    main()
