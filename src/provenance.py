"""Provenance metadata for generated artifacts (calibrations, datasets, models).

Every producer (calibration, and later data generation / training) calls
`write_provenance` once after writing its outputs so that the artifact
directory contains a machine-readable record of how it was made.

Ported from the archive's analysis/provenance.py. It is a dependency of
``src.calibration.run.run_full_pipeline`` (which the task's function list did
not enumerate), so it is carried over here. ``repo_root`` is the parent of
``src/`` (this file lives at ``src/provenance.py``).
"""

from pathlib import Path
import json
import subprocess
import datetime
import sys


def write_provenance(output_dir, config_path, **extra):
    """Write a provenance.json to output_dir documenting how this
    artifact was created."""
    info = {
        "creation_date": datetime.datetime.now().isoformat(),
        "config_path": str(config_path),
        "python_version": sys.version,
        "code_commit": "unknown",
    }

    repo_root = Path(__file__).resolve().parent.parent

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
        ).strip()
        info["code_commit"] = sha
    except Exception:
        pass

    try:
        diff = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            text=True,
        ).strip()
        info["working_tree_dirty"] = bool(diff)
    except Exception:
        pass

    info.update(extra)

    out_path = Path(output_dir) / "provenance.json"
    out_path.write_text(json.dumps(info, indent=2))
