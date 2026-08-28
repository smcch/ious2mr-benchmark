"""Filesystem roots for the benchmark, resolved from environment variables.

The published scripts were written against the authors' working tree. Rather than
rewriting them into a framework, every absolute path was replaced by one of the
roots below, so a user only has to export a handful of variables.

    export IOUS2MR_ROOT=/path/to/working/tree     # holds the per-stage folders
    export IOUS2MR_DATA=/path/to/preprocessed     # ReMIND-derived US/MR volumes
    export IOUS2MR_CKPT=/path/to/weights          # see WEIGHTS.md
    export IOUS2MR_EXTERNAL=/path/to/external     # optional; external pilot only

On Windows use `set` / `$env:` instead of `export`. Defaults point at the current
working directory so that a plain `python -c "import paths"` never crashes, but
every pipeline entry point expects the variables to be set explicitly.
"""
from __future__ import annotations

import os
from pathlib import Path


def _root(var: str, default: str) -> Path:
    return Path(os.environ.get(var, default)).expanduser()


#: Working tree that contains the per-stage folders (COMPARATIVA-3, resvit, ...).
PROJECT_ROOT = _root("IOUS2MR_ROOT", ".")

#: Pre-processed, co-registered ReMIND volumes (US / MR-T2 / MR-FLAIR).
DATA_ROOT = _root("IOUS2MR_DATA", str(PROJECT_ROOT / "data"))

#: Released model weights (see scripts/fetch_weights.py and WEIGHTS.md).
CKPT_ROOT = _root("IOUS2MR_CKPT", str(PROJECT_ROOT / "weights"))

#: External-pilot cohort. Private in the paper; supply your own (docs/external_pilot.md).
EXTERNAL_ROOT = _root("IOUS2MR_EXTERNAL", str(PROJECT_ROOT / "external"))

#: Python interpreters used by the shell orchestrators (two environments, see envs/).
PYTHON_TORCH = os.environ.get("IOUS2MR_PY_TORCH", "python")
PYTHON_TF = os.environ.get("IOUS2MR_PY_TF", "python")

__all__ = [
    "PROJECT_ROOT", "DATA_ROOT", "CKPT_ROOT", "EXTERNAL_ROOT",
    "PYTHON_TORCH", "PYTHON_TF",
]
