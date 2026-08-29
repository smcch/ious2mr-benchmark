#!/usr/bin/env python3
"""Fetch the upstream SynDiff sources this benchmark builds on.

The diffusion arm of the benchmark adapts SynDiff (Özbey et al., 2023), which in
turn builds on NVIDIA's DDGAN. Parts of that code carry the **NVIDIA Source Code
License**, which restricts use of the work *and its derivatives* to
**non-commercial research**. Rather than redistribute that code inside an
Apache-2.0 repository, we fetch it on demand at a pinned commit.

    python scripts/setup_syndiff_upstream.py

This clones icon-lab/SynDiff at the pinned revision into ``src/syndiff/_upstream``
(git-ignored). The first-party code in ``src/syndiff`` — the 3D backbones, the
refiners, the paired single-direction training loops and every evaluation script —
is ours and ships with the repository.

By running this script you acknowledge the upstream licence terms; see
THIRD_PARTY_NOTICES.md and licenses/NVIDIA_SOURCE_CODE_LICENSE.txt.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/icon-lab/SynDiff.git"
# Pinned so the benchmark is reproducible even if upstream moves.
COMMIT = "fff3d8449e8c7ba38339be2f9ffd4aa5572beb4b"  # 2025-05-27
DEST = Path(__file__).resolve().parents[1] / "src" / "syndiff" / "_upstream"

NOTICE = """
------------------------------------------------------------------------------
 LICENCE NOTICE — read before using the diffusion arm
------------------------------------------------------------------------------
 The code about to be downloaded includes components derived from NVIDIA's
 DDGAN, distributed under the NVIDIA Source Code License. That licence limits
 the work AND ANY DERIVATIVE WORKS -- including models you train with it -- to
 NON-COMMERCIAL research or evaluation use, and requires that limitation to
 propagate to anything you redistribute.

 The SynDiff weights released with this benchmark carry the same restriction.
 Everything else in this repository is Apache-2.0.
------------------------------------------------------------------------------
"""


def main() -> int:
    print(NOTICE)
    if DEST.exists() and any(DEST.iterdir()):
        print(f"Already present: {DEST}")
        print("Delete it first if you want a clean re-fetch.")
        return 0
    if shutil.which("git") is None:
        sys.exit("git is required but was not found on PATH.")

    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {REPO} @ {COMMIT[:12]} -> {DEST}")
    subprocess.run(["git", "clone", "--quiet", REPO, str(DEST)], check=True)
    subprocess.run(["git", "-C", str(DEST), "checkout", "--quiet", COMMIT], check=True)
    shutil.rmtree(DEST / ".git", ignore_errors=True)

    print("\nDone. The diffusion training/eval scripts import from this tree.")
    print("If upstream is unavailable, any mirror of that commit works: the only")
    print("requirement is the directory layout (backbones/, utils/op/, dataset.py).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
