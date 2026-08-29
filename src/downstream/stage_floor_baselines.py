#!/usr/bin/env python3
"""Stage the two training-free floor baselines for the downstream endpoint.

FLOOR_US        the co-registered ioUS volume itself, no synthesis at all.
FLOOR_US_HISTM  the ioUS volume after histogram matching to the pooled T2w intensity
                distribution of the TRAINING subjects -- the cheapest possible appearance
                conversion, with no learned component.

Both are written onto the same grid, with the same affine, as the REAL_T2 inputs the frozen
Seg-T2 nnU-Net already consumes, so the three inputs differ only in their content. Run this,
then nnUNetv2_predict on the two new directories (see run_floor_baselines.sh), then
floor_baseline.py to score them.

The reference distribution is built from training subjects only; using the test subjects'
own T2w would leak the target into the baseline.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np

SRC = Path(os.environ.get("IOUS2MR_SOURCE_TREE", r"E:\SINTESIS"))
DS = SRC / "downstream_seg"
US_DIR = SRC / "dataset-registration-corrected-cropped" / "US"
T2_DIR = SRC / "dataset-registration-corrected-cropped" / "MR-T2"
REF_DIR = DS / "test_inputs_T2" / "REAL_T2"
SPLIT = SRC / "resvit" / "subject_split.json"

OUT_RAW = DS / "test_inputs_T2" / "FLOOR_US"
OUT_HM = DS / "test_inputs_T2" / "FLOOR_US_HISTM"

SUFFIX = "_0000.nii.gz"          # nnU-Net channel marker
SAMPLES_PER_VOLUME = 20_000


def studies() -> list[str]:
    return sorted(f.name[: -len(SUFFIX)] for f in REF_DIR.glob(f"*{SUFFIX}"))


def build_reference_cdf() -> np.ndarray:
    """Pooled foreground T2w intensities over the training subjects, sorted."""
    train = set(json.loads(SPLIT.read_text(encoding="utf-8"))["train_subjects"])
    rng = np.random.default_rng(0)
    vals = []
    for f in sorted(T2_DIR.glob("*.nii.gz")):
        if "-".join(f.name.split("-")[:2]) not in train:
            continue
        d = np.asarray(nib.load(str(f)).dataobj, dtype=np.float32)
        d = d[d > 0]
        if d.size:
            vals.append(rng.choice(d, size=min(d.size, SAMPLES_PER_VOLUME), replace=False))
    pool = np.concatenate(vals)
    print(f"reference CDF from {len(vals)} training T2w volumes, {pool.size} samples")
    return np.sort(pool)


def hist_match(src: np.ndarray, ref_sorted: np.ndarray) -> np.ndarray:
    """Map src's foreground intensities onto ref's distribution by rank."""
    out = np.zeros_like(src, dtype=np.float32)
    fg = src > 0
    if not fg.any():
        return out
    v = src[fg]
    q = (np.argsort(np.argsort(v)) + 0.5) / v.size
    ref_q = (np.arange(ref_sorted.size) + 0.5) / ref_sorted.size
    out[fg] = np.interp(q, ref_q, ref_sorted)
    return out


def main() -> int:
    OUT_RAW.mkdir(parents=True, exist_ok=True)
    OUT_HM.mkdir(parents=True, exist_ok=True)
    ref_cdf = build_reference_cdf()

    n = 0
    for s in studies():
        us = US_DIR / f"{s}-us.nii.gz"
        ref = REF_DIR / f"{s}{SUFFIX}"
        if not us.exists():
            print(f"  missing ioUS for {s}")
            continue
        r = nib.load(str(ref))
        d = np.asarray(nib.load(str(us)).dataobj, dtype=np.float32)
        if d.shape != r.shape:
            print(f"  shape mismatch {s}: ioUS {d.shape} vs reference {r.shape}")
            continue
        nib.save(nib.Nifti1Image(d, r.affine, r.header), str(OUT_RAW / f"{s}{SUFFIX}"))
        nib.save(nib.Nifti1Image(hist_match(d, ref_cdf), r.affine, r.header),
                 str(OUT_HM / f"{s}{SUFFIX}"))
        n += 1
    print(f"staged {n} studies into {OUT_RAW.name} and {OUT_HM.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
