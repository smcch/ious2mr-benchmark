#!/usr/bin/env python3
"""How much of the downstream endpoint does synthesis actually buy?

Every retention ratio in this benchmark is measured against a ceiling (real T2w) but not
against a floor. Without one, "the best configuration retains 61 % of the real-T2w lesion
Dice" is uninterpretable: if the frozen segmentation model already recovered a comparable
share from the raw ioUS volume, the 48 experiments would have bought nothing.

This script establishes that floor with two training-free baselines, both fed to the *same*
frozen Seg-T2 nnU-Net on the *same* grid as every synthesis:

  FLOOR_US        the co-registered ioUS volume, unmodified;
  FLOOR_US_HISTM  the ioUS volume after histogram matching to the pooled T2w intensity
                  distribution of the TRAINING subjects -- the cheapest possible appearance
                  conversion, with no learned component at all.

Any configuration that does not clear FLOOR_US_HISTM has not earned its training cost.

Outputs results/downstream/floor_baseline.csv.
"""
from __future__ import annotations

import csv
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
SRC = Path(os.environ.get("IOUS2MR_SOURCE_TREE", r"E:\SINTESIS"))

PRED = SRC / "downstream_seg" / "predictions_T2"
GT = SRC / "Segmentations"
PER_STUDY = SRC / "downstream_seg" / "results_paper_protocol" / "seg_metrics_T2_per_study.csv"
OUT = RESULTS / "downstream" / "floor_baseline.csv"

FLOORS = ["FLOOR_US", "FLOOR_US_HISTM"]
TUMOUR, CAVITY = 1, 2


def gt_path(study: str) -> Path | None:
    for name in (f"{study}-mri-segmentation.nii.gz", f"{study}-mir-segmentation.nii.gz"):
        p = GT / name
        if p.exists():
            return p
    return None


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / s) if s else float("nan")


def nsd(a: np.ndarray, b: np.ndarray, spacing, tau=2.0) -> float:
    """Normalised surface Dice at tolerance tau (mm)."""
    if not a.any() or not b.any():
        return float("nan")
    sa = a & ~binary_erosion(a)
    sb = b & ~binary_erosion(b)
    da = distance_transform_edt(~sa, sampling=spacing)
    db = distance_transform_edt(~sb, sampling=spacing)
    ov = (db[sa] <= tau).sum() + (da[sb] <= tau).sum()
    tot = sa.sum() + sb.sum()
    return float(ov / tot) if tot else float("nan")


def score_floor(name: str):
    rows = []
    d = PRED / name
    for f in sorted(d.glob("*.nii.gz")):
        study = f.name[: -len(".nii.gz")]
        g = gt_path(study)
        if g is None:
            continue
        gi = nib.load(str(g))
        pi = nib.load(str(f))
        gd = np.asarray(gi.dataobj).astype(np.int16)
        pd = np.asarray(pi.dataobj).astype(np.int16)
        if gd.shape != pd.shape:
            print(f"  shape mismatch {study}: {gd.shape} vs {pd.shape}")
            continue
        sp = gi.header.get_zooms()[:3]
        for cls, gm, pm in (
            ("lesion", np.isin(gd, (TUMOUR, CAVITY)), np.isin(pd, (TUMOUR, CAVITY))),
            ("tumor", gd == TUMOUR, pd == TUMOUR),
            ("cavity", gd == CAVITY, pd == CAVITY),
        ):
            rows.append(dict(set=name, study=study, cls=cls,
                             gt_present=int(gm.any()),
                             gt_vox=int(gm.sum()), pred_vox=int(pm.sum()),
                             dice=dice(gm, pm) if gm.any() else float("nan"),
                             nsd=nsd(gm, pm, sp) if gm.any() else float("nan")))
    return rows


def published_reference():
    """Real-T2w ceiling and the leading syntheses, from the released per-study table."""
    per = defaultdict(list)
    for r in csv.DictReader(open(PER_STUDY, newline="", encoding="utf-8")):
        if r["class"] != "lesion" or r["dice"] in ("", "nan"):
            continue
        per[r["set"]].append(float(r["dice"]))
    return {k: st.mean(v) for k, v in per.items() if len(v) >= 25}


def main() -> int:
    rows = []
    for name in FLOORS:
        if not (PRED / name).is_dir():
            print(f"missing predictions for {name}; run the staging + inference step first")
            continue
        rows += score_floor(name)
    if not rows:
        return 1

    ref = published_reference()
    ceiling = ref.get("REAL_T2", float("nan"))
    leader = max((v for k, v in ref.items() if not k.startswith(("REAL_", "FLOOR_"))),
                 default=float("nan"))
    leader_name = max((k for k in ref if not k.startswith(("REAL_", "FLOOR_"))),
                      key=lambda k: ref[k], default="?")

    print("=" * 88)
    print("FLOOR BASELINES ON THE PRIMARY LESION ENDPOINT (frozen Seg-T2, same grid)")
    print(f"   {'input':22s} {'n':>3s} {'Dice':>7s} {'NSD':>7s} {'vol/GT':>7s} {'% of ceiling':>13s}")
    summary = {}
    for name in FLOORS:
        v = [r for r in rows if r["set"] == name and r["cls"] == "lesion"
             and r["gt_present"] == 1 and r["dice"] == r["dice"]]
        if not v:
            continue
        md = st.mean(r["dice"] for r in v)
        mn = st.mean(r["nsd"] for r in v if r["nsd"] == r["nsd"])
        vr = st.median(r["pred_vox"] / r["gt_vox"] for r in v if r["gt_vox"])
        summary[name] = md
        print(f"   {name:22s} {len(v):3d} {md:7.3f} {mn:7.3f} {vr:7.2f} {100*md/ceiling:12.1f}%")
    print(f"   {'REAL_T2 (ceiling)':22s} {'':3s} {ceiling:7.3f}")
    print(f"   {leader_name[:22]:22s} {'':3s} {leader:7.3f} {'':7s} {'':7s} "
          f"{100*leader/ceiling:12.1f}%   (best synthesis)")

    floor = max(summary.values()) if summary else float("nan")
    print(f"\n   Synthesis clears the best training-free floor by "
          f"{leader - floor:+.3f} Dice ({100*(leader-floor)/ceiling:+.1f} points of the ceiling).")
    n_above = sum(1 for k, v in ref.items()
                  if not k.startswith(("REAL_", "FLOOR_")) and v > floor)
    n_tot = sum(1 for k in ref if not k.startswith(("REAL_", "FLOOR_")))
    print(f"   {n_above} of {n_tot} configurations exceed it.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
