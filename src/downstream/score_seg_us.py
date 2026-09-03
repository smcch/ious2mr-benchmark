#!/usr/bin/env python3
"""Score the direct-ioUS reference (Seg-US) on the downstream endpoint.

Seg-US is an nnU-Net trained on the ioUS volumes themselves with the same recipe, the same 117
training studies, the same five subject-level folds and the same MR-drawn labels as Seg-T2, so
that only the input modality differs.  It answers whether translating ultrasound into MR first
buys anything over learning the task directly from ultrasound.

Scored exactly as every other input in the benchmark (floor_baseline.py):
  * lesion (tumour union cavity) as the primary endpoint, tumour and cavity as secondary;
  * present-class rule -- a class is scored only where the reference contains it;
  * Dice and NSD@2mm, on the reference's own grid/spacing;
  * retention = mean / real-T2w mean, over the studies the real-T2w reference covers.

Outputs results/downstream/seg_us_reference.csv (per study x class) and prints the summary
inserted in Section 3.
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

SRC = Path(os.environ.get("IOUS2MR_SOURCE_TREE", r"E:\SINTESIS"))
PRED = SRC / "downstream_seg" / "predictions_US" / "SEG_US_mrlab"
GT = SRC / "Segmentations"
PER_STUDY = SRC / "downstream_seg" / "results_paper_protocol" / "seg_metrics_T2_per_study.csv"
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "downstream" / "seg_us_reference.csv"

TUMOUR, CAVITY = 1, 2
# Two SynDiff variants were run outside the 6 x 4 x 2 factorial grid (as in floor_baseline.py).
OFF_GRID = {"SynDiff-cascade-T2", "SynDiff-joint-T2"}


def gt_path(study: str) -> Path | None:
    for name in (f"{study}-mri-segmentation.nii.gz", f"{study}-mir-segmentation.nii.gz"):
        p = GT / name
        if p.exists():
            return p
    return None


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / s) if s else float("nan")


def nsd(a: np.ndarray, b: np.ndarray, spacing, tau: float = 2.0) -> float:
    if not a.any() or not b.any():
        return float("nan")
    sa = a & ~binary_erosion(a)
    sb = b & ~binary_erosion(b)
    da = distance_transform_edt(~sa, sampling=spacing)
    db = distance_transform_edt(~sb, sampling=spacing)
    ov = (db[sa] <= tau).sum() + (da[sb] <= tau).sum()
    tot = sa.sum() + sb.sum()
    return float(ov / tot) if tot else float("nan")


def score() -> list[dict]:
    rows = []
    for f in sorted(PRED.glob("*.nii.gz")):
        study = f.name[: -len(".nii.gz")]
        g = gt_path(study)
        if g is None:
            print(f"  no reference label for {study}")
            continue
        gi, pi = nib.load(str(g)), nib.load(str(f))
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
            rows.append(dict(set="SEG_US", study=study, cls=cls,
                             phase="post" if study.endswith("-post") else "pre",
                             gt_present=int(gm.any()), gt_vox=int(gm.sum()), pred_vox=int(pm.sum()),
                             dice=dice(gm, pm) if gm.any() else float("nan"),
                             nsd=nsd(gm, pm, sp) if gm.any() else float("nan")))
    return rows


def published(cls: str = "lesion"):
    """Real-T2w reference and every synthesis, from the released per-study table."""
    per = defaultdict(dict)
    for r in csv.DictReader(open(PER_STUDY, newline="", encoding="utf-8")):
        if r["class"] != cls or r["dice"] in ("", "nan"):
            continue
        per[r["set"]][r["study"]] = float(r["dice"])
    base = set(per["REAL_T2"])
    means = {k: st.mean(d[s] for s in d if s in base)
             for k, d in per.items() if k not in OFF_GRID and (set(d) & base)}
    return per, base, means


def main() -> int:
    rows = score()
    if not rows:
        print("no predictions found; run run_seg_us_chain.sh first")
        return 1

    per, base, means = published("lesion")
    ceiling = means["REAL_T2"]
    synth = {k: v for k, v in means.items() if not k.startswith(("REAL_", "FLOOR_"))}
    leader_name = max(synth, key=synth.get)
    leader = synth[leader_name]
    # the null-control baselines live in their own released table, not in the per-study CSV
    floor_csv = REPO / "results" / "downstream" / "floor_baseline.csv"
    fl = defaultdict(list)
    for r in csv.DictReader(open(floor_csv, newline="", encoding="utf-8")):
        if r["cls"] == "lesion" and r["gt_present"] == "1" and r["dice"] not in ("", "nan"):
            fl[r["set"]].append(float(r["dice"]))
    floor = max(st.mean(v) for v in fl.values())

    print("=" * 92)
    print("DIRECT-ioUS REFERENCE (Seg-US) ON THE PRIMARY LESION ENDPOINT")
    print(f"   {'class / phase':22s} {'n':>3s} {'Dice':>7s} {'NSD':>7s} {'vs real T2w':>12s}")
    summary = {}
    for cls in ("lesion", "tumor", "cavity"):
        for phase in ("all", "pre", "post"):
            v = [r for r in rows if r["cls"] == cls and r["gt_present"] == 1
                 and r["dice"] == r["dice"] and (phase == "all" or r["phase"] == phase)]
            if not v:
                continue
            md = st.mean(r["dice"] for r in v)
            mn = st.mean(r["nsd"] for r in v if r["nsd"] == r["nsd"])
            summary[(cls, phase)] = (len(v), md, mn)
            ref = ""
            if cls == "lesion":
                if phase == "all":
                    ref = f"{100 * md / ceiling:11.1f}%"
                else:
                    rv = [per["REAL_T2"][s] for s in per["REAL_T2"]
                          if (s.endswith("-post") if phase == "post" else s.endswith("-pre"))]
                    ref = f"{100 * md / st.mean(rv):11.1f}%"
            print(f"   {cls + ' / ' + phase:22s} {len(v):3d} {md:7.3f} {mn:7.3f} {ref:>12s}")

    n, md, _ = summary[("lesion", "all")]
    print("\n   scale on the primary endpoint (lesion Dice, same 29 studies):")
    print(f"     real T2w (upper reference)      {ceiling:.3f}   100 %")
    print(f"     best synthesis ({leader_name[:28]:28s}) {leader:.3f}   {100*leader/ceiling:.0f} %")
    print(f"     Seg-US (direct ioUS)            {md:.3f}   {100*md/ceiling:.0f} %")
    print(f"     null control (raw ioUS)         {floor:.3f}   {100*floor/ceiling:.0f} %")
    n_below = sum(1 for v in synth.values() if v < md)
    print(f"\n   {n_below} of the {len(synth)} configurations fall below Seg-US; "
          f"{sum(1 for v in synth.values() if v > md)} exceed it.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
