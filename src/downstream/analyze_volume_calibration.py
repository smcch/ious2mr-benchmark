#!/usr/bin/env python3
"""What does the downstream endpoint actually penalise: lost detail, or lost calibration?

The first version of this work attributed the low downstream utility of the GAN family to
over-smoothing that suppresses segmentation-relevant high-frequency anatomy. That mechanism
predicts a segmenter which finds *less* lesion than it should. This script tests the prediction
directly, by decomposing every configuration's lesion Dice into the two quantities Dice
conflates: how much lesion the model finds (recall) and how much of what it finds is lesion
(precision), together with the ratio between predicted and reference lesion volume.

Precision and recall are recovered exactly from the released per-study table, since
    Dice = 2 TP / (|GT| + |P|)  =>  TP = Dice (|GT| + |P|) / 2,
    recall = TP / |GT|,  precision = TP / |P|.

Outputs results/downstream/volume_calibration.csv and prints the summary quoted in the paper.
"""
from __future__ import annotations

import csv
import math
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as tdist

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
SRC = Path(os.environ.get("IOUS2MR_SOURCE_TREE", r"E:\SINTESIS"))

PER_STUDY = SRC / "downstream_seg" / "results_paper_protocol" / "seg_metrics_T2_per_study.csv"
SEG = SRC / "downstream_seg" / "results_paper_protocol" / "seg_results_T2.csv"
FIDELITY = RESULTS / "fidelity" / "all_experiments_metrics.csv"
OUT = RESULTS / "downstream" / "volume_calibration.csv"


def family_of_set(s: str) -> str:
    if s.startswith("REAL_"):
        return "real MR"
    if s.startswith("ResViT"):
        return "ResViT"
    if s.startswith("SynDiff"):
        return "SynDiff"
    if s.startswith("GAN-"):
        return {"pix2pix": "Pix2Pix", "swinpix2pix": "SwinPix2Pix",
                "cyclegan": "CycleGAN", "cut": "CUT"}[s.split("-")[1]]
    return "other"


def seg_key(exp: str, target: str) -> str:
    e, dual = exp.lower(), target == "T2+FLAIR"
    if e.startswith("syndiff"):
        v = ("3D+3D-refine" if "3drefine" in e else "3D" if "full3d" in e
             else "2.5D" if "2.5d" in e else "2D")
        return f"SynDiff-{v}-T2" + ("-from-dual" if dual else "")
    if e.startswith("resvit"):
        v = ("2D+3D-refine" if "2d_3d_refine" in e else "3D" if "full_3d" in e
             else "2.5D" if "2.5d" in e else "2D")
        return f"ResViT-{v}-T2" + ("-from-dual" if dual else "-from-single")
    for f in ("pix2pix", "swinpix2pix", "cyclegan", "cut"):
        if e.startswith(f):
            v = ("2D+3D-post" if "2d_3dpost" in e else
                 "3D" if "_3d" in e.replace("_3dpost", "") else
                 "2.5D" if "_25d" in e else "2D")
            return f"GAN-{f}-{v}-T2" + ("-from-dual" if dual else "-from-single")
    raise ValueError(exp)


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    r = float(np.corrcoef(x, y)[0, 1])
    tv = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
    return r, float(2 * tdist.sf(abs(tv), n - 2)), n


def partial_pearson(x, y, z):
    """Correlation of x and y after removing the linear effect of z."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    def resid(v):
        A = np.column_stack([np.ones_like(z), z])
        beta, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ beta
    rx, ry = resid(x), resid(y)
    n = len(x)
    r = float(np.corrcoef(rx, ry)[0, 1])
    df = n - 3
    tv = r * math.sqrt(df / max(1e-12, 1 - r * r))
    return r, float(2 * tdist.sf(abs(tv), df)), df


def main() -> int:
    # ---- per-study decomposition of the lesion class ----
    per = defaultdict(list)
    for r in csv.DictReader(open(PER_STUDY, newline="", encoding="utf-8")):
        if r["class"] != "lesion" or r["gt_present"] != "1":
            continue
        try:
            gt, pr, dice = float(r["gt_vox"]), float(r["pred_vox"]), float(r["dice"])
        except ValueError:
            continue
        if gt <= 0:
            continue
        tp = dice * (gt + pr) / 2.0
        per[r["set"]].append({
            "ratio": pr / gt,
            "recall": tp / gt,
            "precision": (tp / pr) if pr > 0 else 0.0,
            "dice": dice,
        })

    rows = []
    for s, v in per.items():
        rows.append(dict(
            set=s, family=family_of_set(s), n=len(v),
            median_volume_ratio=round(st.median(x["ratio"] for x in v), 3),
            mean_recall=round(st.mean(x["recall"] for x in v), 3),
            mean_precision=round(st.mean(x["precision"] for x in v), 3),
            mean_dice=round(st.mean(x["dice"] for x in v), 3),
        ))
    rows.sort(key=lambda r: -r["median_volume_ratio"])

    print("=" * 84)
    print("1. DOES THE SEGMENTER FIND TOO LITTLE LESION, OR TOO MUCH?")
    print(f"   {'configuration':44s} {'vol/GT':>7s} {'recall':>7s} {'precis.':>8s} {'Dice':>6s}")
    real = next(r for r in rows if r["set"].startswith("REAL_"))
    show = [real] + [r for r in rows if not r["set"].startswith("REAL_")][:6] + \
           [r for r in rows if not r["set"].startswith("REAL_")][-4:]
    for r in show:
        print(f"   {r['set'][:44]:44s} {r['median_volume_ratio']:7.2f} "
              f"{r['mean_recall']:7.3f} {r['mean_precision']:8.3f} {r['mean_dice']:6.3f}")

    print("\n   By family (median of the per-configuration medians):")
    byfam = defaultdict(list)
    for r in rows:
        byfam[r["family"]].append(r)
    for fam in sorted(byfam, key=lambda f: -st.median(x["median_volume_ratio"] for x in byfam[f])):
        v = byfam[fam]
        print(f"   {fam:14s} volume ratio {st.median(x['median_volume_ratio'] for x in v):5.2f}"
              f"   recall {st.mean(x['mean_recall'] for x in v):.3f}"
              f"   precision {st.mean(x['mean_precision'] for x in v):.3f}")

    # ---- link calibration to fidelity and to utility ----
    seg = list(csv.DictReader(open(SEG, newline="", encoding="utf-8")))
    ref = next(r for r in seg if r["set"] == "REAL_T2" and r["class"] == "lesion"
               and r["phase"] == "all")
    rd, rn = float(ref["dice_mean"]), float(ref["nsd_mean"])
    util = {r["set"]: (float(r["dice_mean"]) / rd, float(r["nsd_mean"]) / rn)
            for r in seg if r["class"] == "lesion" and r["phase"] == "all"
            and not r["set"].startswith("REAL_") and r["dice_mean"] not in ("", "nan")}
    ratio = {r["set"]: r["median_volume_ratio"] for r in rows}

    data = []
    for r in csv.DictReader(open(FIDELITY, newline="", encoding="utf-8")):
        k = seg_key(r["Experiment"], r["Target"])
        if k in util and k in ratio and ratio[k] > 0:
            data.append(dict(ssim=float(r["ssim_t2_mean"]), lpips=float(r["lpips_t2_mean"]),
                             logratio=math.log(ratio[k]),
                             u_dice=util[k][0], u_nsd=util[k][1]))
    print("\n" + "=" * 84)
    print(f"2. VOLUMETRIC CALIBRATION AS THE LINK ({len(data)} experiments)")
    for nm, a, b in (("SSIM  vs log(volume ratio)", "ssim", "logratio"),
                     ("LPIPS vs log(volume ratio)", "lpips", "logratio"),
                     ("log(volume ratio) vs Dice retention", "logratio", "u_dice"),
                     ("log(volume ratio) vs NSD retention", "logratio", "u_nsd")):
        r_, p_, n = pearson([d[a] for d in data], [d[b] for d in data])
        print(f"   {nm:38s} r = {r_:+.3f}  (p = {p_:.2g})")

    print("\n   Association of SSIM with utility, before and after removing calibration:")
    for ep, nm in (("u_dice", "Dice"), ("u_nsd", "NSD")):
        r0, p0, _ = pearson([d["ssim"] for d in data], [d[ep] for d in data])
        r1, p1, _ = partial_pearson([d["ssim"] for d in data], [d[ep] for d in data],
                                    [d["logratio"] for d in data])
        print(f"   {nm:5s}: raw r = {r0:+.3f} (p = {p0:.2g})  ->  "
              f"partialling out log(volume ratio): r = {r1:+.3f} (p = {p1:.2g})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
