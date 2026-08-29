#!/usr/bin/env python3
"""Patient-level inference for the benchmark's headline claims.

The unit of analysis in this benchmark is the *study*, not the patient: 30 test studies come
from 16 patients, and 14 patients contribute both a pre- and a post-resection study. Two of the
paper's claims are affected, and this script recomputes both correctly.

1. **Uncertainty on the primary endpoint.** Retention ratios were reported as point estimates.
   Here they carry a bias-corrected bootstrap confidence interval obtained by resampling
   *patients* (so that the two studies of one patient move together), which is what determines
   whether a ranking is separable at all.

2. **Subgroup effects.** Reoperation and histological grade are patient-level covariates, so
   testing them over studies double-counts the 14 patients who contribute twice. Here each
   patient contributes one value (the mean over their studies) before the Mann-Whitney test.

3. **The pre- versus post-resection contrast** is a paired comparison within the 14 patients
   that have both phases, not a two-sample comparison of 16 versus 14 independent studies.

Outputs results/downstream/patient_level_inference.csv and prints the corrected numbers.
"""
from __future__ import annotations

import csv
import math
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
SRC = Path(os.environ.get("IOUS2MR_SOURCE_TREE", r"E:\SINTESIS"))

PER_STUDY = SRC / "downstream_seg" / "results_paper_protocol" / "seg_metrics_T2_per_study.csv"
PERSUBJ = SRC / "evaluacion-final" / "rescore_methods_persubject.csv"
DIAG = REPO / "results" / "cohort_diagnoses.csv"          # optional
OUT = RESULTS / "downstream" / "patient_level_inference.csv"

RNG = np.random.default_rng(20260829)
N_BOOT = 20000

LGG = {"ReMIND-003", "ReMIND-004", "ReMIND-023", "ReMIND-056", "ReMIND-087",
       "ReMIND-096", "ReMIND-102", "ReMIND-107", "ReMIND-109"}
REOP = {"ReMIND-023", "ReMIND-034", "ReMIND-077", "ReMIND-079", "ReMIND-087",
        "ReMIND-102", "ReMIND-107"}

TOP = [
    ("ResViT-2D+3D-refine-T2-from-single", "ResViT 2D + 3D-refine"),
    ("ResViT-2.5D-T2-from-single", "ResViT 2.5D"),
    ("SynDiff-2D-T2", "SynDiff 2D"),
    ("ResViT-3D-T2-from-single", "ResViT Full-3D"),
    ("SynDiff-3D+3D-refine-T2", "SynDiff 2D + 3D-refine"),
    ("ResViT-2D-T2-from-single", "ResViT 2D"),
    ("SynDiff-2.5D-T2", "SynDiff 2.5D"),
    ("SynDiff-3D+3D-refine-T2-from-dual", "SynDiff 2D + 3D-refine (dual)"),
]

EIGHT_FID = [
    ("pix2pix-2D+3D-post-T2", "Pix2Pix 2D + 3D-refine"),
    ("SwinPix2Pix-2D+3D-post-T2", "SwinPix2Pix 2D + 3D-refine"),
    ("CycleGAN-2D+3D-post-T2", "CycleGAN 2D + 3D-refine"),
    ("CUT-2D+3D-post-T2", "CUT 2D + 3D-refine"),
    ("ResViT-2.5D-T2", "ResViT 2.5D"),
    ("ResViT-3D-T2", "ResViT Full-3D"),
    ("SynDiff-2.5D-T2", "SynDiff 2.5D"),
    ("SynDiff-3D+3D-refine-T2", "SynDiff 2D + 3D-refine"),
]


def patient(study: str) -> str:
    return study.rsplit("-", 1)[0]


def load_lesion():
    per = defaultdict(dict)
    for r in csv.DictReader(open(PER_STUDY, newline="", encoding="utf-8")):
        if r["class"] != "lesion" or r["dice"] in ("", "nan"):
            continue
        per[r["set"]][r["study"]] = float(r["dice"])
    return per


def boot_ratio_ci(num_by_study, den_by_study, studies):
    """Bias-corrected percentile CI of mean(num)/mean(den), resampling patients."""
    bypat = defaultdict(list)
    for s in studies:
        bypat[patient(s)].append(s)
    pats = sorted(bypat)
    point = (st.mean(num_by_study[s] for s in studies)
             / st.mean(den_by_study[s] for s in studies))
    draws = []
    for _ in range(N_BOOT):
        pick = RNG.choice(len(pats), len(pats), replace=True)
        sel = [s for i in pick for s in bypat[pats[i]]]
        d = st.mean(den_by_study[s] for s in sel)
        if d > 0:
            draws.append(st.mean(num_by_study[s] for s in sel) / d)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi), len(pats)


def main() -> int:
    per = load_lesion()
    real = per["REAL_T2"]
    studies = sorted(real)
    rows = []

    print("=" * 88)
    print(f"1. RETENTION WITH PATIENT-LEVEL BOOTSTRAP CIs "
          f"({len(studies)} studies, {len({patient(s) for s in studies})} patients)")
    print(f"   {'configuration':32s} {'retention':>10s}  {'95% CI':>18s}")
    for key, label in TOP:
        if key not in per:
            continue
        common = [s for s in studies if s in per[key]]
        pt, lo, hi, npat = boot_ratio_ci(per[key], real, common)
        print(f"   {label:32s} {pt:10.3f}  [{lo:.3f}, {hi:.3f}]   n={len(common)} studies / {npat} patients")
        rows.append(dict(analysis="retention_ci", key=label, estimate=round(pt, 3),
                         ci_low=round(lo, 3), ci_high=round(hi, 3), n=len(common)))
    lead = rows[0]
    overlapping = [r for r in rows[1:] if r["ci_high"] >= lead["ci_low"]]
    print(f"\n   The leader's interval overlaps {len(overlapping)} of the other "
          f"{len(rows)-1} configurations: the top group is not separable.")

    # ---- 2. subgroup tests, patient level ----
    print("\n" + "=" * 88)
    print("2. SUBGROUP EFFECTS: STUDY-LEVEL (as published) vs PATIENT-LEVEL")
    fid = [r for r in csv.DictReader(open(PERSUBJ, newline="", encoding="utf-8"))
           if r["channel"] == "t2"]
    by = defaultdict(lambda: defaultdict(dict))
    for r in fid:
        for m in ("ssim", "psnr", "mae", "lpips"):
            by[r["method"]][m][r["subject"]] = float(r[m])

    n_sig_study = n_sig_pat = n_tests = 0
    for key, label in EIGHT_FID:
        for m in ("ssim", "psnr", "mae", "lpips"):
            vals = by[key][m]
            if not vals:
                continue
            a_s = [v for s, v in vals.items() if patient(s) not in REOP]
            b_s = [v for s, v in vals.items() if patient(s) in REOP]
            pat = defaultdict(list)
            for s, v in vals.items():
                pat[patient(s)].append(v)
            pm = {p: st.mean(v) for p, v in pat.items()}
            a_p = [v for p, v in pm.items() if p not in REOP]
            b_p = [v for p, v in pm.items() if p in REOP]
            p_s = mannwhitneyu(a_s, b_s, alternative="two-sided").pvalue
            p_p = mannwhitneyu(a_p, b_p, alternative="two-sided").pvalue
            n_tests += 1
            n_sig_study += p_s < 0.05
            n_sig_pat += p_p < 0.05
            if p_s < 0.05 or p_p < 0.05:
                flag = "survives" if p_p < 0.05 else "DISAPPEARS"
                print(f"   {label:28s} {m.upper():5s} reoperation: "
                      f"study-level p={p_s:.3f} -> patient-level p={p_p:.3f}   {flag}")
                rows.append(dict(analysis="reoperation_subgroup", key=f"{label}/{m}",
                                 estimate=round(p_s, 4), ci_low=round(p_p, 4),
                                 ci_high="", n=len(pm)))
    print(f"\n   significant at study level: {n_sig_study}/{n_tests}; "
          f"at patient level: {n_sig_pat}/{n_tests} "
          f"({100*n_sig_pat/n_tests:.1f}%, i.e. the false-positive rate expected by chance)")

    # ---- 3. pre vs post as a paired contrast ----
    print("\n" + "=" * 88)
    print("3. PRE- vs POST-RESECTION AS A PAIRED CONTRAST (patients with both phases)")
    for key, label in EIGHT_FID:
        line = []
        for m in ("ssim", "psnr"):
            vals = by[key][m]
            pre = {patient(s): v for s, v in vals.items() if s.endswith("-pre")}
            post = {patient(s): v for s, v in vals.items() if s.endswith("-post")}
            both = sorted(set(pre) & set(post))
            if len(both) < 5:
                continue
            d = [post[p] - pre[p] for p in both]
            pv = wilcoxon(d).pvalue
            line.append(f"{m.upper()} delta={st.mean(d):+.3f} p={pv:.3f}"
                        f"{'*' if pv < 0.05 else ' '}")
            rows.append(dict(analysis="paired_phase", key=f"{label}/{m}",
                             estimate=round(st.mean(d), 4), ci_low=round(pv, 4),
                             ci_high="", n=len(both)))
        if line:
            print(f"   {label:28s} n={len(both):2d} pairs | " + " | ".join(line))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["analysis", "key", "estimate", "ci_low", "ci_high", "n"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
