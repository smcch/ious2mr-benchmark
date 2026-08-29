#!/usr/bin/env python3
"""Recompute every cross-experiment comparison on a single common cohort.

Two evaluation cohorts exist in this benchmark, for one reason: the multi-task
(T2w + FLAIR) configurations can only be scored where a real FLAIR acquisition exists.
That gives 20 of the 30 test studies for fidelity, and 19 of the 29 evaluable studies for
the downstream analysis. The single-target configurations can be scored on all of them.

Per-experiment tables may legitimately use each experiment's full cohort, but any ranking or
ratio that places single-target and multi-task experiments side by side must not: comparing a
mean over 30 studies with a mean over 20 confounds the comparison with cohort difficulty (the
dual subset is systematically harder). This script therefore recomputes, on the common cohort:

  * the cross-family fidelity leaderboards (SSIM, PSNR, LPIPS);
  * the downstream lesion table and the retention ratios, each against the real-MR ceiling
    computed on the same studies;

and reports how much the published ordering changes.

Outputs: results/fidelity/leaderboards_common_cohort.csv
         results/downstream/lesion_common_cohort.csv
"""
from __future__ import annotations

import csv
import math
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

from scipy.stats import t as tdist

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
SRC = Path(os.environ.get("IOUS2MR_SOURCE_TREE", r"E:\SINTESIS"))

PERSUBJ = SRC / "evaluacion-final" / "rescore_methods_persubject.csv"
PER_STUDY = SRC / "downstream_seg" / "results_paper_protocol" / "seg_metrics_T2_per_study.csv"
METRICS = ["ssim", "psnr", "mae", "lpips"]


def ci(values):
    n = len(values)
    m = st.mean(values)
    if n < 2:
        return m, 0.0
    h = tdist.ppf(0.975, n - 1) * st.stdev(values) / math.sqrt(n)
    return m, h


def pretty(method):
    dual = method.endswith("+FLAIR")
    core = method.replace("-T2+FLAIR", "").replace("-T2", "")
    fam, _, reg = core.partition("-")
    fam = {"pix2pix": "Pix2Pix"}.get(fam, fam)
    reg = reg.replace("2D+3D-post", "2D + 3D-refine").replace("3D+3D-refine", "2D + 3D-refine")
    if reg == "3D":
        reg = "Full-3D"
    return fam, reg, ("T2w + FLAIR" if dual else "T2w only")


# ----------------------------------------------------------- fidelity
def fidelity():
    rows = [r for r in csv.DictReader(open(PERSUBJ, newline="", encoding="utf-8"))
            if r["channel"] == "t2"]
    per = defaultdict(dict)
    for r in rows:
        per[r["method"]][r["subject"]] = {m: float(r[m]) for m in METRICS}
    # the common cohort is the intersection of the multi-task methods' subjects
    dual = [m for m in per if m.endswith("+FLAIR")]
    common = set.intersection(*[set(per[m]) for m in dual])
    full = set.intersection(*[set(per[m]) for m in per if not m.endswith("+FLAIR")
                              and len(per[m]) >= 30])
    print(f"fidelity: common cohort = {len(common)} studies "
          f"(subset of the {len(full)}-study single-target cohort: {common <= full})")

    out = []
    for meth, subs in per.items():
        if not common <= set(subs):
            continue
        fam, reg, tgt = pretty(meth)
        row = dict(method=meth, family=fam, regime=reg, target=tgt, n=len(common))
        for m in METRICS:
            mean, h = ci([subs[s][m] for s in common])
            row[f"{m}_mean"] = round(mean, 4)
            row[f"{m}_ci"] = round(h, 4)
            full_vals = [subs[s][m] for s in subs]
            row[f"{m}_mean_full"] = round(st.mean(full_vals), 4)
            row[f"{m}_n_full"] = len(full_vals)
        out.append(row)

    print(f"\nCross-family leaderboards on the common cohort (n = {len(common)}):")
    for m, better in (("ssim", max), ("psnr", max), ("lpips", min)):
        ranked = sorted(out, key=lambda r: r[f"{m}_mean"], reverse=(better is max))
        print(f"\n  {m.upper()} top-5")
        for i, r in enumerate(ranked[:5], 1):
            pub = sorted(out, key=lambda x: x[f"{m}_mean_full"],
                         reverse=(better is max)).index(r) + 1
            flag = "" if pub == i else f"   (was #{pub} on mixed cohorts)"
            print(f"    {i}. {r['family']} {r['regime']} [{r['target']}] "
                  f"{r[f'{m}_mean']:.4f}{flag}")

    p = RESULTS / "fidelity" / "leaderboards_common_cohort.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"\n  -> {p}")
    return common


# ----------------------------------------------------------- downstream
def downstream():
    rows = [r for r in csv.DictReader(open(PER_STUDY, newline="", encoding="utf-8"))
            if r["class"] == "lesion"]
    per = defaultdict(dict)
    for r in rows:
        if r["dice"] in ("", "nan"):
            continue
        per[r["set"]][r["study"]] = float(r["dice"])
    dual = [s for s in per if "from-dual" in s]
    common = set.intersection(*[set(per[s]) for s in dual])
    print(f"\ndownstream: common cohort = {len(common)} studies")

    real = per["REAL_T2"]
    ceil_common = st.mean(real[s] for s in common)
    ceil_full = st.mean(real.values())
    print(f"  real-T2w lesion ceiling: {ceil_full:.4f} on {len(real)} studies, "
          f"{ceil_common:.4f} on the common {len(common)}")

    out = []
    for s, d in per.items():
        if s.startswith("REAL_") or not common <= set(d):
            continue
        mc = st.mean(d[x] for x in common)
        mf = st.mean(d.values())
        out.append(dict(set=s, n_common=len(common), dice_common=round(mc, 4),
                        retention_common=round(mc / ceil_common, 3),
                        n_full=len(d), dice_full=round(mf, 4),
                        retention_vs_full_ceiling=round(mf / ceil_full, 3)))
    out.sort(key=lambda r: -r["dice_common"])

    print(f"\n  Lesion leaderboard on the common cohort (n = {len(common)}, "
          f"ceiling {ceil_common:.3f}):")
    for i, r in enumerate(out[:8], 1):
        print(f"    {i}. {r['set'][:46]:46s} {r['dice_common']:.3f} "
              f"({100*r['retention_common']:.0f}%)")
    gan = [r for r in out if r["set"].startswith("GAN-")]
    if gan:
        rank = out.index(gan[0]) + 1
        print(f"    best GAN: #{rank} {gan[0]['set']} {gan[0]['dice_common']:.3f} "
              f"({100*gan[0]['retention_common']:.0f}%)")

    p = RESULTS / "downstream" / "lesion_common_cohort.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"\n  -> {p}")


if __name__ == "__main__":
    fidelity()
    downstream()
