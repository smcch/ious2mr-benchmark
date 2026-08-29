#!/usr/bin/env python3
"""Is the fidelity-utility association a property of the metric, or of the paradigm?

The pooled correlation between an image-fidelity metric and downstream utility, computed
across all 48 experiments, mixes two things: variation *between* architectural families and
variation *within* them. Because the families occupy disjoint SSIM ranges, a pooled
coefficient is largely a two-group contrast. This script separates the two.

For every metric x endpoint it reports:
  * pooled       Pearson r across all experiments (what the first version of the paper reported)
  * within-family partial r, adjusting for family with fixed effects
  * R2 of the family label alone, as a reference for how much either metric adds
and it repeats the analysis on the ROI-restricted metrics (lesion + 5 mm margin), which measure
the same quantity inside the surgical region instead of over the whole foreground.

Outputs: results/downstream/metric_utility_decomposition.csv and a printed summary.
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
# Original working-tree locations are used when the repo copies are absent.
SRC = Path(os.environ.get("IOUS2MR_SOURCE_TREE", r"E:\SINTESIS"))

FIDELITY = RESULTS / "fidelity" / "all_experiments_metrics.csv"
ROI = SRC / "evaluacion-final" / "roi_methods_summary.csv"
SEG = SRC / "downstream_seg" / "results_paper_protocol" / "seg_results_T2.csv"
OUT = RESULTS / "downstream" / "metric_utility_decomposition.csv"

METRICS = ["ssim", "psnr", "mae", "lpips"]


# ----------------------------------------------------------------- helpers
def family_of(exp: str) -> str:
    e = exp.lower()
    if e.startswith("resvit"):
        return "ResViT"
    if e.startswith("syndiff"):
        return "SynDiff"
    for f in ("pix2pix", "swinpix2pix", "cyclegan", "cut"):
        if e.startswith(f):
            return {"pix2pix": "Pix2Pix", "swinpix2pix": "SwinPix2Pix",
                    "cyclegan": "CycleGAN", "cut": "CUT"}[f]
    raise ValueError(exp)


def paradigm_of(fam: str) -> str:
    return "GAN" if fam in ("Pix2Pix", "SwinPix2Pix", "CycleGAN", "CUT") else "transformer/diffusion"


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


def roi_key(method: str) -> str:
    dual = method.endswith("+FLAIR")
    core = method.replace("-T2+FLAIR", "").replace("-T2", "")
    if core.startswith("ResViT"):
        return f"ResViT-{core.split('-', 1)[1]}-T2" + ("-from-dual" if dual else "-from-single")
    if core.startswith("SynDiff"):
        return f"SynDiff-{core.split('-', 1)[1]}-T2" + ("-from-dual" if dual else "")
    fam, _, v = core.partition("-")
    fam = {"pix2pix": "pix2pix", "SwinPix2Pix": "swinpix2pix",
           "CycleGAN": "cyclegan", "CUT": "cut"}[fam]
    return f"GAN-{fam}-{v}-T2" + ("-from-dual" if dual else "-from-single")


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    r = float(np.corrcoef(x, y)[0, 1])
    tv = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
    return r, float(2 * tdist.sf(abs(tv), n - 2)), n


def residualise(values, groups):
    """Remove group means (fixed effects)."""
    means = defaultdict(list)
    for v, g in zip(values, groups):
        means[g].append(v)
    m = {g: st.mean(v) for g, v in means.items()}
    return [v - m[g] for v, g in zip(values, groups)]


def partial_r(x, y, groups):
    """Within-group partial correlation, with df adjusted for the group dummies."""
    rx, ry = residualise(x, groups), residualise(y, groups)
    k = len(set(groups))
    n = len(x)
    r = float(np.corrcoef(rx, ry)[0, 1])
    df = n - k - 1
    tv = r * math.sqrt(df / max(1e-12, 1 - r * r))
    return r, float(2 * tdist.sf(abs(tv), df)), df


def r2_of_labels(y, groups):
    """Fraction of variance in y explained by the group label alone."""
    y = np.asarray(y, float)
    sst = float(((y - y.mean()) ** 2).sum())
    res = residualise(list(y), groups)
    sse = float((np.asarray(res) ** 2).sum())
    return 1 - sse / sst


# ----------------------------------------------------------------- data
def load():
    seg = list(csv.DictReader(open(SEG, newline="", encoding="utf-8")))
    real = next(r for r in seg if r["set"] == "REAL_T2" and r["class"] == "lesion"
                and r["phase"] == "all")
    rd, rn = float(real["dice_mean"]), float(real["nsd_mean"])
    util = {r["set"]: (float(r["dice_mean"]) / rd, float(r["nsd_mean"]) / rn)
            for r in seg if r["class"] == "lesion" and r["phase"] == "all"
            and not r["set"].startswith("REAL_") and r["dice_mean"] not in ("", "nan")}

    rows = []
    for r in csv.DictReader(open(FIDELITY, newline="", encoding="utf-8")):
        k = seg_key(r["Experiment"], r["Target"])
        if k not in util:
            continue
        fam = family_of(r["Experiment"])
        rows.append(dict(
            exp=r["Experiment"], target=r["Target"], family=fam, paradigm=paradigm_of(fam),
            single=(r["Target"] == "T2 only"),
            **{m: float(r[f"{m}_t2_mean"]) for m in METRICS},
            u_dice=util[k][0], u_nsd=util[k][1], seg_set=k,
        ))

    roi = {}
    for r in csv.DictReader(open(ROI, newline="", encoding="utf-8")):
        if (r["roi"], r["dilation_mm"], r["channel"], r["phase"]) == ("lesion", "5.0", "t2", "all"):
            roi[roi_key(r["method"])] = {m: float(r[f"{m}_mean"]) for m in METRICS}
    for row in rows:
        row["roi"] = roi.get(row["seg_set"])
    return rows


# ----------------------------------------------------------------- report
def main() -> int:
    rows = load()
    print(f"experiments matched: {len(rows)}\n")

    # 1. support: do the paradigms overlap at all on SSIM?
    print("=" * 78)
    print("1. SUPPORT OF THE FIDELITY AXIS")
    for p in ("GAN", "transformer/diffusion"):
        v = [r["ssim"] for r in rows if r["paradigm"] == p]
        print(f"   SSIM, {p:22s} n={len(v):2d}  [{min(v):.4f}, {max(v):.4f}]")
    gan = [r["ssim"] for r in rows if r["paradigm"] == "GAN"]
    oth = [r["ssim"] for r in rows if r["paradigm"] != "GAN"]
    if min(gan) > max(oth):
        print(f"   -> disjoint: no experiment lies in ({max(oth):.3f}, {min(gan):.3f})")

    # 2. how much does the label alone explain?
    print("\n" + "=" * 78)
    print("2. VARIANCE EXPLAINED BY THE ARCHITECTURE LABEL ALONE")
    for lbl, key in (("paradigm (2 levels)", "paradigm"), ("family (6 levels)", "family")):
        for ep, nm in (("u_dice", "Dice"), ("u_nsd", "NSD")):
            r2 = r2_of_labels([r[ep] for r in rows], [r[key] for r in rows])
            print(f"   {lbl:20s} -> R2({nm} retention) = {r2:.3f}")

    # 3. the decomposition
    print("\n" + "=" * 78)
    print("3. POOLED vs WITHIN-FAMILY ASSOCIATION (global metrics)")
    out = []
    header = f"   {'metric':6s} {'endpoint':5s} {'pooled r':>10s} {'p':>10s} " \
             f"{'within r':>10s} {'p':>10s} {'R2 metric':>10s}"
    print(header)
    for m in METRICS:
        for ep, nm in (("u_dice", "Dice"), ("u_nsd", "NSD")):
            x = [r[m] for r in rows]
            y = [r[ep] for r in rows]
            fams = [r["family"] for r in rows]
            rp, pp, n = pearson(x, y)
            rw, pw, df = partial_r(x, y, fams)
            print(f"   {m.upper():6s} {nm:5s} {rp:+10.3f} {pp:10.2g} {rw:+10.3f} {pw:10.2g} {rp*rp:10.3f}")
            out.append(dict(scope="global", metric=m.upper(), endpoint=nm, n=n,
                            pooled_r=round(rp, 3), pooled_p=f"{pp:.3g}",
                            within_family_r=round(rw, 3), within_family_p=f"{pw:.3g}",
                            within_family_df=df))

    # 4. same thing inside the surgical ROI
    print("\n" + "=" * 78)
    print("4. THE SAME METRICS, RESTRICTED TO THE LESION + 5 mm MARGIN")
    sub = [r for r in rows if r["roi"]]
    print(f"   (experiments with ROI scores: {len(sub)})")
    print(header)
    for m in METRICS:
        for ep, nm in (("u_dice", "Dice"), ("u_nsd", "NSD")):
            x = [r["roi"][m] for r in sub]
            y = [r[ep] for r in sub]
            fams = [r["family"] for r in sub]
            rp, pp, n = pearson(x, y)
            rw, pw, df = partial_r(x, y, fams)
            print(f"   {m.upper():6s} {nm:5s} {rp:+10.3f} {pp:10.2g} {rw:+10.3f} {pw:10.2g} {rp*rp:10.3f}")
            out.append(dict(scope="roi_lesion_5mm", metric=m.upper(), endpoint=nm, n=n,
                            pooled_r=round(rp, 3), pooled_p=f"{pp:.3g}",
                            within_family_r=round(rw, 3), within_family_p=f"{pw:.3g}",
                            within_family_df=df))

    # 5. single-target only: one cohort, one target, no multi-task confound
    print("\n" + "=" * 78)
    print("5. SENSITIVITY: SINGLE-TARGET EXPERIMENTS ONLY (n = 24, one cohort)")
    s = [r for r in rows if r["single"]]
    for m in ("ssim", "lpips"):
        for ep, nm in (("u_dice", "Dice"), ("u_nsd", "NSD")):
            rp, pp, n = pearson([r[m] for r in s], [r[ep] for r in s])
            rw, pw, df = partial_r([r[m] for r in s], [r[ep] for r in s],
                                   [r["family"] for r in s])
            print(f"   {m.upper():6s} {nm:5s} pooled {rp:+.3f} (p={pp:.2g}) | "
                  f"within-family {rw:+.3f} (p={pw:.2g}, n={n})")
            out.append(dict(scope="single_target_only", metric=m.upper(), endpoint=nm, n=n,
                            pooled_r=round(rp, 3), pooled_p=f"{pp:.3g}",
                            within_family_r=round(rw, 3), within_family_p=f"{pw:.3g}",
                            within_family_df=df))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
