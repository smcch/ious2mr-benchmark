r"""Regenerate Fig_multitask_FLAIR_singleVsDual (v2, 2026-08-28).

Fixes vs the original figure:
  - Panel B now compares single-target vs multi-task T2 quality on the COMMON
    cohort of 20 dual-target test studies (before: single bars were n=30 means
    and the SynDiff dual bars were the wrong n=30 means).
  - SynDiff dual values come from evaluacion-final/rescore_methods_persubject.csv
    (the unified rescore), i.e. n=20, consistent with GAN/ResViT dual rows.
  - Regime label unified to "2D+3D-refine" (paper nomenclature) instead of
    "2D+3D-post".

Panel A (FLAIR forest, n=20) is unchanged in substance: rebuilt from
paper_assets/all_experiments_metrics.csv (FLAIR channel; matches the rescore).

Outputs (300 dpi):
  paper_assets/figures_journal_v2/Fig_multitask_FLAIR_singleVsDual.jpg/.tiff
  $IOUS2MR_ROOT\latex_manuscript\us_sintesis\figures\Fig_multitask_FLAIR_singleVsDual.jpg
Originals are backed up to figures_journal_v2/superseded_20260828/.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from common.paths import PROJECT_ROOT, DATA_ROOT, CKPT_ROOT, EXTERNAL_ROOT  # noqa: E402
import os  # noqa: E402,F811
import csv
import os
import shutil
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import t as tdist

BASE = str(PROJECT_ROOT)
ASSETS = os.path.join(BASE, "paper_assets")
PERSUBJ = os.path.join(BASE, "evaluacion-final", "rescore_methods_persubject.csv")
ALLEXP = os.path.join(ASSETS, "all_experiments_metrics.csv")
OUTDIR = os.path.join(ASSETS, "figures_journal_v2")
MANUSCRIPT_FIG = os.path.join(BASE, "latex_manuscript", "us_sintesis", "figures",
                              "Fig_multitask_FLAIR_singleVsDual.jpg")

FAM_COLOR = {"Pix2Pix": "#1f77b4", "SwinPix2Pix": "#ff7f0e", "CycleGAN": "#2ca02c",
             "CUT": "#d62728", "ResViT": "#9467bd", "SynDiff": "#8c564b"}
REGIME_MARKER = {"2D": "o", "2.5D": "s", "2D+3D-refine": "D", "Full-3D": "^"}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
})


def norm_regime(variant):
    v = variant.replace(" ", "")
    if v in ("2D+3D-post", "2D+3D-refine", "2D+3Drefine", "3D+3D-refine"):
        return "2D+3D-refine"
    if v.lower() in ("full-3d", "3d", "full3d"):
        return "Full-3D"
    return variant


def load_panelA():
    rows = []
    with open(ALLEXP, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["ssim_flair_n"] == "0":
                continue
            fam = {"pix2pix": "Pix2Pix", "swinpix2pix": "SwinPix2Pix",
                   "cyclegan": "CycleGAN", "cut": "CUT"}.get(r["Family"].lower(), r["Family"])
            regime = norm_regime(r["Variant"])
            rows.append(dict(
                fam=fam, regime=regime,
                ssim=(float(r["ssim_flair_mean"]), float(r["ssim_flair_lo"]), float(r["ssim_flair_hi"])),
                psnr=(float(r["psnr_flair_mean"]), float(r["psnr_flair_lo"]), float(r["psnr_flair_hi"])),
                lpips=(float(r["lpips_flair_mean"]), float(r["lpips_flair_lo"]), float(r["lpips_flair_hi"])),
            ))
    rows.sort(key=lambda d: d["ssim"][0], reverse=True)
    return rows


def load_common20():
    """Per-subject T2-channel scores restricted to the 20 dual-cohort studies."""
    rows = [r for r in csv.DictReader(open(PERSUBJ, newline="", encoding="utf-8"))
            if r["channel"] == "t2"]
    cohort = {r["subject"] for r in rows if r["method"] == "SynDiff-2D-T2+FLAIR"}
    assert len(cohort) == 20, f"dual cohort != 20 ({len(cohort)})"

    def stats(method, metric):
        v = [float(r[metric]) for r in rows
             if r["method"] == method and r["subject"] in cohort]
        assert len(v) == 20, f"{method}/{metric}: n={len(v)}"
        mean = st.mean(v)
        h = tdist.ppf(0.975, len(v) - 1) * st.stdev(v) / len(v) ** 0.5
        return mean, h
    return stats


PAIRS = [  # (display label, single method, dual method)
    ("Pix2Pix 2D+3D-refine", "pix2pix-2D+3D-post-T2", "pix2pix-2D+3D-post-T2+FLAIR"),
    ("SwinPix2Pix 2D+3D-refine", "SwinPix2Pix-2D+3D-post-T2", "SwinPix2Pix-2D+3D-post-T2+FLAIR"),
    ("CUT 2D+3D-refine", "CUT-2D+3D-post-T2", "CUT-2D+3D-post-T2+FLAIR"),
    ("CycleGAN 2D+3D-refine", "CycleGAN-2D+3D-post-T2", "CycleGAN-2D+3D-post-T2+FLAIR"),
    ("ResViT 2.5D", "ResViT-2.5D-T2", "ResViT-2.5D-T2+FLAIR"),
    ("SynDiff 2.5D", "SynDiff-2.5D-T2", "SynDiff-2.5D-T2+FLAIR"),
]


def main():
    panelA = load_panelA()
    stats = load_common20()

    fig = plt.figure(figsize=(15.0, 12.4))
    gsA = fig.add_gridspec(1, 3, left=0.17, right=0.985, top=0.905, bottom=0.515, wspace=0.10)
    gsB = fig.add_gridspec(1, 3, left=0.075, right=0.985, top=0.345, bottom=0.115, wspace=0.28)

    # ---------------- Panel A: FLAIR forest ----------------
    names = [f"{d['fam']} – {d['regime']}" for d in panelA]
    ys = range(len(panelA) - 1, -1, -1)
    axesA = []
    for j, (metric, xlabel) in enumerate([("ssim", "FLAIR SSIM (↑)"),
                                          ("psnr", "FLAIR PSNR [dB] (↑)"),
                                          ("lpips", "FLAIR LPIPS (↓)")]):
        ax = fig.add_subplot(gsA[0, j])
        axesA.append(ax)
        for d, y in zip(panelA, ys):
            mean, lo, hi = d[metric]
            ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]],
                        fmt=REGIME_MARKER[d["regime"]], color=FAM_COLOR[d["fam"]],
                        ms=9, capsize=3, lw=1.4, mec=FAM_COLOR[d["fam"]])
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.8, len(panelA) - 0.2)
        ax.grid(axis="x", ls=":", alpha=0.5)
        if j == 0:
            ax.set_yticks(list(ys))
            ax.set_yticklabels(names, fontsize=11)
        else:
            ax.set_yticks(list(ys))
            ax.set_yticklabels([])
    axesA[1].set_title("Synthetic FLAIR quality on multi-task T2+FLAIR runs "
                       "(mean ± 95 % CI; $N=20$)", pad=14)

    handles = [Line2D([], [], marker="s", ls="", color=c, ms=10, label=f)
               for f, c in FAM_COLOR.items()]
    handles += [Line2D([], [], marker=m, ls="", color="0.45", ms=10, label=reg)
                for reg, m in REGIME_MARKER.items()]
    fig.legend(handles=handles, loc="upper center", ncol=10, frameon=False,
               bbox_to_anchor=(0.55, 0.985), fontsize=12, handletextpad=0.3,
               columnspacing=1.1)
    fig.text(0.005, 0.975, "A", fontsize=34, fontweight="bold")

    # ---------------- Panel B: single vs dual on the common 20 ----------------
    C_SINGLE, C_DUAL = "steelblue", "tan"
    labels = [p[0] for p in PAIRS]
    x = range(len(PAIRS))
    w = 0.38
    for j, (metric, title, ylim) in enumerate([("ssim", "SSIM (↑)", (0, 0.88)),
                                               ("psnr", "PSNR [dB] (↑)", (0, 16.8)),
                                               ("lpips", "LPIPS (↓)", (0, 0.30))]):
        ax = fig.add_subplot(gsB[0, j])
        for i, (_lab, m_single, m_dual) in enumerate(PAIRS):
            ms, hs = stats(m_single, metric)
            md, hd = stats(m_dual, metric)
            ax.bar(i - w / 2, ms, w, yerr=hs, color=C_SINGLE, capsize=3,
                   error_kw=dict(lw=1.2), label="T2 only (single)" if i == 0 else None)
            ax.bar(i + w / 2, md, w, yerr=hd, color=C_DUAL, capsize=3,
                   error_kw=dict(lw=1.2), label="T2+FLAIR (dual)" if i == 0 else None)
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10.5)
        ax.grid(axis="y", ls=":", alpha=0.5)
        if j == 0:
            ax.legend(loc="lower left", fontsize=11, frameon=True)
    fig.text(0.005, 0.43, "B", fontsize=34, fontweight="bold")
    fig.text(0.53, 0.435, "Single-target vs multi-task (T2+FLAIR) — synthetic T2 quality\n"
             "on the common subset of $N=20$ dual-target studies (mean ± 95 % CI)",
             ha="center", va="top", fontsize=15)

    os.makedirs(os.path.join(OUTDIR, "superseded_20260828"), exist_ok=True)
    for name in ("Fig_multitask_FLAIR_singleVsDual.jpg", "Fig_multitask_FLAIR_singleVsDual.tiff"):
        src = os.path.join(OUTDIR, name)
        dst = os.path.join(OUTDIR, "superseded_20260828", name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    jpg = os.path.join(OUTDIR, "Fig_multitask_FLAIR_singleVsDual.jpg")
    fig.savefig(jpg, dpi=300)
    fig.savefig(os.path.join(OUTDIR, "Fig_multitask_FLAIR_singleVsDual.tiff"), dpi=300)
    shutil.copy2(jpg, MANUSCRIPT_FIG)
    print("written:", jpg)
    print("written:", MANUSCRIPT_FIG)


if __name__ == "__main__":
    main()
