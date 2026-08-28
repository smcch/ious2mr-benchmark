r"""Regenerate the three May-era manuscript figures from canonical data, with
unified regime nomenclature ("2D+3D-refine") and corrected terminology
(pre-/post-RESECTION, not dural opening).

  Fig_T2_quality_overview   A: family x regime heatmaps (SSIM/PSNR/MAE/LPIPS)
                            B: forest plot, 24 single-target configs x 3 metrics
  Fig_pre_post_resection    3 panels, paired pre/post bars + 95% t-CI, 8 models
  Fig_subgroup_synthesis    2x3 grid: grade (top) / reoperation (bottom)

Values come from paper_assets/all_experiments_metrics.csv (single-target rows)
and evaluacion-final/rescore_methods_persubject.csv. Originals backed up to
figures_journal_v2/superseded_20260828/.
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
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import t as tdist

BASE = str(PROJECT_ROOT)
OUTDIR = os.path.join(BASE, "paper_assets", "figures_journal_v2")
MANUS = os.path.join(BASE, "latex_manuscript", "us_sintesis", "figures")
EXPM = os.path.join(BASE, "paper_assets", "all_experiments_metrics.csv")
PERSUBJ = os.path.join(BASE, "evaluacion-final", "rescore_methods_persubject.csv")

FAMS = ["Pix2Pix", "SwinPix2Pix", "CycleGAN", "CUT", "ResViT", "SynDiff"]
REGS = ["2D", "2.5D", "2D+3D-refine", "Full-3D"]
FAM_COLOR = {"Pix2Pix": "#1f77b4", "SwinPix2Pix": "#ff7f0e", "CycleGAN": "#2ca02c",
             "CUT": "#d62728", "ResViT": "#9467bd", "SynDiff": "#8c564b"}
REG_MARKER = {"2D": "o", "2.5D": "s", "2D+3D-refine": "D", "Full-3D": "^"}
plt.rcParams.update({"font.family": "serif", "font.size": 13,
                     "axes.titlesize": 15, "axes.labelsize": 14})


def save(fig, name):
    os.makedirs(os.path.join(OUTDIR, "superseded_20260828"), exist_ok=True)
    for ext in (".jpg", ".tiff"):
        src = os.path.join(OUTDIR, name + ext)
        dst = os.path.join(OUTDIR, "superseded_20260828", name + ext)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
    fig.savefig(os.path.join(OUTDIR, name + ".jpg"), dpi=300)
    fig.savefig(os.path.join(OUTDIR, name + ".tiff"), dpi=300)
    shutil.copy2(os.path.join(OUTDIR, name + ".jpg"), os.path.join(MANUS, name + ".jpg"))
    plt.close(fig)
    print("written:", name)


def norm_reg(v):
    v = v.replace(" ", "")
    if v in ("2D+3D-post", "2D+3D-refine", "3D+3D-refine"):
        return "2D+3D-refine"
    if v.lower() in ("full-3d", "3d", "full3d"):
        return "Full-3D"
    return v


def norm_fam(f):
    return {"pix2pix": "Pix2Pix", "swinpix2pix": "SwinPix2Pix",
            "cyclegan": "CycleGAN", "cut": "CUT"}.get(f.lower(), f)


# ---------- data: single-target experiment means/CIs ----------
single = {}
for r in csv.DictReader(open(EXPM, newline="", encoding="utf-8")):
    if r["Target"] != "T2 only":
        continue
    fam, reg = norm_fam(r["Family"]), norm_reg(r["Variant"])
    single[(fam, reg)] = {m: (float(r[f"{m}_t2_mean"]), float(r[f"{m}_t2_lo"]),
                              float(r[f"{m}_t2_hi"]))
                          for m in ("ssim", "psnr", "mae", "lpips")}
assert len(single) == 24, len(single)

# ================= Fig_T2_quality_overview =================
fig = plt.figure(figsize=(15, 17.6))
gsA = fig.add_gridspec(2, 2, left=0.09, right=0.97, top=0.945, bottom=0.545,
                       wspace=0.32, hspace=0.35)
gsB = fig.add_gridspec(1, 3, left=0.175, right=0.985, top=0.44, bottom=0.055,
                       wspace=0.12)
panels = [("ssim", "SSIM ($\\uparrow$)", "viridis", False),
          ("psnr", "PSNR [dB] ($\\uparrow$)", "viridis", False),
          ("mae", "MAE ($\\downarrow$)", "viridis_r", True),
          ("lpips", "LPIPS ($\\downarrow$)", "viridis_r", True)]
for k, (m, ttl, cmap, lower) in enumerate(panels):
    ax = fig.add_subplot(gsA[k // 2, k % 2])
    M = np.array([[single[(f, g)][m][0] for g in REGS] for f in FAMS])
    im = ax.imshow(M, cmap=cmap, aspect="auto")
    for i in range(len(FAMS)):
        for j in range(len(REGS)):
            v = M[i, j]
            vm = (v - M.min()) / (M.max() - M.min() + 1e-9)
            dark = vm < 0.45 if cmap == "viridis" else vm > 0.55
            fmt = "%.2f" if m == "psnr" else "%.3f"
            ax.text(j, i, fmt % v, ha="center", va="center", fontsize=11,
                    color="white" if (vm < 0.5 if cmap == "viridis" else vm > 0.5) else "black")
    ax.set_xticks(range(len(REGS)))
    ax.set_xticklabels(REGS, fontsize=11)
    ax.set_yticks(range(len(FAMS)))
    ax.set_yticklabels(FAMS, fontsize=11)
    ax.set_title(ttl, fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
fig.text(0.01, 0.965, "A", fontsize=32, fontweight="bold")
fig.text(0.53, 0.975, "Family $\\times$ regime — synthetic T2w metrics (single-target runs)",
         ha="center", fontsize=16)

order = [(f, g) for f in FAMS for g in REGS]
names = [f"{f} – {g}" for f, g in order]
ys = range(len(order) - 1, -1, -1)
for j, (m, xl) in enumerate([("ssim", "SSIM ($\\uparrow$)"),
                             ("psnr", "PSNR [dB] ($\\uparrow$)"),
                             ("lpips", "LPIPS ($\\downarrow$)")]):
    ax = fig.add_subplot(gsB[0, j])
    for (f, g), y in zip(order, ys):
        mean, lo, hi = single[(f, g)][m]
        ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]],
                    fmt=REG_MARKER[g], color=FAM_COLOR[f], ms=8, capsize=3,
                    lw=1.3, mec=FAM_COLOR[f])
    ax.set_xlabel(xl)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(names if j == 0 else [], fontsize=10)
handles = [Line2D([], [], marker="s", ls="", color=c, ms=10, label=f)
           for f, c in FAM_COLOR.items()]
handles += [Line2D([], [], marker=mk, ls="", color="0.45", ms=10, label=g)
            for g, mk in REG_MARKER.items()]
fig.legend(handles=handles, loc="upper center", ncol=10, frameon=False,
           bbox_to_anchor=(0.55, 0.475), fontsize=11.5, handletextpad=0.3,
           columnspacing=1.0)
fig.text(0.01, 0.455, "B", fontsize=32, fontweight="bold")
save(fig, "Fig_T2_quality_overview")

# ---------- per-subject helpers ----------
rows_ps = [r for r in csv.DictReader(open(PERSUBJ, newline="", encoding="utf-8"))
           if r["channel"] == "t2"]
EIGHT = [("pix2pix-2D+3D-post-T2", "Pix2Pix\n2D+3D-refine"),
         ("SwinPix2Pix-2D+3D-post-T2", "SwinPix2Pix\n2D+3D-refine"),
         ("CycleGAN-2D+3D-post-T2", "CycleGAN\n2D+3D-refine"),
         ("CUT-2D+3D-post-T2", "CUT\n2D+3D-refine"),
         ("ResViT-2.5D-T2", "ResViT\n2.5D"),
         ("ResViT-3D-T2", "ResViT\nFull-3D"),
         ("SynDiff-2.5D-T2", "SynDiff\n2.5D"),
         ("SynDiff-3D+3D-refine-T2", "SynDiff\n2D+3D-refine")]


def stats(method, metric, cond):
    v = [float(r[metric]) for r in rows_ps
         if r["method"] == method and cond(r["subject"])]
    mean = st.mean(v)
    h = tdist.ppf(0.975, len(v) - 1) * st.stdev(v) / len(v) ** 0.5
    return mean, h


# ================= Fig_pre_post_resection =================
C_PRE, C_POST = "#7f9fc4", "#d98d8f"
fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
fig.subplots_adjust(left=0.055, right=0.99, top=0.84, bottom=0.26, wspace=0.24)
x = np.arange(len(EIGHT))
w = 0.38
for j, (m, ttl) in enumerate([("ssim", "SSIM ($\\uparrow$)"),
                              ("psnr", "PSNR [dB] ($\\uparrow$)"),
                              ("lpips", "LPIPS ($\\downarrow$)")]):
    ax = axes[j]
    for i, (meth, lab) in enumerate(EIGHT):
        mp, hp = stats(meth, m, lambda s: s.endswith("-pre"))
        mq, hq = stats(meth, m, lambda s: s.endswith("-post"))
        ax.bar(i - w / 2, mp, w, yerr=hp, color=C_PRE, capsize=3, zorder=3,
               edgecolor="black", linewidth=0.4,
               label="Pre-resection" if i == 0 else None)
        ax.bar(i + w / 2, mq, w, yerr=hq, color=C_POST, capsize=3, zorder=3,
               edgecolor="black", linewidth=0.4,
               label="Post-resection" if i == 0 else None)
    ax.set_title(ttl, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _m, lab in EIGHT], rotation=45, ha="right", fontsize=10)
    ax.grid(axis="y", ls=":", alpha=0.5, zorder=0)
    if m == "ssim":
        ax.set_ylim(0.62, 0.87)
        ax.legend(loc="upper right", fontsize=11)
    elif m == "psnr":
        ax.set_ylim(13.0, 16.5)
    else:
        ax.set_ylim(0.12, 0.30)
fig.suptitle("Pre- versus post-resection synthetic T2w quality "
             "(mean $\\pm$ 95\u2009% CI; $n_{\\mathrm{pre}}=16$, $n_{\\mathrm{post}}=14$)",
             fontsize=15)
save(fig, "Fig_pre_post_resection")

# ================= Fig_subgroup_synthesis =================
LGG = {"ReMIND-003", "ReMIND-004", "ReMIND-023", "ReMIND-056", "ReMIND-087",
       "ReMIND-096", "ReMIND-102", "ReMIND-107", "ReMIND-109"}
REOP = {"ReMIND-023", "ReMIND-034", "ReMIND-077", "ReMIND-079", "ReMIND-087",
        "ReMIND-102", "ReMIND-107"}
C1, C2 = "#3d6f9e", "#c34a4d"


def pid(s):
    return s.rsplit("-", 1)[0]


fig, axes = plt.subplots(2, 3, figsize=(15, 9.4), sharex="col")
fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.17,
                    wspace=0.24, hspace=0.18)
rowspec = [(("LGG", lambda s: pid(s) in LGG), ("HGG", lambda s: pid(s) not in LGG), "by grade"),
           (("No reop.", lambda s: pid(s) not in REOP), ("Reop.", lambda s: pid(s) in REOP), "by reoperation")]
for r_i, ((l1, c1f), (l2, c2f), rowlab) in enumerate(rowspec):
    for j, (m, ttl, ylim) in enumerate([("ssim", "SSIM ($\\uparrow$)", (0.65, 0.85)),
                                        ("psnr", "PSNR [dB] ($\\uparrow$)", (13.5, 15.8)),
                                        ("lpips", "LPIPS ($\\downarrow$)", (0.14, 0.30))]):
        ax = axes[r_i][j]
        for i, (meth, lab) in enumerate(EIGHT):
            m1 = st.mean(float(r[m]) for r in rows_ps if r["method"] == meth and c1f(r["subject"]))
            m2 = st.mean(float(r[m]) for r in rows_ps if r["method"] == meth and c2f(r["subject"]))
            ax.bar(i - w / 2, m1, w, color=C1, zorder=3, edgecolor="black", linewidth=0.4,
                   label=l1 if i == 0 else None)
            ax.bar(i + w / 2, m2, w, color=C2, zorder=3, edgecolor="black", linewidth=0.4,
                   label=l2 if i == 0 else None)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", ls=":", alpha=0.5, zorder=0)
        if r_i == 0:
            ax.set_title(ttl, fontsize=14)
        if j == 2:
            ax.legend(loc="upper right", fontsize=10.5)
        if j == 0:
            ax.set_ylabel(rowlab, fontsize=13, fontweight="bold")
        if r_i == 1:
            ax.set_xticks(x)
            ax.set_xticklabels([lab for _m2, lab in EIGHT], rotation=45,
                               ha="right", fontsize=10)
fig.suptitle("Synthetic T2w quality stratified by histological grade (top) and "
             "reoperation history (bottom) — eight strongest models", fontsize=15)
save(fig, "Fig_subgroup_synthesis")
