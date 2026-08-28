r"""Figure for the external pilot section (Fig_external_pilot).

(A) Rank-transfer scatters: internal (ReMIND, pre-resection) vs external
    (external pilot) mean SSIM per experiment; T2 channel (n=48) and FLAIR (n=24).
    GAN fidelity = triplanar-harmonised (preds_tri*).
(B) External downstream lesion-Dice leaderboards (Seg-T2 and Seg-FLAIR):
    top-8 configurations vs the domain-shifted real-MR reference (dashed).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from common.paths import PROJECT_ROOT, DATA_ROOT, CKPT_ROOT, EXTERNAL_ROOT  # noqa: E402
import os  # noqa: E402,F811
import csv
import os
import shutil
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

# External-pilot working tree. The cohort used in the paper is private (see DATA.md);
# point IOUS2MR_EXTERNAL at your own external cohort organised as docs/external_pilot.md
# describes, or skip this figure — it is the only one that needs non-public data.
BENCH = os.path.join(str(EXTERNAL_ROOT), "bench")
EP = os.path.join(BENCH, "external_pilot")
OUTDIR = os.path.join(str(PROJECT_ROOT), "paper_assets", "figures_journal_v2")
MANUS = os.path.join(str(PROJECT_ROOT), "latex_manuscript", "us_sintesis", "figures")
REMIND_PREOP = os.path.join(str(PROJECT_ROOT), "evaluacion-final", "rescore_preop_summary.csv")

FAM_STYLE = {"Pix2Pix": ("#4477aa", "o"), "SwinPix2Pix": ("#55a868", "s"),
             "CycleGAN": ("#9467bd", "^"), "CUT": ("#ff8c1a", "D"),
             "ResViT": ("#d62728", "P"), "SynDiff": ("#222222", "*")}
plt.rcParams.update({"font.family": "serif", "font.size": 13,
                     "axes.titlesize": 15, "axes.labelsize": 14})


def fam_of(name):
    n = name.lower()
    if n.startswith("resvit"):
        return "ResViT"
    if n.startswith("syndiff"):
        return "SynDiff"
    if n.startswith("pix2pix"):
        return "Pix2Pix"
    if n.startswith("swinpix2pix"):
        return "SwinPix2Pix"
    if n.startswith("cyclegan"):
        return "CycleGAN"
    return "CUT"


def b2r(exp, channel):
    e = exp.lower()
    dual = e.endswith("_flair")
    if e.startswith("syndiff"):
        v = ("3D+3D-refine" if "3drefine" in e else "3D" if "-3d-" in e or e.endswith("-3d-t2") or "3d-t2_flair" in e
             else "2.5D" if "2.5d" in e else "2D")
        base = f"SynDiff-{v}-T2"
    elif e.startswith("resvit"):
        v = ("2D+3D-refine" if "2d_3d_refine" in e else "3D" if "full_3d" in e
             else "2.5D" if "2.5d" in e else "2D")
        base = f"ResViT-{v}-T2"
    else:
        for fam, p in (("pix2pix", "pix2pix"), ("swinpix2pix", "SwinPix2Pix"),
                       ("cyclegan", "CycleGAN"), ("cut", "CUT")):
            if e.startswith(fam):
                v = ("2D+3D-post" if "2d_3dpost" in e else
                     "3D" if "_3d" in e.replace("_3dpost", "") else
                     "2.5D" if "_25d" in e else "2D")
                base = f"{p}-{v}-T2"
                break
    if channel == "flair":
        return base.replace("-T2", "-T2") + "+FLAIR"
    return base + ("+FLAIR" if dual else "")


def merged_means(channel):
    if channel == "t2":
        ax = list(csv.DictReader(open(os.path.join(BENCH, "preds", "bench_persubject_conemasked.csv"),
                                      newline="", encoding="utf-8")))
        tri = list(csv.DictReader(open(os.path.join(BENCH, "preds_tri", "bench_persubject_conemasked.csv"),
                                       newline="", encoding="utf-8")))
    else:
        ax = list(csv.DictReader(open(os.path.join(BENCH, "preds_flair", "bench_persubject_conemasked_flair.csv"),
                                      newline="", encoding="utf-8")))
        tri = list(csv.DictReader(open(os.path.join(BENCH, "preds_tri_flair", "bench_persubject_conemasked_flair.csv"),
                                       newline="", encoding="utf-8")))
    tri_m = {r["method"] for r in tri}
    merged = [r for r in ax if r["method"] not in tri_m] + tri
    per = defaultdict(list)
    for r in merged:
        per[r["method"]].append(float(r["ssim"]))
    return {k: st.mean(v) for k, v in per.items()}


def remind_means(channel):
    out = {}
    for r in csv.DictReader(open(REMIND_PREOP, newline="", encoding="utf-8")):
        if r["channel"] == channel:
            out[r["method"]] = float(r["ssim_mean"])
    return out


fig = plt.figure(figsize=(15, 12.6))
gsA = fig.add_gridspec(1, 2, left=0.07, right=0.985, top=0.92, bottom=0.585, wspace=0.22)
gsB = fig.add_gridspec(1, 2, left=0.30, right=0.985, top=0.455, bottom=0.115,
                       wspace=0.75, width_ratios=[1.0, 1.0])

# ---------- Panel A ----------
for j, (ch, n_lbl) in enumerate([("t2", "T2w channel ($n = 48$)"),
                                 ("flair", "FLAIR channel ($n = 24$)")]):
    ax = fig.add_subplot(gsA[0, j])
    bra = merged_means(ch)
    rem = remind_means(ch)
    xs, ys = [], []
    for k, v in bra.items():
        rk = b2r(k, ch)
        if rk not in rem:
            continue
        c, mk = FAM_STYLE[fam_of(k)]
        ax.scatter(rem[rk], v, color=c, marker=mk, s=110 if mk == "*" else 65,
                   alpha=0.9, edgecolor="white", linewidth=0.6, zorder=3)
        xs.append(rem[rk])
        ys.append(v)
    rho, p = spearmanr(xs, ys)
    a = (sum((x - st.mean(xs)) * (y - st.mean(ys)) for x, y in zip(xs, ys))
         / sum((x - st.mean(xs)) ** 2 for x in xs))
    b = st.mean(ys) - a * st.mean(xs)
    xr = [min(xs), max(xs)]
    ax.plot(xr, [a * x + b for x in xr], color="grey", ls="--", lw=1.2, zorder=2)
    ptxt = "$p$ < 0.001" if p < 0.001 else f"$p$ = {p:.3f}"
    ax.text(0.03, 0.97, f"Spearman $\\rho$ = {rho:+.2f} ({ptxt})",
            transform=ax.transAxes, va="top", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="grey", alpha=0.92))
    ax.set_xlabel("Internal SSIM (ReMIND, pre-resection)")
    if j == 0:
        ax.set_ylabel("External SSIM (pilot, cone-masked)")
    ax.set_title(n_lbl, fontsize=13.5)
    ax.grid(ls=":", alpha=0.4, zorder=0)
fig.text(0.01, 0.955, "A", fontsize=30, fontweight="bold")
fig.text(0.53, 0.965, "Fidelity ranking transfer: internal vs external cohort",
         ha="center", fontsize=16)

# ---------- Panel B ----------
summ = list(csv.DictReader(open(os.path.join(EP, "external_downstream_summary.csv"),
                                newline="", encoding="utf-8")))


def disp(s):
    fam = fam_of(s)
    reg = ("2D+3D-refine" if ("2d_3dpost" in s or "2d_3d_refine" in s or "3drefine" in s)
           else "Full-3D" if ("full_3d" in s or "_3d" in s.replace("_3dpost", "") or "-3d-" in s)
           else "2.5D" if ("25d" in s or "2.5d" in s) else "2D")
    dual = s.endswith("_flair")
    return f"{fam} {reg}" + (" (dual)" if dual else "")


for j, ch in enumerate(["T2", "FLAIR"]):
    ax = fig.add_subplot(gsB[0, j])
    rows = [r for r in summ if r["channel"] == ch]
    real = next(r for r in rows if r["set"].startswith("REAL"))
    synth = sorted([r for r in rows if not r["set"].startswith("REAL")],
                   key=lambda r: -float(r["dice_mean"]))[:8]
    labels = [f"{i}. {disp(r['set'])}" for i, r in enumerate(synth, 1)]
    vals = [float(r["dice_mean"]) for r in synth]
    cols = [FAM_STYLE[fam_of(r["set"])][0] for r in synth]
    hats = ["//" if fam_of(r["set"]) in ("Pix2Pix", "SwinPix2Pix", "CycleGAN", "CUT")
            else None for r in synth]
    y = range(len(synth) - 1, -1, -1)
    bars = ax.barh(list(y), vals, color=cols, height=0.62, zorder=3)
    for bbar, h in zip(bars, hats):
        if h:
            bbar.set_hatch(h)
    rv = float(real["dice_mean"])
    for yi, v in zip(y, vals):
        ax.text(v + 0.006, yi, f"{v:.3f} ({100*v/rv:.0f}\u2009%)", va="center", fontsize=10)
    ax.axvline(rv, color="0.35", ls="--", lw=1.5, zorder=4)
    ax.text(rv + 0.004, -0.45, f"real ({rv:.2f})", rotation=90, va="bottom",
            ha="left", fontsize=10, color="0.35")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlim(0, max(vals) * 1.30)
    ax.set_xlabel("Lesion Dice")
    ax.set_title(f"Seg-{ch} ($n = 9$ sweeps)", fontsize=13.5)
    ax.grid(axis="x", ls=":", alpha=0.5, zorder=0)
fig.text(0.01, 0.49, "B", fontsize=30, fontweight="bold")
fig.text(0.53, 0.50, "External downstream lesion segmentation vs the domain-shifted real-MR reference",
         ha="center", fontsize=16)

handles = [Line2D([], [], marker=mk, ls="", color=c, ms=12 if mk == "*" else 9, label=f)
           for f, (c, mk) in FAM_STYLE.items()]
fig.legend(handles=handles, loc="lower center", ncol=6, frameon=True,
           bbox_to_anchor=(0.53, 0.004), fontsize=12)

os.makedirs(os.path.join(OUTDIR, "superseded_20260828"), exist_ok=True)
for ext in (".jpg", ".tiff"):
    fig.savefig(os.path.join(OUTDIR, "Fig_external_pilot" + ext), dpi=300)
shutil.copy2(os.path.join(OUTDIR, "Fig_external_pilot.jpg"),
             os.path.join(MANUS, "Fig_external_pilot.jpg"))
print("written Fig_external_pilot")
