r"""Regenerate the two downstream figures on the LESION-primary endpoint (A3).

Fig 1  Fig_downstream_utility_correlation.jpg
   (A) lesion Dice leaderboards (Seg-T2 top-8 + best GAN; Seg-FLAIR top-4)
       with the real-MR ceiling and retention labels.
   (B) 2x2 scatters: LPIPS/SSIM/PSNR/MAE (T2 channel, 48 experiments) vs the
       lesion NSD2mm retention ratio, with Pearson/Spearman annotations.

Fig 2  Fig_subgroup_downstream.jpg
   2x2: lesion (left) / cavity (right) Dice by grade (top) and reoperation
   (bottom) for 9 configurations, with per-stratum real-T2w dotted ceilings.

Inputs: results_paper_protocol/seg_results_{T2,FLAIR}.csv, seg_metrics_T2_per_study.csv,
paper_assets/all_experiments_metrics.csv (SynDiff-dual fixed), ReMIND_subset_diagnoses.csv.
Originals backed up to figures_journal_v2/superseded_20260828/.
"""
import csv
import math
import os
import shutil
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import t as tdist

BASE = r"$IOUS2MR_ROOT"
RES = os.path.join(BASE, "downstream_seg", "results_paper_protocol")
ASSETS = os.path.join(BASE, "paper_assets")
OUTDIR = os.path.join(ASSETS, "figures_journal_v2")
MANUS = os.path.join(BASE, "latex_manuscript", "us_sintesis", "figures")

# Null control: the same frozen segmentation model applied to the raw co-registered ioUS
# volume (src/downstream/floor_baseline.py).  Direct-ioUS reference: an nnU-Net trained on
# the ioUS volumes under the identical protocol (src/downstream/score_seg_us.py).  Only
# Seg-T2 has both, because the input is the ultrasound and there is no FLAIR equivalent.
FLOOR_DICE = {"T2": 0.251}
SEG_US_DICE = {"T2": 0.401}

FAM_STYLE = {  # colour, marker  (matches the previous figure's language)
    "Pix2Pix": ("#4477aa", "o"),
    "SwinPix2Pix": ("#55a868", "s"),
    "CycleGAN": ("#9467bd", "^"),
    "CUT": ("#ff8c1a", "D"),
    "ResViT": ("#d62728", "P"),
    "SynDiff": ("#222222", "*"),
}
plt.rcParams.update({"font.family": "serif", "font.size": 13,
                     "axes.titlesize": 15, "axes.labelsize": 14})


def load(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


seg_t2 = load(os.path.join(RES, "seg_results_T2.csv"))
seg_fl = load(os.path.join(RES, "seg_results_FLAIR.csv"))
per_study = load(os.path.join(RES, "seg_metrics_T2_per_study.csv"))
expm = load(os.path.join(ASSETS, "all_experiments_metrics.csv"))
diag = load(os.path.join(ASSETS, "ReMIND_subset_diagnoses.csv"))

f = lambda x: float(x) if x not in ("", "nan", None) else math.nan
is_bench = lambda s: "cascade" not in s and "joint" not in s


def get(rows, s, cls, ph):
    for r in rows:
        if r["set"] == s and r["class"] == cls and r["phase"] == ph:
            return r


def family_of(s):
    if s.startswith("GAN-"):
        return {"pix2pix": "Pix2Pix", "swinpix2pix": "SwinPix2Pix",
                "cyclegan": "CycleGAN", "cut": "CUT"}[s.split("-")[1]]
    return "ResViT" if s.startswith("ResViT") else "SynDiff"


def disp(s):
    fam = family_of(s)
    dual = "-from-dual" in s or s.endswith("T2_FLAIR")
    core = (s.replace("GAN-", "").replace("-from-single", "").replace("-from-dual", "")
             .replace("-T2", "").replace("-FLAIR", ""))
    reg = core.split("-", 1)[1] if "-" in core else core
    reg = (reg.replace("2D+3D-post", "2D+3D-refine").replace("3D+3D-refine", "2D+3D-refine"))
    if reg == "3D":
        reg = "Full-3D"
    return f"{fam} {reg}" + (" (dual)" if dual else "")


# ================= Figure 1 =================
def lesion_rank(seg_rows):
    rows = [r for r in seg_rows if r["class"] == "lesion" and r["phase"] == "all"
            and not r["set"].startswith("REAL_") and is_bench(r["set"])
            and r["dice_mean"] not in ("", "nan")]
    rows.sort(key=lambda r: -f(r["dice_mean"]))
    return rows


def map_exp(exp, target, channel):
    e = exp.lower()
    dual = target == "T2+FLAIR"
    if "syndiff" in e:
        v = ("3D+3D-refine" if "3drefine" in e else "3D" if "full3d" in e
             else "2.5D" if "2.5d" in e else "2D")
        return (f"SynDiff-{v}-T2" + ("-from-dual" if dual else "")) if channel == "T2" \
            else f"SynDiff-{v}-FLAIR"
    if "resvit" in e:
        v = ("2D+3D-refine" if "2d_3d_refine" in e else "3D" if "full_3d" in e
             else "2.5D" if "2.5d" in e else "2D")
        return (f"ResViT-{v}-T2" + ("-from-dual" if dual else "-from-single")) if channel == "T2" \
            else f"ResViT-{v}-FLAIR"
    for fam in ("pix2pix", "swinpix2pix", "cyclegan", "cut"):
        if e.startswith(fam):
            v = ("2D+3D-post" if "2d_3dpost" in e else "3D" if "_3d" in e
                 else "2.5D" if "_25d" in e else "2D")
            return (f"GAN-{fam}-{v}-T2" + ("-from-dual" if dual else "-from-single")) if channel == "T2" \
                else f"GAN-{fam}-{v}-FLAIR"


def pearson_t(xs, ys):
    n = len(xs)
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    r = num / den
    tv = r * math.sqrt((n - 2) / (1 - r * r))
    return r, 2 * tdist.sf(abs(tv), n - 2)


def within_family_r(points, metric):
    """Partial correlation of `metric` with utility after removing family means."""
    by_fam = defaultdict(list)
    for vals, y, fam in points:
        by_fam[fam].append((vals[metric], y))
    rx, ry = [], []
    for fam, vs in by_fam.items():
        mx = st.mean(v[0] for v in vs)
        my = st.mean(v[1] for v in vs)
        rx += [v[0] - mx for v in vs]
        ry += [v[1] - my for v in vs]
    n, k = len(rx), len(by_fam)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    r = num / den
    df = n - k - 1
    tv = r * math.sqrt(df / max(1e-12, 1 - r * r))
    return r, float(2 * tdist.sf(abs(tv), df))


def spearman_t(xs, ys):
    def rk(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        for i, idx in enumerate(order):
            out[idx] = i
        return out
    return pearson_t(rk(xs), rk(ys))


def fig_downstream():
    fig = plt.figure(figsize=(15, 18.2))
    gsA = fig.add_gridspec(1, 2, left=0.24, right=0.97, top=0.955, bottom=0.755,
                           wspace=0.55, width_ratios=[1.15, 1.0])
    gsB = fig.add_gridspec(2, 2, left=0.075, right=0.97, top=0.66, bottom=0.085,
                           wspace=0.22, hspace=0.33)

    # ---- panel A ----
    for j, (seg_rows, mod, topn) in enumerate([(seg_t2, "T2", 8), (seg_fl, "FLAIR", 4)]):
        ax = fig.add_subplot(gsA[0, j])
        rows = lesion_rank(seg_rows)
        sel = rows[:topn]
        gan_rank = next((i for i, r in enumerate(rows, 1) if r["set"].startswith("GAN-")), None)
        if gan_rank and gan_rank > topn:
            sel = sel + [rows[gan_rank - 1]]
        real = f(get(seg_rows, f"REAL_{mod}", "lesion", "all")["dice_mean"])
        labels, vals, cols, hat = [], [], [], []
        for i, r in enumerate(sel, 1):
            rank = i if i <= topn else gan_rank
            labels.append(f"{rank}. {disp(r['set'])}")
            vals.append(f(r["dice_mean"]))
            cols.append(FAM_STYLE[family_of(r["set"])][0])
            hat.append("//" if r["set"].startswith("GAN-") else None)
        y = range(len(sel) - 1, -1, -1)
        bars = ax.barh(list(y), vals, color=cols, height=0.62, zorder=3)
        for b, h in zip(bars, hat):
            if h:
                b.set_hatch(h)
        for yi, v in zip(y, vals):
            ax.text(v + 0.006, yi, f"{v:.3f} ({100*v/real:.0f}\u2009%)",
                    va="center", fontsize=10.5)
        ax.axvline(real, color="0.35", ls="--", lw=1.4, zorder=4)
        ax.text(real - 0.006, len(sel) - 0.45, f"real {mod}w ({real:.2f})" if mod == "T2"
                else f"real FLAIR ({real:.2f})",
                rotation=90, va="top", ha="right", fontsize=10.5, color="0.35")
        floor = FLOOR_DICE.get(mod)
        segus = SEG_US_DICE.get(mod)
        if floor:
            # both reference lines cross every bar, so their labels go in the empty band
            # below the last bar rather than inside the plotted area
            ax.axvline(floor, color="#b03a2e", ls="-.", lw=1.4, zorder=4)
            ax.set_ylim(-1.55, len(sel) - 0.35)
            ax.text(floor + 0.008, -0.75, f"raw ioUS, no synthesis ({floor:.2f})",
                    va="center", ha="left", fontsize=10.5, color="#b03a2e")
        if segus:
            ax.axvline(segus, color="#1f6f4a", ls=(0, (4, 2)), lw=1.4, zorder=4)
            ax.text(segus + 0.008, -1.30, f"segmentation trained on ioUS ({segus:.2f})",
                    va="center", ha="left", fontsize=10.5, color="#1f6f4a")
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=11.5)
        ax.set_xlim(0, real * 1.13)
        ax.set_xlabel("Lesion Dice")
        n_lab = "$n=29$" if mod == "T2" else "$n=19$"
        ax.set_title(f"Seg-{mod} ({n_lab})", fontsize=14)
        ax.grid(axis="x", ls=":", alpha=0.5, zorder=0)
    fig.text(0.01, 0.965, "A", fontsize=32, fontweight="bold")
    fig.text(0.55, 0.985, "Downstream lesion segmentation (tumour $\\cup$ cavity) "
             "from synthetic MRI", ha="center", fontsize=16)

    # ---- panel B ----
    um = {}
    real_nsd = f(get(seg_t2, "REAL_T2", "lesion", "all")["nsd_mean"])
    for r in seg_t2:
        if r["class"] == "lesion" and r["phase"] == "all" and not r["set"].startswith("REAL_") \
                and r["nsd_mean"] not in ("", "nan"):
            um[r["set"]] = f(r["nsd_mean"]) / real_nsd
    pts = []
    for r in expm:
        if r["Target"] not in ("T2 only", "T2+FLAIR"):
            continue
        mp = map_exp(r["Experiment"], r["Target"], "T2")
        if mp not in um:
            continue
        try:
            vals = {m: float(r[f"{m}_t2_mean"]) for m in ("lpips", "ssim", "psnr", "mae")}
        except ValueError:
            continue
        pts.append((vals, um[mp], family_of(mp)))
    assert len(pts) == 48, len(pts)

    panels = [("lpips", "LPIPS  (lower is better)", "(a) LPIPS vs downstream utility"),
              ("ssim", "SSIM  (higher is better)", "(b) SSIM vs downstream utility"),
              ("psnr", "PSNR (dB)  (higher is better)", "(c) PSNR vs downstream utility"),
              ("mae", "MAE  (lower is better)", "(d) MAE vs downstream utility")]
    for k, (m, xl, ttl) in enumerate(panels):
        ax = fig.add_subplot(gsB[k // 2, k % 2])
        xs = [p[0][m] for p in pts]
        ys = [p[1] for p in pts]
        for (vals, yv, fam) in pts:
            c, mk = FAM_STYLE[fam]
            ax.scatter(vals[m], yv, color=c, marker=mk,
                       s=120 if mk == "*" else 70, alpha=0.9,
                       edgecolor="white", linewidth=0.6, zorder=3)
        # Pooled fit, drawn only over the range where experiments actually exist, plus one
        # thin within-family fit per family: the pooled slope is a between-paradigm contrast
        # and the two paradigms occupy disjoint ranges on some metrics.
        def fit(px, py):
            mx, my = st.mean(px), st.mean(py)
            sa = sum((a_ - mx) * (b_ - my) for a_, b_ in zip(px, py))
            sb = sum((a_ - mx) ** 2 for a_ in px)
            a_ = sa / sb if sb else 0.0
            return a_, my - a_ * mx

        a, b = fit(xs, ys)
        xx = [min(xs), max(xs)]
        ax.plot(xx, [a * x + b for x in xx], color="grey", ls="--", lw=1.4, zorder=2,
                label="pooled")
        for fam in sorted({p[2] for p in pts}):
            fx = [p[0][m] for p in pts if p[2] == fam]
            fy = [p[1] for p in pts if p[2] == fam]
            if len(fx) < 3 or max(fx) == min(fx):
                continue
            fa, fb = fit(fx, fy)
            ax.plot([min(fx), max(fx)], [fa * min(fx) + fb, fa * max(fx) + fb],
                    color=FAM_STYLE[fam][0], ls="-", lw=1.1, alpha=0.65, zorder=2)
        r_, p_ = pearson_t(xs, ys)
        rw, pw = within_family_r(pts, m)
        ptxt = "$p$ < 0.001" if p_ < 0.001 else f"$p$ = {p_:.3f}"
        pwtxt = "$p$ < 0.001" if pw < 0.001 else f"$p$ = {pw:.3f}"
        loc = (0.97, 0.97, "right", "top") if m in ("lpips", "mae") else (0.03, 0.03, "left", "bottom")
        ax.text(loc[0], loc[1],
                f"pooled r = {r_:+.2f} ({ptxt})\n"
                f"within-family r = {rw:+.2f} ({pwtxt})",
                transform=ax.transAxes, ha=loc[2], va=loc[3], fontsize=10.5,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="grey", alpha=0.92))
        ax.set_xlabel(xl)
        if k % 2 == 0:
            ax.set_ylabel("Lesion NSD$_{2\\mathrm{mm}}$ retention (synth / real)")
        ax.set_title(ttl, fontsize=13.5)
        ax.grid(ls=":", alpha=0.4, zorder=0)
    fig.text(0.01, 0.675, "B", fontsize=32, fontweight="bold")
    fig.text(0.53, 0.695, "Image fidelity vs downstream lesion utility "
             "(T2w channel, 48 experiments)", ha="center", fontsize=16)
    handles = [Line2D([], [], marker=mk, ls="", color=c,
                      ms=13 if mk == "*" else 9, label=fam)
               for fam, (c, mk) in FAM_STYLE.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=True,
               bbox_to_anchor=(0.53, 0.005), fontsize=12.5)
    return fig


# ================= Figure 2: subgroups =================
LGG = {r["ID"] for r in diag if r["Grade"] == "Low"}
REOP = {r["ID"] for r in diag if r["Reoperation"] == "Yes"}
SUB_SETS = [
    "GAN-pix2pix-2D+3D-post-T2-from-single",
    "GAN-swinpix2pix-2D+3D-post-T2-from-single",
    "GAN-cyclegan-2D+3D-post-T2-from-single",
    "GAN-cut-2D+3D-post-T2-from-single",
    "ResViT-2.5D-T2-from-single",
    "ResViT-2D+3D-refine-T2-from-single",
    "ResViT-3D-T2-from-single",
    "SynDiff-2.5D-T2",
    "SynDiff-3D+3D-refine-T2",
]


def strat_means(setname, cls):
    vals = defaultdict(list)
    for r in per_study:
        if r["set"] != setname or r["class"] != cls or r["gt_present"] != "1":
            continue
        d = f(r["dice"])
        if d != d:
            continue
        pid = r["study"].rsplit("-", 1)[0]
        vals["LGG" if pid in LGG else "HGG"].append(d)
        vals["Reop" if pid in REOP else "NoReop"].append(d)
    return {k: st.mean(v) for k, v in vals.items()}


def fig_subgroup():
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.6), sharex="col")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.86, bottom=0.20,
                        wspace=0.18, hspace=0.22)
    C1, C2 = "#3d6f9e", "#c34a4d"
    combos = [("lesion", ("LGG", "HGG"), axes[0][0], "Lesion Dice $\\uparrow$  (by grade)", ("LGG", "HGG")),
              ("cavity", ("LGG", "HGG"), axes[0][1], "Cavity Dice $\\uparrow$  (by grade)", ("LGG", "HGG")),
              ("lesion", ("NoReop", "Reop"), axes[1][0], "Lesion Dice $\\uparrow$  (by reoperation)", ("No reop.", "Reop.")),
              ("cavity", ("NoReop", "Reop"), axes[1][1], "Cavity Dice $\\uparrow$  (by reoperation)", ("No reop.", "Reop."))]
    x = range(len(SUB_SETS))
    w = 0.38
    for cls, (k1, k2), ax, ttl, (l1, l2) in combos:
        real = strat_means("REAL_T2", cls)
        for i, s in enumerate(SUB_SETS):
            sm = strat_means(s, cls)
            ax.bar(i - w / 2, sm.get(k1, math.nan), w, color=C1, zorder=3,
                   edgecolor="black", linewidth=0.4, label=l1 if i == 0 else None)
            ax.bar(i + w / 2, sm.get(k2, math.nan), w, color=C2, zorder=3,
                   edgecolor="black", linewidth=0.4, label=l2 if i == 0 else None)
        ax.axhline(real[k1], color=C1, ls=":", lw=2.0, zorder=2)
        ax.axhline(real[k2], color=C2, ls=":", lw=2.0, zorder=2)
        ax.set_title(ttl, fontsize=14)
        ax.set_ylim(0, 0.85)
        ax.grid(axis="y", ls=":", alpha=0.4, zorder=0)
        if cls == "lesion":
            ax.set_ylabel("Dice")
        hleg = [Line2D([], [], color=C1, ls=":", lw=2, label=f"{l1} (real T2w ref.)"),
                Line2D([], [], color=C2, ls=":", lw=2, label=f"{l2} (real T2w ref.)")]
        ax.legend(handles=ax.get_legend_handles_labels()[0] + hleg,
                  loc="upper right", fontsize=9.5, ncol=2, frameon=True)
    labels = [disp(s).replace(" ", "\n", 1) for s in SUB_SETS]
    for ax in axes[1]:
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=10.5)
    fig.suptitle("Downstream lesion (primary) and cavity (secondary) segmentation Dice "
                 "from synthetic T2w, stratified by grade and reoperation\n"
                 "(dotted lines = nnU-Net on real T2w in the same stratum, upper-bound reference)",
                 fontsize=14.5)
    return fig


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
    print("written:", name)


save(fig_downstream(), "Fig_downstream_utility_correlation")
save(fig_subgroup(), "Fig_subgroup_downstream")
