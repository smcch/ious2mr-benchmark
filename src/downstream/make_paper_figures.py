"""Generate individual figure panels for the downstream-segmentation rewrite.

Produces one self-contained PNG per panel under
   $IOUS2MR_ROOT\\downstream_seg\\results_paper_protocol\\figures\\

So the user can assemble multi-panel figures later as needed.

Panels:
  fig01_lesion_dice_by_phase_T2.png       T2w lesion Dice (real vs best synth vs inter-obs ref) × {all, preop, postop}
  fig02_lesion_dice_by_phase_FLAIR.png    FLAIR lesion Dice, same layout
  fig03_lesion_retention_overview.png     Retention (synth/real) per modality × phase
  fig04_tumor_dice_by_phase.png           Secondary: tumour Dice per modality × phase
  fig05_cavity_dice_by_phase.png          Secondary: cavity Dice per modality × phase
  fig06_inter_observer_reference.png      Inter-observer Dice per class × phase (reference of task difficulty)
  fig07a_top_models_lesion_T2.png         Horizontal-bar top-12 synth methods on T2w lesion
  fig07b_top_models_lesion_FLAIR.png      Horizontal-bar top-12 synth methods on FLAIR lesion
  fig08_fidelity_vs_utility_LPIPS.png     Scatter: LPIPS vs lesion utility ratio (T2w experiments)
  fig09_fidelity_vs_utility_SSIM.png      Scatter: SSIM vs lesion utility
  fig10_fidelity_vs_utility_PSNR.png      Scatter
  fig11_fidelity_vs_utility_MAE.png       Scatter
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from common.paths import PROJECT_ROOT, DATA_ROOT, CKPT_ROOT, EXTERNAL_ROOT  # noqa: E402
import os  # noqa: E402,F811
import os, csv, re, math
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(str(PROJECT_ROOT), "downstream_seg")
RES  = os.path.join(ROOT, "results_paper_protocol")
OUT  = os.path.join(RES, "figures")
os.makedirs(OUT, exist_ok=True)

PAPER_ASSETS = os.path.join(str(PROJECT_ROOT), "paper_assets")

# ------------------------- shared style ----------------------------------------
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})

C_REAL  = "#1f77b4"   # blue  -- real MR
C_SYNTH = "#d62728"   # red   -- best synth
C_HUMAN = "#7f7f7f"   # grey  -- inter-observer reference
C_TUMOR = "#2ca02c"
C_CAVITY= "#ff7f0e"
C_LESION= "#9467bd"

FAMILY_COLOR = {
    "Pix2Pix":      "#1f77b4",
    "SwinPix2Pix":  "#ff7f0e",
    "CycleGAN":     "#2ca02c",
    "CUT":          "#d62728",
    "ResViT":       "#9467bd",
    "SynDiff":      "#8c564b",
}

def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print(f"-> {p}")


# ------------------------- load data ------------------------------------------
def load_csv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))

headline_lesion = load_csv(os.path.join(RES, "headline_lesion_primary.csv"))
headline_tc     = load_csv(os.path.join(RES, "headline_tumor_cavity_secondary.csv"))
interobs_sum    = load_csv(os.path.join(RES, "interobs_MR_summary.csv"))
seg_results_t2  = load_csv(os.path.join(RES, "seg_results_T2.csv"))
seg_results_fl  = load_csv(os.path.join(RES, "seg_results_FLAIR.csv"))
exp_metrics     = load_csv(os.path.join(PAPER_ASSETS, "all_experiments_metrics.csv"))


def float_or_nan(x):
    try: return float(x)
    except (ValueError, TypeError): return float("nan")


def human_lesion_dice():
    """Return dict scope -> dice mean. Scopes: test_no_copies, test_preop, test_postop."""
    out = {}
    for r in interobs_sum:
        if r["class"] == "lesion":
            out[r["scope"]] = float_or_nan(r.get("dice_mean", float("nan")))
    return out


def human_dice(cls, scope_alias):
    """scope_alias in {all, preop, postop} -> test_no_copies, test_preop, test_postop"""
    mapping = {"all": "test_no_copies", "preop": "test_preop", "postop": "test_postop"}
    sc = mapping[scope_alias]
    for r in interobs_sum:
        if r["class"] == cls and r["scope"] == sc:
            return float_or_nan(r.get("dice_mean", float("nan")))
    return float("nan")


def headline_lookup(rows, mod, phase, cls=None):
    for r in rows:
        if r["modality"] == mod and r["phase"] == phase and (cls is None or r["class"] == cls):
            return r
    return None


# ============================================================================
# PANEL 1 & 2: Lesion Dice by phase, per modality
# ============================================================================
def panel_lesion_by_phase(mod, fname):
    phases = ["all", "preop", "postop"]
    real = [float(headline_lookup(headline_lesion, mod, p)["ref_dice_mean"]) for p in phases]
    synth = [float(headline_lookup(headline_lesion, mod, p)["best_synth_dice_mean"]) for p in phases]
    human = [human_dice("lesion", p) for p in phases]
    n_ds = [headline_lookup(headline_lesion, mod, p)["n_evaluable"] for p in phases]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    x = np.arange(len(phases))
    w = 0.32
    b1 = ax.bar(x - w, real,  w, color=C_REAL,  label=f"Real {mod}",  zorder=3)
    b2 = ax.bar(x,     synth, w, color=C_SYNTH, label="Best synth",   zorder=3)
    b3 = ax.bar(x + w, human, w, color=C_HUMAN, label="Inter-obs (ref.)", zorder=3, alpha=0.85)

    for rects, vals in [(b1, real), (b2, synth), (b3, human)]:
        for r, v in zip(rects, vals):
            if not math.isnan(v):
                ax.text(r.get_x() + r.get_width()/2, v + 0.015, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"all\n(n={n_ds[0]})", f"preop\n(n={n_ds[1]})", f"postop\n(n={n_ds[2]})"])
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{mod} — lesion (tumour $\\cup$ cavity)")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(loc="upper right", frameon=False)
    save(fig, fname)


panel_lesion_by_phase("T2",    "fig01_lesion_dice_by_phase_T2.png")
panel_lesion_by_phase("FLAIR", "fig02_lesion_dice_by_phase_FLAIR.png")


# ============================================================================
# PANEL 3: Retention overview (synth/real) per modality × phase
# ============================================================================
def panel_retention_overview():
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    cells = []
    for mod in ("T2", "FLAIR"):
        for phase in ("all", "preop", "postop"):
            r = headline_lookup(headline_lesion, mod, phase)
            cells.append((f"{mod}\n{phase}", float(r["retention_pct"])))
    labels, vals = zip(*cells)
    colors = ["#1f77b4"]*3 + ["#ff7f0e"]*3
    bars = ax.bar(range(len(vals)), vals, color=colors, zorder=3)
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width()/2, v + 1.0, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=9)
    ax.axhline(100, color="grey", linestyle="--", linewidth=0.8, zorder=1)
    ax.text(0.5, 100.8, "real-MR ceiling (100%)", color="grey", fontsize=8, ha="center", va="bottom")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Retention vs. real-MR Dice (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Lesion-Dice retention by modality × phase")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    # modality is already clear from the x-tick labels, no separate legend
    save(fig, "fig03_lesion_retention_overview.png")


panel_retention_overview()


# ============================================================================
# PANEL 4 & 5: Secondary endpoints — tumor and cavity
# ============================================================================
def panel_secondary(cls, fname):
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    phases = ["all", "preop", "postop"]
    bars_per_cell = []  # (label, real, synth, human, n)
    for mod in ("T2", "FLAIR"):
        for phase in phases:
            r = headline_lookup(headline_tc, mod, phase, cls=cls)
            if r is None: continue
            bars_per_cell.append((
                f"{mod}\n{phase}",
                float(r["ref_dice_mean"]),
                float(r["best_synth_dice_mean"]),
                human_dice(cls, phase),
                int(r["n_evaluable"]),
            ))
    labels = [b[0] for b in bars_per_cell]
    real   = [b[1] for b in bars_per_cell]
    synth  = [b[2] for b in bars_per_cell]
    human  = [b[3] for b in bars_per_cell]
    ns     = [b[4] for b in bars_per_cell]
    x = np.arange(len(labels))
    w = 0.27
    ax.bar(x - w, real,  w, color=C_REAL,  label="Real",       zorder=3)
    ax.bar(x,     synth, w, color=C_SYNTH, label="Best synth", zorder=3)
    ax.bar(x + w, human, w, color=C_HUMAN, label="Inter-obs (ref.)", zorder=3, alpha=0.85)
    ax.set_xticks(x)
    # combined label: "T2\npreop\n(n=16)"
    combined_labels = [f"{lbl}\n(n={ni})" for lbl, ni in zip(labels, ns)]
    ax.set_xticklabels(combined_labels)
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Secondary endpoint — {cls}")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(loc="upper right", frameon=False, ncol=3)
    # Mark very small evaluable cohorts with a star above the synth bar
    for xi, ni, sv in zip(x, ns, synth):
        if ni <= 6:
            top = max(real[list(x).index(xi)], sv, human[list(x).index(xi)] if not math.isnan(human[list(x).index(xi)]) else 0)
            ax.annotate("★", xy=(xi, top + 0.06), ha="center", va="bottom",
                        fontsize=14, color="orange")
    save(fig, fname)


panel_secondary("tumor",  "fig04_tumor_dice_by_phase.png")
panel_secondary("cavity", "fig05_cavity_dice_by_phase.png")


# ============================================================================
# PANEL 6: Inter-observer reference summary (class × phase, test cohort)
# ============================================================================
def panel_inter_observer():
    classes = ["lesion", "tumor", "cavity"]
    scopes  = [("test_no_copies", "all"), ("test_preop", "preop"), ("test_postop", "postop")]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    x = np.arange(len(classes))
    w = 0.27
    for i, (sc, lbl) in enumerate(scopes):
        vals = []
        for cls in classes:
            for r in interobs_sum:
                if r["class"] == cls and r["scope"] == sc:
                    vals.append(float_or_nan(r.get("dice_mean", float("nan"))))
                    break
            else:
                vals.append(float("nan"))
        ax.bar(x + (i-1)*w, vals, w, label=lbl, zorder=3,
               color=["#888888", "#1f77b4", "#d62728"][i])
        for xi, v in zip(x + (i-1)*w, vals):
            if not math.isnan(v):
                ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Inter-observer Dice")
    ax.set_ylim(0, 1.0)
    ax.set_title("Inter-observer reference (test cohort)")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    save(fig, "fig06_inter_observer_reference.png")


panel_inter_observer()


# ============================================================================
# PANEL 7: Top-12 synth methods on lesion (T2w + FLAIR)
# ============================================================================
def family_of(set_name):
    if set_name.startswith("GAN-"):
        return set_name.split("-")[1].title().replace("Cyclegan", "CycleGAN").replace("Pix2Pix", "Pix2Pix").replace("Swinpix2Pix", "SwinPix2Pix").replace("Cut", "CUT")
    if set_name.startswith("ResViT"):  return "ResViT"
    if set_name.startswith("SynDiff"): return "SynDiff"
    return "Other"


def short_label(set_name):
    s = set_name.replace("GAN-", "").replace("-T2-from-single", "-single").replace("-T2-from-dual", "-dual").replace("-T2", "")
    s = s.replace("-FLAIR", "-FLAIR")
    return s


def panel_top_models(seg_rows, mod, fname, top_n=12):
    rows = [r for r in seg_rows if r["class"] == "lesion" and r["phase"] == "all" and not r["set"].startswith("REAL_")]
    rows = [r for r in rows if not (r["dice_mean"] == "" or r["dice_mean"] == "nan")]
    rows.sort(key=lambda r: -float(r["dice_mean"]))
    rows = rows[:top_n]
    real_dice = next(float(r["dice_mean"]) for r in seg_rows if r["set"] == f"REAL_{mod}" and r["class"] == "lesion" and r["phase"] == "all")
    human_d = human_dice("lesion", "all")
    fig, ax = plt.subplots(figsize=(6.6, max(3.5, 0.32*top_n)))
    labels = [short_label(r["set"]) for r in rows]
    dices  = [float(r["dice_mean"]) for r in rows]
    fams   = [family_of(r["set"]) for r in rows]
    colors = [FAMILY_COLOR.get(f, "#666666") for f in fams]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, dices, color=colors, zorder=3)
    for yi, v in zip(y, dices):
        ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=8)
    ax.axvline(real_dice, color=C_REAL,  linestyle="--", linewidth=1.2, zorder=4, label=f"Real {mod} ({real_dice:.2f})")
    ax.axvline(human_d,   color=C_HUMAN, linestyle=":",  linewidth=1.2, zorder=4, label=f"Inter-obs ref. ({human_d:.2f})")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Lesion Dice (all phases)")
    ax.set_xlim(0, max(real_dice, max(dices))*1.08 + 0.05)
    ax.set_title(f"Top-{top_n} synthetic methods on {mod} lesion")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    # family legend
    from matplotlib.patches import Patch
    fam_handles = [Patch(color=FAMILY_COLOR[f], label=f) for f in sorted(set(fams))]
    ax.legend(handles=fam_handles + [
        plt.Line2D([0],[0], color=C_REAL, linestyle="--", label=f"Real {mod} ({real_dice:.2f})"),
        plt.Line2D([0],[0], color=C_HUMAN, linestyle=":", label=f"Inter-obs ref. ({human_d:.2f})"),
    ], loc="lower right", frameon=False, fontsize=8)
    save(fig, fname)


panel_top_models(seg_results_t2, "T2",    "fig07a_top_models_lesion_T2.png", top_n=12)
panel_top_models(seg_results_fl, "FLAIR", "fig07b_top_models_lesion_FLAIR.png", top_n=12)


# ============================================================================
# PANEL 8-11: Fidelity vs lesion-utility correlation (T2w experiments)
# ============================================================================
def map_experiment_to_set(exp_name, target, channel="T2"):
    """Map experiment-csv name to seg-result set name.

    GANs use lowercase_with_underscores (e.g. 'pix2pix_2d_t2', 'cut_2d_3dpost_t2_flair').
    ResViT and SynDiff use PascalCase with hyphens (e.g. 'SynDiff-2D-T2', 'SynDiff-3Drefine-T2_FLAIR').
    The two naming conventions coexist in all_experiments_metrics.csv. This mapper
    handles both.
    """
    e_raw = exp_name
    e = exp_name.lower()
    is_dual = target == "T2+FLAIR"

    if "syndiff" in e:
        # Pascal-case form: 'SynDiff-2D-T2', 'SynDiff-3Drefine-T2', 'SynDiff-full3D-T2_FLAIR'
        # Decide variant from the canonical form
        if   "3drefine" in e: variant = "3D+3D-refine"     # SynDiff's analogue of 2D+3D-refine
        elif "full3d"   in e: variant = "3D"
        elif "2.5d"     in e or "25d" in e: variant = "2.5D"
        elif "-2d-"     in e or e.endswith("-2d-t2") or e.endswith("-2d-t2_flair") or "syndiff-2d" in e: variant = "2D"
        else: variant = "2D"
        if channel == "T2":
            tag = f"-{variant}-T2" + ("-from-dual" if is_dual else "")
        else:
            tag = f"-{variant}-FLAIR"
        return f"SynDiff{tag}"

    if "resvit" in e:
        # Pascal-case form: 'ResViT-2D-T2', 'ResViT-2.5D-T2', 'ResViT-2D+3D-refine-T2', 'ResViT-Full-3D-T2'
        if   "2d+3d-refine" in e or "2d_3drefine" in e: variant = "2D+3D-refine"
        elif "full-3d"      in e or "full_3d"     in e: variant = "3D"
        elif "-3d-"         in e and "2d+3d"      not in e: variant = "3D"
        elif "2.5d"         in e or "25d"         in e: variant = "2.5D"
        else: variant = "2D"
        if channel == "T2":
            tag = f"-{variant}-T2" + ("-from-dual" if is_dual else "-from-single")
        else:
            tag = f"-{variant}-FLAIR"
        return f"ResViT{tag}"

    # GANs (lowercase_underscore form)
    for fam in ("pix2pix", "swinpix2pix", "cyclegan", "cut"):
        if e.startswith(fam):
            if   "2d_3dpost" in e: variant = "2D+3D-post"
            elif "_3d_" in e or e.endswith("_3d"): variant = "3D"
            elif "_25d" in e: variant = "2.5D"
            else: variant = "2D"
            if channel == "T2":
                tag = f"-{variant}-T2" + ("-from-dual" if is_dual else "-from-single")
            else:
                tag = f"-{variant}-FLAIR"
            return f"GAN-{fam}{tag}"
    return None


def build_correlation_data(channel="T2", endpoint="dice"):
    """Yield (lpips, ssim, psnr, mae, util, family, variant) per experiment.

    endpoint = 'dice' or 'nsd' (both on lesion class, phase=all).
    """
    seg_rows = seg_results_t2 if channel == "T2" else seg_results_fl
    real_set = f"REAL_{channel}"
    metric_col = "dice_mean" if endpoint == "dice" else "nsd_mean"
    real_score = next(float(r[metric_col]) for r in seg_rows if r["set"] == real_set and r["class"] == "lesion" and r["phase"] == "all")
    set_to_score = {r["set"]: float(r[metric_col]) for r in seg_rows
                    if r["class"] == "lesion" and r["phase"] == "all" and r[metric_col] not in ("", "nan")}
    out = []
    for r in exp_metrics:
        if channel == "T2" and r["Target"] == "T2+FLAIR":
            mapped = map_experiment_to_set(r["Experiment"], r["Target"], "T2")
            lpips_key, ssim_key, psnr_key, mae_key = "lpips_t2_mean", "ssim_t2_mean", "psnr_t2_mean", "mae_t2_mean"
        elif channel == "T2" and r["Target"] == "T2 only":
            mapped = map_experiment_to_set(r["Experiment"], r["Target"], "T2")
            lpips_key, ssim_key, psnr_key, mae_key = "lpips_t2_mean", "ssim_t2_mean", "psnr_t2_mean", "mae_t2_mean"
        elif channel == "FLAIR" and r["Target"] == "T2+FLAIR":
            mapped = map_experiment_to_set(r["Experiment"], r["Target"], "FLAIR")
            lpips_key, ssim_key, psnr_key, mae_key = "lpips_flair_mean", "ssim_flair_mean", "psnr_flair_mean", "mae_flair_mean"
        else:
            continue
        if mapped not in set_to_score: continue
        try:
            lpips = float(r[lpips_key]); ssim = float(r[ssim_key])
            psnr = float(r[psnr_key]);   mae  = float(r[mae_key])
        except (ValueError, KeyError):
            continue
        util = set_to_score[mapped] / real_score
        fam = r["Family"]
        out.append((lpips, ssim, psnr, mae, util, fam, r["Variant"]))
    return out


def pearson(xs, ys):
    n = len(xs)
    if n < 3: return float("nan"), float("nan")
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx  = math.sqrt(sum((x-mx)**2 for x in xs))
    dy  = math.sqrt(sum((y-my)**2 for y in ys))
    if dx == 0 or dy == 0: return float("nan"), float("nan")
    r = num/(dx*dy)
    # approximate two-sided p
    t = r * math.sqrt((n-2) / max(1e-12, 1 - r*r))
    from math import erf, sqrt
    # crude normal approximation for p (good enough for figure annotations)
    p = 2*(1 - 0.5*(1 + erf(abs(t)/math.sqrt(2))))
    return r, p


def panel_fidelity_scatter_single_channel(metric_index, metric_label, xlabel, fname,
                                          lower_is_better, endpoint, channel):
    rows = build_correlation_data(channel, endpoint=endpoint)
    if not rows:
        print(f"  (skip {fname}: no data)")
        return
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    used_fams = set()
    xs = [r[metric_index] for r in rows]
    ys = [r[4] for r in rows]
    fams = [r[5] for r in rows]
    for x, y, fam in zip(xs, ys, fams):
        ax.scatter(x, y, color=FAMILY_COLOR.get(fam, "#666666"),
                   marker="o", s=46, alpha=0.80, edgecolor="white", linewidth=0.6)
        used_fams.add(fam)
    rcorr, pcorr = pearson(xs, ys)
    if len(xs) >= 3:
        m, b = np.polyfit(xs, ys, 1)
        xx = np.linspace(min(xs), max(xs), 100)
        ax.plot(xx, m*xx + b, color="grey", linestyle="--", linewidth=1, alpha=0.7)
    p_text = "< 0.001" if pcorr < 0.001 else f"= {pcorr:.3f}"
    ax.text(0.04, 0.04,
            f"Channel: {channel}  (n={len(xs)})\nPearson r = {rcorr:+.2f}\np {p_text}",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="grey"))
    util_label = "Dice" if endpoint == "dice" else "NSD$_{2\\mathrm{mm}}$"
    ax.set_xlabel(xlabel + ("  (lower is better)" if lower_is_better else "  (higher is better)"))
    ax.set_ylabel(f"Lesion utility ratio  (synth {util_label} / real {util_label})")
    ax.set_title(f"{metric_label} vs. lesion-{util_label} utility — {channel} channel")
    ax.grid(alpha=0.25)
    from matplotlib.lines import Line2D
    fam_handles = [Line2D([0],[0], marker="o", color="w",
                          markerfacecolor=FAMILY_COLOR.get(f, "#666666"), markersize=7,
                          label=f) for f in sorted(used_fams)]
    ax.legend(handles=fam_handles, loc="upper right", frameon=False, fontsize=8, title="Family")
    save(fig, fname)


# 4 metrics × 2 endpoints × 2 channels = 16 panels
METRICS = [
    (0, "LPIPS", "LPIPS-AlexNet", True),
    (1, "SSIM",  "SSIM",          False),
    (2, "PSNR",  "PSNR (dB)",     False),
    (3, "MAE",   "MAE",           True),
]
ENDPOINTS = [("dice", "Dice"), ("nsd", "NSD")]
CHANNELS  = ["T2", "FLAIR"]

for mi, mlbl, xl, lower in METRICS:
    for ep, ep_short in ENDPOINTS:
        for ch in CHANNELS:
            fname = f"fig08_{mlbl}_vs_lesion_{ep_short}_utility_{ch}.png" if mlbl == "LPIPS" else \
                    f"fig09_{mlbl}_vs_lesion_{ep_short}_utility_{ch}.png" if mlbl == "SSIM" else \
                    f"fig10_{mlbl}_vs_lesion_{ep_short}_utility_{ch}.png" if mlbl == "PSNR" else \
                    f"fig11_{mlbl}_vs_lesion_{ep_short}_utility_{ch}.png"
            panel_fidelity_scatter_single_channel(mi, mlbl, xl, fname, lower, ep, ch)

# Also write the actual correlation coefficients out to a CSV so the paper text
# can quote them without re-running anything.
with open(os.path.join(OUT, "correlations_lesion.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["endpoint", "metric", "n_T2", "n_FLAIR", "n_pooled", "pearson_r_pooled", "p_value_pooled",
                "pearson_r_T2", "pearson_r_FLAIR"])
    for endpoint in ("dice", "nsd"):
        rows_t2 = build_correlation_data("T2", endpoint=endpoint)
        rows_fl = build_correlation_data("FLAIR", endpoint=endpoint)
        for idx, name in enumerate(("LPIPS", "SSIM", "PSNR", "MAE")):
            all_x = [r[idx] for r in rows_t2 + rows_fl]
            all_y = [r[4]   for r in rows_t2 + rows_fl]
            rcorr, pcorr = pearson(all_x, all_y)
            r_t2, _ = pearson([r[idx] for r in rows_t2], [r[4] for r in rows_t2])
            r_fl, _ = pearson([r[idx] for r in rows_fl], [r[4] for r in rows_fl])
            w.writerow([endpoint, name, len(rows_t2), len(rows_fl), len(all_x),
                        f"{rcorr:.3f}", f"{pcorr:.4g}", f"{r_t2:.3f}", f"{r_fl:.3f}"])
print("-> correlations_lesion.csv")


print("\nAll panels written to:", OUT)
