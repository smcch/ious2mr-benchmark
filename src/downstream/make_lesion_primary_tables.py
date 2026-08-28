r"""Lesion-primary downstream bundle for the manuscript rewrite (A3).

Reads ONLY canonical inputs:
  results_paper_protocol/seg_results_{T2,FLAIR}.csv        (present-class protocol, jun-16)
  results_paper_protocol/seg_metrics_T2_per_study.csv      (per-study, for subgroups)
  results_paper_protocol/seg_wilcoxon_{T2,FLAIR}.csv
  ../paper_assets/all_experiments_metrics.csv              (fidelity means; SynDiff-dual fixed)
  ../paper_assets/ReMIND_subset_diagnoses.csv              (strata)

Writes results_paper_protocol/lesion_primary/:
  table_lesion_T2.csv / table_lesion_FLAIR.csv   (top models + REAL, per-phase, retention)
  correlations_lesion_corrected.csv              (4 metrics x dice/nsd x T2/FLAIR, exact t p)
  subgroup_lesion_T2.csv                         (per top config x strata: mean Dice + utility)
  secondary_tumor_cavity.csv                     (ceilings/best/retention per phase)
and prints a human-readable report with LaTeX-ready rows.

Utility/retention definition (kept from the jun-16 protocol): ratio of cohort
means, U = mean(synth Dice) / mean(real Dice), lesion class, present-class
aggregation; no per-study threshold.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from common.paths import PROJECT_ROOT, DATA_ROOT, CKPT_ROOT, EXTERNAL_ROOT  # noqa: E402
import os  # noqa: E402,F811
import csv
import math
import os
import statistics as st
from collections import defaultdict

from scipy.stats import t as tdist

ROOT = os.path.join(str(PROJECT_ROOT), "downstream_seg")
RES = os.path.join(ROOT, "results_paper_protocol")
OUT = os.path.join(RES, "lesion_primary")
os.makedirs(OUT, exist_ok=True)
ASSETS = os.path.join(str(PROJECT_ROOT), "paper_assets")


def load(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


seg_t2 = load(os.path.join(RES, "seg_results_T2.csv"))
seg_fl = load(os.path.join(RES, "seg_results_FLAIR.csv"))
per_study = load(os.path.join(RES, "seg_metrics_T2_per_study.csv"))
wil_t2 = load(os.path.join(RES, "seg_wilcoxon_T2.csv"))
wil_fl = load(os.path.join(RES, "seg_wilcoxon_FLAIR.csv"))
expm = load(os.path.join(ASSETS, "all_experiments_metrics.csv"))
diag = load(os.path.join(ASSETS, "ReMIND_subset_diagnoses.csv"))

EXCLUDE = ("cascade", "joint")  # not part of the 48-experiment benchmark


def is_bench(setname):
    return not any(e in setname for e in EXCLUDE)


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


# ---------------- display names ----------------
def display(setname):
    s = setname
    dual = "-from-dual" in s or s.endswith("T2_FLAIR")
    target = "T2w + FLAIR" if dual else "T2w only"
    s = (s.replace("GAN-", "").replace("-from-single", "").replace("-from-dual", "")
           .replace("-T2", "").replace("-FLAIR", ""))
    fam_map = {"pix2pix": "Pix2Pix", "swinpix2pix": "SwinPix2Pix",
               "cyclegan": "CycleGAN", "cut": "CUT"}
    for k, v in fam_map.items():
        if s.lower().startswith(k):
            s = v + s[len(k):]
    s = (s.replace("2D+3D-post", "2D + 3D-refine")
           .replace("3D+3D-refine", "2D + 3D-refine")
           .replace("2D+3D-refine", "2D + 3D-refine"))
    # bare "-3D" regime = Full-3D (but not the refine label)
    if s.endswith("-3D") and "refine" not in s:
        s = s[:-3] + "-Full-3D"
    fam, _, reg = s.partition("-")
    return fam, reg.replace("-", " ").replace("2D + 3D refine", "2D + 3D-refine"), target


def get(rows, setname, cls, phase):
    for r in rows:
        if r["set"] == setname and r["class"] == cls and r["phase"] == phase:
            return r
    return None


# ---------------- 1. lesion leaderboards ----------------
def lesion_table(seg_rows, mod, top_n):
    real = get(seg_rows, f"REAL_{mod}", "lesion", "all")
    real_d = f(real["dice_mean"])
    rows = [r for r in seg_rows if r["class"] == "lesion" and r["phase"] == "all"
            and not r["set"].startswith("REAL_") and is_bench(r["set"])
            and r["dice_mean"] not in ("", "nan")]
    rows.sort(key=lambda r: -f(r["dice_mean"]))
    out = []

    def mk(r, rank):
        s = r["set"]
        pre = get(seg_rows, s, "lesion", "preop")
        post = get(seg_rows, s, "lesion", "postop")
        return dict(rank=rank, set=s,
                    dice=f(r["dice_mean"]), nsd=f(r["nsd_mean"]),
                    hd95=f(r["hd95_median"]),
                    ret=100 * f(r["dice_mean"]) / real_d,
                    dice_pre=f(pre["dice_mean"]) if pre else math.nan,
                    dice_post=f(post["dice_mean"]) if post else math.nan)

    real_pre = get(seg_rows, f"REAL_{mod}", "lesion", "preop")
    real_post = get(seg_rows, f"REAL_{mod}", "lesion", "postop")
    out.append(dict(rank=0, set=f"REAL_{mod}", dice=real_d, nsd=f(real["nsd_mean"]),
                    hd95=f(real["hd95_median"]), ret=100.0,
                    dice_pre=f(real_pre["dice_mean"]), dice_post=f(real_post["dice_mean"])))
    for i, r in enumerate(rows[:top_n], 1):
        out.append(mk(r, i))
    # first GAN
    for i, r in enumerate(rows, 1):
        if r["set"].startswith("GAN-"):
            out.append(mk(r, i))
            break
    return out, rows


tab_t2, all_t2 = lesion_table(seg_t2, "T2", 8)
tab_fl, all_fl = lesion_table(seg_fl, "FLAIR", 4)

for name, tab in [("table_lesion_T2.csv", tab_t2), ("table_lesion_FLAIR.csv", tab_fl)]:
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=list(tab[0]))
        w.writeheader()
        w.writerows(tab)

print("=" * 100)
print("LESION TABLE (T2, n=29: 16 pre / 13 post)   [LaTeX rows]")
for r in tab_t2:
    if r["set"].startswith("REAL_"):
        lab = ("Real T2w", "---", "---")
    else:
        lab = display(r["set"])
    print(f"{r['rank']:>2} & {lab[0]} & {lab[1]} & {lab[2]} & "
          f"{r['dice']:.3f} & {r['nsd']:.2f} & {r['hd95']:.1f} & {r['ret']:.0f}\\,\\% & "
          f"{r['dice_pre']:.3f} & {r['dice_post']:.3f} \\\\")
print()
print("LESION TABLE (FLAIR, n=19: 12 pre / 7 post)   [LaTeX rows]")
for r in tab_fl:
    if r["set"].startswith("REAL_"):
        lab = ("Real FLAIR", "---", "---")
    else:
        lab = display(r["set"])
    print(f"{r['rank']:>2} & {lab[0]} & {lab[1]} & "
          f"{r['dice']:.3f} & {r['nsd']:.2f} & {r['hd95']:.1f} & {r['ret']:.0f}\\,\\% & "
          f"{r['dice_pre']:.3f} & {r['dice_post']:.3f} \\\\")

n_gan_top = sum(1 for r in all_t2[:8] if r["set"].startswith("GAN-"))
print(f"\nGAN baselines in T2 lesion top-8: {n_gan_top}")

# ---------------- 2. secondary tumour/cavity ----------------
sec_rows = []
print("\n" + "=" * 100)
print("SECONDARY tumour/cavity (paper protocol)")
for mod, seg_rows in [("T2", seg_t2), ("FLAIR", seg_fl)]:
    for cls in ("tumor", "cavity"):
        for phase in ("all", "preop", "postop"):
            real = get(seg_rows, f"REAL_{mod}", cls, phase)
            if not real:
                continue
            cand = [r for r in seg_rows if r["class"] == cls and r["phase"] == phase
                    and not r["set"].startswith("REAL_") and is_bench(r["set"])
                    and r["dice_mean"] not in ("", "nan")]
            cand.sort(key=lambda r: -f(r["dice_mean"]))
            best = cand[0]
            row = dict(modality=mod, cls=cls, phase=phase,
                       n=real["n_gt_present"], real_dice=f(real["dice_mean"]),
                       real_nsd=f(real["nsd_mean"]),
                       best_set=best["set"], best_dice=f(best["dice_mean"]),
                       best_nsd=f(best["nsd_mean"]),
                       ret=100 * f(best["dice_mean"]) / f(real["dice_mean"]))
            sec_rows.append(row)
            print(f"{mod:5s} {cls:6s} {phase:6s} n={row['n']:>2} real={row['real_dice']:.3f} "
                  f"best={best['set'][:42]:42s} {row['best_dice']:.3f} ({row['ret']:.0f}%)")
# where does the lesion-top model sit on cavity?
top_set = tab_t2[1]["set"]
cav = get(seg_t2, top_set, "cavity", "all")
tum = get(seg_t2, top_set, "tumor", "all")
print(f"\nTop lesion model {top_set}: tumour Dice {f(tum['dice_mean']):.3f}, "
      f"cavity Dice {f(cav['dice_mean']):.3f}")
with open(os.path.join(OUT, "secondary_tumor_cavity.csv"), "w", newline="", encoding="utf-8") as fo:
    w = csv.DictWriter(fo, fieldnames=list(sec_rows[0]))
    w.writeheader()
    w.writerows(sec_rows)

# ---------------- 3. Wilcoxon summaries ----------------
print("\n" + "=" * 100)
print("WILCOXON vs real (paper protocol), benchmark sets only")
for mod, wil in [("T2", wil_t2), ("FLAIR", wil_fl)]:
    for cls in ("lesion", "tumor", "cavity"):
        ps = [f(r["p_value"]) for r in wil
              if r["class"] == cls and r["phase"] == "all" and is_bench(r["set"])
              and r["p_value"] not in ("", "nan")]
        ps = [p for p in ps if p == p]
        if ps:
            print(f"{mod:5s} {cls:6s}: n_sets={len(ps):2d}  max p={max(ps):.3g}  "
                  f"n>=0.05: {sum(1 for p in ps if p >= 0.05)}")

# ---------------- 4. corrected experiment-level correlations ----------------
def map_exp(exp, target, channel):
    e = exp.lower()
    dual = target == "T2+FLAIR"
    if "syndiff" in e:
        v = ("3D+3D-refine" if "3drefine" in e else
             "3D" if "full3d" in e else "2.5D" if "2.5d" in e else "2D")
        return (f"SynDiff-{v}-T2" + ("-from-dual" if dual else "")) if channel == "T2" \
            else f"SynDiff-{v}-FLAIR"
    if "resvit" in e:
        v = ("2D+3D-refine" if "2d_3d_refine" in e or "2d+3d" in e else
             "3D" if "full_3d" in e or "full-3d" in e else
             "2.5D" if "2.5d" in e else "2D")
        return (f"ResViT-{v}-T2" + ("-from-dual" if dual else "-from-single")) if channel == "T2" \
            else f"ResViT-{v}-FLAIR"
    for fam in ("pix2pix", "swinpix2pix", "cyclegan", "cut"):
        if e.startswith(fam):
            v = ("2D+3D-post" if "2d_3dpost" in e else
                 "3D" if "_3d_" in e or "_3d" == e[-3:] else
                 "2.5D" if "_25d" in e else "2D")
            return (f"GAN-{fam}-{v}-T2" + ("-from-dual" if dual else "-from-single")) if channel == "T2" \
                else f"GAN-{fam}-{v}-FLAIR"
    return None


def util_map(seg_rows, mod, endpoint):
    col = "dice_mean" if endpoint == "dice" else "nsd_mean"
    real = f(get(seg_rows, f"REAL_{mod}", "lesion", "all")[col])
    return {r["set"]: f(r[col]) / real for r in seg_rows
            if r["class"] == "lesion" and r["phase"] == "all"
            and not r["set"].startswith("REAL_") and r[col] not in ("", "nan")}


def pearson_t(xs, ys):
    n = len(xs)
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    r = num / den
    tval = r * math.sqrt((n - 2) / (1 - r * r))
    p = 2 * tdist.sf(abs(tval), n - 2)
    return r, p, n


def spearman_t(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        for i, idx in enumerate(order):
            rk[idx] = i
        return rk
    return pearson_t(rank(xs), rank(ys))


print("\n" + "=" * 100)
print("CORRECTED experiment-level correlations (fidelity mean vs lesion utility = ratio of means)")
corr_rows = []
for channel, seg_rows in [("T2", seg_t2), ("FLAIR", seg_fl)]:
    for endpoint in ("dice", "nsd"):
        um = util_map(seg_rows, channel, endpoint)
        data = []
        for r in expm:
            if channel == "T2":
                key = ("t2",)
                if r["Target"] not in ("T2 only", "T2+FLAIR"):
                    continue
            else:
                if r["Target"] != "T2+FLAIR":
                    continue
            mapped = map_exp(r["Experiment"], r["Target"], channel)
            if mapped not in um:
                continue
            pref = "t2" if channel == "T2" else "flair"
            try:
                vals = {m: float(r[f"{m}_{pref}_mean"]) for m in ("lpips", "ssim", "psnr", "mae")}
            except ValueError:
                continue
            data.append((vals, um[mapped], r["Family"]))
        for m in ("lpips", "ssim", "psnr", "mae"):
            xs = [d[0][m] for d in data]
            ys = [d[1] for d in data]
            r_, p_, n_ = pearson_t(xs, ys)
            rho, prho, _ = spearman_t(xs, ys)
            corr_rows.append(dict(channel=channel, endpoint=endpoint, metric=m.upper(),
                                  n=n_, pearson_r=round(r_, 3), p=f"{p_:.3g}",
                                  spearman_rho=round(rho, 3), p_rho=f"{prho:.3g}"))
            print(f"{channel:5s} {endpoint:4s} {m.upper():5s} n={n_:2d} "
                  f"r={r_:+.3f} (p={p_:.2g})  rho={rho:+.3f} (p={prho:.2g})")
with open(os.path.join(OUT, "correlations_lesion_corrected.csv"), "w", newline="", encoding="utf-8") as fo:
    w = csv.DictWriter(fo, fieldnames=list(corr_rows[0]))
    w.writeheader()
    w.writerows(corr_rows)

# ---------------- 5. subgroup lesion (and cavity) utilities, T2 ----------------
LGG = {r["ID"] for r in diag if r["Grade"] == "Low"}
REOP = {r["ID"] for r in diag if r["Reoperation"] == "Yes"}


def stratum_of(study):
    pid = study.rsplit("-", 1)[0]
    return ("LGG" if pid in LGG else "HGG", "Reop" if pid in REOP else "NoReop")


def subgroup_means(setname, cls):
    vals = defaultdict(list)
    for r in per_study:
        if r["set"] != setname or r["class"] != cls or r["gt_present"] != "1":
            continue
        d = f(r["dice"])
        if d != d:
            continue
        g, o = stratum_of(r["study"])
        vals[g].append(d)
        vals[o].append(d)
        vals["all"].append(d)
    return {k: (st.mean(v), len(v)) for k, v in vals.items()}


print("\n" + "=" * 100)
print("SUBGROUP lesion Dice (T2; paper protocol), ratio-of-means utility per stratum")
real_sub = subgroup_means("REAL_T2", "lesion")
print("REAL_T2 lesion per stratum:",
      {k: (round(v[0], 3), v[1]) for k, v in sorted(real_sub.items())})
sub_rows = []
top8_sets = [r["set"] for r in tab_t2[1:9]]
for s in top8_sets:
    sm = subgroup_means(s, "lesion")
    row = dict(set=s)
    for k in ("LGG", "HGG", "NoReop", "Reop", "all"):
        row[f"dice_{k}"] = round(sm[k][0], 3)
        row[f"U_{k}"] = round(sm[k][0] / real_sub[k][0], 2)
        row[f"n_{k}"] = sm[k][1]
    sub_rows.append(row)
    lab = display(s)
    print(f"{lab[0]:12s} {lab[1]:16s} {lab[2]:12s} "
          f"U: LGG {row['U_LGG']:.2f} HGG {row['U_HGG']:.2f} "
          f"NoReop {row['U_NoReop']:.2f} Reop {row['U_Reop']:.2f} | "
          f"Dice: {row['dice_LGG']:.3f}/{row['dice_HGG']:.3f}/"
          f"{row['dice_NoReop']:.3f}/{row['dice_Reop']:.3f}")
print("strata n:", {k: real_sub[k][1] for k in ("LGG", "HGG", "NoReop", "Reop")})
print("\nSUBGROUP cavity Dice (secondary, same sets):")
real_cav = subgroup_means("REAL_T2", "cavity")
print("REAL_T2 cavity per stratum:",
      {k: (round(v[0], 3), v[1]) for k, v in sorted(real_cav.items())})
for s in top8_sets:
    sm = subgroup_means(s, "cavity")
    lab = display(s)
    parts = []
    for k in ("LGG", "HGG", "NoReop", "Reop"):
        if k in sm and k in real_cav:
            parts.append(f"{k} {sm[k][0]:.3f} (U {sm[k][0]/real_cav[k][0]:.2f}, n={sm[k][1]})")
    print(f"{lab[0]:12s} {lab[1]:16s} {lab[2]:12s} | " + "  ".join(parts))
with open(os.path.join(OUT, "subgroup_lesion_T2.csv"), "w", newline="", encoding="utf-8") as fo:
    w = csv.DictWriter(fo, fieldnames=list(sub_rows[0]))
    w.writeheader()
    w.writerows(sub_rows)

print("\nbundle ->", OUT)
