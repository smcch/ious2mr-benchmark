"""Robustness checks for the fidelity-vs-utility correlation under pseudoreplication.

The 48 experiments in the benchmark share architecture families (6), inference
regimes (4) and target configurations (2). A naive Pearson r across all 48 (or
72, after splitting T2w/FLAIR channels) experiments treats them as independent,
inflating the effective sample size. This script reports three complementary
estimates that respect that structure:

  (a) Naive pooled r  (= what the paper currently reports)
  (b) Within-subject Pearson r (computed across experiments, within each test
      subject), then aggregated as median + 25/75 IQR across subjects
  (c) Cluster-bootstrap-by-family 95 % CI for the pooled r (5000 draws,
      resampling the 6 architecture families with replacement)
  (d) Linear mixed-effects model:  utility ~ fidelity + (1 | family) + (1 | subject)
      reports the fixed-effect coefficient and its t-statistic + p-value

Both lesion endpoints (Dice and NSD_2mm) and both channels (T2w, FLAIR) are
analysed independently because the utility ratio is defined channel-wise.

Inputs:
  $IOUS2MR_ROOT\\per_subject_all_models.csv
       columns: family, experiment, target, subject, ssim_t2, psnr_t2, mae_t2,
                lpips_t2, ssim_flair, psnr_flair, mae_flair, lpips_flair
  $IOUS2MR_ROOT\\downstream_seg\\results_paper_protocol\\
       seg_metrics_T2_per_study.csv      per-study downstream Dice/HD95/NSD
       seg_metrics_FLAIR_per_study.csv

Output:
  $IOUS2MR_ROOT\\downstream_seg\\results_paper_protocol\\correlation_robustness.csv
  (one row per metric × endpoint × channel × estimator)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from common.paths import PROJECT_ROOT, DATA_ROOT, CKPT_ROOT, EXTERNAL_ROOT  # noqa: E402
import os  # noqa: E402,F811
import os, csv, math, random, re
from collections import defaultdict
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

ROOT  = str(PROJECT_ROOT)
DSDIR = os.path.join(ROOT, "downstream_seg")
RES   = os.path.join(DSDIR, "results_paper_protocol")
PER_SUBJECT_FID = os.path.join(ROOT, "evaluacion-final", "rescore_methods_persubject.csv")
SEG_T2  = os.path.join(RES, "seg_metrics_T2_per_study.csv")
SEG_FL  = os.path.join(RES, "seg_metrics_FLAIR_per_study.csv")
OUT_CSV = os.path.join(RES, "correlation_robustness.csv")

random.seed(2026)
np.random.seed(2026)
N_BOOT = 5000


def load_csv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------- experiment-name normalisation ----------------------------
def map_method_to_set(method_name, channel):
    """Map rescore-CSV 'method' string to seg-result set name.

    rescore method examples:
      'pix2pix-2D-T2', 'pix2pix-2D-T2+FLAIR'
      'ResViT-2.5D-T2', 'ResViT-2.5D-T2+FLAIR'
      'SynDiff-2D-T2', 'SynDiff-2D-T2+FLAIR'
      'SynDiff-ResViT->SynDiff-cascade-T2', 'SynDiff-ResViT+SynDiff-joint-T2'
      'SynDiff-2D-untuned-T2'  (excluded)

    channel is 't2' or 'flair' (the row's output channel).
    """
    # Exclude non-canonical ensemble / extra runs: only the 48 canonical
    # experiments (6 families × 4 regimes × 2 targets) enter the correlation
    # analysis. The SynDiff cascade / joint / untuned variants are excluded
    # because they combine two networks and do not fit the cross-family axis.
    if "untuned" in method_name: return None
    if "cascade" in method_name: return None
    if "joint"   in method_name: return None
    is_dual = method_name.endswith("T2+FLAIR")
    base = method_name.replace("-T2+FLAIR", "").replace("-T2", "")
    # base now looks like "pix2pix-2D" or "ResViT-2.5D" or "SynDiff-2D"
    parts = base.split("-")
    family_raw = parts[0]
    variant = "-".join(parts[1:])
    # FLAIR only exists in dual-target configurations, and the downstream-eval
    # set names for FLAIR do NOT carry the -from-dual suffix.
    if family_raw.lower() in ("pix2pix", "swinpix2pix", "cyclegan", "cut"):
        fam = family_raw.lower()
        if channel == "flair":
            return f"GAN-{fam}-{variant}-FLAIR"
        suffix = "-from-dual" if is_dual else "-from-single"
        return f"GAN-{fam}-{variant}-T2{suffix}"
    if family_raw == "ResViT":
        if channel == "flair":
            return f"ResViT-{variant}-FLAIR"
        suffix = "-from-dual" if is_dual else "-from-single"
        return f"ResViT-{variant}-T2{suffix}"
    if family_raw == "SynDiff":
        if channel == "flair":
            return f"SynDiff-{variant}-FLAIR"
        suffix = "-from-dual" if is_dual else ""
        return f"SynDiff-{variant}-T2{suffix}"
    return None


def family_of_method(method, family_col):
    """Use the family column from the CSV when available, else parse."""
    if family_col:
        f = family_col.strip().lower()
        if "syndiff" in f: return "SynDiff"
        if "resvit" in f:  return "ResViT"
        for lc, nice in [("pix2pix","Pix2Pix"), ("swinpix2pix","SwinPix2Pix"),
                          ("cyclegan","CycleGAN"), ("cut","CUT")]:
            if lc in f: return nice
    return "Other"


# ---------------- assemble per-subject dataset ----------------------------
def build_dataset(channel="T2", endpoint="dice"):
    """Return list of dicts: family, method, subject, fidelity{ssim,psnr,mae,lpips}, utility.

    Restricted to the 48 canonical experiments (6 families × 4 regimes × 2 targets).
    Channel = 'T2' or 'FLAIR'.  Endpoint = 'dice' or 'nsd' on the lesion class.
    """
    fid_rows = load_csv(PER_SUBJECT_FID)
    seg_rows = load_csv(SEG_T2 if channel == "T2" else SEG_FL)
    metric_col = "dice" if endpoint == "dice" else "nsd2mm"
    chan_lc = "t2" if channel == "T2" else "flair"

    # (set, study) -> downstream score (lesion class only)
    ds = {}
    for r in seg_rows:
        if r["class"] != "lesion": continue
        v = r[metric_col]
        if v in ("", "nan"): continue
        try: ds[(r["set"], r["study"])] = float(v)
        except ValueError: continue
    real_set = f"REAL_{channel}"
    real_by_subj = {st: v for (s, st), v in ds.items() if s == real_set}

    out = []
    seen_methods = set()
    for fr in fid_rows:
        if fr["channel"] != chan_lc:
            continue
        method = fr["method"]
        mapped = map_method_to_set(method, fr["channel"])
        if mapped is None: continue
        if mapped == real_set: continue
        subj = fr["subject"]
        synth_score = ds.get((mapped, subj))
        if synth_score is None: continue
        real_score = real_by_subj.get(subj)
        if not real_score or real_score < 0.05: continue
        try:
            ssim  = float(fr["ssim"])
            psnr  = float(fr["psnr"])
            mae   = float(fr["mae"])
            lpips = float(fr["lpips"])
        except (ValueError, KeyError):
            continue
        if math.isnan(lpips) or math.isnan(ssim): continue
        out.append({
            "family":  family_of_method(method, fr.get("family", "")),
            "method":  method,
            "mapped":  mapped,
            "target":  fr["target"],
            "subject": subj,
            "ssim":    ssim,  "psnr": psnr,
            "mae":     mae,   "lpips": lpips,
            "utility": synth_score / real_score,
        })
        seen_methods.add(method)
    return out


# ---------------- estimators --------------------------------------------------
def pearson(xs, ys):
    n = len(xs); xs = np.asarray(xs); ys = np.asarray(ys)
    if n < 3: return float("nan"), float("nan")
    if np.std(xs) == 0 or np.std(ys) == 0: return float("nan"), float("nan")
    r = float(np.corrcoef(xs, ys)[0, 1])
    t = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return r, p


def within_subject_r(rows, fid_field):
    """Pearson r computed within each subject across experiments, then aggregated."""
    by_subj = defaultdict(list)
    for r in rows:
        by_subj[r["subject"]].append((r[fid_field], r["utility"]))
    rs = []
    for subj, pairs in by_subj.items():
        if len(pairs) < 4: continue
        xs, ys = zip(*pairs)
        if np.std(xs) == 0 or np.std(ys) == 0: continue
        r_subj = float(np.corrcoef(xs, ys)[0, 1])
        rs.append(r_subj)
    rs = np.array(rs)
    if len(rs) == 0:
        return None
    return {
        "n_subjects": len(rs),
        "median": float(np.median(rs)),
        "q25":    float(np.quantile(rs, 0.25)),
        "q75":    float(np.quantile(rs, 0.75)),
        "frac_same_sign": float((np.sign(rs) == np.sign(np.median(rs))).mean()),
    }


def cluster_bootstrap_by_family(rows, fid_field, n_boot=N_BOOT):
    """Resample the families with replacement; recompute pooled r over experiment means."""
    # collapse to experiment-level means first (this is what the original r was on)
    by_exp = defaultdict(lambda: {"fids": [], "utils": [], "family": None})
    for r in rows:
        k = (r["family"], r["method"], r["target"])
        by_exp[k]["fids"].append(r[fid_field])
        by_exp[k]["utils"].append(r["utility"])
        by_exp[k]["family"] = r["family"]
    exp_means = []
    for k, v in by_exp.items():
        exp_means.append({"family": v["family"], "fid": np.mean(v["fids"]),
                          "util": np.mean(v["utils"])})
    families = sorted({e["family"] for e in exp_means})
    by_fam = {f: [e for e in exp_means if e["family"] == f] for f in families}

    boot_rs = []
    for _ in range(n_boot):
        sampled_fams = [random.choice(families) for _ in families]
        sample_exps = [e for f in sampled_fams for e in by_fam[f]]
        xs = [e["fid"] for e in sample_exps]
        ys = [e["util"] for e in sample_exps]
        rb, _ = pearson(xs, ys)
        if not math.isnan(rb): boot_rs.append(rb)
    boot_rs = np.array(boot_rs)
    if len(boot_rs) == 0:
        return None
    naive_r, naive_p = pearson([e["fid"] for e in exp_means],
                                [e["util"] for e in exp_means])
    return {
        "n_experiments": len(exp_means),
        "naive_r":  naive_r,
        "naive_p":  naive_p,
        "boot_mean_r": float(np.mean(boot_rs)),
        "boot_ci_lo":  float(np.quantile(boot_rs, 0.025)),
        "boot_ci_hi":  float(np.quantile(boot_rs, 0.975)),
        "frac_same_sign": float((np.sign(boot_rs) == np.sign(naive_r)).mean()),
    }


def mixed_effects_model(rows, fid_field):
    """Fit a linear mixed model with subject and family random intercepts.

    utility_ij = beta0 + beta1 * fidelity_ij + u_family[i] + u_subject[j] + eps
    Reports beta1 (slope on the fidelity covariate), partial r, and p-value.
    """
    import pandas as pd
    df = pd.DataFrame([{"util": r["utility"], "fid": r[fid_field],
                        "family": r["family"], "subject": r["subject"]} for r in rows])
    if len(df) < 20: return None
    # z-score covariate so coefficients are comparable across metrics
    df["fid_z"]  = (df["fid"]  - df["fid"].mean())  / df["fid"].std(ddof=1)
    df["util_z"] = (df["util"] - df["util"].mean()) / df["util"].std(ddof=1)
    # Two random effects: subject and family. Use VC (variance components) syntax.
    try:
        md = smf.mixedlm("util_z ~ fid_z", data=df, groups=df["subject"],
                          re_formula="~1",
                          vc_formula={"fam": "0 + C(family)"})
        mdf = md.fit(method="lbfgs", reml=True, disp=False)
        beta = float(mdf.params["fid_z"])
        se   = float(mdf.bse["fid_z"])
        tval = float(mdf.tvalues["fid_z"])
        pval = float(mdf.pvalues["fid_z"])
        # partial r approximation: t / sqrt(t^2 + df_resid)
        try:
            df_resid = float(mdf.df_resid)
        except Exception:
            df_resid = len(df) - 3
        partial_r = float(tval / math.sqrt(tval * tval + df_resid)) if df_resid > 0 else float("nan")
        return {"beta_z": beta, "se_z": se, "t": tval, "p": pval,
                "partial_r": partial_r, "n_obs": int(len(df))}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------- main ----------------------------------------------------------
def main():
    rows_out = []
    for channel in ("T2", "FLAIR"):
        for endpoint in ("dice", "nsd"):
            rows = build_dataset(channel=channel, endpoint=endpoint)
            n_obs = len(rows)
            n_subjects = len({r["subject"] for r in rows})
            n_experiments = len({(r["family"], r["method"], r["target"]) for r in rows})
            print(f"\n=== Channel={channel}  Endpoint={endpoint}  "
                  f"obs={n_obs}  subjects={n_subjects}  experiments={n_experiments} ===")
            for fid, fid_label in [("lpips", "LPIPS"), ("ssim", "SSIM"),
                                    ("psnr",  "PSNR"),  ("mae",  "MAE")]:
                # within-subject
                ws = within_subject_r(rows, fid)
                cb = cluster_bootstrap_by_family(rows, fid)
                lm = mixed_effects_model(rows, fid)
                base_row = {"channel": channel, "endpoint": endpoint,
                            "metric": fid_label, "n_obs": n_obs,
                            "n_subjects": n_subjects, "n_experiments": n_experiments}
                if cb:
                    rows_out.append({**base_row, "estimator": "naive_pooled_r",
                                     "value": f"{cb['naive_r']:+.3f}",
                                     "ci_low": "", "ci_high": "",
                                     "extra": f"p={cb['naive_p']:.4g}"})
                    rows_out.append({**base_row, "estimator": "cluster_bootstrap_by_family",
                                     "value": f"{cb['boot_mean_r']:+.3f}",
                                     "ci_low": f"{cb['boot_ci_lo']:+.3f}",
                                     "ci_high": f"{cb['boot_ci_hi']:+.3f}",
                                     "extra": f"frac_same_sign={cb['frac_same_sign']:.2f}"})
                if ws:
                    rows_out.append({**base_row, "estimator": "within_subject_r_median",
                                     "value": f"{ws['median']:+.3f}",
                                     "ci_low": f"{ws['q25']:+.3f}",
                                     "ci_high": f"{ws['q75']:+.3f}",
                                     "extra": f"n_subjects_w_r={ws['n_subjects']}; "
                                              f"frac_same_sign={ws['frac_same_sign']:.2f}"})
                if lm and "error" not in lm:
                    rows_out.append({**base_row, "estimator": "mixed_effects_partial_r",
                                     "value": f"{lm['partial_r']:+.3f}",
                                     "ci_low": "", "ci_high": "",
                                     "extra": f"beta_z={lm['beta_z']:+.3f}; "
                                              f"t={lm['t']:+.2f}; "
                                              f"p={lm['p']:.4g}; n_obs={lm['n_obs']}"})
                # console summary
                line = f"  {fid_label:6s}"
                if cb:
                    line += f"  naive r={cb['naive_r']:+.3f}  bootCI=[{cb['boot_ci_lo']:+.2f},{cb['boot_ci_hi']:+.2f}]"
                if ws:
                    line += f"  within-subj med r={ws['median']:+.3f} (IQR {ws['q25']:+.2f}..{ws['q75']:+.2f}; same sign {ws['frac_same_sign']*100:.0f}%)"
                if lm and "error" not in lm:
                    line += f"  mixed r={lm['partial_r']:+.3f} (p={lm['p']:.3g})"
                print(line)

    # write CSV
    keys = ["channel", "endpoint", "metric", "estimator", "value", "ci_low", "ci_high",
            "n_obs", "n_subjects", "n_experiments", "extra"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows_out:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"\n-> {OUT_CSV}  rows={len(rows_out)}")


if __name__ == "__main__":
    main()
