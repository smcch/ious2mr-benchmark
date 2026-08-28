"""
Build the final paper-ready unified comparison CSVs:

  results/unified_summary_final.csv     — one row per method, mean/sd/ci95 for
                                          all metrics (T2 + FLAIR) using exactly
                                          the same schema as
                                          resvit/results_eval/global_summary_with_one.csv
  results/unified_per_subject_final.csv — long format, one row per
                                          (method, subject), with full metrics
                                          for statistical tests

Methods included:
  - Comparativa-3: pix2pix / cut / swinpix2pix / cyclegan × {2d, 2.5d, 3d, 2d_3dpost}
                   × {t2, t2_flair} baselines. Ablations dropped.
  - ResViT: ResViT-{2d, 2.5d, 2d_3d_refine, full_3d} × {t2, t2_flair} baselines.
            Ablations dropped (anything with `abl` or `comparativa-3-one`).
  - SynDiff (paired): one row per training × (best epoch by max SSIM_T2). 8 rows:
      2D / 2.5D / 3D-refine / full_3d, both single (T2) and dual (T2+FLAIR).

We deliberately do NOT include the original bidirectional SynDiff or any of
the multi-epoch checkpoints — only the canonical "winning" config per
training, matching how the user wants to write up the comparative.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from common.paths import PROJECT_ROOT, DATA_ROOT, CKPT_ROOT, EXTERNAL_ROOT  # noqa: E402
import os  # noqa: E402,F811
import os, sys, csv, math, glob
import numpy as np

ROOT = os.path.join(str(PROJECT_ROOT), "synthdiff")
RESVIT_DIR = os.path.join(str(PROJECT_ROOT), "resvit", "results_eval")

# ----------------------------------------------------------------------------
# 1) SynDiff bests — (exp prefix, group label, eval per_subject csv path,
#                     experiment_name_in_summary_csv, group_for_unified)
#
# For each variant we point to the per_subject.csv of the BEST epoch (the one
# whose volumes were saved + LPIPS computed). LPIPS rows have already been
# merged in-place into per_subject.csv by eval_lpips*.py.
# ----------------------------------------------------------------------------
SYNDIFF_BESTS = [
    # (label, group, dual?, per_subj_csv path)
    ("SynDiff-2D-T2",            "syndiff-paired", False,
     f"{ROOT}/results/syndiff_us_t2_paired_resvit_protocol_ep40/per_subject.csv"),
    ("SynDiff-2.5D-T2",          "syndiff-paired", False,
     f"{ROOT}/results/syndiff_us_t2_paired_25d_resvit_protocol_ep40/per_subject.csv"),
    ("SynDiff-3Drefine-T2",      "syndiff-paired", False,
     f"{ROOT}/results/syndiff_us_t2_paired_3drefine_resvit_protocol_ep20/per_subject.csv"),
    ("SynDiff-full3D-T2",        "syndiff-paired", False,
     f"{ROOT}/results/syndiff_us_t2_paired_3d_resvit_protocol_ep140/per_subject.csv"),
    ("SynDiff-2D-T2_FLAIR",      "syndiff-paired-dual", True,
     f"{ROOT}/results/syndiff_us_t2flair_paired_resvit_protocol_ep40/per_subject.csv"),
    ("SynDiff-2.5D-T2_FLAIR",    "syndiff-paired-dual", True,
     f"{ROOT}/results/syndiff_us_t2flair_paired_25d_resvit_protocol_ep60/per_subject.csv"),
    ("SynDiff-3Drefine-T2_FLAIR","syndiff-paired-dual", True,
     f"{ROOT}/results/syndiff_us_t2flair_paired_3drefine_resvit_protocol_ep20/per_subject.csv"),
    ("SynDiff-full3D-T2_FLAIR",  "syndiff-paired-dual", True,
     f"{ROOT}/results/syndiff_us_t2flair_paired_3d_resvit_protocol_ep100/per_subject.csv"),
]


def _maybe_float(s):
    try:
        if s == "" or s is None: return float("nan")
        return float(s)
    except Exception:
        return float("nan")


def _fnum(arr):
    """ Returns (n, mean, sd, ci95) ignoring NaN. """
    a = np.array([x for x in arr if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=np.float64)
    n = len(a)
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan")
    if n == 1:
        return 1, float(a[0]), 0.0, 0.0
    mean = float(a.mean())
    sd = float(a.std(ddof=1))
    ci = float(1.96 * sd / math.sqrt(n))
    return n, mean, sd, ci


def load_resvit_global_summary():
    """Read ResViT's canonical global_summary_with_one.csv. Drop ablation rows.
    Returns list of dicts using global_summary schema."""
    src = os.path.join(RESVIT_DIR, "global_summary_with_one.csv")
    rows = []
    with open(src) as f:
        for r in csv.DictReader(f):
            grp = r.get("group", "")
            exp = r.get("experiment", "")
            # Drop ablations explicitly
            if "abl" in exp.lower(): continue
            if grp == "comparativa-3-one":
                # `-one` rows are the single-axis-only ablations of Comparativa-3
                # baselines; drop them so the canonical baseline is the
                # `comparativa-3` group only.
                continue
            rows.append(r)
    return rows


def load_resvit_per_subject():
    """Read ResViT's canonical per-subject CSV. Drop ablation rows."""
    src = os.path.join(RESVIT_DIR, "global_per_subject_with_one.csv")
    rows = []
    with open(src) as f:
        for r in csv.DictReader(f):
            grp = r.get("group", "")
            exp = r.get("experiment", "")
            if "abl" in exp.lower(): continue
            if grp == "comparativa-3-one": continue
            rows.append(r)
    return rows


# ----------------------------------------------------------------------------
# 2) Build SynDiff per-subject rows for the best ckpts and aggregate
# ----------------------------------------------------------------------------
def syndiff_per_subject_rows():
    out = []
    for label, group, dual, ps_csv in SYNDIFF_BESTS:
        if not os.path.exists(ps_csv):
            print(f"  [warn] missing {ps_csv}")
            continue
        with open(ps_csv) as f:
            for r in csv.DictReader(f):
                # Single-target rows have ssim_t2, psnr_t2, mae_t2, lpips_t2 only.
                # Dual rows additionally have ssim_flair, psnr_flair, mae_flair, lpips_flair.
                row = {
                    "group": group,
                    "experiment": label,
                    "target": "t2_flair" if dual else "t2",
                    "subject": r["subject"],
                    "ssim_t2":  _maybe_float(r.get("ssim_t2", "")),
                    "psnr_t2":  _maybe_float(r.get("psnr_t2", "")),
                    "mae_t2":   _maybe_float(r.get("mae_t2", "")),
                    "lpips_t2": _maybe_float(r.get("lpips_t2", "")),
                    "ssim_flair":  _maybe_float(r.get("ssim_flair", "") if dual else ""),
                    "psnr_flair":  _maybe_float(r.get("psnr_flair", "") if dual else ""),
                    "mae_flair":   _maybe_float(r.get("mae_flair", "") if dual else ""),
                    "lpips_flair": _maybe_float(r.get("lpips_flair", "") if dual else ""),
                }
                out.append(row)
    return out


def aggregate_syndiff_summary(per_subj_rows):
    """Group SynDiff per-subject rows by (group, experiment) -> single summary
    row using the global_summary schema."""
    by_exp = {}
    for r in per_subj_rows:
        key = (r["group"], r["experiment"])
        by_exp.setdefault(key, []).append(r)

    out = []
    for (group, exp), rows in by_exp.items():
        ssim_t2  = [r["ssim_t2"]  for r in rows]
        psnr_t2  = [r["psnr_t2"]  for r in rows]
        mae_t2   = [r["mae_t2"]   for r in rows]
        lpips_t2 = [r["lpips_t2"] for r in rows]
        ssim_fl  = [r["ssim_flair"]  for r in rows]
        psnr_fl  = [r["psnr_flair"]  for r in rows]
        mae_fl   = [r["mae_flair"]   for r in rows]
        lpips_fl = [r["lpips_flair"] for r in rows]

        n_t2,  ssim_t2_m, ssim_t2_sd, ssim_t2_ci  = _fnum(ssim_t2)
        _,     psnr_t2_m, psnr_t2_sd, psnr_t2_ci  = _fnum(psnr_t2)
        _,     mae_t2_m,  mae_t2_sd,  mae_t2_ci   = _fnum(mae_t2)
        _,     lp_t2_m,   lp_t2_sd,   lp_t2_ci    = _fnum(lpips_t2)
        n_fl,  ssim_fl_m, ssim_fl_sd, ssim_fl_ci  = _fnum(ssim_fl)
        _,     psnr_fl_m, psnr_fl_sd, psnr_fl_ci  = _fnum(psnr_fl)
        _,     mae_fl_m,  mae_fl_sd,  mae_fl_ci   = _fnum(mae_fl)
        _,     lp_fl_m,   lp_fl_sd,   lp_fl_ci    = _fnum(lpips_fl)
        out.append({
            "group": group, "experiment": exp,
            "n_t2": n_t2,
            "ssim_t2_mean": ssim_t2_m, "ssim_t2_sd": ssim_t2_sd, "ssim_t2_ci95": ssim_t2_ci,
            "psnr_t2_mean": psnr_t2_m, "psnr_t2_sd": psnr_t2_sd, "psnr_t2_ci95": psnr_t2_ci,
            "mae_t2_mean":  mae_t2_m,  "mae_t2_sd":  mae_t2_sd,  "mae_t2_ci95":  mae_t2_ci,
            "lpips_t2_mean": lp_t2_m,  "lpips_t2_sd": lp_t2_sd,  "lpips_t2_ci95": lp_t2_ci,
            "n_flair": n_fl,
            "ssim_fl_mean": ssim_fl_m, "ssim_fl_sd": ssim_fl_sd, "ssim_fl_ci95": ssim_fl_ci,
            "psnr_fl_mean": psnr_fl_m, "psnr_fl_sd": psnr_fl_sd, "psnr_fl_ci95": psnr_fl_ci,
            "mae_fl_mean":  mae_fl_m,  "mae_fl_sd":  mae_fl_sd,  "mae_fl_ci95":  mae_fl_ci,
            "lpips_fl_mean": lp_fl_m,  "lpips_fl_sd": lp_fl_sd,  "lpips_fl_ci95": lp_fl_ci,
        })
    return out


# ----------------------------------------------------------------------------
# 3) Stitch everything together
# ----------------------------------------------------------------------------
def main():
    out_dir = os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)

    # Per-subject merge (ResViT+C-3 already in canonical schema; SynDiff added)
    resvit_psubj = load_resvit_per_subject()
    syn_psubj    = syndiff_per_subject_rows()
    print(f"[per-subject] resvit/c-3={len(resvit_psubj)}  syndiff={len(syn_psubj)}")

    # Use exactly the same column schema as ResViT canonical per_subject
    columns_ps = ["group", "experiment", "target", "subject",
                  "ssim_t2", "psnr_t2", "mae_t2", "lpips_t2",
                  "ssim_flair", "psnr_flair", "mae_flair", "lpips_flair"]
    out_ps = os.path.join(out_dir, "unified_per_subject_final.csv")
    with open(out_ps, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns_ps)
        w.writeheader()
        for r in resvit_psubj:
            row = {c: r.get(c, "") for c in columns_ps}
            w.writerow(row)
        for r in syn_psubj:
            row = {c: ("" if (c not in r or (isinstance(r[c], float) and math.isnan(r[c]))) else r[c])
                   for c in columns_ps}
            w.writerow(row)
    print(f"[done] {out_ps}")

    # Global summary merge (ResViT canonical schema; SynDiff aggregated)
    resvit_summ = load_resvit_global_summary()
    syn_summ    = aggregate_syndiff_summary(syn_psubj)
    print(f"[summary] resvit/c-3={len(resvit_summ)}  syndiff={len(syn_summ)}")

    columns_summ = [
        "group", "experiment", "n_t2",
        "ssim_t2_mean", "ssim_t2_sd", "ssim_t2_ci95",
        "psnr_t2_mean", "psnr_t2_sd", "psnr_t2_ci95",
        "mae_t2_mean",  "mae_t2_sd",  "mae_t2_ci95",
        "lpips_t2_mean","lpips_t2_sd","lpips_t2_ci95",
        "n_flair",
        "ssim_fl_mean", "ssim_fl_sd", "ssim_fl_ci95",
        "psnr_fl_mean", "psnr_fl_sd", "psnr_fl_ci95",
        "mae_fl_mean",  "mae_fl_sd",  "mae_fl_ci95",
        "lpips_fl_mean","lpips_fl_sd","lpips_fl_ci95",
    ]
    out_summ = os.path.join(out_dir, "unified_summary_final.csv")
    with open(out_summ, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns_summ)
        w.writeheader()
        for r in resvit_summ:
            row = {c: r.get(c, "") for c in columns_summ}
            w.writerow(row)
        for r in syn_summ:
            row = {c: ("" if (c not in r or (isinstance(r[c], float) and math.isnan(r[c]))) else r[c])
                   for c in columns_summ}
            w.writerow(row)
    print(f"[done] {out_summ}")

    # Quick ranking print (T2 SSIM)
    print("\n=== Final ranking by SSIM_T2 (top 25) ===")
    all_rows = []
    for r in resvit_summ + [{**s, **{}} for s in syn_summ]:
        try:
            ssim = float(r.get("ssim_t2_mean", "nan"))
        except Exception:
            ssim = float("nan")
        if not math.isnan(ssim):
            all_rows.append((ssim, r))
    all_rows.sort(key=lambda t: -t[0])
    print(f"{'#':>3} {'group':18s} {'experiment':36s} {'n_t2':>4s} {'SSIM_T2':>8s} {'PSNR':>6s} {'MAE':>6s} {'LPIPS':>6s}")
    for i, (s, r) in enumerate(all_rows[:30], 1):
        psnr_v = float(r.get("psnr_t2_mean", "nan") or "nan")
        mae_v  = float(r.get("mae_t2_mean",  "nan") or "nan")
        lp_v   = float(r.get("lpips_t2_mean","nan") or "nan")
        print(f"{i:3d} {r['group']:18s} {r['experiment']:36s} "
              f"{r.get('n_t2',''):>4s if isinstance(r.get('n_t2'),str) else 4} "
              f"{s:.4f}  {psnr_v:6.2f}  {mae_v:.4f}  {lp_v:.4f}"
              .replace("nan", "  - "))


if __name__ == "__main__":
    main()
