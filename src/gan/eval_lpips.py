"""
run_lpips_eval.py — Compute SSIM/PSNR/MAE/LPIPS for all 32 experiments.
Uses PyTorch + lpips package. Run in nnunet conda env.

Reads predictions (.nii.gz) + ground truth NIfTIs.

Modes:
  (default)       → reads predictions/ for every experiment.
                    Outputs: results_lpips_per_subject.csv,
                             final_comparison_t2_flair.csv
  --single-axis   → reads predictions_one/ for non-3D variants and
                    predictions/ for 3D variants.
                    Outputs: results_lpips_per_subject_one.csv,
                             final_comparison_t2_flair_one.csv

Summary CSV includes mean, SD, and 95% CI (t-distribution) for every metric.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from common.paths import PROJECT_ROOT, DATA_ROOT, CKPT_ROOT, EXTERNAL_ROOT  # noqa: E402
import os  # noqa: E402,F811
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import nibabel as nib
import torch
import lpips
from scipy import stats as _stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(str(PROJECT_ROOT), "data_cropped_192")
SPLIT_FILE = os.path.join(BASE_DIR, "subject_split.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Data loading (standalone, no TF dependency)
# =============================================================================
def _norm_us(vol, plow=2, phigh=98):
    fg = vol > vol.max() * 0.01
    if fg.sum() < 100:
        return np.full_like(vol, -1.0)
    fv = vol[fg]
    lo, hi = np.percentile(fv, plow), np.percentile(fv, phigh)
    out = np.full_like(vol, -1.0)
    if hi - lo > 1e-8:
        c = np.clip(vol, lo, hi)
        s = (c - lo) / (hi - lo) * 2 - 1
        out[fg] = s[fg]
    return out.astype(np.float32)


def _norm_mri(vol, clip_std=3.0):
    fg = vol > vol.max() * 0.01
    if fg.sum() < 100:
        return np.full_like(vol, -1.0)
    fv = vol[fg]
    m, s = np.mean(fv), np.std(fv)
    out = np.full_like(vol, -1.0)
    if s > 1e-8:
        z = np.clip((vol - m) / s, -clip_std, clip_std) / clip_std
        out[fg] = z[fg]
    return out.astype(np.float32)


def load_all_data(base_dir):
    us_dir = os.path.join(base_dir, "US")
    t2_dir = os.path.join(base_dir, "MR-T2")
    fl_dir = os.path.join(base_dir, "MR-FLAIR")
    data = {}
    for f in sorted(os.listdir(us_dir)):
        if not f.endswith(".nii.gz"):
            continue
        sid = f.replace("-us.nii.gz", "")
        us = nib.load(os.path.join(us_dir, f)).get_fdata().astype(np.float32)
        t2f = f.replace("-us.nii.gz", "-mri.nii.gz")
        t2_path = os.path.join(t2_dir, t2f)
        if not os.path.exists(t2_path):
            continue
        t2 = nib.load(t2_path).get_fdata().astype(np.float32)
        us_n = _norm_us(us)
        t2_n = _norm_mri(t2)
        entry = {"us": us_n, "t2": t2_n}
        flair_f = f.replace("-us.nii.gz", "-mri.nii.gz")
        flair_path = os.path.join(fl_dir, flair_f)
        if os.path.exists(flair_path):
            fl = nib.load(flair_path).get_fdata().astype(np.float32)
            entry["flair"] = _norm_mri(fl)
        else:
            entry["flair"] = None
        data[sid] = entry
    return data


# =============================================================================
# Metrics (numpy, no TF)
# =============================================================================
def ssim_3d(target, pred):
    from skimage.metrics import structural_similarity
    t01 = (target + 1.0) / 2.0
    p01 = (pred + 1.0) / 2.0
    vals = []
    for z in range(t01.shape[2]):
        t_sl, p_sl = t01[:, :, z], p01[:, :, z]
        if np.mean(t_sl > 0.025) < 0.01:
            continue
        vals.append(structural_similarity(t_sl, p_sl, data_range=1.0))
    return float(np.mean(vals)) if vals else 0.0


def psnr_3d(target, pred):
    t01 = (target + 1.0) / 2.0
    p01 = (pred + 1.0) / 2.0
    fg = t01 > 0.025
    if fg.sum() < 100:
        return 0.0
    mse = float(np.mean((t01[fg] - p01[fg]) ** 2))
    if mse < 1e-10:
        return 50.0
    return float(10.0 * np.log10(1.0 / mse))


def mae_3d(target, pred):
    t01 = (target + 1.0) / 2.0
    p01 = (pred + 1.0) / 2.0
    fg = t01 > 0.025
    if fg.sum() < 100:
        return 1.0
    return float(np.mean(np.abs(t01[fg] - p01[fg])))


# =============================================================================
# LPIPS (original, AlexNet)
# =============================================================================
_LPIPS_FN = None


def _get_lpips():
    global _LPIPS_FN
    if _LPIPS_FN is None:
        _LPIPS_FN = lpips.LPIPS(net="alex").to(DEVICE)
        _LPIPS_FN.eval()
    return _LPIPS_FN


def _to_lpips_tensor(slice_01):
    """Convert a [0,1] HxW grayscale slice to LPIPS-ready (1,3,H,W) tensor in [-1,1]."""
    s = slice_01 * 2.0 - 1.0  # back to [-1,1] for LPIPS
    t = torch.from_numpy(s).float()
    if t.ndim == 2:
        t = t.unsqueeze(0).expand(3, -1, -1)  # (3, H, W)
    return t.unsqueeze(0).to(DEVICE)  # (1, 3, H, W)


def lpips_3d(target, pred):
    """LPIPS on 3D volumes, slice-by-slice along depth, foreground-only."""
    fn = _get_lpips()
    t01 = (target + 1.0) / 2.0
    p01 = (pred + 1.0) / 2.0
    vals = []
    for z in range(t01.shape[2]):
        t_sl, p_sl = t01[:, :, z], p01[:, :, z]
        if np.mean(t_sl > 0.025) < 0.01:
            continue
        with torch.no_grad():
            d = fn(_to_lpips_tensor(t_sl), _to_lpips_tensor(p_sl))
        vals.append(float(d.item()))
    return float(np.mean(vals)) if vals else 0.0


# =============================================================================
# Main
# =============================================================================
def _log(msg):
    print(msg, flush=True)


def _mean_sd_ci95(values):
    """Return dict with mean, sd (ddof=1), and 95% CI half-width (t-dist)."""
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)],
                     dtype=np.float64)
    n = arr.size
    if n == 0:
        return {"mean": float("nan"), "sd": float("nan"),
                "ci95_lo": float("nan"), "ci95_hi": float("nan"), "n": 0}
    if n == 1:
        return {"mean": float(arr[0]), "sd": 0.0,
                "ci95_lo": float(arr[0]), "ci95_hi": float(arr[0]), "n": 1}
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    se = sd / np.sqrt(n)
    tcrit = float(_stats.t.ppf(0.975, n - 1))
    return {"mean": mean, "sd": sd,
            "ci95_lo": mean - tcrit * se,
            "ci95_hi": mean + tcrit * se,
            "n": n}


def main():
    single_axis = "--single-axis" in sys.argv[1:]
    if single_axis:
        _log("Mode: --single-axis (predictions_one/ for non-3D, "
             "predictions/ for 3D)")
    else:
        _log("Mode: default (predictions/ for all experiments)")
    _log("Loading data...")
    all_data = load_all_data(DATA_DIR)
    _log(f"  {len(all_data)} studies")

    with open(SPLIT_FILE) as f:
        split = json.load(f)
    test_studies = set(split["test"])
    test_data = {k: v for k, v in all_data.items() if k in test_studies}
    _log(f"  {len(test_data)} test studies")

    _log(f"Loading LPIPS (AlexNet) on {DEVICE}...")
    _ = _get_lpips()
    _log("  Ready")

    experiments = []
    for arch in ["pix2pix", "swinpix2pix", "cyclegan", "cut"]:
        for var in ["25d", "2d", "2d_3dpost", "3d"]:
            for tgt in ["t2", "t2_flair"]:
                experiments.append((f"{arch}_{var}_{tgt}", arch, var, tgt))

    all_rows = []
    for exp_name, arch, variant, target in experiments:
        exp_dir = os.path.join(BASE_DIR, exp_name)
        is_multitask = (target == "t2_flair")

        if is_multitask:
            valid_test = {k: v for k, v in test_data.items()
                          if v.get("flair") is not None}
        else:
            valid_test = test_data

        if single_axis and variant != "3d":
            pred_dir = os.path.join(exp_dir, "predictions_one")
        else:
            pred_dir = os.path.join(exp_dir, "predictions")
        if not os.path.isdir(pred_dir):
            _log(f"\n  {exp_name}: [SKIP] no dir {pred_dir}")
            continue
        _log(f"\n  {exp_name} ← {os.path.basename(pred_dir)} "
             f"({len(valid_test)} subjects)")
        n_done = 0

        for study_id in sorted(valid_test.keys()):
            gt_t2 = valid_test[study_id]["t2"]
            gt_flair = valid_test[study_id].get("flair")

            # Load predictions from NIfTI (available for all variants)
            pred_t2_path = os.path.join(pred_dir, f"{study_id}_pred_t2.nii.gz")
            if not os.path.exists(pred_t2_path):
                continue
            pred_t2 = nib.load(pred_t2_path).get_fdata().astype(np.float32)

            row = {
                "experiment": exp_name, "arch": arch, "variant": variant,
                "target": target, "subject": study_id,
                "ssim_t2": ssim_3d(gt_t2, pred_t2),
                "psnr_t2": psnr_3d(gt_t2, pred_t2),
                "mae_t2": mae_3d(gt_t2, pred_t2),
                "lpips_t2": lpips_3d(gt_t2, pred_t2),
            }

            if is_multitask:
                pred_fl_path = os.path.join(pred_dir, f"{study_id}_pred_flair.nii.gz")
                if os.path.exists(pred_fl_path) and gt_flair is not None:
                    pred_fl = nib.load(pred_fl_path).get_fdata().astype(np.float32)
                    row.update({
                        "ssim_flair": ssim_3d(gt_flair, pred_fl),
                        "psnr_flair": psnr_3d(gt_flair, pred_fl),
                        "mae_flair": mae_3d(gt_flair, pred_fl),
                        "lpips_flair": lpips_3d(gt_flair, pred_fl),
                    })
                else:
                    row.update({k: np.nan for k in
                                ["ssim_flair", "psnr_flair", "mae_flair", "lpips_flair"]})
            else:
                row.update({k: np.nan for k in
                            ["ssim_flair", "psnr_flair", "mae_flair", "lpips_flair"]})

            n_done += 1
            fl_str = ""
            if is_multitask and not np.isnan(row.get("ssim_flair", np.nan)):
                fl_str = (f" | FL SSIM={row['ssim_flair']:.4f} "
                          f"LPIPS={row['lpips_flair']:.4f}")
            _log(f"    {study_id}: T2 SSIM={row['ssim_t2']:.4f} "
                 f"LPIPS={row['lpips_t2']:.4f}{fl_str}")
            all_rows.append(row)

        _log(f"    → {n_done} subjects evaluated")

    suffix = "_one" if single_axis else ""

    # Save per-subject
    fieldnames = [
        "experiment", "arch", "variant", "target", "subject",
        "ssim_t2", "psnr_t2", "mae_t2", "lpips_t2",
        "ssim_flair", "psnr_flair", "mae_flair", "lpips_flair",
    ]
    per_subj = os.path.join(
        BASE_DIR, f"results_lpips_per_subject{suffix}.csv")
    with open(per_subj, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    _log(f"\nPer-subject → {per_subj}")

    # Aggregate
    agg = defaultdict(list)
    for r in all_rows:
        agg[r["experiment"]].append(r)

    t2_metrics = ["ssim_t2", "psnr_t2", "mae_t2", "lpips_t2"]
    fl_metrics = ["ssim_flair", "psnr_flair", "mae_flair", "lpips_flair"]

    def _stat_cols(prefix, stats):
        return {
            f"{prefix}_mean": stats["mean"],
            f"{prefix}_sd": stats["sd"],
            f"{prefix}_ci95_lo": stats["ci95_lo"],
            f"{prefix}_ci95_hi": stats["ci95_hi"],
        }

    summary = []
    for exp_name, rows in agg.items():
        r0 = rows[0]
        sr = {"experiment": exp_name, "arch": r0["arch"],
              "variant": r0["variant"], "target": r0["target"],
              "n_t2": len(rows)}
        for m in t2_metrics:
            sr.update(_stat_cols(m, _mean_sd_ci95([r[m] for r in rows])))

        fl = [r for r in rows if not np.isnan(r.get("ssim_flair", np.nan))]
        sr["n_flair"] = len(fl)
        for m in fl_metrics:
            sr.update(_stat_cols(m, _mean_sd_ci95(
                [r[m] for r in fl] if fl else [])))
        summary.append(sr)

    # Sort by T2 SSIM descending
    summary.sort(key=lambda r: -r["ssim_t2_mean"])

    summary_path = os.path.join(
        BASE_DIR, f"final_comparison_t2_flair{suffix}.csv")
    sf = ["experiment", "arch", "variant", "target", "n_t2"]
    for m in t2_metrics:
        sf += [f"{m}_mean", f"{m}_sd", f"{m}_ci95_lo", f"{m}_ci95_hi"]
    sf += ["n_flair"]
    for m in fl_metrics:
        sf += [f"{m}_mean", f"{m}_sd", f"{m}_ci95_lo", f"{m}_ci95_hi"]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sf, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)
    _log(f"Summary → {summary_path}")

    def _fmt(mean, sd, lo, hi):
        if np.isnan(mean):
            return "     n/a     "
        return f"{mean:.4f}±{sd:.3f} [{lo:.3f},{hi:.3f}]"

    _log("\n" + "=" * 150)
    _log("  FINAL COMPARISON — T2 (sorted by SSIM ↓)   mean±SD [95% CI]")
    _log("=" * 150)
    _log(f"{'Experiment':<40} {'SSIM':>26} {'PSNR':>26} "
         f"{'MAE':>26} {'LPIPS↓':>26} {'N':>4}")
    _log("-" * 150)
    for sr in summary:
        _log(f"{sr['experiment']:<40} "
             f"{_fmt(sr['ssim_t2_mean'], sr['ssim_t2_sd'], sr['ssim_t2_ci95_lo'], sr['ssim_t2_ci95_hi']):>26} "
             f"{_fmt(sr['psnr_t2_mean'], sr['psnr_t2_sd'], sr['psnr_t2_ci95_lo'], sr['psnr_t2_ci95_hi']):>26} "
             f"{_fmt(sr['mae_t2_mean'], sr['mae_t2_sd'], sr['mae_t2_ci95_lo'], sr['mae_t2_ci95_hi']):>26} "
             f"{_fmt(sr['lpips_t2_mean'], sr['lpips_t2_sd'], sr['lpips_t2_ci95_lo'], sr['lpips_t2_ci95_hi']):>26} "
             f"{sr['n_t2']:>4}")

    flair = [s for s in summary if s.get("n_flair", 0) > 0
             and not np.isnan(s.get("ssim_flair_mean", np.nan))]
    if flair:
        flair.sort(key=lambda r: -r["ssim_flair_mean"])
        _log("\n" + "=" * 150)
        _log("  FINAL COMPARISON — FLAIR (sorted by SSIM ↓)   mean±SD [95% CI]")
        _log("=" * 150)
        _log(f"{'Experiment':<40} {'SSIM':>26} {'PSNR':>26} "
             f"{'MAE':>26} {'LPIPS↓':>26} {'N':>4}")
        _log("-" * 150)
        for sr in flair:
            _log(f"{sr['experiment']:<40} "
                 f"{_fmt(sr['ssim_flair_mean'], sr['ssim_flair_sd'], sr['ssim_flair_ci95_lo'], sr['ssim_flair_ci95_hi']):>26} "
                 f"{_fmt(sr['psnr_flair_mean'], sr['psnr_flair_sd'], sr['psnr_flair_ci95_lo'], sr['psnr_flair_ci95_hi']):>26} "
                 f"{_fmt(sr['mae_flair_mean'], sr['mae_flair_sd'], sr['mae_flair_ci95_lo'], sr['mae_flair_ci95_hi']):>26} "
                 f"{_fmt(sr['lpips_flair_mean'], sr['lpips_flair_sd'], sr['lpips_flair_ci95_lo'], sr['lpips_flair_ci95_hi']):>26} "
                 f"{sr.get('n_flair', 0):>4}")

    _log("\nDONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
