"""
eval_resvit_metrics.py
Evaluate all ResViT experiments that have a predictions/ directory.

Computes SSIM, PSNR, MAE, LPIPS (AlexNet) per subject and aggregates
(mean, sd, ci95) to match the schema of
COMPARATIVA-3/final_comparison_t2_flair.csv.

Metric definitions mirror COMPARATIVA-3/run_lpips_eval.py. Ground truth
and prediction are loaded from the ResViT-saved NIfTI sidecars
(pred_t2.nii.gz, tgt_t2.nii.gz, pred_fl.nii.gz, tgt_fl.nii.gz) - both are
already normalized to [-1, 1] with the same convention, so metrics are
directly comparable with COMPARATIVA-3.

Run in the nnunet conda env (has torch + lpips).
"""
import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np
import nibabel as nib
import torch
import lpips
from skimage.metrics import structural_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RESULTS_DIR = os.path.join(BASE_DIR, "results_eval")
os.makedirs(RESULTS_DIR, exist_ok=True)

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.paths import PROJECT_ROOT  # noqa: E402

# Per-subject GAN-baseline metrics, produced by src/gan/eval_lpips.py.
COMPARATIVA_PERSUBJ = os.path.join(
    str(PROJECT_ROOT), "COMPARATIVA-3", "results_lpips_per_subject.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_canonical_cohorts(per_subj_csv):
    """Return (t2_subjects, flair_subjects) - same cohorts used in COMPARATIVA-3.

    - t2_subjects: subjects evaluated for any `_t2` variant (T2 is present).
    - flair_subjects: subjects with FLAIR ground truth - the `_t2_flair`
      variants evaluate BOTH T2 and FLAIR on this subset (matching
      COMPARATIVA-3/run_lpips_eval.py behaviour).
    """
    import csv as _csv
    t2_subs, fl_subs = set(), set()
    with open(per_subj_csv) as f:
        for r in _csv.DictReader(f):
            if r["target"] == "t2":
                t2_subs.add(r["subject"])
            elif r["target"] == "t2_flair":
                v = r.get("ssim_flair", "")
                if v and v.lower() != "nan":
                    fl_subs.add(r["subject"])
    return sorted(t2_subs), sorted(fl_subs)


def ssim_3d(target, pred):
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


_LPIPS_FN = None


def _get_lpips():
    global _LPIPS_FN
    if _LPIPS_FN is None:
        _LPIPS_FN = lpips.LPIPS(net="alex").to(DEVICE)
        _LPIPS_FN.eval()
    return _LPIPS_FN


def _to_lpips_tensor(slice_01):
    s = slice_01 * 2.0 - 1.0
    t = torch.from_numpy(np.ascontiguousarray(s)).float()
    if t.ndim == 2:
        t = t.unsqueeze(0).expand(3, -1, -1)
    return t.unsqueeze(0).to(DEVICE)


def lpips_3d(target, pred):
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


def _ci95(std, n):
    if n < 2:
        return 0.0
    return float(1.96 * std / math.sqrt(n))


def discover_experiments(root):
    exps = []
    for name in sorted(os.listdir(root)):
        if not name.startswith("ResViT-"):
            continue
        exp_dir = os.path.join(root, name)
        pred_dir = os.path.join(exp_dir, "predictions")
        if not os.path.isdir(pred_dir):
            continue
        # Target is "t2_flair" if that token is in the name (handles both
        # plain "ResViT-<variant>-t2_flair" and ablations like
        # "ResViT-<variant>-t2_flair-ablX"). Otherwise default to "t2".
        target = "t2_flair" if ("-t2_flair" in name) else "t2"
        exps.append((name, pred_dir, target))
    return exps


def iter_subjects(pred_dir):
    for d in sorted(os.listdir(pred_dir)):
        sub = os.path.join(pred_dir, d)
        if not os.path.isdir(sub):
            continue
        if not d.startswith("ReMIND-"):
            continue
        yield d, sub


def load_nii(path):
    return nib.load(path).get_fdata().astype(np.float32)


def eval_experiment(exp_name, pred_dir, target, t2_cohort, flair_cohort):
    is_multitask = (target == "t2_flair")
    cohort = set(flair_cohort) if is_multitask else set(t2_cohort)
    rows = []
    missing = []
    print(f"\n[ {exp_name} ] target={target} | cohort_size={len(cohort)}",
          flush=True)
    for study_id in sorted(cohort):
        subdir = os.path.join(pred_dir, study_id)
        pred_t2_p = os.path.join(subdir, "pred_t2.nii.gz")
        tgt_t2_p = os.path.join(subdir, "tgt_t2.nii.gz")
        if not (os.path.exists(pred_t2_p) and os.path.exists(tgt_t2_p)):
            missing.append(study_id)
            continue
        pred_t2 = load_nii(pred_t2_p)
        tgt_t2 = load_nii(tgt_t2_p)
        row = {
            "experiment": exp_name,
            "target": target,
            "subject": study_id,
            "ssim_t2": ssim_3d(tgt_t2, pred_t2),
            "psnr_t2": psnr_3d(tgt_t2, pred_t2),
            "mae_t2": mae_3d(tgt_t2, pred_t2),
            "lpips_t2": lpips_3d(tgt_t2, pred_t2),
            "ssim_flair": np.nan, "psnr_flair": np.nan,
            "mae_flair": np.nan, "lpips_flair": np.nan,
        }
        if is_multitask:
            pred_fl_p = os.path.join(subdir, "pred_fl.nii.gz")
            tgt_fl_p = os.path.join(subdir, "tgt_fl.nii.gz")
            if os.path.exists(pred_fl_p) and os.path.exists(tgt_fl_p):
                pred_fl = load_nii(pred_fl_p)
                tgt_fl = load_nii(tgt_fl_p)
                row.update({
                    "ssim_flair": ssim_3d(tgt_fl, pred_fl),
                    "psnr_flair": psnr_3d(tgt_fl, pred_fl),
                    "mae_flair": mae_3d(tgt_fl, pred_fl),
                    "lpips_flair": lpips_3d(tgt_fl, pred_fl),
                })
            else:
                missing.append(f"{study_id}(flair)")
        fl_str = ""
        if is_multitask and not np.isnan(row["ssim_flair"]):
            fl_str = (f" | FL SSIM={row['ssim_flair']:.4f} "
                      f"LPIPS={row['lpips_flair']:.4f}")
        print(f"  {study_id}: T2 SSIM={row['ssim_t2']:.4f} "
              f"PSNR={row['psnr_t2']:.2f} LPIPS={row['lpips_t2']:.4f}{fl_str}",
              flush=True)
        rows.append(row)
    if missing:
        print(f"  [warn] missing predictions: {missing}", flush=True)
    return rows


def summarize(all_rows):
    agg = defaultdict(list)
    for r in all_rows:
        agg[r["experiment"]].append(r)
    out = []
    for exp_name, rows in agg.items():
        sr = {"experiment": exp_name, "n_t2": len(rows)}
        for key in ("ssim_t2", "psnr_t2", "mae_t2", "lpips_t2"):
            vals = np.array([r[key] for r in rows], dtype=np.float64)
            mean = float(np.mean(vals))
            sd = float(np.std(vals))
            sr[f"{key}_mean"] = mean
            sr[f"{key}_sd"] = sd
            sr[f"{key}_ci95"] = _ci95(sd, len(vals))
        fl_rows = [r for r in rows if not np.isnan(r["ssim_flair"])]
        if fl_rows:
            sr["n_flair"] = len(fl_rows)
            for key_src, key_dst in [("ssim_flair", "ssim_fl"),
                                      ("psnr_flair", "psnr_fl"),
                                      ("mae_flair", "mae_fl"),
                                      ("lpips_flair", "lpips_fl")]:
                vals = np.array([r[key_src] for r in fl_rows], dtype=np.float64)
                mean = float(np.mean(vals))
                sd = float(np.std(vals))
                sr[f"{key_dst}_mean"] = mean
                sr[f"{key_dst}_sd"] = sd
                sr[f"{key_dst}_ci95"] = _ci95(sd, len(vals))
        else:
            sr["n_flair"] = ""
            for key_dst in ("ssim_fl", "psnr_fl", "mae_fl", "lpips_fl"):
                sr[f"{key_dst}_mean"] = ""
                sr[f"{key_dst}_sd"] = ""
                sr[f"{key_dst}_ci95"] = ""
        out.append(sr)
    out.sort(key=lambda r: -r["ssim_t2_mean"])
    return out


def write_per_subject(all_rows, path):
    fieldnames = [
        "experiment", "target", "subject",
        "ssim_t2", "psnr_t2", "mae_t2", "lpips_t2",
        "ssim_flair", "psnr_flair", "mae_flair", "lpips_flair",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)


SUMMARY_FIELDS = [
    "experiment",
    "n_t2",
    "ssim_t2_mean", "ssim_t2_sd", "ssim_t2_ci95",
    "psnr_t2_mean", "psnr_t2_sd", "psnr_t2_ci95",
    "mae_t2_mean", "mae_t2_sd", "mae_t2_ci95",
    "lpips_t2_mean", "lpips_t2_sd", "lpips_t2_ci95",
    "n_flair",
    "ssim_fl_mean", "ssim_fl_sd", "ssim_fl_ci95",
    "psnr_fl_mean", "psnr_fl_sd", "psnr_fl_ci95",
    "mae_fl_mean", "mae_fl_sd", "mae_fl_ci95",
    "lpips_fl_mean", "lpips_fl_sd", "lpips_fl_ci95",
]


def write_summary(summary, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)


def main():
    pa = argparse.ArgumentParser("ResViT metrics evaluator")
    pa.add_argument("--search_dir", default=OUTPUT_DIR,
                    help="Directory containing ResViT-* experiment folders")
    pa.add_argument("--out_prefix", default="resvit",
                    help="Prefix for output CSVs in results_eval/")
    args = pa.parse_args()

    print(f"ResViT eval - device={DEVICE}", flush=True)
    print(f"Search dir: {args.search_dir}")
    print(f"Output prefix: {args.out_prefix}")

    t2_cohort, flair_cohort = load_canonical_cohorts(COMPARATIVA_PERSUBJ)
    print(f"Canonical cohorts (from COMPARATIVA-3):")
    print(f"  T2 cohort (_t2 variants):        n={len(t2_cohort)}")
    print(f"  T2+FLAIR cohort (_t2_flair):     n={len(flair_cohort)}")

    exps = discover_experiments(args.search_dir)
    print(f"\nFound {len(exps)} experiments with predictions/:", flush=True)
    for n, _, t in exps:
        print(f"  - {n} [{t}]", flush=True)

    if not exps:
        print("No experiments found. Exiting.")
        return 0

    print("\nLoading LPIPS (AlexNet)...", flush=True)
    _get_lpips()

    all_rows = []
    for exp_name, pred_dir, target in exps:
        rows = eval_experiment(exp_name, pred_dir, target,
                               t2_cohort, flair_cohort)
        all_rows.extend(rows)

    per_subj = os.path.join(RESULTS_DIR, f"{args.out_prefix}_per_subject.csv")
    write_per_subject(all_rows, per_subj)
    print(f"\nPer-subject -> {per_subj}", flush=True)

    summary = summarize(all_rows)
    summary_path = os.path.join(
        RESULTS_DIR, f"{args.out_prefix}_final_comparison_t2_flair.csv")
    write_summary(summary, summary_path)
    print(f"Summary -> {summary_path}", flush=True)

    print("\n" + "=" * 110)
    print("  ResViT COMPARISON - T2 (sorted by SSIM (low))")
    print("=" * 110)
    print(f"{'Experiment':<40} {'SSIM (mean+/-sd)':>18} {'PSNR':>8} "
          f"{'MAE':>8} {'LPIPS(low)':>18} {'N':>4}")
    print("-" * 110)
    for sr in summary:
        print(f"{sr['experiment']:<40} "
              f"{sr['ssim_t2_mean']:.4f}+/-{sr['ssim_t2_sd']:.3f}   "
              f"{sr['psnr_t2_mean']:>7.2f} "
              f"{sr['mae_t2_mean']:>7.4f} "
              f"{sr['lpips_t2_mean']:.4f}+/-{sr['lpips_t2_sd']:.3f}   "
              f"{sr['n_t2']:>4}")

    fl = [s for s in summary if s.get("n_flair")]
    if fl:
        fl.sort(key=lambda r: -r["ssim_fl_mean"])
        print("\n" + "=" * 110)
        print("  ResViT COMPARISON - FLAIR (sorted by SSIM (low))")
        print("=" * 110)
        print(f"{'Experiment':<40} {'SSIM (mean+/-sd)':>18} {'PSNR':>8} "
              f"{'MAE':>8} {'LPIPS(low)':>18} {'N':>4}")
        print("-" * 110)
        for sr in fl:
            print(f"{sr['experiment']:<40} "
                  f"{sr['ssim_fl_mean']:.4f}+/-{sr['ssim_fl_sd']:.3f}   "
                  f"{sr['psnr_fl_mean']:>7.2f} "
                  f"{sr['mae_fl_mean']:>7.4f} "
                  f"{sr['lpips_fl_mean']:.4f}+/-{sr['lpips_fl_sd']:.3f}   "
                  f"{sr['n_flair']:>4}")

    print("\nDONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
