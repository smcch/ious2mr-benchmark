"""Build two global CSVs covering every experiment:

- global_per_subject.csv:  one row per (experiment, subject) with all
                           metrics plus a `group` column identifying the
                           source (comparativa-3 / resvit-baseline /
                           resvit-ablation).
- global_summary.csv:      one row per experiment with mean, sd, ci95 for
                           each of the 4 metrics on each of the 2 targets.

Sources merged:
  - COMPARATIVA-3/results_lpips_per_subject.csv          (baselines pix2pix/cut/cyclegan/swinpix2pix)
  - resvit/results_eval/resvit_baselines_per_subject.csv (ResViT 2d/2.5d/2d_3d_refine/full_3d)
  - resvit/results_eval/abl_t2flair_win_v2_per_subject.csv (ResViT ablations)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from common.paths import PROJECT_ROOT, DATA_ROOT, CKPT_ROOT, EXTERNAL_ROOT  # noqa: E402
import os  # noqa: E402,F811
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, "results_eval")

CMP3_SUBJ = os.path.join(str(PROJECT_ROOT), "COMPARATIVA-3", "results_lpips_per_subject.csv")
RESVIT_SUBJ = os.path.join(RESULTS_DIR, "resvit_baselines_per_subject.csv")
ABL_SUBJ = os.path.join(RESULTS_DIR, "abl_t2flair_win_v2_per_subject.csv")

OUT_SUBJ = os.path.join(RESULTS_DIR, "global_per_subject.csv")
OUT_SUMMARY = os.path.join(RESULTS_DIR, "global_summary.csv")

METRIC_KEYS = ("ssim_t2", "psnr_t2", "mae_t2", "lpips_t2",
               "ssim_flair", "psnr_flair", "mae_flair", "lpips_flair")


def _f(v):
    if v is None:
        return float("nan")
    if isinstance(v, float):
        return v
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_subjects(path, group):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            out = {
                "group": group,
                "experiment": r["experiment"],
                "target": r.get("target", ""),
                "subject": r["subject"],
            }
            for k in METRIC_KEYS:
                out[k] = _f(r.get(k))
            rows.append(out)
    return rows


def ci95(sd, n):
    if n <= 1:
        return 0.0
    return float(1.96 * sd / math.sqrt(n))


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["experiment"]].append(r)

    summary = []
    for exp, exp_rows in groups.items():
        # group identifier is constant per experiment -> take first
        g = exp_rows[0]["group"]
        sr = {"experiment": exp, "group": g}

        t2_rows = [r for r in exp_rows if not math.isnan(r["ssim_t2"])]
        sr["n_t2"] = len(t2_rows)
        for k_src, k_dst in [("ssim_t2", "ssim_t2"), ("psnr_t2", "psnr_t2"),
                             ("mae_t2", "mae_t2"), ("lpips_t2", "lpips_t2")]:
            if t2_rows:
                vals = np.array([r[k_src] for r in t2_rows], dtype=np.float64)
                sr[f"{k_dst}_mean"] = float(vals.mean())
                sr[f"{k_dst}_sd"] = float(vals.std(ddof=0))
                sr[f"{k_dst}_ci95"] = ci95(float(vals.std(ddof=0)), len(vals))
            else:
                sr[f"{k_dst}_mean"] = ""
                sr[f"{k_dst}_sd"] = ""
                sr[f"{k_dst}_ci95"] = ""

        fl_rows = [r for r in exp_rows if not math.isnan(r["ssim_flair"])]
        sr["n_flair"] = len(fl_rows) if fl_rows else ""
        for k_src, k_dst in [("ssim_flair", "ssim_fl"), ("psnr_flair", "psnr_fl"),
                             ("mae_flair", "mae_fl"), ("lpips_flair", "lpips_fl")]:
            if fl_rows:
                vals = np.array([r[k_src] for r in fl_rows], dtype=np.float64)
                sr[f"{k_dst}_mean"] = float(vals.mean())
                sr[f"{k_dst}_sd"] = float(vals.std(ddof=0))
                sr[f"{k_dst}_ci95"] = ci95(float(vals.std(ddof=0)), len(vals))
            else:
                sr[f"{k_dst}_mean"] = ""
                sr[f"{k_dst}_sd"] = ""
                sr[f"{k_dst}_ci95"] = ""
        summary.append(sr)

    # Sort: group ("resvit-ablation" first, then "resvit-baseline", then "comparativa-3")
    # then by LPIPS_t2 ascending.
    group_order = {"resvit-ablation": 0, "resvit-baseline": 1, "comparativa-3": 2}
    def _sort_key(r):
        lpips = r.get("lpips_t2_mean")
        return (group_order.get(r["group"], 99),
                lpips if isinstance(lpips, (int, float)) else 1e9)
    summary.sort(key=_sort_key)
    return summary


SUBJ_FIELDS = [
    "group", "experiment", "target", "subject",
    "ssim_t2", "psnr_t2", "mae_t2", "lpips_t2",
    "ssim_flair", "psnr_flair", "mae_flair", "lpips_flair",
]

SUMMARY_FIELDS = [
    "group", "experiment",
    "n_t2",
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


def _fmt(v, digits=4):
    if isinstance(v, float) and not math.isnan(v):
        return f"{v:.{digits}f}"
    if v == "" or v is None:
        return ""
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def write_subjects(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUBJ_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = dict(r)
            for k in METRIC_KEYS:
                v = out.get(k)
                if isinstance(v, float) and math.isnan(v):
                    out[k] = ""
            w.writerow(out)


def write_summary(summary, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in summary:
            w.writerow(r)


def main():
    print(f"Loading sources ...")
    cmp3 = load_subjects(CMP3_SUBJ, "comparativa-3")
    base = load_subjects(RESVIT_SUBJ, "resvit-baseline")
    abl = load_subjects(ABL_SUBJ, "resvit-ablation")
    print(f"  comparativa-3  : {len(cmp3)} subject-rows, {len({r['experiment'] for r in cmp3})} experiments")
    print(f"  resvit-baseline: {len(base)} subject-rows, {len({r['experiment'] for r in base})} experiments")
    print(f"  resvit-ablation: {len(abl)} subject-rows, {len({r['experiment'] for r in abl})} experiments")

    all_rows = cmp3 + base + abl
    print(f"  TOTAL          : {len(all_rows)} subject-rows, {len({r['experiment'] for r in all_rows})} experiments")

    write_subjects(all_rows, OUT_SUBJ)
    print(f"\nGlobal per-subject CSV: {OUT_SUBJ}")

    summary = summarize(all_rows)
    write_summary(summary, OUT_SUMMARY)
    print(f"Global summary CSV:    {OUT_SUMMARY}")

    # Pretty print top of each group
    print("\n" + "=" * 135)
    print(f"  GLOBAL SUMMARY (n rows = {len(summary)}; columns: group, experiment, n_t2, LPIPS(t2) mean+/-ci95, n_fl, LPIPS(fl) mean+/-ci95)")
    print("=" * 135)
    print(f"  {'group':<18} {'experiment':<45} {'n_t2':>4} "
          f"{'LPIPS_t2 mean+/-ci95':<22} {'SSIM_t2':<16}  "
          f"{'n_fl':>4} {'LPIPS_fl mean+/-ci95':<22} {'SSIM_fl':<16}")
    print("-" * 135)
    for s in summary:
        l_t2 = _fmt(s.get("lpips_t2_mean"), 4)
        l_t2c = _fmt(s.get("lpips_t2_ci95"), 3)
        s_t2 = _fmt(s.get("ssim_t2_mean"), 4)
        s_t2c = _fmt(s.get("ssim_t2_ci95"), 3)
        l_fl = _fmt(s.get("lpips_fl_mean"), 4)
        l_flc = _fmt(s.get("lpips_fl_ci95"), 3)
        s_fl = _fmt(s.get("ssim_fl_mean"), 4)
        s_flc = _fmt(s.get("ssim_fl_ci95"), 3)
        fl_col = f"{l_fl}+/-{l_flc}" if l_fl else "-"
        fls_col = f"{s_fl}+/-{s_flc}" if s_fl else "-"
        print(f"  {s['group']:<18} {s['experiment']:<45} {str(s.get('n_t2','')):>4} "
              f"{l_t2}+/-{l_t2c:<9} {s_t2}+/-{s_t2c:<8}  "
              f"{str(s.get('n_flair','')):>4} {fl_col:<22} {fls_col:<16}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
