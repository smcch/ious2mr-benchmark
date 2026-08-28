r"""Patch the T2-channel columns of the 4 SynDiff dual rows in
paper_assets/all_experiments_metrics.csv and per_experiment_ci.csv.

Those rows were averaged over the full 30-study T2 cohort instead of the
20-study dual cohort (every other dual row is n=20). Correct per-subject
values live in evaluacion-final/rescore_methods_persubject.csv (channel=t2,
methods SynDiff-*-T2+FLAIR). CIs are Student-t, matching the rest of the file.

Originals are backed up as *_SUPERSEDED_syndiffdual_n30.csv (once).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from common.paths import PROJECT_ROOT, DATA_ROOT, CKPT_ROOT, EXTERNAL_ROOT  # noqa: E402
import os  # noqa: E402,F811
import csv
import os
import shutil
import statistics as st

from scipy.stats import t as tdist

ASSETS = os.path.join(str(PROJECT_ROOT), "paper_assets")
PERSUBJ = os.path.join(str(PROJECT_ROOT), "evaluacion-final", "rescore_methods_persubject.csv")

# experiment name in the CSVs -> method name in the rescore per-subject file
MAP = {
    "SynDiff-2D-T2_FLAIR": "SynDiff-2D-T2+FLAIR",
    "SynDiff-2.5D-T2_FLAIR": "SynDiff-2.5D-T2+FLAIR",
    "SynDiff-3Drefine-T2_FLAIR": "SynDiff-3D+3D-refine-T2+FLAIR",
    "SynDiff-full3D-T2_FLAIR": "SynDiff-3D-T2+FLAIR",
}
METRICS = ["ssim", "psnr", "mae", "lpips"]
FMT = {"ssim": "{:.3f}", "psnr": "{:.2f}", "mae": "{:.3f}", "lpips": "{:.3f}"}


def stats_for(method):
    rows = [r for r in csv.DictReader(open(PERSUBJ, newline="", encoding="utf-8"))
            if r["channel"] == "t2" and r["method"] == method]
    out = {}
    for m in METRICS:
        v = [float(r[m]) for r in rows]
        n = len(v)
        assert n == 20, f"{method}/{m}: n={n}"
        mean, sd = st.mean(v), st.stdev(v)
        h = tdist.ppf(0.975, n - 1) * sd / n ** 0.5
        out[m] = dict(mean=mean, sd=sd, lo=mean - h, hi=mean + h, n=n)
    return out


def patch(path, exp_col, has_str):
    bak = path.replace(".csv", "_SUPERSEDED_syndiffdual_n30.csv")
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print("backup ->", bak)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    n_patched = 0
    for r in rows:
        exp = r[exp_col]
        if exp not in MAP:
            continue
        s = stats_for(MAP[exp])
        for m in METRICS:
            d = s[m]
            r[f"{m}_t2_mean"] = repr(d["mean"])
            r[f"{m}_t2_sd"] = repr(d["sd"])
            r[f"{m}_t2_lo"] = repr(d["lo"])
            r[f"{m}_t2_hi"] = repr(d["hi"])
            r[f"{m}_t2_n"] = "20"
            if has_str:
                f_ = FMT[m]
                r[f"{m}_t2_str"] = (f_.format(d["mean"]) + " ("
                                    + f_.format(d["lo"]) + "\u2013"
                                    + f_.format(d["hi"]) + ")")
        n_patched += 1
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"patched {n_patched} rows in {os.path.basename(path)}")


patch(os.path.join(ASSETS, "all_experiments_metrics.csv"), "Experiment", has_str=True)
patch(os.path.join(ASSETS, "per_experiment_ci.csv"), "experiment", has_str=False)

# quick verification
for r in csv.DictReader(open(os.path.join(ASSETS, "all_experiments_metrics.csv"),
                             newline="", encoding="utf-8")):
    if r["Experiment"] in MAP:
        print(r["Experiment"], "ssim:", r["ssim_t2_str"], "n:", r["ssim_t2_n"],
              "lpips:", r["lpips_t2_str"])
