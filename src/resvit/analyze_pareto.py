"""Who offers the best LPIPS/SSIM balance? For each cohort:
- compute Pareto front on (low LPIPS, high SSIM)
- compute a balanced z-score rank: -z(LPIPS) + z(SSIM)
"""
import csv
import math
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(BASE, "results_eval", "global_summary.csv")


def f(v):
    try:
        return float(v) if v not in ("", None) else None
    except Exception:
        return None


def pareto(rows, lpips_key, ssim_key):
    """Return subset on the Pareto front: min(lpips), max(ssim)."""
    front = []
    for r in rows:
        dominated = False
        for r2 in rows:
            if r2 is r:
                continue
            if (r2[lpips_key] <= r[lpips_key]
                    and r2[ssim_key] >= r[ssim_key]
                    and (r2[lpips_key] < r[lpips_key]
                         or r2[ssim_key] > r[ssim_key])):
                dominated = True
                break
        if not dominated:
            front.append(r)
    return front


def zscore(vals):
    n = len(vals)
    mu = sum(vals) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / n)
    if sd < 1e-9:
        return [0.0] * n
    return [(v - mu) / sd for v in vals]


def rank_cohort(rows, lpips_key, ssim_key, label):
    rows = [dict(r) for r in rows if r.get(lpips_key) is not None and r.get(ssim_key) is not None]
    if not rows:
        return
    for r in rows:
        r[lpips_key] = float(r[lpips_key])
        r[ssim_key] = float(r[ssim_key])

    zl = zscore([r[lpips_key] for r in rows])
    zs = zscore([r[ssim_key] for r in rows])
    for r, a, b in zip(rows, zl, zs):
        r["_score"] = -a + b   # lower LPIPS is better, higher SSIM is better

    front_set = {id(r) for r in pareto(rows, lpips_key, ssim_key)}

    rows.sort(key=lambda r: -r["_score"])

    print("=" * 130)
    print(f"  {label} -- balanced rank (score = -z(LPIPS) + z(SSIM); * = Pareto front)")
    print("=" * 130)
    print(f"  {'rank':>4} {'★':<2} {'group':<16} {'experiment':<45} "
          f"{'LPIPS':<10} {'SSIM':<10} {'score':>7}")
    print("-" * 130)
    for i, r in enumerate(rows, 1):
        star = "*" if id(r) in front_set else " "
        print(f"  {i:>4} {star:<2} {r['group']:<16} {r['experiment']:<45} "
              f"{r[lpips_key]:<10.4f} {r[ssim_key]:<10.4f} {r['_score']:>7.2f}")
    print()


def main():
    all_rows = []
    with open(SUMMARY) as fh:
        for r in csv.DictReader(fh):
            row = dict(r)
            for k in ("lpips_t2_mean", "ssim_t2_mean", "lpips_fl_mean", "ssim_fl_mean"):
                row[k] = f(row.get(k))
            row["n_t2"] = row.get("n_t2", "")
            row["n_flair"] = row.get("n_flair", "")
            all_rows.append(row)

    # Cohort 1: T2 single-target (n_t2=30)
    t2_only = [r for r in all_rows if r["n_t2"] in ("30", "30.0")]
    rank_cohort(t2_only, "lpips_t2_mean", "ssim_t2_mean",
                "T2 single-target (n=30)")

    # Cohort 2: T2 in multitask (n_t2=20 and n_flair=20)
    t2_mt = [r for r in all_rows if r["n_t2"] in ("20", "20.0") and r["n_flair"] in ("20", "20.0")]
    rank_cohort(t2_mt, "lpips_t2_mean", "ssim_t2_mean",
                "T2 in multitask t2_flair (n=20)")

    # Cohort 3: FLAIR (n_flair=20)
    fl = [r for r in all_rows if r["n_flair"] in ("20", "20.0")]
    rank_cohort(fl, "lpips_fl_mean", "ssim_fl_mean",
                "FLAIR (n=20)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
