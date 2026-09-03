#!/usr/bin/env python3
"""Seg-US (direct ioUS) versus the synthesis pipeline on the primary lesion endpoint.

Paired, per study, on the 29 evaluable test studies:
  * Seg-US vs real T2w, vs the null control, and vs every one of the 48 configurations
    (Wilcoxon signed-rank; Holm-corrected within the 48-test family);
  * patient-level percentile bootstrap (20 000 resamples, patients kept whole) on the
    Seg-US - best-synthesis difference and on the Seg-US retention ratio;
  * phase breakdown and per-class comparison.
"""
from __future__ import annotations

import csv
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

SRC = Path(os.environ.get("IOUS2MR_SOURCE_TREE", r"E:\SINTESIS"))
PER_STUDY = SRC / "downstream_seg" / "results_paper_protocol" / "seg_metrics_T2_per_study.csv"
REPO = Path(__file__).resolve().parents[2]
SEG_US = REPO / "results" / "downstream" / "seg_us_reference.csv"
FLOOR = REPO / "results" / "downstream" / "floor_baseline.csv"
OFF_GRID = {"SynDiff-cascade-T2", "SynDiff-joint-T2"}
RESAMPLES = 20_000


def load(cls="lesion"):
    per = defaultdict(dict)
    for r in csv.DictReader(open(PER_STUDY, newline="", encoding="utf-8")):
        if r["class"] == cls and r["dice"] not in ("", "nan"):
            per[r["set"]][r["study"]] = float(r["dice"])
    for r in csv.DictReader(open(SEG_US, newline="", encoding="utf-8")):
        if r["cls"] == cls and r["gt_present"] == "1" and r["dice"] not in ("", "nan"):
            per["SEG_US"][r["study"]] = float(r["dice"])
    for r in csv.DictReader(open(FLOOR, newline="", encoding="utf-8")):
        if r["cls"] == cls and r["gt_present"] == "1" and r["dice"] not in ("", "nan"):
            per[r["set"]][r["study"]] = float(r["dice"])
    return per


def holm(pairs):
    """pairs: list of (name, p) -> dict name -> adjusted p"""
    out, m = {}, len(pairs)
    run = 0.0
    for i, (name, p) in enumerate(sorted(pairs, key=lambda x: x[1])):
        run = max(run, min(1.0, (m - i) * p))
        out[name] = run
    return out


def boot_diff(a, b, studies, rng):
    """patient-level bootstrap of mean(a) - mean(b) over the given studies"""
    pat = defaultdict(list)
    for s in studies:
        pat["-".join(s.split("-")[:2])].append(s)
    keys = list(pat)
    d = []
    for _ in range(RESAMPLES):
        pick = rng.choice(len(keys), len(keys), replace=True)
        ss = [s for i in pick for s in pat[keys[i]]]
        d.append(st.mean(a[s] for s in ss) - st.mean(b[s] for s in ss))
    return np.percentile(d, [2.5, 97.5])


def boot_ratio(a, ref, studies, rng):
    pat = defaultdict(list)
    for s in studies:
        pat["-".join(s.split("-")[:2])].append(s)
    keys = list(pat)
    d = []
    for _ in range(RESAMPLES):
        pick = rng.choice(len(keys), len(keys), replace=True)
        ss = [s for i in pick for s in pat[keys[i]]]
        d.append(st.mean(a[s] for s in ss) / st.mean(ref[s] for s in ss))
    return np.percentile(d, [2.5, 97.5])


def main():
    rng = np.random.default_rng(0)
    per = load("lesion")
    base = sorted(per["REAL_T2"])
    us = per["SEG_US"]
    assert set(base) == set(us), set(base) ^ set(us)
    synth = {k: v for k, v in per.items()
             if not k.startswith(("REAL_", "FLOOR_", "SEG_US")) and k not in OFF_GRID
             and set(base) <= set(v)}
    print(f"{len(base)} studies, {len(synth)} configurations on the full cohort")

    m_us = st.mean(us[s] for s in base)
    m_real = st.mean(per["REAL_T2"][s] for s in base)
    m_floor = st.mean(per["FLOOR_US"][s] for s in base)
    ranked = sorted(synth, key=lambda k: st.mean(synth[k][s] for s in base), reverse=True)
    best = ranked[0]
    m_best = st.mean(synth[best][s] for s in base)

    print(f"\nlesion Dice (n={len(base)}):  real {m_real:.3f} | best synth {m_best:.3f} ({best})"
          f" | Seg-US {m_us:.3f} | null control {m_floor:.3f}")
    lo, hi = boot_ratio(us, per["REAL_T2"], base, rng)
    print(f"Seg-US retention {100*m_us/m_real:.1f}%  (95% CI {100*lo:.1f}-{100*hi:.1f}%)")
    lo, hi = boot_diff(us, synth[best], base, rng)
    print(f"Seg-US - best synthesis: {m_us-m_best:+.3f} Dice (95% CI {lo:+.3f} to {hi:+.3f})")

    # paired tests
    print("\npaired Wilcoxon (Seg-US vs ...):")
    for name, other in (("real T2w", per["REAL_T2"]), ("null control", per["FLOOR_US"]),
                        (f"best synthesis ({best})", synth[best])):
        a = [us[s] for s in base]; b = [other[s] for s in base]
        stat, p = wilcoxon(a, b)
        print(f"   {name:34s} diff {st.mean(a)-st.mean(b):+.3f}  p = {p:.4f}")

    # Seg-US vs each configuration, Holm within the family
    raw = []
    for k in synth:
        a = [us[s] for s in base]; b = [synth[k][s] for s in base]
        raw.append((k, wilcoxon(a, b).pvalue))
    adj = holm(raw)
    above = [k for k in synth if st.mean(synth[k][s] for s in base) > m_us]
    sig_below = [k for k in synth if adj[k] < 0.05 and st.mean(synth[k][s] for s in base) < m_us]
    sig_above = [k for k in adj if adj[k] < 0.05 and k in above]
    print(f"\nconfigurations above Seg-US in mean: {len(above)} ({', '.join(above) or '-'})")
    print(f"significantly below Seg-US after Holm: {len(sig_below)} of {len(synth)}")
    print(f"significantly above Seg-US after Holm: {len(sig_above)} ({', '.join(sig_above) or 'none'})")
    print(f"raw p for the best synthesis vs Seg-US: {dict(raw)[best]:.4f} "
          f"(Holm-adjusted {adj[best]:.4f})")

    # phase breakdown
    print("\nby phase (lesion Dice):")
    for phase, sel in (("pre", [s for s in base if s.endswith("-pre")]),
                       ("post", [s for s in base if s.endswith("-post")])):
        bp = max(synth, key=lambda k: st.mean(synth[k][s] for s in sel))
        print(f"   {phase:4s} n={len(sel):2d}  real {st.mean(per['REAL_T2'][s] for s in sel):.3f}"
              f" | Seg-US {st.mean(us[s] for s in sel):.3f}"
              f" | best synth {st.mean(synth[bp][s] for s in sel):.3f} ({bp})"
              f" | null {st.mean(per['FLOOR_US'][s] for s in sel):.3f}")
        a = [us[s] for s in sel]; b = [synth[bp][s] for s in sel]
        print(f"        Seg-US vs that best: {st.mean(a)-st.mean(b):+.3f}, p = {wilcoxon(a, b).pvalue:.4f}")

    # secondary classes
    for cls in ("tumor", "cavity"):
        p2 = load(cls)
        b2 = sorted(set(p2["REAL_T2"]) & set(p2["SEG_US"]))
        s2 = {k: v for k, v in p2.items()
              if not k.startswith(("REAL_", "FLOOR_", "SEG_US")) and k not in OFF_GRID
              and set(b2) <= set(v)}
        if not s2:
            continue
        bk = max(s2, key=lambda k: st.mean(s2[k][s] for s in b2))
        print(f"\n{cls} (n={len(b2)}): real {st.mean(p2['REAL_T2'][s] for s in b2):.3f}"
              f" | Seg-US {st.mean(p2['SEG_US'][s] for s in b2):.3f}"
              f" | best synth {st.mean(s2[bk][s] for s in b2):.3f} ({bk})")
        nb = sum(1 for k in s2 if st.mean(s2[k][s] for s in b2) > st.mean(p2["SEG_US"][s] for s in b2))
        print(f"   configurations above Seg-US: {nb} of {len(s2)}")


if __name__ == "__main__":
    main()
