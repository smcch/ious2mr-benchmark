#!/usr/bin/env python3
"""What the present-class aggregation rule hides.

Subject-level scores in this benchmark average only over classes whose ground-truth
structure exists, so a model is not penalised for a label that is anatomically absent.
That is the right choice for a summary statistic and the wrong one for safety: the failure
with the clearest clinical consequence is a synthetic MRI that shows residual tumour where
the resection was macroscopically complete, and no Dice-based summary can express it,
because there is no ground truth to compare against.

This script measures it directly. For every configuration and every test study in which the
tumour class is absent, it records whether a tumour was nevertheless predicted and how large
that spurious prediction was, then aggregates by generator family and compares against the
same frozen model applied to real T2w.

Outputs results/downstream/absent_class_false_positives.csv.
"""
from __future__ import annotations

import csv
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
SRC = Path(os.environ.get("IOUS2MR_SOURCE_TREE", r"E:\SINTESIS"))

PER_STUDY = SRC / "downstream_seg" / "results_paper_protocol" / "seg_metrics_T2_per_study.csv"
OUT = RESULTS / "downstream" / "absent_class_false_positives.csv"


def family_of(s: str) -> str:
    if s.startswith("REAL_"):
        return "real MR"
    if s.startswith("FLOOR_"):
        return "floor (raw ioUS)"
    if s.startswith("ResViT"):
        return "ResViT"
    if s.startswith("SynDiff"):
        return "SynDiff"
    if s.startswith("GAN-"):
        return "GAN"
    return "other"


def main() -> int:
    rows = []
    for r in csv.DictReader(open(PER_STUDY, newline="", encoding="utf-8")):
        if r["class"] != "tumor" or r["gt_present"] != "0":
            continue
        try:
            pred = float(r["pred_vox"])
        except (KeyError, ValueError):
            continue
        rows.append(dict(set=r["set"], family=family_of(r["set"]), study=r["study"],
                         pred_vox=int(pred), false_positive=int(pred > 0)))

    if not rows:
        print("no absent-tumour studies found in the per-study table")
        return 1

    print("=" * 88)
    print("FALSE POSITIVES ON AN ABSENT CLASS (tumour absent -> tumour predicted)")
    print(f"   {'family':18s} {'cases':>6s} {'FP':>6s} {'rate':>7s} {'median false volume':>21s}")
    byfam = defaultdict(list)
    for r in rows:
        byfam[r["family"]].append(r)
    order = ["real MR", "floor (raw ioUS)", "GAN", "ResViT", "SynDiff"]
    for fam in [f for f in order if f in byfam] + [f for f in byfam if f not in order]:
        v = byfam[fam]
        fp = [x for x in v if x["false_positive"]]
        med = st.median(x["pred_vox"] for x in fp) if fp else 0
        print(f"   {fam:18s} {len(v):6d} {len(fp):6d} {100*len(fp)/len(v):6.0f}% "
              f"{med:21,.0f} vox")

    print("\n   Per-configuration extremes (configurations with >= 4 absent-tumour studies):")
    bycfg = defaultdict(list)
    for r in rows:
        bycfg[r["set"]].append(r)
    ranked = []
    for s, v in bycfg.items():
        if len(v) < 4:
            continue
        fp = [x for x in v if x["false_positive"]]
        ranked.append((len(fp) / len(v),
                       st.median(x["pred_vox"] for x in fp) if fp else 0, s, len(v)))
    ranked.sort()
    for rate, med, s, n in ranked[:3] + ranked[-3:]:
        print(f"     {s[:46]:46s} {100*rate:4.0f}%  median {med:8,.0f} vox  (n={n})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
