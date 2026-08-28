"""Inter-observer agreement on MR-space segmentations.

Two expert observers segmented tumor & resection cavity on the same MR-space
volumes (shape/affine match the MR grid; the files happen to be stored under
`*/US-seg/` and `Segmentations/US/` because they share the FOV-cropped grid
with the US volumes, but the labels themselves were drawn on the MR).

Observer 1: $IOUS2MR_ROOT\\dataset-registration-corrected-cropped\\US-seg\\<s>-us.nii.gz
Observer 2: $IOUS2MR_ROOT\\Segmentations\\US\\<s>-us-segmentation.nii.gz
            (obs2 ≈ GT used by the downstream nnU-Net eval in Segmentations\\MRI)

This script provides the **human ceiling** for the downstream segmentation
analysis: it answers "if even a second expert were given the same MR volume,
what Dice would they reach vs the one used as GT in the synth evaluation?".

Filtered to the **test cohort (16 subjects, 31 paired studies)** that matches
the downstream evaluation, and stratified by phase (preop/postop), so the
human ceiling is directly comparable to REAL-MR and synth-MR Dices reported
in `results_paper_protocol/headline_*.csv`.

Outputs (under results_paper_protocol/):
  interobs_MR_per_study.csv      Dice/HD95/NSD per study × class
  interobs_MR_summary.csv        aggregated per (class × phase × scope)
  human_ceiling_vs_models.csv    side-by-side: human ceiling vs REAL-MR vs best synth
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from common.paths import PROJECT_ROOT, DATA_ROOT, CKPT_ROOT, EXTERNAL_ROOT  # noqa: E402
import os  # noqa: E402,F811
import os, csv, math, json, sys
from collections import defaultdict
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, binary_erosion

ROOT = str(PROJECT_ROOT)
OBS1_DIR = os.path.join(ROOT, "dataset-registration-corrected-cropped", "US-seg")
OBS2_DIR = os.path.join(ROOT, "Segmentations", "US")
SPLIT_JSON = os.path.join(ROOT, "resvit", "subject_split.json")
OUT = os.path.join(ROOT, "downstream_seg", "results_paper_protocol")
PAPER_HEADLINES = os.path.join(OUT, "headline_lesion_primary.csv")
PAPER_HEADLINES_TC = os.path.join(OUT, "headline_tumor_cavity_secondary.csv")

# Bit-identical copies obs1 -> obs2 (filled obs2 gaps) — exclude from analysis
# so trivial perfect agreement does not inflate the ceiling.
COPIES = {
    "ReMIND-002-pre", "ReMIND-004-pre", "ReMIND-004-post",
    "ReMIND-049-post", "ReMIND-079-post", "ReMIND-109-pre",
}


def obs1(s): return os.path.join(OBS1_DIR, f"{s}-us.nii.gz")
def obs2(s): return os.path.join(OBS2_DIR, f"{s}-us-segmentation.nii.gz")


def dice(g, p):
    s = g.sum() + p.sum()
    return float("nan") if s == 0 else 2.0 * float(np.logical_and(g, p).sum()) / float(s)


def hd95(g, p, sp):
    if g.sum() == 0 or p.sum() == 0:
        return float("nan")
    dt_g = distance_transform_edt(~g, sampling=sp)
    dt_p = distance_transform_edt(~p, sampling=sp)
    d_all = np.concatenate([dt_g[p].ravel(), dt_p[g].ravel()])
    return float(np.percentile(d_all, 95))


def nsd(g, p, sp, tol=2.0):
    if g.sum() == 0 or p.sum() == 0:
        return float("nan")
    bg = g & ~binary_erosion(g, iterations=1)
    bp = p & ~binary_erosion(p, iterations=1)
    if bg.sum() == 0 or bp.sum() == 0:
        return float("nan")
    dt_g = distance_transform_edt(~bg, sampling=sp)
    dt_p = distance_transform_edt(~bp, sampling=sp)
    ok_p = (dt_g[bp] <= tol).sum()
    ok_g = (dt_p[bg] <= tol).sum()
    return float(ok_p + ok_g) / float(bp.sum() + bg.sum())


def phase_of(s): return "preop" if s.endswith("-pre") else "postop"


def main():
    split = json.load(open(SPLIT_JSON))
    # subject_split.json: `train`/`test` are STUDY lists; `train_subjects`/`test_subjects` are subject lists.
    train_studies = set(split["train"])
    test_studies  = set(split["test"])

    obs1_studies = {f.replace("-us.nii.gz", "") for f in os.listdir(OBS1_DIR) if f.endswith("-us.nii.gz")}
    obs2_studies = {f.replace("-us-segmentation.nii.gz", "") for f in os.listdir(OBS2_DIR) if f.endswith("-us-segmentation.nii.gz")}
    common = sorted(obs1_studies & obs2_studies)
    print(f"Total common studies: {len(common)} (of {len(obs1_studies)} obs1 / {len(obs2_studies)} obs2)")

    rows = []
    for s in common:
        subj = "-".join(s.split("-")[:2])
        cohort = "train" if s in train_studies else ("test" if s in test_studies else "other")
        phase = phase_of(s)
        is_copy = int(s in COPIES)

        i1 = nib.load(obs1(s)); i2 = nib.load(obs2(s))
        if i1.shape != i2.shape:
            print(f"  shape mismatch {s}: {i1.shape} vs {i2.shape}", file=sys.stderr)
            continue
        a = np.rint(i1.get_fdata()).astype(np.uint8)
        b = np.rint(i2.get_fdata()).astype(np.uint8)
        sp = tuple(abs(v) for v in np.diag(i1.affine)[:3])

        for cls_name in ("tumor", "cavity", "lesion"):
            if cls_name == "tumor":
                g = (a == 1); p = (b == 1)
            elif cls_name == "cavity":
                g = (a == 2); p = (b == 2)
            else:
                g = (a > 0); p = (b > 0)
            gt_present = int(g.sum() > 0)
            obs2_present = int(p.sum() > 0)
            # Paper §2.7 protocol: only evaluate when obs1 has the label present
            # (obs1 plays the role of the GT here).
            if gt_present:
                d = dice(g, p)
                h = hd95(g, p, sp) if obs2_present else float("nan")
                ns = nsd(g, p, sp, tol=2.0) if obs2_present else float("nan")
            else:
                d = h = ns = float("nan")
            rows.append({
                "study": s, "subject": subj, "cohort": cohort, "phase": phase,
                "is_copy": is_copy, "class": cls_name,
                "obs1_present": gt_present, "obs2_present": obs2_present,
                "dice": d, "hd95_mm": h, "nsd2mm": ns,
                "vox_obs1": int(g.sum()), "vox_obs2": int(p.sum()),
                "spacing": ",".join(f"{v:.4f}" for v in sp),
            })

    per_path = os.path.join(OUT, "interobs_MR_per_study.csv")
    fields = list(rows[0].keys())
    with open(per_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        [w.writerow(r) for r in rows]
    print(f"-> {per_path}  rows={len(rows)}")

    def stats(values):
        v = np.array([x for x in values if not (x is None or (isinstance(x, float) and math.isnan(x)))], dtype=np.float64)
        if len(v) == 0: return None
        return dict(n=int(len(v)), mean=float(v.mean()),
                    sd=float(v.std(ddof=1)) if len(v) >= 2 else 0.0,
                    median=float(np.median(v)),
                    min=float(v.min()), max=float(v.max()))

    # Aggregations: by class × scope (cohort+phase). Exclude copies from agreement stats.
    agg = []
    scopes = [
        ("all_no_copies",       lambda r: not r["is_copy"]),
        ("test_no_copies",      lambda r: r["cohort"] == "test" and not r["is_copy"]),
        ("test_preop",          lambda r: r["cohort"] == "test" and r["phase"] == "preop"  and not r["is_copy"]),
        ("test_postop",         lambda r: r["cohort"] == "test" and r["phase"] == "postop" and not r["is_copy"]),
        ("train_no_copies",     lambda r: r["cohort"] == "train" and not r["is_copy"]),
        ("all_preop",           lambda r: r["phase"] == "preop"  and not r["is_copy"]),
        ("all_postop",          lambda r: r["phase"] == "postop" and not r["is_copy"]),
        ("all_with_copies",     lambda r: True),  # for reference (will be slightly higher)
    ]
    for cls in ("lesion", "tumor", "cavity"):
        sub = [r for r in rows if r["class"] == cls]
        for scope, pred in scopes:
            sel = [r for r in sub if pred(r)]
            n_total = len(sel)
            n_obs1_present = sum(1 for r in sel if r["obs1_present"])
            n_pair_present = sum(1 for r in sel if r["obs1_present"] and r["obs2_present"])
            srow = {"class": cls, "scope": scope,
                    "n_total": n_total,
                    "n_obs1_present": n_obs1_present,
                    "n_both_present": n_pair_present}
            for met in ("dice", "hd95_mm", "nsd2mm"):
                st = stats([r[met] for r in sel])
                if st:
                    for k, v in st.items():
                        srow[f"{met}_{k}"] = v
            agg.append(srow)

    sum_path = os.path.join(OUT, "interobs_MR_summary.csv")
    if agg:
        keys = sorted({k for r in agg for k in r.keys()})
        with open(sum_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            [w.writerow(r) for r in agg]
    print(f"-> {sum_path}  rows={len(agg)}")

    # Build the human-ceiling-vs-models comparison table.
    # Joins:
    #   - human inter-observer Dice (test cohort) on (class, phase)
    #   - REAL-MR Dice and best-synth Dice from headline_lesion + tumor_cavity tables (T2 and FLAIR)
    headline_rows = []
    for p in (PAPER_HEADLINES, PAPER_HEADLINES_TC):
        if os.path.exists(p):
            headline_rows.extend(list(csv.DictReader(open(p))))

    def human_stats(cls, scope):
        for r in agg:
            if r["class"] == cls and r["scope"] == scope:
                return r
        return None

    cmp_rows = []
    scope_for_phase = {"all": "test_no_copies", "preop": "test_preop", "postop": "test_postop"}
    for h in headline_rows:
        cls = h["class"]; phase = h["phase"]; mod = h["modality"]
        sc = scope_for_phase.get(phase)
        hs = human_stats(cls, sc)
        if hs is None: continue
        cmp_rows.append({
            "modality": mod, "phase": phase, "class": cls,
            "n_human_pairs": hs["n_total"],
            "human_dice_mean": hs.get("dice_mean", float("nan")),
            "human_dice_sd":   hs.get("dice_sd", float("nan")),
            "human_dice_median": hs.get("dice_median", float("nan")),
            "human_hd95_median": hs.get("hd95_mm_median", float("nan")),
            "human_nsd_mean":  hs.get("nsd2mm_mean", float("nan")),
            "n_real":  h["n_evaluable"],
            "real_dice_mean":    h["ref_dice_mean"],
            "real_dice_median":  h["ref_dice_median"],
            "real_hd95_median":  h["ref_hd95_median"],
            "real_nsd_mean":     h["ref_nsd_mean"],
            "best_synth_set":    h["best_synth_set"],
            "best_synth_dice":   h["best_synth_dice_mean"],
            "best_synth_hd95_median": h["best_synth_hd95_median"],
            "best_synth_nsd":    h["best_synth_nsd_mean"],
            "retention_pct_vs_real":   h["retention_pct"],
            # how far the model is from the human ceiling
            "retention_pct_vs_human":  (float(h["best_synth_dice_mean"]) / hs["dice_mean"] * 100.0) if (hs.get("dice_mean") and not math.isnan(float(hs["dice_mean"])) and float(hs["dice_mean"]) > 0) else float("nan"),
            "gap_real_vs_human": (float(h["ref_dice_mean"]) - float(hs["dice_mean"])) if hs.get("dice_mean") and not math.isnan(float(hs["dice_mean"])) else float("nan"),
        })

    cmp_path = os.path.join(OUT, "human_ceiling_vs_models.csv")
    if cmp_rows:
        with open(cmp_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cmp_rows[0].keys()))
            w.writeheader()
            [w.writerow(r) for r in cmp_rows]
        print(f"-> {cmp_path}  rows={len(cmp_rows)}")

    # Short human-readable summary
    print("\n=== Inter-observer ceiling (test cohort, no copies) ===")
    for cls in ("lesion", "tumor", "cavity"):
        for sc in ("test_no_copies", "test_preop", "test_postop"):
            hs = human_stats(cls, sc)
            if hs and hs["n_total"]:
                print(f"  {cls:7s} / {sc:15s} n={hs['n_total']:>2}  "
                      f"Dice {hs.get('dice_mean', float('nan')):.3f} ± {hs.get('dice_sd', 0):.3f}  "
                      f"median {hs.get('dice_median', float('nan')):.3f}  "
                      f"HD95med {hs.get('hd95_mm_median', float('nan')):.2f} mm  "
                      f"NSD {hs.get('nsd2mm_mean', float('nan')):.3f}")


if __name__ == "__main__":
    main()
