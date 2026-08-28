"""Recompute segmentation metrics under the paper §2.7 protocol.

Difference vs `compute_seg_metrics.py`:
  - Dice / HD95 / NSD are set to NaN whenever the GT class is anatomically
    absent (gt_vox == 0), so the model is NOT penalised for a false positive
    on an absent label. This honours the explicit claim in §2.7:
        "Subject-level aggregation averaged the per-class scores across only
         those classes for which a ground-truth structure was present, so
         that the model was not penalised when a label was anatomically
         absent."
  - 'lesion' (tumor ∪ cavity) is the **primary** endpoint; tumor and cavity
    are kept as **secondary** per-class metrics.
  - Pre / post phase is treated as a first-class stratification axis.

Outputs are written to $IOUS2MR_ROOT\\downstream_seg\\results_paper_protocol\\
and are intended to support the rewrite of the downstream section of the
paper. The original CSVs in the parent folder are NOT overwritten — both
protocols coexist for transparency.

Outputs:
  results_paper_protocol/
    seg_metrics_T2_per_study.csv       long format, includes gt_present/pred_present
    seg_metrics_FLAIR_per_study.csv
    seg_results_T2.csv                 aggregated per (set, class, phase)
    seg_results_FLAIR.csv
    seg_wilcoxon_T2.csv                paired vs best-Dice set
    seg_wilcoxon_FLAIR.csv
    seg_dual_T2_vs_FLAIR.csv           cross-modal table (dual methods)
    headline_lesion_primary.csv        ready-for-paper table on lesion class
    headline_tumor_cavity_secondary.csv ready-for-paper table on tumor & cavity
    phase_breakdown.csv                per-class evaluable cohort sizes per phase
    protocol_comparison.csv            old vs new Dice side-by-side for REAL
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from common.paths import PROJECT_ROOT, DATA_ROOT, CKPT_ROOT, EXTERNAL_ROOT  # noqa: E402
import os  # noqa: E402,F811
import os, sys, json, csv, math, glob
from collections import defaultdict
import numpy as np
import nibabel as nib

try:
    from scipy.ndimage import distance_transform_edt
except Exception:
    distance_transform_edt = None

ROOT = str(PROJECT_ROOT)
DS = os.path.join(ROOT, "downstream_seg")
SEG_DIR = os.path.join(ROOT, "Segmentations")
OUT_DIR = os.path.join(DS, "results_paper_protocol")
os.makedirs(OUT_DIR, exist_ok=True)

LABELS = [(1, "tumor"), (2, "cavity"), ("lesion", "lesion")]


def seg_path(s):
    p = os.path.join(SEG_DIR, f"{s}-mri-segmentation.nii.gz")
    if os.path.exists(p):
        return p
    p2 = os.path.join(SEG_DIR, f"{s}-mir-segmentation.nii.gz")
    return p2 if os.path.exists(p2) else None


def dice(gt, pr):
    inter = float(np.logical_and(gt, pr).sum())
    s = float(gt.sum() + pr.sum())
    if s == 0:
        return float("nan")
    return 2.0 * inter / s


def hd95(gt, pr, spacing):
    if distance_transform_edt is None:
        return float("nan")
    if gt.sum() == 0 or pr.sum() == 0:
        return float("nan")
    sp = np.array(spacing, dtype=np.float32)
    dt_gt = distance_transform_edt(~gt, sampling=sp)
    dt_pr = distance_transform_edt(~pr, sampling=sp)
    d_all = np.concatenate([dt_gt[pr].ravel(), dt_pr[gt].ravel()])
    return float(np.percentile(d_all, 95))


def nsd(gt, pr, spacing, tol=2.0):
    if distance_transform_edt is None:
        return float("nan")
    if gt.sum() == 0 or pr.sum() == 0:
        return float("nan")
    from scipy.ndimage import binary_erosion
    sp = np.array(spacing, dtype=np.float32)
    bg = gt & ~binary_erosion(gt, iterations=1)
    bp = pr & ~binary_erosion(pr, iterations=1)
    if bg.sum() == 0 or bp.sum() == 0:
        return float("nan")
    dt_g = distance_transform_edt(~bg, sampling=sp)
    dt_p = distance_transform_edt(~bp, sampling=sp)
    ok_p = (dt_g[bp] <= tol).sum()
    ok_g = (dt_p[bg] <= tol).sum()
    return float(ok_p + ok_g) / float(bp.sum() + bg.sum())


def phase_of(study):
    return "preop" if study.endswith("-pre") else "postop"


def evaluate_set(pred_dir, modality):
    rows = []
    for pred_path in sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz"))):
        study = os.path.basename(pred_path).replace(".nii.gz", "")
        gt_path = seg_path(study)
        if gt_path is None:
            continue
        gt_img = nib.load(gt_path)
        pr_img = nib.load(pred_path)
        if gt_img.shape != pr_img.shape:
            print(f"  shape mismatch {study}", file=sys.stderr)
            continue
        gt = np.rint(gt_img.get_fdata()).astype(np.uint8)
        pr = np.rint(pr_img.get_fdata()).astype(np.uint8)
        sp = tuple(abs(v) for v in np.diag(gt_img.affine)[:3])
        for cls_id, cls_name in LABELS:
            if cls_id == "lesion":
                gtm = (gt > 0); prm = (pr > 0)
            else:
                gtm = (gt == cls_id); prm = (pr == cls_id)
            present_gt = int(gtm.sum() > 0); present_pr = int(prm.sum() > 0)
            # Paper §2.7 protocol: skip studies where the GT class is absent.
            # Dice/HD95/NSD set to NaN -> excluded from aggregation.
            if present_gt:
                d = dice(gtm, prm)
                h = hd95(gtm, prm, sp) if present_pr else float("nan")
                n = nsd(gtm, prm, sp, tol=2.0) if present_pr else float("nan")
            else:
                d = float("nan"); h = float("nan"); n = float("nan")
            rows.append({
                "modality": modality, "study": study, "phase": phase_of(study),
                "class_id": cls_id, "class": cls_name,
                "gt_present": present_gt, "pred_present": present_pr,
                "dice": d, "hd95_mm": h, "nsd2mm": n,
                "gt_vox": int(gtm.sum()), "pred_vox": int(prm.sum()),
                "spacing": ",".join(f"{v:.4f}" for v in sp),
            })
    return rows


def aggregate(rows):
    by = defaultdict(list)
    for r in rows:
        for phase in ("all", r["phase"]):
            by[(r["set"], r["class"], phase)].append(r)
    out = []
    for (set_, cls, phase), rs in by.items():
        d = np.array([x["dice"] for x in rs], dtype=np.float64); d = d[~np.isnan(d)]
        h = np.array([x["hd95_mm"] for x in rs], dtype=np.float64); h = h[~np.isnan(h)]
        ns = np.array([x["nsd2mm"] for x in rs], dtype=np.float64); ns = ns[~np.isnan(ns)]
        n_gt_present = sum(1 for x in rs if x["gt_present"])
        n_pred_when_absent = sum(1 for x in rs if (not x["gt_present"]) and x["pred_present"])
        out.append({
            "set": set_, "class": cls, "phase": phase,
            "n_studies_total": len(rs),
            "n_gt_present": n_gt_present,
            "n_pred_when_gt_absent": n_pred_when_absent,
            "n_dice_evaluable": len(d),
            "dice_mean":   float(d.mean())   if len(d) else float("nan"),
            "dice_sd":     float(d.std(ddof=1)) if len(d) >= 2 else float("nan"),
            "dice_median": float(np.median(d)) if len(d) else float("nan"),
            "dice_min":    float(d.min())    if len(d) else float("nan"),
            "dice_max":    float(d.max())    if len(d) else float("nan"),
            "hd95_mean":   float(h.mean())   if len(h) else float("nan"),
            "hd95_median": float(np.median(h)) if len(h) else float("nan"),
            "nsd_mean":    float(ns.mean())  if len(ns) else float("nan"),
            "nsd_median":  float(np.median(ns)) if len(ns) else float("nan"),
        })
    return out


def wilcoxon_vs_best(rows):
    from scipy.stats import wilcoxon
    by = defaultdict(dict)
    for r in rows:
        for phase in ("all", r["phase"]):
            by[(r["set"], r["class"], phase)][r["study"]] = r["dice"]
    classes = sorted({r["class"] for r in rows})
    phases = ["all", "preop", "postop"]
    out = []
    for cls in classes:
        for phase in phases:
            sets = sorted({s for (s, c, p) in by.keys() if c == cls and p == phase})
            means = []
            for st in sets:
                vals = [v for v in by[(st, cls, phase)].values() if not (v is None or math.isnan(v))]
                means.append((st, float(np.mean(vals)) if vals else float("-inf")))
            if not means:
                continue
            best = max(means, key=lambda x: x[1])[0]
            for st in sets:
                if st == best:
                    continue
                common = sorted(set(by[(st, cls, phase)].keys()) & set(by[(best, cls, phase)].keys()))
                a = np.array([by[(st, cls, phase)][s] for s in common], dtype=np.float64)
                b = np.array([by[(best, cls, phase)][s] for s in common], dtype=np.float64)
                m = ~(np.isnan(a) | np.isnan(b))
                a, b = a[m], b[m]
                if len(a) < 2:
                    stat, p = float("nan"), float("nan")
                else:
                    try:
                        s_, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
                        stat = float(s_)
                    except Exception:
                        stat, p = float("nan"), float("nan")
                out.append({"class": cls, "phase": phase, "best_set": best, "set": st,
                            "n_paired": int(m.sum()),
                            "dice_mean_set": float(a.mean()) if len(a) else float("nan"),
                            "dice_mean_best": float(b.mean()) if len(b) else float("nan"),
                            "wilcoxon_stat": stat, "p_value": p if p == p else float("nan")})
    return out


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_headlines(agg_rows, modality, ref_set):
    """Build two paper-ready summary tables.

    headline_lesion_primary: per phase, ref Dice, best synth + retention %, Wilcoxon p
    headline_tumor_cavity_secondary: same but for tumor and cavity
    """
    def row(set_name, cls, phase):
        for r in agg_rows:
            if r["set"] == set_name and r["class"] == cls and r["phase"] == phase:
                return r
        return None

    # find best synth per (class, phase): highest dice_mean among non-ref sets
    def best_synth(cls, phase):
        cands = [r for r in agg_rows if r["class"] == cls and r["phase"] == phase and r["set"] != ref_set]
        cands = [r for r in cands if not math.isnan(r["dice_mean"])]
        if not cands:
            return None
        cands.sort(key=lambda r: -r["dice_mean"])
        return cands[0]

    lesion_rows = []
    tc_rows = []
    for phase in ("all", "preop", "postop"):
        for cls in ("lesion", "tumor", "cavity"):
            ref = row(ref_set, cls, phase)
            best = best_synth(cls, phase)
            if ref is None or best is None:
                continue
            retention = (best["dice_mean"] / ref["dice_mean"] * 100.0) if ref["dice_mean"] > 0 else float("nan")
            out = {
                "modality": modality, "phase": phase, "class": cls,
                "ref_set": ref_set,
                "n_evaluable": ref["n_dice_evaluable"],
                "n_gt_present": ref["n_gt_present"],
                "n_total": ref["n_studies_total"],
                "ref_dice_mean": ref["dice_mean"], "ref_dice_sd": ref["dice_sd"],
                "ref_dice_median": ref["dice_median"],
                "ref_hd95_median": ref["hd95_median"],
                "ref_nsd_mean": ref["nsd_mean"],
                "best_synth_set": best["set"],
                "best_synth_dice_mean": best["dice_mean"],
                "best_synth_dice_sd": best["dice_sd"],
                "best_synth_hd95_median": best["hd95_median"],
                "best_synth_nsd_mean": best["nsd_mean"],
                "retention_pct": retention,
            }
            if cls == "lesion":
                lesion_rows.append(out)
            else:
                tc_rows.append(out)
    return lesion_rows, tc_rows


def build_phase_breakdown(rows):
    """Cohort composition by (modality, class, phase) — how many studies have
    GT class present, how many got a false-positive prediction, etc."""
    out = []
    by = defaultdict(list)
    for r in rows:
        for phase in ("all", r["phase"]):
            by[(r["modality"], r["class"], phase)].append(r)
    for (mod, cls, phase), rs in by.items():
        # use the REAL set as the reference for the cohort composition
        ref_rows = [r for r in rs if r["set"].startswith("REAL_")]
        if not ref_rows:
            ref_rows = rs
        # dedup by study (cohort size shouldn't depend on # of sets)
        seen = {}
        for r in ref_rows:
            seen[r["study"]] = r
        seen_studies = list(seen.values())
        n_total = len(seen_studies)
        n_gt = sum(1 for r in seen_studies if r["gt_present"])
        n_absent_total = n_total - n_gt
        out.append({
            "modality": mod, "class": cls, "phase": phase,
            "n_studies_total": n_total,
            "n_gt_present": n_gt,
            "n_gt_absent": n_absent_total,
            "pct_gt_present": (n_gt / n_total * 100) if n_total else float("nan"),
        })
    return out


def build_protocol_comparison(rows_new, rows_old_csv):
    """Compare the paper-protocol Dice (new) vs the old protocol on REAL set,
    so the rewrite of the paper can show the magnitude of the correction."""
    if not os.path.exists(rows_old_csv):
        return []
    old_rows = list(csv.DictReader(open(rows_old_csv)))
    new_index = {(r["modality"], r["set"], r["study"], r["class"], r["phase"]): r for r in rows_new}
    out = []
    for ph in ("all", "preop", "postop"):
        for cls in ("lesion", "tumor", "cavity"):
            # old mean (REAL only) — what compute_seg_metrics.py produced
            real_old = [r for r in old_rows if r["set"].startswith("REAL_") and r["class"] == cls and (ph == "all" or r["phase"] == ph)]
            old_dices = []
            for r in real_old:
                try:
                    v = float(r["dice"])
                    if not math.isnan(v):
                        old_dices.append(v)
                except (ValueError, KeyError):
                    pass
            # new mean (REAL only, paper protocol)
            new_real = [v for k, r in new_index.items() if r["set"].startswith("REAL_") and r["class"] == cls and (ph == "all" or r["phase"] == ph) for v in [r["dice"]] if not math.isnan(v)]
            out.append({
                "class": cls, "phase": ph,
                "ref_set": "REAL",
                "n_dice_old_protocol": len(old_dices),
                "dice_mean_old_protocol": float(np.mean(old_dices)) if old_dices else float("nan"),
                "n_dice_paper_protocol": len(new_real),
                "dice_mean_paper_protocol": float(np.mean(new_real)) if new_real else float("nan"),
                "delta_dice": (float(np.mean(new_real)) - float(np.mean(old_dices))) if (new_real and old_dices) else float("nan"),
            })
    return out


def build_dual_cross(t2_persubj_csv, fl_persubj_csv):
    if not (os.path.exists(t2_persubj_csv) and os.path.exists(fl_persubj_csv)):
        return []
    t2 = list(csv.DictReader(open(t2_persubj_csv)))
    fl = list(csv.DictReader(open(fl_persubj_csv)))
    def base(set_name):
        return set_name.replace("-T2-from-dual", "").replace("-T2-from-single", "") \
                       .replace("-T2", "").replace("-FLAIR", "")
    t2_dual = [r for r in t2 if "from-dual" in r["set"]]
    fl_index = defaultdict(list)
    for rf in fl:
        fl_index[(base(rf["set"]), rf["study"], rf["class"])].append(rf)
    pairs = []
    for r in t2_dual:
        key = (base(r["set"]), r["study"], r["class"])
        if fl_index.get(key):
            mf = fl_index[key][0]
            pairs.append({
                "method_base": base(r["set"]),
                "study": r["study"], "phase": r["phase"], "class": r["class"],
                "dice_T2": r["dice"], "dice_FLAIR": mf["dice"],
                "hd95_T2": r["hd95_mm"], "hd95_FLAIR": mf["hd95_mm"],
                "T2_set": r["set"], "FLAIR_set": mf["set"],
            })
    return pairs


def main():
    print("=== Computing seg metrics — PAPER §2.7 PROTOCOL ===")
    print(f"Output dir: {OUT_DIR}")
    all_modal_rows = []
    for modality, pred_root, ref_set in [
        ("T2",    os.path.join(DS, "predictions_T2"),    "REAL_T2"),
        ("FLAIR", os.path.join(DS, "predictions_FLAIR"), "REAL_FLAIR"),
    ]:
        if not os.path.isdir(pred_root):
            print(f"skip {modality}: no predictions dir"); continue
        all_rows = []
        for set_dir in sorted(os.listdir(pred_root)):
            full = os.path.join(pred_root, set_dir)
            if not os.path.isdir(full):
                continue
            print(f"  {modality} :: {set_dir} ...", end=" ", flush=True)
            rs = evaluate_set(full, modality)
            for r in rs:
                r["set"] = set_dir
            all_rows.extend(rs)
            print(f"{len(rs)//len(LABELS)} studies")

        out_persubj = os.path.join(OUT_DIR, f"seg_metrics_{modality}_per_study.csv")
        out_agg     = os.path.join(OUT_DIR, f"seg_results_{modality}.csv")
        out_wil     = os.path.join(OUT_DIR, f"seg_wilcoxon_{modality}.csv")
        write_csv(out_persubj, all_rows)
        agg = aggregate(all_rows)
        write_csv(out_agg, agg)
        wil = wilcoxon_vs_best(all_rows)
        write_csv(out_wil, wil)
        print(f"  -> {out_persubj}  rows={len(all_rows)}")
        print(f"  -> {out_agg}  rows={len(agg)}")
        print(f"  -> {out_wil}  rows={len(wil)}")

        # headlines
        lesion_h, tc_h = build_headlines(agg, modality, ref_set)
        all_modal_rows.append((modality, all_rows, lesion_h, tc_h))

    # Combine headline tables across modalities into single CSVs
    all_lesion = sum((m[2] for m in all_modal_rows), [])
    all_tc     = sum((m[3] for m in all_modal_rows), [])
    write_csv(os.path.join(OUT_DIR, "headline_lesion_primary.csv"), all_lesion)
    write_csv(os.path.join(OUT_DIR, "headline_tumor_cavity_secondary.csv"), all_tc)
    print(f"  -> headline_lesion_primary.csv ({len(all_lesion)} rows)")
    print(f"  -> headline_tumor_cavity_secondary.csv ({len(all_tc)} rows)")

    # Phase breakdown (cohort composition)
    rows_concat = sum((m[1] for m in all_modal_rows), [])
    pb = build_phase_breakdown(rows_concat)
    write_csv(os.path.join(OUT_DIR, "phase_breakdown.csv"), pb)
    print(f"  -> phase_breakdown.csv ({len(pb)} rows)")

    # Protocol comparison vs the original (incorrect) computation
    cmp_rows = []
    for modality, _, _, _ in all_modal_rows:
        old_csv = os.path.join(DS, f"seg_metrics_{modality}_per_study.csv")
        new_rows = [r for r in rows_concat if r["modality"] == modality]
        rows_cmp = build_protocol_comparison(new_rows, old_csv)
        for r in rows_cmp:
            r["modality"] = modality
        cmp_rows.extend(rows_cmp)
    if cmp_rows:
        # reorder so modality comes first
        keys = ["modality"] + [k for k in cmp_rows[0].keys() if k != "modality"]
        cmp_rows = [{k: r[k] for k in keys} for r in cmp_rows]
        write_csv(os.path.join(OUT_DIR, "protocol_comparison.csv"), cmp_rows)
        print(f"  -> protocol_comparison.csv ({len(cmp_rows)} rows)")

    # Cross-modal dual table
    pairs = build_dual_cross(
        os.path.join(OUT_DIR, "seg_metrics_T2_per_study.csv"),
        os.path.join(OUT_DIR, "seg_metrics_FLAIR_per_study.csv"),
    )
    write_csv(os.path.join(OUT_DIR, "seg_dual_T2_vs_FLAIR.csv"), pairs)
    print(f"  -> seg_dual_T2_vs_FLAIR.csv ({len(pairs)} rows)")

    print("DONE.")


if __name__ == "__main__":
    main()
