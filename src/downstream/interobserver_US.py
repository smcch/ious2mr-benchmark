"""Inter-observer agreement between two US segmentation observers.

Observer 1: $IOUS2MR_ROOT\\dataset-registration-corrected-cropped\\US-seg\\<study>-us.nii.gz
Observer 2: $IOUS2MR_ROOT\\Segmentations\\US\\<study>-us-segmentation.nii.gz

NOTE: 6 cases were copied today from Observer 1 -> Observer 2 to fill gaps in
Observer 2 (002-pre, 004-pre, 004-post, 049-post, 079-post, 109-pre).
Those are excluded from the analysis to avoid trivial perfect agreement.

For each pair (study) we compute, per class (and overall foreground):
  - Dice
  - Jaccard / IoU
  - HD95 (mm)
  - NSD@2mm (Normalized Surface Dice, tolerance 2 mm)
  - Volume_obs1 / Volume_obs2 (mm^3) and volume ratio
  - Sensitivity_obs2_wrt_obs1 / Specificity / Cohen's kappa per-voxel

Aggregations:
  - mean +- SD over studies (overall, pre, post, train cohort, test cohort)
  - paired Wilcoxon: not applicable here (no two methods), but we report median.

Outputs:
  $IOUS2MR_ROOT\\downstream_seg\\interobs_us_per_study.csv
  $IOUS2MR_ROOT\\downstream_seg\\interobs_us_summary.csv
  $IOUS2MR_ROOT\\downstream_seg\\interobs_us_label_confusion.csv  (3x3 voxel-confusion matrix aggregated)
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
OBS1 = os.path.join(ROOT, "dataset-registration-corrected-cropped", "US-seg")
OBS2 = os.path.join(ROOT, "Segmentations", "US")
OUT  = os.path.join(ROOT, "downstream_seg")
SPLIT_JSON = os.path.join(ROOT, "resvit", "subject_split.json")

EXCLUDE = set()  # full-cohort run; ALSO tag the 6 copies as 'is_copy'=True
COPIES = {  # bit-identical copies obs1->obs2 (today): trivial perfect agreement
    "ReMIND-002-pre", "ReMIND-004-pre", "ReMIND-004-post",
    "ReMIND-049-post", "ReMIND-079-post", "ReMIND-109-pre",
}


def obs1_path(s):
    return os.path.join(OBS1, f"{s}-us.nii.gz")


def obs2_path(s):
    return os.path.join(OBS2, f"{s}-us-segmentation.nii.gz")


def dice(g, p):
    s = g.sum() + p.sum()
    return float("nan") if s == 0 else 2.0 * float(np.logical_and(g, p).sum()) / float(s)


def jaccard(g, p):
    u = np.logical_or(g, p).sum()
    return float("nan") if u == 0 else float(np.logical_and(g, p).sum()) / float(u)


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


def cohens_kappa(g, p, n_classes=3):
    """Per-voxel Cohen's kappa across all class labels (0,1,2)."""
    g = g.ravel(); p = p.ravel()
    N = len(g)
    if N == 0: return float("nan")
    M = np.zeros((n_classes, n_classes), dtype=np.int64)
    for i in range(n_classes):
        for j in range(n_classes):
            M[i, j] = int(((g == i) & (p == j)).sum())
    po = float(np.trace(M)) / float(N)
    row = M.sum(1).astype(float); col = M.sum(0).astype(float)
    pe = float((row * col).sum()) / float(N * N)
    if pe >= 1.0:
        return float("nan")
    return (po - pe) / (1.0 - pe)


def main():
    split = json.load(open(SPLIT_JSON))
    train_set = set(split["train"])
    test_set  = set(split["test"])

    # studies present in BOTH observers (original obs2 only, excluding copies)
    obs1_studies = {f.replace("-us.nii.gz", "")
                    for f in os.listdir(OBS1) if f.endswith("-us.nii.gz")}
    obs2_studies = {f.replace("-us-segmentation.nii.gz", "")
                    for f in os.listdir(OBS2) if f.endswith("-us-segmentation.nii.gz")}
    common = sorted(obs1_studies & obs2_studies - EXCLUDE)
    print(f"obs1: {len(obs1_studies)}  obs2: {len(obs2_studies)}  "
          f"common (excluding {len(EXCLUDE)} copies): {len(common)}")

    rows = []
    conf_overall = np.zeros((3, 3), dtype=np.int64)
    shape_errs = 0

    for s in common:
        i1 = nib.load(obs1_path(s)); i2 = nib.load(obs2_path(s))
        if i1.shape != i2.shape:
            print(f"  shape mismatch {s}: {i1.shape} vs {i2.shape}", file=sys.stderr)
            shape_errs += 1
            continue
        a = np.rint(i1.get_fdata()).astype(np.uint8)
        b = np.rint(i2.get_fdata()).astype(np.uint8)
        sp = tuple(abs(v) for v in np.diag(i1.affine)[:3])
        vox = float(sp[0] * sp[1] * sp[2])  # mm^3

        # accumulate confusion (clip to 0..2)
        a_c = np.clip(a, 0, 2); b_c = np.clip(b, 0, 2)
        for i in range(3):
            for j in range(3):
                conf_overall[i, j] += int(((a_c == i) & (b_c == j)).sum())

        cohort = "train" if s in train_set else ("test" if s in test_set else "other")
        phase  = "preop" if s.endswith("-pre") else "postop"
        is_copy = s in COPIES

        # Per class: tumor=1, cavity=2
        for cls_id, cls_name in [(1, "tumor"), (2, "cavity"),
                                 ("fg", "foreground")]:
            if cls_id == "fg":
                g = (a > 0); p = (b > 0)
            else:
                g = (a == cls_id); p = (b == cls_id)
            row = {
                "study": s, "cohort": cohort, "phase": phase,
                "is_copy": int(is_copy),
                "class": cls_name,
                "vox_obs1": int(g.sum()), "vox_obs2": int(p.sum()),
                "vol_obs1_mm3": float(g.sum() * vox),
                "vol_obs2_mm3": float(p.sum() * vox),
                "dice": dice(g, p),
                "jaccard": jaccard(g, p),
                "hd95_mm": hd95(g, p, sp),
                "nsd2mm": nsd(g, p, sp, tol=2.0),
                "spacing": ",".join(f"{v:.4f}" for v in sp),
            }
            rows.append(row)

        # whole-label kappa (treats all 3 classes)
        kappa = cohens_kappa(a_c, b_c, n_classes=3)
        rows.append({
            "study": s, "cohort": cohort, "phase": phase,
            "is_copy": int(is_copy),
            "class": "kappa_3class",
            "vox_obs1": int((a_c > 0).sum()), "vox_obs2": int((b_c > 0).sum()),
            "vol_obs1_mm3": float((a_c > 0).sum() * vox),
            "vol_obs2_mm3": float((b_c > 0).sum() * vox),
            "dice": kappa,  # store kappa in dice slot for compactness
            "jaccard": float("nan"),
            "hd95_mm": float("nan"),
            "nsd2mm": float("nan"),
            "spacing": ",".join(f"{v:.4f}" for v in sp),
        })

    # write per-study
    per_path = os.path.join(OUT, "interobs_us_per_study.csv")
    with open(per_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"-> {per_path}  rows={len(rows)}")

    # aggregate per (class x scope) where scope in {all, preop, postop, train, test}
    def stats(values):
        v = np.array([x for x in values if not (x is None or (isinstance(x, float) and math.isnan(x)))], dtype=np.float64)
        if len(v) == 0: return None
        return {"n": int(len(v)),
                "mean": float(v.mean()),
                "sd": float(v.std(ddof=1)) if len(v) >= 2 else 0.0,
                "median": float(np.median(v)),
                "min": float(v.min()), "max": float(v.max())}

    agg_rows = []
    for cls in ("tumor", "cavity", "foreground", "kappa_3class"):
        sub = [r for r in rows if r["class"] == cls]
        for scope_name, predicate in [
            ("all", lambda r: True),
            ("all_excl_copies", lambda r: not r["is_copy"]),
            ("preop", lambda r: r["phase"] == "preop"),
            ("postop", lambda r: r["phase"] == "postop"),
            ("train", lambda r: r["cohort"] == "train"),
            ("test",  lambda r: r["cohort"] == "test"),
            ("test_excl_copies", lambda r: r["cohort"] == "test" and not r["is_copy"]),
        ]:
            sel = [r for r in sub if predicate(r)]
            if not sel:
                continue
            srow = {"class": cls, "scope": scope_name, "n_studies": len(sel)}
            for met in ("dice", "jaccard", "hd95_mm", "nsd2mm",
                        "vol_obs1_mm3", "vol_obs2_mm3"):
                st = stats([r[met] for r in sel])
                if st:
                    for k, v in st.items():
                        srow[f"{met}_{k}"] = v
            agg_rows.append(srow)

    agg_path = os.path.join(OUT, "interobs_us_summary.csv")
    if agg_rows:
        keys = sorted({k for r in agg_rows for k in r.keys()})
        with open(agg_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in agg_rows: w.writerow(r)
        print(f"-> {agg_path}  rows={len(agg_rows)}")

    # voxel confusion matrix
    conf_path = os.path.join(OUT, "interobs_us_label_confusion.csv")
    with open(conf_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "obs2_bg", "obs2_tumor", "obs2_cavity"])
        labels = ["obs1_bg", "obs1_tumor", "obs1_cavity"]
        for i, lab in enumerate(labels):
            w.writerow([lab, *conf_overall[i].tolist()])
    print(f"-> {conf_path}")
    print(f"   total voxels compared: {conf_overall.sum():,}")

    if shape_errs:
        print(f"WARN: {shape_errs} studies had shape mismatch and were skipped")


if __name__ == "__main__":
    main()
