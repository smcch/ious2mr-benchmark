"""
rescore_all.py  --  UNIFIED RE-SCORE of all US->MRI synthesis METHODS (no ablations)

One identical metric harness (the "resvit-protocol", lifted from
resvit/eval_resvit_metrics.py) is applied to every method:
  - COMPARATIVA-3 GAN baselines (pix2pix / cut / cyclegan / swinpix2pix x {2d,25d,2d_3dpost,3d} x {t2,t2_flair})  -> recomputed from NIfTI
  - ResViT ({2d, 2.5d, 2d_3d_refine, full_3d} x {t2, t2_flair})  -> recomputed from NIfTI
  - SynDiff "syndiff_us_t2" untuned full model  -> recomputed from saved NIfTI volumes (best epoch)
  - SynDiff ensembles (t2 single / t2 dual / fl dual, alpha 0.5)  -> recomputed from saved NIfTI volumes
  - Other SynDiff methods (paired 2D/2.5D/3D/3Drefine single & dual, ResViT->SynDiff cascade, ResViT+SynDiff joint)
        -> the *_resvit_protocol_ep* folders contain NO NIfTI, only per_subject.csv produced by the
           SAME protocol (eval_resvit_protocol.py mirrors eval_resvit_metrics.py); we read those per-subject
           rows at the chosen best epoch. LPIPS taken from those CSVs where present, else from
           synthdiff/results/unified_per_subject_final.csv (3Drefine rows), else flagged.

Outputs (new file names; nothing existing is overwritten) under E:/SINTESIS/evaluacion-final/:
  rescore_methods_persubject.csv, rescore_methods_summary.csv,
  rescore_methods_rankings_t2.csv, rescore_methods_rankings_flair.csv,
  rescore_wilcoxon.csv, master_benchmark_table.csv (+ _flair),
  rescore_methods.xlsx (only if openpyxl importable), RESCORE_README.md
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from common.paths import PROJECT_ROOT, DATA_ROOT, CKPT_ROOT, EXTERNAL_ROOT  # noqa: E402
import os  # noqa: E402,F811
import os, sys, csv, math, json, glob
from collections import defaultdict
import numpy as np
import nibabel as nib

ROOT = str(PROJECT_ROOT)
OUT = os.path.join(ROOT, "evaluacion-final")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "resvit"))
from eval_resvit_metrics import load_canonical_cohorts  # noqa

DEVICE = None
_LPIPS_FN = None
try:
    import torch
    import lpips
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception as e:
    print("[warn] torch/lpips unavailable:", e)


# ---------------------------------------------------------------- metric core
# All functions take arrays already in [0,1] (foreground = >0.025).
def _ssim01(t01, p01):
    from skimage.metrics import structural_similarity
    vals = []
    for z in range(t01.shape[2]):
        ts, ps = t01[:, :, z], p01[:, :, z]
        if np.mean(ts > 0.025) < 0.01:
            continue
        vals.append(structural_similarity(ts, ps, data_range=1.0))
    return float(np.mean(vals)) if vals else 0.0


def _ssim01_gauss(t01, p01):
    from skimage.metrics import structural_similarity
    vals = []
    for z in range(t01.shape[2]):
        ts, ps = t01[:, :, z], p01[:, :, z]
        if np.mean(ts > 0.025) < 0.01:
            continue
        vals.append(structural_similarity(ts, ps, data_range=1.0,
                                          gaussian_weights=True, sigma=1.5,
                                          use_sample_covariance=False))
    return float(np.mean(vals)) if vals else 0.0


def _psnr01(t01, p01):
    fg = t01 > 0.025
    if fg.sum() < 100:
        return 0.0
    mse = float(np.mean((t01[fg] - p01[fg]) ** 2))
    if mse < 1e-10:
        return 50.0
    return float(10.0 * np.log10(1.0 / mse))


def _mae01(t01, p01):
    fg = t01 > 0.025
    if fg.sum() < 100:
        return 1.0
    return float(np.mean(np.abs(t01[fg] - p01[fg])))


def _get_lpips():
    global _LPIPS_FN
    if _LPIPS_FN is None:
        _LPIPS_FN = lpips.LPIPS(net="alex").to(DEVICE)
        _LPIPS_FN.eval()
    return _LPIPS_FN


def _lpips_tensor01(slice01):
    s = slice01 * 2.0 - 1.0
    t = torch.from_numpy(np.ascontiguousarray(s)).float()
    if t.ndim == 2:
        t = t.unsqueeze(0).expand(3, -1, -1)
    return t.unsqueeze(0).to(DEVICE)


def _lpips01(t01, p01):
    if DEVICE is None:
        return float("nan")
    fn = _get_lpips()
    vals = []
    for z in range(t01.shape[2]):
        ts, ps = t01[:, :, z], p01[:, :, z]
        if np.mean(ts > 0.025) < 0.01:
            continue
        with torch.no_grad():
            d = fn(_lpips_tensor01(ts), _lpips_tensor01(ps))
        vals.append(float(d.item()))
    return float(np.mean(vals)) if vals else 0.0


def metrics01(t01, p01, want_gauss=True):
    return {
        "ssim": _ssim01(t01, p01),
        "ssim_gauss": _ssim01_gauss(t01, p01) if want_gauss else float("nan"),
        "psnr": _psnr01(t01, p01),
        "mae": _mae01(t01, p01),
        "lpips": _lpips01(t01, p01),
    }


def load01(path, src_range):
    """Load NIfTI, return array in [0,1]. src_range in {'m11','01'}."""
    a = nib.load(path).get_fdata().astype(np.float32)
    if src_range == "m11":
        return (a + 1.0) / 2.0
    return a  # already [0,1]


# ---------------------------------------------------------------- label maps
GAN_ARCH = {"2d": "2D", "25d": "2.5D", "2d_3dpost": "2D+3D-post", "3d": "3D"}
GAN_FAM_LABEL = {"pix2pix": "pix2pix", "cut": "CUT", "cyclegan": "CycleGAN",
                 "swinpix2pix": "SwinPix2Pix"}
RESVIT_ARCH = {"2d": "2D", "2.5d": "2.5D", "2d_3d_refine": "2D+3D-refine",
               "full_3d": "3D"}

t2_cohort, flair_cohort = load_canonical_cohorts(
    os.path.join(ROOT, "COMPARATIVA-3", "results_lpips_per_subject.csv"))
T2SET, FLSET = set(t2_cohort), set(flair_cohort)
print(f"Cohorts: T2 n={len(t2_cohort)}  FLAIR n={len(flair_cohort)}")

PERSUBJ = []   # rows: method, family, backbone, architecture, target, channel, subject, ssim, ssim_gauss, psnr, mae, lpips, src
MISSING_FLAGS = []


def add_persubj(method, family, backbone, architecture, target, channel, subject, m, src):
    PERSUBJ.append({
        "method": method, "family": family, "backbone": backbone,
        "architecture": architecture, "target": target, "channel": channel,
        "subject": subject, "ssim": m["ssim"], "ssim_gauss": m.get("ssim_gauss", float("nan")),
        "psnr": m["psnr"], "mae": m["mae"], "lpips": m["lpips"], "src": src,
    })


# ============================================================ A. COMPARATIVA-3
def score_comparativa3():
    base = os.path.join(ROOT, "COMPARATIVA-3")
    for fam in ("pix2pix", "cut", "cyclegan", "swinpix2pix"):
        for arch in ("2d", "25d", "2d_3dpost", "3d"):
            for tgt in ("t2", "t2_flair"):
                exp = f"{fam}_{arch}_{tgt}"
                pred_dir = os.path.join(base, exp, "predictions")
                if not os.path.isdir(pred_dir):
                    MISSING_FLAGS.append(f"COMPARATIVA-3 missing dir: {exp}")
                    continue
                cohort = FLSET if tgt == "t2_flair" else T2SET
                method = f"{GAN_FAM_LABEL[fam]}-{GAN_ARCH[arch]}-{'T2+FLAIR' if tgt=='t2_flair' else 'T2'}"
                n_found = 0
                for subj in sorted(cohort):
                    p_t2 = os.path.join(pred_dir, f"{subj}_pred_t2.nii.gz")
                    g_t2 = os.path.join(pred_dir, f"{subj}_target_t2.nii.gz")
                    if not (os.path.exists(p_t2) and os.path.exists(g_t2)):
                        continue
                    n_found += 1
                    t01 = load01(g_t2, "m11"); p01 = load01(p_t2, "m11")
                    add_persubj(method, GAN_FAM_LABEL[fam], "GAN", GAN_ARCH[arch],
                                tgt, "t2", subj, metrics01(t01, p01), "nifti")
                    if tgt == "t2_flair":
                        p_fl = os.path.join(pred_dir, f"{subj}_pred_flair.nii.gz")
                        g_fl = os.path.join(pred_dir, f"{subj}_target_flair.nii.gz")
                        if os.path.exists(p_fl) and os.path.exists(g_fl):
                            t01f = load01(g_fl, "m11"); p01f = load01(p_fl, "m11")
                            add_persubj(method, GAN_FAM_LABEL[fam], "GAN", GAN_ARCH[arch],
                                        tgt, "flair", subj, metrics01(t01f, p01f), "nifti")
                        else:
                            MISSING_FLAGS.append(f"{exp}: {subj} missing flair sidecars")
                exp_n = len(cohort)
                print(f"  [C-3] {exp}: n={n_found}/{exp_n}")
                if n_found < exp_n:
                    MISSING_FLAGS.append(f"{method}: n={n_found} < expected {exp_n}")


# ============================================================ B. ResViT
def score_resvit():
    base = os.path.join(ROOT, "resvit", "output")
    variants = {
        "ResViT-2d-t2": ("2d", "t2"), "ResViT-2.5d-t2": ("2.5d", "t2"),
        "ResViT-2d_3d_refine-t2": ("2d_3d_refine", "t2"),
        "ResViT-full_3d-t2": ("full_3d", "t2"),
        "ResViT-2d-t2_flair": ("2d", "t2_flair"), "ResViT-2.5d-t2_flair": ("2.5d", "t2_flair"),
        "ResViT-2d_3d_refine-t2_flair": ("2d_3d_refine", "t2_flair"),
        "ResViT-full_3d-t2_flair": ("full_3d", "t2_flair"),
    }
    for vname, (arch, tgt) in variants.items():
        pred_dir = os.path.join(base, vname, "predictions")
        if not os.path.isdir(pred_dir):
            MISSING_FLAGS.append(f"ResViT missing dir: {vname}")
            continue
        cohort = FLSET if tgt == "t2_flair" else T2SET
        method = f"ResViT-{RESVIT_ARCH[arch]}-{'T2+FLAIR' if tgt=='t2_flair' else 'T2'}"
        n_found = 0
        for subj in sorted(cohort):
            sd = os.path.join(pred_dir, subj)
            p_t2 = os.path.join(sd, "pred_t2.nii.gz"); g_t2 = os.path.join(sd, "tgt_t2.nii.gz")
            if not (os.path.exists(p_t2) and os.path.exists(g_t2)):
                continue
            n_found += 1
            t01 = load01(g_t2, "m11"); p01 = load01(p_t2, "m11")
            add_persubj(method, "ResViT", "ResViT", RESVIT_ARCH[arch], tgt, "t2", subj,
                        metrics01(t01, p01), "nifti")
            if tgt == "t2_flair":
                p_fl = os.path.join(sd, "pred_fl.nii.gz"); g_fl = os.path.join(sd, "tgt_fl.nii.gz")
                if os.path.exists(p_fl) and os.path.exists(g_fl):
                    t01f = load01(g_fl, "m11"); p01f = load01(p_fl, "m11")
                    add_persubj(method, "ResViT", "ResViT", RESVIT_ARCH[arch], tgt, "flair", subj,
                                metrics01(t01f, p01f), "nifti")
                else:
                    MISSING_FLAGS.append(f"{vname}: {subj} missing flair sidecars")
        exp_n = len(cohort)
        print(f"  [ResViT] {vname}: n={n_found}/{exp_n}")
        if n_found < exp_n:
            MISSING_FLAGS.append(f"{method}: n={n_found} < expected {exp_n}")


# ============================================================ C1. SynDiff untuned (NIfTI volumes)
def score_syndiff_untuned():
    # syndiff_us_t2 (full untuned SynDiff). Volumes saved per epoch in
    # synthdiff/results/syndiff_us_t2_ep<N>/volumes/<subj>_predT2.nii.gz (already [0,1]).
    # Best epoch by SSIM from summary: ep15 (0.6968) per syndiff_us_t2_resvit_protocol_summary.csv
    best_ep = 15
    vdir = os.path.join(ROOT, "synthdiff", "results", f"syndiff_us_t2_ep{best_ep}", "volumes")
    method = "SynDiff-2D-untuned-T2"
    if not os.path.isdir(vdir):
        # fallback: pick any available
        cands = sorted(glob.glob(os.path.join(ROOT, "synthdiff", "results", "syndiff_us_t2_ep*", "volumes")))
        if cands:
            vdir = cands[-1]
            best_ep = os.path.basename(os.path.dirname(vdir)).split("ep")[-1]
        else:
            MISSING_FLAGS.append("SynDiff untuned: no volume folders")
            return None
    n_found = 0
    for subj in sorted(T2SET):
        p = os.path.join(vdir, f"{subj}_predT2.nii.gz")
        g = os.path.join(vdir, f"{subj}_gtT2.nii.gz")
        if not (os.path.exists(p) and os.path.exists(g)):
            continue
        n_found += 1
        t01 = load01(g, "01"); p01 = load01(p, "01")
        add_persubj(method, "SynDiff", "SynDiff", "2D (full SynDiff, untuned)", "t2", "t2", subj,
                    metrics01(t01, p01), f"nifti(ep{best_ep})")
    print(f"  [SynDiff untuned] ep{best_ep}: n={n_found}/{len(T2SET)}")
    if n_found < len(T2SET):
        MISSING_FLAGS.append(f"{method}: n={n_found} < {len(T2SET)}")
    return best_ep


# ============================================================ C2. SynDiff ensembles (NIfTI volumes)
def score_ensembles():
    specs = [
        # (folder, method label, target, channel-name, vol-suffix)
        ("ensemble_t2_25Dsingle_a0.5", "Ensemble-ResViT+SynDiff-2.5Dsingle-T2", "t2", "t2", "predEnsemble"),
        ("ensemble_t2_25Ddual_a0.5", "Ensemble-ResViT+SynDiff-2.5Ddual-T2", "t2", "t2", "predEnsemble"),
        ("ensemble_fl_25Ddual_a0.5", "Ensemble-ResViT+SynDiff-2.5Ddual-FLAIR", "t2_flair", "flair", "predEnsemble"),
    ]
    for folder, method, tgt, chan, suf in specs:
        vdir = os.path.join(ROOT, "synthdiff", "results", folder, "volumes")
        cohort = FLSET if tgt == "t2_flair" else T2SET
        if not os.path.isdir(vdir):
            MISSING_FLAGS.append(f"Ensemble missing dir: {folder}")
            continue
        n_found = 0
        for subj in sorted(cohort):
            p = os.path.join(vdir, f"{subj}_{suf}.nii.gz")
            g = os.path.join(vdir, f"{subj}_gt.nii.gz")
            if not (os.path.exists(p) and os.path.exists(g)):
                continue
            n_found += 1
            t01 = load01(g, "01"); p01 = load01(p, "01")
            add_persubj(method, "SynDiff", "ResViT+SynDiff", "ResViT+SynDiff ensemble (alpha=0.5)",
                        tgt, chan, subj, metrics01(t01, p01), "nifti")
        print(f"  [Ensemble] {folder}: n={n_found}/{len(cohort)}")
        if n_found < len(cohort):
            MISSING_FLAGS.append(f"{method}: n={n_found} < {len(cohort)}")


# ============================================================ C3. SynDiff CSV-only methods
def _read_csv_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(x):
    try:
        if x is None or x == "" or str(x).lower() == "nan":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _lpips_lookup_from_unified():
    """subject -> {(label): lpips_t2, lpips_flair} from unified_per_subject_final.csv."""
    path = os.path.join(ROOT, "synthdiff", "results", "unified_per_subject_final.csv")
    out = defaultdict(dict)
    for r in _read_csv_rows(path):
        out[r["experiment"]][r["subject"]] = (_f(r.get("lpips_t2")), _f(r.get("lpips_flair")))
    return out


def score_syndiff_csv():
    """Methods whose resvit-protocol per-subject CSV exists at a chosen best epoch.
    (label, family-arch-label, target, csv path rel to results/, dual?, unified-label-for-lpips-fallback)
    Best epochs chosen by SSIM (T2 channel) from each *_resvit_protocol_summary.csv;
    cross-checked against build_unified_final.py / unified_per_subject_final.csv.
    """
    R = os.path.join(ROOT, "synthdiff", "results")
    UNI = _lpips_lookup_from_unified()
    specs = [
        # T2-only paired
        ("SynDiff-2D-T2", "2D", "t2", "syndiff_us_t2_paired_resvit_protocol_ep40/per_subject.csv", False, None),
        ("SynDiff-2.5D-T2", "2.5D", "t2", "syndiff_us_t2_paired_25d_resvit_protocol_ep40/per_subject.csv", False, None),
        ("SynDiff-3D-T2", "3D", "t2", "syndiff_us_t2_paired_3d_resvit_protocol_ep140/per_subject.csv", False, None),
        ("SynDiff-3D+3D-refine-T2", "3D+3D-refine", "t2", "syndiff_us_t2_paired_3drefine_resvit_protocol_ep180_ref20/per_subject.csv", False, "SynDiff-3Drefine-T2"),
        ("SynDiff-ResViT->SynDiff-cascade-T2", "ResViT->SynDiff cascade", "t2", "syndiff_us_t2_paired_resvit_refine_resvit_protocol_ep10/per_subject.csv", False, None),
        ("SynDiff-ResViT+SynDiff-joint-T2", "ResViT+SynDiff joint", "t2", "syndiff_us_t2_joint_finetune_resvit_protocol_ep30/per_subject.csv", False, None),
        # dual
        ("SynDiff-2D-T2+FLAIR", "2D", "t2_flair", "syndiff_us_t2flair_paired_resvit_protocol_ep40/per_subject.csv", True, None),
        ("SynDiff-2.5D-T2+FLAIR", "2.5D", "t2_flair", "syndiff_us_t2flair_paired_25d_resvit_protocol_ep60/per_subject.csv", True, None),
        ("SynDiff-3D-T2+FLAIR", "3D", "t2_flair", "syndiff_us_t2flair_paired_3d_resvit_protocol_ep100/per_subject.csv", True, None),
        ("SynDiff-3D+3D-refine-T2+FLAIR", "3D+3D-refine", "t2_flair", "syndiff_us_t2flair_paired_3drefine_resvit_protocol_ep40_ref20/per_subject.csv", True, "SynDiff-3Drefine-T2_FLAIR"),
    ]
    for label, archlab, tgt, rel, dual, uni_label in specs:
        path = os.path.join(R, rel)
        if not os.path.exists(path):
            MISSING_FLAGS.append(f"SynDiff CSV missing: {rel}")
            continue
        cohort = FLSET if dual else T2SET
        rows = _read_csv_rows(path)
        by_subj = {r["subject"]: r for r in rows}
        n_t2 = 0; lp_missing = 0
        for subj in sorted(cohort):
            r = by_subj.get(subj)
            if r is None:
                continue
            n_t2 += 1
            lp = _f(r.get("lpips_t2"))
            if math.isnan(lp) and uni_label and subj in UNI.get(uni_label, {}):
                lp = UNI[uni_label][subj][0]
            if math.isnan(lp):
                lp_missing += 1
            add_persubj(label, "SynDiff", "SynDiff", archlab, tgt, "t2", subj, {
                "ssim": _f(r.get("ssim_t2")), "ssim_gauss": float("nan"),
                "psnr": _f(r.get("psnr_t2")), "mae": _f(r.get("mae_t2")), "lpips": lp,
            }, "csv:" + os.path.dirname(rel))
            if dual:
                sf = _f(r.get("ssim_flair"))
                if not math.isnan(sf):
                    lpf = _f(r.get("lpips_flair"))
                    if math.isnan(lpf) and uni_label and subj in UNI.get(uni_label, {}):
                        lpf = UNI[uni_label][subj][1]
                    add_persubj(label, "SynDiff", "SynDiff", archlab, tgt, "flair", subj, {
                        "ssim": sf, "ssim_gauss": float("nan"),
                        "psnr": _f(r.get("psnr_flair")), "mae": _f(r.get("mae_flair")), "lpips": lpf,
                    }, "csv:" + os.path.dirname(rel))
        exp_n = len(cohort)
        print(f"  [SynDiff CSV] {label}: n={n_t2}/{exp_n}  lpips_missing={lp_missing}")
        if n_t2 < exp_n:
            MISSING_FLAGS.append(f"{label}: n={n_t2} < {exp_n}")
        if lp_missing:
            MISSING_FLAGS.append(f"{label}: {lp_missing} subjects without per-subject LPIPS")


# ============================================================ aggregation
def ci95(sd, n):
    return float(1.96 * sd / math.sqrt(n)) if n >= 2 else 0.0


def summarize():
    by = defaultdict(list)
    for r in PERSUBJ:
        by[(r["method"], r["family"], r["backbone"], r["architecture"], r["target"], r["channel"])].append(r)
    out = []
    for key, rows in by.items():
        method, family, backbone, arch, tgt, chan = key
        sr = {"method": method, "family": family, "backbone": backbone,
              "architecture": arch, "target": tgt, "channel": chan, "n": len(rows)}
        for met in ("ssim", "psnr", "mae", "lpips", "ssim_gauss"):
            vals = np.array([r[met] for r in rows], dtype=np.float64)
            vals = vals[~np.isnan(vals)]
            n = len(vals)
            if n == 0:
                sr[f"{met}_mean"] = sr[f"{met}_sd"] = sr[f"{met}_ci95"] = ""
                sr[f"{met}_median"] = sr[f"{met}_min"] = sr[f"{met}_max"] = ""
                continue
            m = float(vals.mean()); sd = float(vals.std(ddof=1)) if n >= 2 else 0.0
            sr[f"{met}_mean"] = m; sr[f"{met}_sd"] = sd; sr[f"{met}_ci95"] = ci95(sd, n)
            sr[f"{met}_median"] = float(np.median(vals))
            sr[f"{met}_min"] = float(vals.min()); sr[f"{met}_max"] = float(vals.max())
        out.append(sr)
    return out


# ============================================================ benchmark joins
def parse_bench():
    """Return dict keyed by (backbone_norm, architecture_norm, target_norm) -> dict."""
    out = {}
    # 1) _bench_results.json (ResViT + SynDiff measured)
    bj = json.load(open(os.path.join(ROOT, "paper_assets", "_bench_results.json")))["results"]
    for r in bj:
        out[("json", r["backbone"], r["variant"], r["target"])] = r
    return out


def parse_compute_md():
    """Parse the main table in compute_benchmark.md -> list of dict rows."""
    path = os.path.join(ROOT, "paper_assets", "compute_benchmark.md")
    rows = []
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    in_tbl = False
    for ln in lines:
        if ln.startswith("| Backbone | Variant | Target"):
            in_tbl = True
            continue
        if in_tbl:
            if not ln.startswith("|"):
                in_tbl = False
                continue
            if set(ln.strip()) <= set("|-: "):
                continue
            cells = [c.strip().strip("*") for c in ln.strip().strip("|").split("|")]
            if len(cells) < 5:
                continue
            rows.append(cells)
    return rows


# helper to extract first integer-like number from a cell string
import re
def _first_int(s):
    m = re.search(r"[\d,]{3,}", s)
    if not m:
        return ""
    return int(m.group(0).replace(",", ""))


def _first_float(s):
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group(0)) if m else ""


def build_master(summary):
    """Left-join T2-channel summary (and a FLAIR sheet) with compute + training info."""
    md_rows = parse_compute_md()
    # md_rows columns: Backbone | Variant | Target | Gen params | Disc params | Total params | GFLOPs | Latency | VRAM | Per-volume
    # build lookup: normalized (backbone, arch, target)
    def norm_bb(b):
        b = b.replace("GAN: ", "").replace("**", "").strip()
        if b.startswith("SynDiff joint"):
            return "SynDiffJoint"
        return b
    md_lut = {}
    for c in md_rows:
        bb = norm_bb(c[0]); variant = c[1].strip(); tgt = c[2].strip()
        md_lut[(bb.lower(), variant.lower(), tgt.lower())] = c
    # training wall-clock from benchmarking_report.md (hard-coded parse of section 4 tables)
    TRAIN = {
        # GAN: family x arch -> approx
        ("GAN", "pix2pix", "2D"): "~46 min", ("GAN", "pix2pix", "2.5D"): "~46 min",
        ("GAN", "pix2pix", "2D+3D-post"): "~0.3-0.7 h (2D+refiner)", ("GAN", "pix2pix", "3D"): "~4.7-5.0 h",
        ("GAN", "CUT", "2D"): "~35-46 min", ("GAN", "CUT", "2.5D"): "~36 min",
        ("GAN", "CUT", "2D+3D-post"): "~1-1.3 h", ("GAN", "CUT", "3D"): "~3.6-4.0 h",
        ("GAN", "SwinPix2Pix", "2D"): "~70 min", ("GAN", "SwinPix2Pix", "2.5D"): "~71 min",
        ("GAN", "SwinPix2Pix", "2D+3D-post"): "~1.1-1.5 h", ("GAN", "SwinPix2Pix", "3D"): "~4.9-5.0 h",
        ("GAN", "CycleGAN", "2D"): "~4.5-5 h", ("GAN", "CycleGAN", "2.5D"): "~4.5 h",
        ("GAN", "CycleGAN", "2D+3D-post"): "~43-51 min", ("GAN", "CycleGAN", "3D"): "~9.6-12.4 h",
        # ResViT
        ("ResViT", "2D", "t2"): "~5 h", ("ResViT", "2D", "t2_flair"): "~4.3 h",
        ("ResViT", "2.5D", "t2"): "~3.85 h", ("ResViT", "2.5D", "t2_flair"): "~3.9 h",
        ("ResViT", "2D+3D-refine", "t2"): "~5.5 h", ("ResViT", "2D+3D-refine", "t2_flair"): "~7-8 h",
        ("ResViT", "3D", "t2"): "~6 h", ("ResViT", "3D", "t2_flair"): "~6.5 h",
        # SynDiff
        ("SynDiff", "2D", "t2"): "~13.1 h", ("SynDiff", "2D", "t2_flair"): "~11.2 h",
        ("SynDiff", "2.5D", "t2"): "~13.2 h", ("SynDiff", "2.5D", "t2_flair"): "~6.9 h",
        ("SynDiff", "3D", "t2"): "~5.4 h", ("SynDiff", "3D", "t2_flair"): "~4.7 h",
        ("SynDiff", "3D+3D-refine", "t2"): "~5.4 h + ~4.0 h refiner", ("SynDiff", "3D+3D-refine", "t2_flair"): "~4.7 h + ~4.1 h refiner",
        ("SynDiff", "ResViT->SynDiff cascade", "t2"): "~3.85 h ResViT + ~6.5 h SynDiff-refine",
        ("SynDiff", "ResViT+SynDiff joint", "t2"): "~3.85 h ResViT + ~4.7 h joint-finetune",
        ("SynDiff", "2D (full SynDiff, untuned)", "t2"): "~13 h (ngf64 reference)",
        ("ResViT+SynDiff", "ResViT+SynDiff ensemble (alpha=0.5)", "t2"): "(no extra training; combines ResViT-2.5D + SynDiff-2.5D)",
        ("ResViT+SynDiff", "ResViT+SynDiff ensemble (alpha=0.5)", "t2_flair"): "(no extra training)",
    }

    # ssim ranks per channel
    def rank_within(channel):
        rows = [s for s in summary if s["channel"] == channel and isinstance(s.get("ssim_mean"), float)]
        rows.sort(key=lambda r: -r["ssim_mean"])
        return {r["method"]: i + 1 for i, r in enumerate(rows)}
    ssim_rank = {"t2": rank_within("t2"), "flair": rank_within("flair")}

    master = {"t2": [], "flair": []}
    for s in summary:
        chan = s["channel"]
        # map to compute md
        bb = s["backbone"]; arch = s["architecture"]; tgt = s["target"]
        # md target token
        md_tgt = "t2" if tgt == "t2" else "t2+flair"
        # md variant: GAN uses GAN_ARCH labels (2D/2.5D/2D+3D-post/full-3D), ResViT/SynDiff similar
        # try a few candidate keys
        cand = []
        if bb == "GAN":
            # md backbone like "pix2pix" etc; variant "2D"/"2.5D"/"2D + 3D-refine"... actually GAN 3D rows weird
            cand.append((s["family"].lower(), arch.lower(), md_tgt))
            cand.append((("pix2pix-3d" if "3D" == arch and s["family"] in ("pix2pix",) else s["family"].lower()), "full-3d", md_tgt))
        elif bb == "ResViT":
            v = {"2D": "2d", "2.5D": "2.5d", "2D+3D-refine": "2d + 3d-refine", "3D": "full-3d"}[arch]
            cand.append(("resvit", v, md_tgt))
        elif bb == "SynDiff":
            v = {"2D": "2d", "2.5D": "2.5d", "3D": "full-3d", "3D+3D-refine": "2d + 3d-refine",
                 "2D (full SynDiff, untuned)": "2d"}.get(arch, arch.lower())
            cand.append(("syndiff", v, md_tgt))
        c = None
        for k in cand:
            if k in md_lut:
                c = md_lut[k]; break
        gen_p = disc_p = tot_p = gflops = infer_s = ""
        if c:
            gen_p = _first_int(c[3]); disc_p = _first_int(c[4]); tot_p = _first_int(c[5])
            gflops = _first_float(c[6]);
            # per-volume inference: cell 9
            infer_s = c[9] if len(c) > 9 else ""
        # training
        tk = None
        if bb == "GAN":
            tk = ("GAN", s["family"], arch)
        elif bb == "ResViT":
            tk = ("ResViT", arch, tgt)
        elif bb == "SynDiff":
            tk = ("SynDiff", arch, tgt)
        elif bb == "ResViT+SynDiff":
            tk = ("ResViT+SynDiff", arch, tgt)
        train_time = TRAIN.get(tk, "")
        row = {
            "method": s["method"], "backbone": bb, "architecture": arch, "target": tgt,
            "gen_params": gen_p, "disc_params": disc_p, "total_params": tot_p,
            "gflops_fwd": gflops, "train_time": train_time, "infer_s_per_vol": infer_s,
            "ssim_mean": s.get("ssim_mean"), "ssim_sd": s.get("ssim_sd"),
            "psnr_mean": s.get("psnr_mean"), "psnr_sd": s.get("psnr_sd"),
            "mae_mean": s.get("mae_mean"), "mae_sd": s.get("mae_sd"),
            "lpips_mean": s.get("lpips_mean"), "lpips_sd": s.get("lpips_sd"),
            "n": s["n"], "ssim_rank": ssim_rank[chan].get(s["method"], ""),
        }
        master["t2" if chan == "t2" else "flair"].append(row)
    for ch in master:
        master[ch].sort(key=lambda r: (r["ssim_rank"] if isinstance(r["ssim_rank"], int) else 999))
    return master


# ============================================================ Wilcoxon
def wilcoxon_tests():
    try:
        from scipy.stats import wilcoxon
    except Exception as e:
        print("[warn] scipy unavailable:", e)
        return []
    # per-subject dict: method -> {subj: {ssim,psnr,lpips}} for T2 channel
    by = defaultdict(dict)
    for r in PERSUBJ:
        if r["channel"] != "t2":
            continue
        by[r["method"]][r["subject"]] = r
    methods = list(by.keys())
    # find best-SSIM method (T2) by mean
    means = {m: np.nanmean([by[m][s]["ssim"] for s in by[m]]) for m in methods}
    best = max(means, key=means.get)
    results = []

    def do_pair(a, b, label):
        common = sorted(set(by[a]) & set(by[b]))
        if len(common) < 5:
            return
        for met in ("ssim", "psnr", "lpips"):
            xa = np.array([by[a][s][met] for s in common], dtype=float)
            xb = np.array([by[b][s][met] for s in common], dtype=float)
            mask = ~(np.isnan(xa) | np.isnan(xb))
            xa, xb = xa[mask], xb[mask]
            if len(xa) < 5 or np.allclose(xa, xb):
                results.append({"comparison": label, "metric": met, "n": len(xa),
                                "mean_A": float(np.mean(xa)) if len(xa) else "",
                                "mean_B": float(np.mean(xb)) if len(xb) else "",
                                "stat": "", "p": "", "note": "insufficient/identical"})
                continue
            try:
                st, p = wilcoxon(xa, xb)
            except Exception as ex:
                results.append({"comparison": label, "metric": met, "n": len(xa),
                                "mean_A": float(np.mean(xa)), "mean_B": float(np.mean(xb)),
                                "stat": "", "p": "", "note": f"err:{ex}"})
                continue
            results.append({"comparison": label, "metric": met, "n": len(xa),
                            "mean_A": float(np.mean(xa)), "mean_B": float(np.mean(xb)),
                            "stat": float(st), "p": float(p), "note": ""})

    # (i) every method vs best
    for m in methods:
        if m == best:
            continue
        do_pair(m, best, f"{m}  vs  BEST({best})")
    # (ii) within-backbone: each non-2D variant vs that backbone's plain 2D, T2 channel
    fam_groups = {
        "GAN-pix2pix": ("pix2pix-2D-T2", ["pix2pix-2.5D-T2", "pix2pix-2D+3D-post-T2", "pix2pix-3D-T2"]),
        "GAN-CUT": ("CUT-2D-T2", ["CUT-2.5D-T2", "CUT-2D+3D-post-T2", "CUT-3D-T2"]),
        "GAN-CycleGAN": ("CycleGAN-2D-T2", ["CycleGAN-2.5D-T2", "CycleGAN-2D+3D-post-T2", "CycleGAN-3D-T2"]),
        "GAN-SwinPix2Pix": ("SwinPix2Pix-2D-T2", ["SwinPix2Pix-2.5D-T2", "SwinPix2Pix-2D+3D-post-T2", "SwinPix2Pix-3D-T2"]),
        "ResViT": ("ResViT-2D-T2", ["ResViT-2.5D-T2", "ResViT-2D+3D-refine-T2", "ResViT-3D-T2"]),
        "SynDiff": ("SynDiff-2D-T2", ["SynDiff-2.5D-T2", "SynDiff-3D-T2", "SynDiff-3D+3D-refine-T2",
                                       "SynDiff-ResViT->SynDiff-cascade-T2", "SynDiff-ResViT+SynDiff-joint-T2"]),
    }
    for fam, (base_m, others) in fam_groups.items():
        if base_m not in by:
            continue
        for o in others:
            if o not in by:
                continue
            do_pair(o, base_m, f"[arch effect {fam}] {o}  vs  {base_m}")
    return results, best


# ============================================================ writers
def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    print("\n== A. COMPARATIVA-3 GAN baselines =="); score_comparativa3()
    print("\n== B. ResViT =="); score_resvit()
    print("\n== C1. SynDiff untuned =="); score_syndiff_untuned()
    print("\n== C2. Ensembles =="); score_ensembles()
    print("\n== C3. SynDiff CSV-only methods =="); score_syndiff_csv()

    # per-subject
    ps_fields = ["method", "family", "backbone", "architecture", "target", "channel",
                 "subject", "ssim", "ssim_gauss", "psnr", "mae", "lpips", "src"]
    write_csv(os.path.join(OUT, "rescore_methods_persubject.csv"), PERSUBJ, ps_fields)

    # summary
    summary = summarize()
    sm_fields = ["method", "family", "backbone", "architecture", "target", "channel", "n"]
    for met in ("ssim", "psnr", "mae", "lpips", "ssim_gauss"):
        for stat in ("mean", "sd", "ci95", "median", "min", "max"):
            sm_fields.append(f"{met}_{stat}")
    # sort summary: t2 channel by ssim desc, then flair
    summary.sort(key=lambda r: (0 if r["channel"] == "t2" else 1,
                                -(r["ssim_mean"] if isinstance(r.get("ssim_mean"), float) else -1)))
    write_csv(os.path.join(OUT, "rescore_methods_summary.csv"), summary, sm_fields)

    # rankings
    def make_ranking(channel):
        rows = [s for s in summary if s["channel"] == channel and isinstance(s.get("ssim_mean"), float)]
        rows = sorted(rows, key=lambda r: -r["ssim_mean"])
        # compute psnr/mae/lpips ranks
        def rk(key, reverse):
            order = sorted(rows, key=lambda r: (r.get(key) if isinstance(r.get(key), float) else (1e9 if not reverse else -1e9)),
                           reverse=reverse)
            return {id(r): i + 1 for i, r in enumerate(order)}
        psnr_r = rk("psnr_mean", True); mae_r = rk("mae_mean", False); lp_r = rk("lpips_mean", False)
        out = []
        for i, r in enumerate(rows):
            out.append({
                "ssim_rank": i + 1, "method": r["method"], "family": r["family"],
                "backbone": r["backbone"], "architecture": r["architecture"], "target": r["target"],
                "n": r["n"], "ssim_mean": r["ssim_mean"], "ssim_sd": r["ssim_sd"], "ssim_ci95": r["ssim_ci95"],
                "psnr_mean": r["psnr_mean"], "psnr_rank": psnr_r[id(r)],
                "mae_mean": r["mae_mean"], "mae_rank": mae_r[id(r)],
                "lpips_mean": r.get("lpips_mean"), "lpips_rank": lp_r[id(r)],
            })
        return out
    rank_t2 = make_ranking("t2"); rank_fl = make_ranking("flair")
    rk_fields = ["ssim_rank", "method", "family", "backbone", "architecture", "target", "n",
                 "ssim_mean", "ssim_sd", "ssim_ci95", "psnr_mean", "psnr_rank",
                 "mae_mean", "mae_rank", "lpips_mean", "lpips_rank"]
    write_csv(os.path.join(OUT, "rescore_methods_rankings_t2.csv"), rank_t2, rk_fields)
    write_csv(os.path.join(OUT, "rescore_methods_rankings_flair.csv"), rank_fl, rk_fields)

    # wilcoxon
    wres = wilcoxon_tests()
    if wres:
        wr, best_m = wres
        write_csv(os.path.join(OUT, "rescore_wilcoxon.csv"), wr,
                  ["comparison", "metric", "n", "mean_A", "mean_B", "stat", "p", "note"])
    else:
        wr, best_m = [], "?"

    # master benchmark
    master = build_master(summary)
    mb_fields = ["method", "backbone", "architecture", "target", "gen_params", "disc_params",
                 "total_params", "gflops_fwd", "train_time", "infer_s_per_vol",
                 "ssim_mean", "ssim_sd", "psnr_mean", "psnr_sd", "mae_mean", "mae_sd",
                 "lpips_mean", "lpips_sd", "n", "ssim_rank"]
    write_csv(os.path.join(OUT, "master_benchmark_table.csv"), master["t2"], mb_fields)
    write_csv(os.path.join(OUT, "master_benchmark_table_flair.csv"), master["flair"], mb_fields)

    # xlsx (optional)
    try:
        import openpyxl
        from openpyxl import Workbook
        wb = Workbook(); wb.remove(wb.active)
        def add_sheet(name, rows, fields):
            ws = wb.create_sheet(name[:31]); ws.append(fields)
            for r in rows:
                ws.append([r.get(c, "") for c in fields])
        add_sheet("per_subject", PERSUBJ, ps_fields)
        add_sheet("summary", summary, sm_fields)
        add_sheet("ranking_t2", rank_t2, rk_fields)
        add_sheet("ranking_flair", rank_fl, rk_fields)
        add_sheet("wilcoxon", wr, ["comparison", "metric", "n", "mean_A", "mean_B", "stat", "p", "note"])
        add_sheet("master_t2", master["t2"], mb_fields)
        add_sheet("master_flair", master["flair"], mb_fields)
        wb.save(os.path.join(OUT, "rescore_methods.xlsx"))
        print("wrote rescore_methods.xlsx")
    except Exception as e:
        print("[info] skipping xlsx:", e)

    # README
    write_readme(summary, rank_t2, rank_fl, wr, best_m)

    # console summary
    print("\n" + "=" * 80)
    print("EXPERIMENTS SCORED (channel=t2 unless flair-only):")
    seen = set()
    for s in summary:
        k = (s["method"], s["channel"])
        print(f"  {s['method']:<42} channel={s['channel']:<6} n={s['n']:<3}  "
              f"SSIM={s.get('ssim_mean',''):.4f}" if isinstance(s.get('ssim_mean'), float) else f"  {s['method']}")
    print("\n--- T2 RANKING ---")
    for r in rank_t2:
        print(f"{r['ssim_rank']:>3}. {r['method']:<42} SSIM={r['ssim_mean']:.4f}  PSNR={r['psnr_mean']:.2f}  "
              f"MAE={r['mae_mean']:.4f}  LPIPS={'%.4f'%r['lpips_mean'] if isinstance(r['lpips_mean'],float) else 'NA'}  n={r['n']}")
    print("\n--- FLAIR RANKING ---")
    for r in rank_fl:
        print(f"{r['ssim_rank']:>3}. {r['method']:<42} SSIM={r['ssim_mean']:.4f}  PSNR={r['psnr_mean']:.2f}  "
              f"MAE={r['mae_mean']:.4f}  LPIPS={'%.4f'%r['lpips_mean'] if isinstance(r['lpips_mean'],float) else 'NA'}  n={r['n']}")
    print(f"\n--- KEY WILCOXON (best method = {best_m}) ---")
    for r in wr:
        if r["metric"] == "ssim" and r["note"] == "":
            print(f"  {r['comparison']:<60} ssim n={r['n']} stat={r['stat']:.1f} p={r['p']:.4g}")
    if MISSING_FLAGS:
        print("\n--- FLAGS ---")
        for m in MISSING_FLAGS:
            print("  *", m)


def write_readme(summary, rank_t2, rank_fl, wr, best_m):
    p = os.path.join(OUT, "RESCORE_README.md")
    L = []
    L.append("# Unified Re-Score of US->MRI Synthesis Methods\n")
    L.append("Generated by `rescore_all.py`. All metrics computed with ONE identical harness "
             "(the *resvit-protocol*, lifted verbatim from `resvit/eval_resvit_metrics.py`).\n")
    L.append("## Metric protocol\n")
    L.append("- Volumes loaded in [-1,1] (or [0,1] for SynDiff/ensemble NIfTI which are stored pre-scaled); "
             "everything mapped to `[0,1]` before metrics.\n"
             "- **SSIM**: per axial slice z, skip if `mean(t01[:,:,z] > 0.025) < 0.01`; "
             "`skimage.structural_similarity(t_sl, p_sl, data_range=1.0)` with the skimage default uniform window; "
             "SSIM = mean over kept slices. An extra `ssim_gauss` column (Gaussian window, sigma=1.5, "
             "`use_sample_covariance=False`) is computed for NIfTI-recomputed methods only.\n"
             "- **PSNR**: foreground `t01 > 0.025`; if <100 voxels skip; `mse = mean((t01[fg]-p01[fg])^2)`; "
             "`psnr = 10*log10(1/mse)` (data_range = 1).\n"
             "- **MAE**: `mean(|t01[fg] - p01[fg]|)`.\n"
             "- **LPIPS**: AlexNet (`lpips.LPIPS(net='alex')`, cuda, no_grad), per kept axial slice (same fg-slice "
             "criterion), slice mapped to [-1,1] and replicated to 3 channels; LPIPS = mean over kept slices.\n")
    L.append("\n## Cohorts\n")
    L.append(f"- T2 cohort: n={len(t2_cohort)} subjects (the `_t2` variants).\n"
             f"- FLAIR cohort: n={len(flair_cohort)} subjects with FLAIR GT; the `_t2_flair` (dual) variants are "
             "scored on these 20 subjects for BOTH the T2 channel and the FLAIR channel.\n"
             "- `n` is reported per (method, channel); any method missing cohort subjects is flagged below.\n")
    L.append("\n## Methods included (NO ablations)\n")
    L.append("**A. COMPARATIVA-3 GAN baselines** — `COMPARATIVA-3/<exp>/predictions/*.nii.gz`, recomputed from NIfTI: "
             "`{pix2pix,cut,cyclegan,swinpix2pix} x {2d->2D, 25d->2.5D, 2d_3dpost->2D+3D-post, 3d->3D} x {t2, t2_flair}` "
             "(32 experiments). The `predictions_one/` axial-only variants exist but were EXCLUDED.\n")
    L.append("**B. ResViT** — `resvit/output/ResViT-<variant>-<target>/predictions/<subj>/{pred_t2,tgt_t2,pred_fl,tgt_fl}.nii.gz`, "
             "recomputed from NIfTI: `{2d->2D, 2.5d->2.5D, 2d_3d_refine->2D+3D-refine, full_3d->3D} x {t2, t2_flair}` "
             "(8 experiments). EXCLUDED: `output/ablation/*`, `*_fullres`, `*_mrspace`, `*_eval_staging*`.\n")
    L.append("**C. SynDiff** — `synthdiff/results/`. The `*_resvit_protocol_ep*` folders contain only `per_subject.csv` "
             "(produced by `eval_resvit_protocol.py`, which mirrors `eval_resvit_metrics.py` bit-for-bit), NOT NIfTI; "
             "for those we read the per-subject rows at the chosen best epoch. The untuned full SynDiff and the ensembles "
             "DO have saved NIfTI volumes and were recomputed from scratch.\n")
    L.append("\nBest-epoch / best-config choices for SynDiff (by max SSIM on the T2 channel, cross-checked against "
             "`synthdiff/results/unified_per_subject_final.csv` and `synthdiff/build_unified_final.py`):\n")
    L.append("| SynDiff method | label | architecture label | chosen ckpt | source |\n|---|---|---|---|---|\n"
             "| syndiff_us_t2_paired | SynDiff-2D-T2 | 2D | ep40 | csv (LPIPS in csv) |\n"
             "| syndiff_us_t2_paired_25d | SynDiff-2.5D-T2 | 2.5D | ep40 | csv |\n"
             "| syndiff_us_t2_paired_3d | SynDiff-3D-T2 | 3D | ep140 | csv |\n"
             "| syndiff_us_t2_paired_3drefine | SynDiff-3D+3D-refine-T2 | 3D+3D-refine | ep180_ref20 | csv; per-subject LPIPS from unified_per_subject_final.csv |\n"
             "| syndiff_us_t2_paired_resvit_refine | SynDiff-ResViT->SynDiff-cascade-T2 | ResViT->SynDiff cascade | ep10 (refiner) | csv |\n"
             "| syndiff_us_t2_joint_finetune | SynDiff-ResViT+SynDiff-joint-T2 | ResViT+SynDiff joint | ep30 | csv |\n"
             "| syndiff_us_t2 (untuned full SynDiff, ngf64) | SynDiff-2D-untuned-T2 | 2D (full SynDiff, untuned) | ep15 | recomputed from saved NIfTI volumes |\n"
             "| syndiff_us_t2flair_paired | SynDiff-2D-T2+FLAIR | 2D | ep40 | csv |\n"
             "| syndiff_us_t2flair_paired_25d | SynDiff-2.5D-T2+FLAIR | 2.5D | ep60 | csv |\n"
             "| syndiff_us_t2flair_paired_3d | SynDiff-3D-T2+FLAIR | 3D | ep100 | csv |\n"
             "| syndiff_us_t2flair_paired_3drefine | SynDiff-3D+3D-refine-T2+FLAIR | 3D+3D-refine | ep40_ref20 | csv; per-subject LPIPS from unified_per_subject_final.csv |\n"
             "| ensemble_t2_25Dsingle_a0.5 | Ensemble-ResViT+SynDiff-2.5Dsingle-T2 | ResViT+SynDiff ensemble (alpha=0.5) | alpha=0.5 | recomputed from saved NIfTI volumes |\n"
             "| ensemble_t2_25Ddual_a0.5 | Ensemble-ResViT+SynDiff-2.5Ddual-T2 | ResViT+SynDiff ensemble (alpha=0.5) | alpha=0.5 | recomputed from NIfTI |\n"
             "| ensemble_fl_25Ddual_a0.5 | Ensemble-ResViT+SynDiff-2.5Ddual-FLAIR | ResViT+SynDiff ensemble (alpha=0.5) | alpha=0.5 | recomputed from NIfTI (FLAIR channel) |\n")
    L.append("\n### Label conventions\n")
    L.append("- GAN families capitalised (pix2pix, CUT, CycleGAN, SwinPix2Pix); architectures: 2D / 2.5D / 2D+3D-post / 3D.\n"
             "- ResViT architectures: 2D / 2.5D / 2D+3D-refine / 3D.\n"
             "- SynDiff: paired->2D, paired_25d->2.5D, paired_3d->3D, paired_3drefine->'3D+3D-refine' (3D diffusion + 3D ResNet refiner), "
             "paired_resvit_refine->'ResViT->SynDiff cascade', joint_finetune->'ResViT+SynDiff joint', "
             "syndiff_us_t2->'2D (full SynDiff, untuned)' (ngf64 bidirectional reference, not tuned), "
             "ensembles->'ResViT+SynDiff ensemble (alpha=0.5)'.\n"
             "- Method names append '-T2' or '-T2+FLAIR' for the trained target; dual methods produce both a t2-channel "
             "and a flair-channel row.\n")
    L.append("\n## Top of rankings (T2 channel, all methods, sorted by SSIM)\n")
    L.append("| # | method | family | architecture | SSIM | PSNR | MAE | LPIPS | n |\n|---|---|---|---|---|---|---|---|---|\n")
    for r in rank_t2:
        lp = "%.4f" % r["lpips_mean"] if isinstance(r["lpips_mean"], float) else "NA"
        L.append(f"| {r['ssim_rank']} | {r['method']} | {r['family']} | {r['architecture']} | "
                 f"{r['ssim_mean']:.4f} | {r['psnr_mean']:.2f} | {r['mae_mean']:.4f} | {lp} | {r['n']} |\n")
    L.append("\n## Top of rankings (FLAIR channel, dual methods, sorted by SSIM)\n")
    L.append("| # | method | family | architecture | SSIM | PSNR | MAE | LPIPS | n |\n|---|---|---|---|---|---|---|---|---|\n")
    for r in rank_fl:
        lp = "%.4f" % r["lpips_mean"] if isinstance(r["lpips_mean"], float) else "NA"
        L.append(f"| {r['ssim_rank']} | {r['method']} | {r['family']} | {r['architecture']} | "
                 f"{r['ssim_mean']:.4f} | {r['psnr_mean']:.2f} | {r['mae_mean']:.4f} | {lp} | {r['n']} |\n")
    L.append(f"\n## Wilcoxon signed-rank tests (per-subject, T2 channel)\n")
    L.append(f"Best-SSIM method (T2): **{best_m}**. Full results in `rescore_wilcoxon.csv` "
             "(comparison (i): every method vs the best; (ii): within-backbone arch-effect, each non-2D variant vs the "
             "backbone's plain 2D). Pairs use only subjects present in both members. SSIM-row highlights:\n")
    L.append("| comparison | n | mean_A | mean_B | stat | p |\n|---|---|---|---|---|---|\n")
    for r in wr:
        if r["metric"] == "ssim" and r["note"] == "" and ("arch effect" in r["comparison"] or "BEST" in r["comparison"]):
            L.append(f"| {r['comparison']} | {r['n']} | {r['mean_A']:.4f} | {r['mean_B']:.4f} | "
                     f"{r['stat']:.1f} | {r['p']:.4g} |\n")
    L.append("\n## Master benchmarking table\n")
    L.append("`master_benchmark_table.csv` (T2 channel) and `master_benchmark_table_flair.csv` left-join the summary with "
             "compute numbers parsed from `paper_assets/compute_benchmark.md` (gen/disc/total params, GFLOPs/forward, "
             "per-volume inference) and training wall-clock parsed from `paper_assets/benchmarking_report.md` section 4. "
             "GAN GFLOPs are unmeasured (TF env unavailable) and left blank; GAN per-volume inference and training times "
             "are log-derived estimates. Diffusion GFLOPs are per single forward (one 4-step sample ~= 4x).\n")
    if MISSING_FLAGS:
        L.append("\n## Missing-subject / data flags\n")
        for m in MISSING_FLAGS:
            L.append(f"- {m}\n")
    else:
        L.append("\n## Missing-subject / data flags\nNone.\n")
    L.append("\n## Outputs in this folder\n"
             "- `rescore_methods_persubject.csv` — one row per (method, channel, subject).\n"
             "- `rescore_methods_summary.csv` — mean/sd/ci95/median/min/max/n per (method, channel) for SSIM, PSNR, MAE, LPIPS (+ssim_gauss).\n"
             "- `rescore_methods_rankings_t2.csv`, `rescore_methods_rankings_flair.csv` — sorted by SSIM, with PSNR/MAE/LPIPS ranks.\n"
             "- `rescore_wilcoxon.csv` — paired Wilcoxon signed-rank results.\n"
             "- `master_benchmark_table.csv`, `master_benchmark_table_flair.csv` — metrics + compute + training.\n"
             "- `rescore_methods.xlsx` — all of the above as sheets (only if openpyxl is available).\n"
             "- `rescore_all.py` — the script that produced everything here.\n")
    with open(p, "w", encoding="utf-8") as f:
        f.write("".join(L))
    print("wrote RESCORE_README.md")


if __name__ == "__main__":
    main()
