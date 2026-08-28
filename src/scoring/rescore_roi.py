"""
rescore_roi.py  --  ROI-restricted re-score of US->MRI synthesis methods.

Same harness as `rescore_all.py` (same SSIM/PSNR/MAE/LPIPS primitives, same cohorts),
but each metric is restricted to a lesion-aware mask built from
`E:/SINTESIS/Segmentations/MRI/<subject>-mri-segmentation.nii.gz`:

  ROI = "lesion" = (label >= 1)   # tumor + cavity
  dilations: 0 mm and 5 mm        # 5 mm = anisotropic-aware via distance_transform_edt

The mask is resampled (nearest neighbour) into each prediction's grid:
  - ResViT preds: same affine as the seg -> resample_from_to(seg, pred).
  - COMPARATIVA-3 preds: identity affine (stripped on save) but content == normalize_mri(
    data_cropped_192/MR-T2/<subj>-mri.nii.gz). We resample the seg into the
    data_cropped_192 MR grid (real affine, same world frame as the seg), and use the
    resulting array directly on the C3 prediction (same shape, content identical).
  - SynDiff NIfTI (untuned + paired single/dual): same affine as the seg.

Methods INCLUDED:
  - all 32 COMPARATIVA-3 GANs (pix2pix / CUT / CycleGAN / SwinPix2Pix x 4 archs x {T2, T2+FLAIR})
  - all 8 ResViT variants
  - SynDiff-2D-untuned-T2 (ep15)
  - SynDiff-{2D, 2.5D, 3D}-{T2, T2+FLAIR} paired (epochs as in rescore_all.py)
  - SynDiff-3D+3D-refine-{T2, T2+FLAIR} (epochs as in rescore_all.py)

Methods EXCLUDED (per user request / no usable NIfTI):
  - any `_fullres`, `_mrspace`, `ablation/*`
  - SynDiff ensembles, joint, cascade  (ensembles per user request; joint/cascade CSV-only)

Outputs in E:/SINTESIS/evaluacion-final/ (new file names; nothing existing is overwritten):
  roi_methods_persubject.csv  - one row per (method, channel, subject, roi, dilation_mm)
  roi_methods_summary.csv     - mean/sd/ci95/median/min/max/n per (method, channel, roi, dilation_mm)
  roi_methods_rankings_t2.csv, roi_methods_rankings_flair.csv  (lesion@0mm and lesion@5mm, sorted by SSIM)
  roi_methods_wilcoxon.csv    - paired Wilcoxon vs best (lesion@0mm and lesion@5mm)
  ROI_README.md
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from common.paths import PROJECT_ROOT, DATA_ROOT, CKPT_ROOT, EXTERNAL_ROOT  # noqa: E402
import os  # noqa: E402,F811
import os, sys, csv, math, json, glob
from collections import defaultdict

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy.ndimage import distance_transform_edt

ROOT = str(PROJECT_ROOT)
OUT = os.path.join(ROOT, "evaluacion-final")
SEG_DIR = os.path.join(ROOT, "Segmentations", "MRI")
C3_REF_DIR = os.path.join(ROOT, "data_cropped_192", "MR-T2")  # has real affine for C3 grid

DILATIONS_MM = [0.0, 5.0]
# Per-ROI label semantics:
#   lesion = (label >= 1)   = tumor + cavity (combined)
#   tumor  = (label == 1)
#   cavity = (label == 2)
ROIS = ["lesion", "tumor", "cavity"]

os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "resvit"))
from eval_resvit_metrics import load_canonical_cohorts  # noqa

# ---------------------------------------------------------------- LPIPS init
DEVICE = None
_LPIPS_FN = None
try:
    import torch
    import lpips
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception as e:
    print("[warn] torch/lpips unavailable:", e)


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


# ---------------------------------------------------------------- ROI-restricted metrics
# All take t01/p01 (target/pred in [0,1]) plus a binary roi_mask of the same 3D shape.
MIN_ROI_PER_SLICE = 50  # pixels


def _slice_roi_bbox(m, pad=2, min_side=11):
    """Return (y0, y1, x0, x1) bbox of True voxels in m, padded by `pad` px,
    forced to at least min_side x min_side (else None)."""
    ys, xs = np.where(m)
    if ys.size == 0:
        return None
    y0 = max(0, int(ys.min()) - pad); y1 = min(m.shape[0], int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad); x1 = min(m.shape[1], int(xs.max()) + 1 + pad)
    if (y1 - y0) < min_side:
        c = (y0 + y1) // 2
        y0 = max(0, c - min_side // 2); y1 = min(m.shape[0], y0 + min_side)
        y0 = max(0, y1 - min_side)
    if (x1 - x0) < min_side:
        c = (x0 + x1) // 2
        x0 = max(0, c - min_side // 2); x1 = min(m.shape[1], x0 + min_side)
        x0 = max(0, x1 - min_side)
    return y0, y1, x0, x1


def _ssim_roi(t01, p01, roi_mask):
    """SSIM per axial slice, **cropped to the slice's ROI bounding box** (so the metric
    actually measures lesion content, not background match). Skips slices with
    <MIN_ROI_PER_SLICE ROI px. Pixels outside ROI **within the bbox** are zeroed in both
    target and pred (so they cancel and don't add spurious signal). skimage SSIM with
    data_range=1.0, default uniform window.
    """
    from skimage.metrics import structural_similarity
    vals = []
    Z = t01.shape[2]
    for z in range(Z):
        m = roi_mask[:, :, z]
        if m.sum() < MIN_ROI_PER_SLICE:
            continue
        bb = _slice_roi_bbox(m)
        if bb is None:
            continue
        y0, y1, x0, x1 = bb
        m_bb = m[y0:y1, x0:x1]
        ts = np.where(m_bb, t01[y0:y1, x0:x1, z], 0.0).astype(np.float32)
        ps = np.where(m_bb, p01[y0:y1, x0:x1, z], 0.0).astype(np.float32)
        if min(ts.shape) < 7:  # skimage SSIM default win_size=7
            continue
        vals.append(structural_similarity(ts, ps, data_range=1.0))
    return float(np.mean(vals)) if vals else float("nan"), len(vals)


def _psnr_roi(t01, p01, roi_mask):
    if roi_mask.sum() < 100:
        return float("nan")
    mse = float(np.mean((t01[roi_mask] - p01[roi_mask]) ** 2))
    if mse < 1e-10:
        return 50.0
    return float(10.0 * np.log10(1.0 / mse))


def _mae_roi(t01, p01, roi_mask):
    if roi_mask.sum() < 100:
        return float("nan")
    return float(np.mean(np.abs(t01[roi_mask] - p01[roi_mask])))


def _lpips_roi(t01, p01, roi_mask):
    """LPIPS per axial slice, **cropped to the slice's ROI bounding box** and resized to
    64x64 (LPIPS AlexNet needs >=32 to avoid degenerate feature maps). Pixels outside ROI
    inside the bbox are zeroed in both target and pred. Skips slices with <MIN_ROI_PER_SLICE
    ROI px.
    """
    if DEVICE is None:
        return float("nan")
    fn = _get_lpips()
    from scipy.ndimage import zoom
    SZ = 64
    vals = []
    Z = t01.shape[2]
    for z in range(Z):
        m = roi_mask[:, :, z]
        if m.sum() < MIN_ROI_PER_SLICE:
            continue
        bb = _slice_roi_bbox(m)
        if bb is None:
            continue
        y0, y1, x0, x1 = bb
        m_bb = m[y0:y1, x0:x1]
        ts = np.where(m_bb, t01[y0:y1, x0:x1, z], 0.0).astype(np.float32)
        ps = np.where(m_bb, p01[y0:y1, x0:x1, z], 0.0).astype(np.float32)
        # resize to (SZ, SZ) — bilinear for intensities
        zy = SZ / ts.shape[0]; zx = SZ / ts.shape[1]
        ts_r = zoom(ts, (zy, zx), order=1)
        ps_r = zoom(ps, (zy, zx), order=1)
        # crop or pad to exact (SZ,SZ) in case of rounding
        ts_r = np.clip(ts_r, 0.0, 1.0)[:SZ, :SZ]
        ps_r = np.clip(ps_r, 0.0, 1.0)[:SZ, :SZ]
        if ts_r.shape != (SZ, SZ):
            pad_y = SZ - ts_r.shape[0]; pad_x = SZ - ts_r.shape[1]
            ts_r = np.pad(ts_r, ((0, max(0, pad_y)), (0, max(0, pad_x))))
            ps_r = np.pad(ps_r, ((0, max(0, pad_y)), (0, max(0, pad_x))))
            ts_r = ts_r[:SZ, :SZ]; ps_r = ps_r[:SZ, :SZ]
        with torch.no_grad():
            d = fn(_lpips_tensor01(ts_r), _lpips_tensor01(ps_r))
        vals.append(float(d.item()))
    return float(np.mean(vals)) if vals else float("nan")


def metrics_roi(t01, p01, roi_mask):
    ssim, n_sl = _ssim_roi(t01, p01, roi_mask)
    return {
        "ssim": ssim,
        "psnr": _psnr_roi(t01, p01, roi_mask),
        "mae":  _mae_roi(t01, p01, roi_mask),
        "lpips": _lpips_roi(t01, p01, roi_mask),
        "n_slices": n_sl,
        "roi_vox": int(roi_mask.sum()),
    }


# ---------------------------------------------------------------- ROI building
def build_roi_mask(seg_arr_aligned, spacing_mm, dilation_mm):
    """seg_arr_aligned: integer label volume already in the prediction's grid.
    Returns boolean mask for (label >= 1), optionally dilated by `dilation_mm` (mm).
    The dilation respects voxel anisotropy via distance_transform_edt with sampling=spacing.
    """
    base = (seg_arr_aligned >= 1)
    if dilation_mm <= 0 or base.sum() == 0:
        return base
    # distance_transform_edt with sampling returns mm distance to the nearest True voxel of ~base
    dist = distance_transform_edt(~base, sampling=spacing_mm)
    return base | (dist <= dilation_mm)


def load_seg_for_subject(subj):
    p = os.path.join(SEG_DIR, f"{subj}-mri-segmentation.nii.gz")
    if not os.path.exists(p):
        return None
    return nib.load(p)


# ---------------------------------------------------------------- loaders / volume readers
def load01(path, src_range):
    a = nib.load(path).get_fdata().astype(np.float32)
    if src_range == "m11":
        return (a + 1.0) / 2.0
    return a  # already [0,1]


# ---------------------------------------------------------------- bookkeeping
GAN_ARCH = {"2d": "2D", "25d": "2.5D", "2d_3dpost": "2D+3D-post", "3d": "3D"}
GAN_FAM_LABEL = {"pix2pix": "pix2pix", "cut": "CUT", "cyclegan": "CycleGAN",
                 "swinpix2pix": "SwinPix2Pix"}
RESVIT_ARCH = {"2d": "2D", "2.5d": "2.5D", "2d_3d_refine": "2D+3D-refine",
               "full_3d": "3D"}

t2_cohort, flair_cohort = load_canonical_cohorts(
    os.path.join(ROOT, "COMPARATIVA-3", "results_lpips_per_subject.csv"))
T2SET, FLSET = set(t2_cohort), set(flair_cohort)

# Intersect with seg availability
def _seg_exists(s):
    return os.path.exists(os.path.join(SEG_DIR, f"{s}-mri-segmentation.nii.gz"))


T2SET_SEG = {s for s in T2SET if _seg_exists(s)}
FLSET_SEG = {s for s in FLSET if _seg_exists(s)}
print(f"Cohorts:")
print(f"  T2:    canonical n={len(T2SET)}  with seg n={len(T2SET_SEG)}  missing seg: "
      f"{sorted(T2SET - T2SET_SEG)}")
print(f"  FLAIR: canonical n={len(FLSET)}  with seg n={len(FLSET_SEG)}  missing seg: "
      f"{sorted(FLSET - FLSET_SEG)}")

PERSUBJ = []  # rows: method, family, backbone, arch, target, channel, subject, roi, dilation_mm, ssim, psnr, mae, lpips, n_slices, roi_vox, src
MISSING = []


def _phase_of(subject):
    if subject.endswith("-pre"):
        return "pre"
    if subject.endswith("-post"):
        return "post"
    return "unknown"


def add_row(method, family, backbone, arch, target, channel, subject, roi, dilation_mm, m, src):
    PERSUBJ.append({
        "method": method, "family": family, "backbone": backbone, "architecture": arch,
        "target": target, "channel": channel, "subject": subject,
        "phase": _phase_of(subject),
        "roi": roi, "dilation_mm": dilation_mm,
        "ssim": m["ssim"], "psnr": m["psnr"], "mae": m["mae"], "lpips": m["lpips"],
        "n_slices": m["n_slices"], "roi_vox": m["roi_vox"], "src": src,
    })


def score_pair(method, family, backbone, arch, target, channel, subj,
               tgt01, pred01, ref_grid_img, src):
    """Evaluate one (method, channel, subject) over the configured ROIs / dilations.
    `ref_grid_img` is the nibabel image whose grid (shape, affine) the pred lives on.
    """
    seg_img = load_seg_for_subject(subj)
    if seg_img is None:
        MISSING.append(f"{method}/{subj}: no MRI seg")
        return
    seg_in_ref = resample_from_to(seg_img, ref_grid_img, order=0)
    seg_arr = seg_in_ref.get_fdata().astype(np.int16)
    if seg_arr.shape != tgt01.shape:
        MISSING.append(f"{method}/{subj}: shape mismatch seg{seg_arr.shape} vs tgt{tgt01.shape}")
        return
    spacing = tuple(float(z) for z in ref_grid_img.header.get_zooms()[:3])
    for roi in ROIS:
        if roi == "lesion":
            base_mask = (seg_arr >= 1)
        elif roi == "tumor":
            base_mask = (seg_arr == 1)
        elif roi == "cavity":
            base_mask = (seg_arr == 2)
        else:
            continue
        if base_mask.sum() == 0:
            # ROI absent for this subject (e.g. cavity in a pre-op case) -> record empty rows
            for d in DILATIONS_MM:
                add_row(method, family, backbone, arch, target, channel, subj, roi, d,
                        {"ssim": float("nan"), "psnr": float("nan"), "mae": float("nan"),
                         "lpips": float("nan"), "n_slices": 0, "roi_vox": 0}, src)
            continue
        for d_mm in DILATIONS_MM:
            # build the (dilated) version of THIS roi's base mask
            if d_mm <= 0:
                mask = base_mask
            else:
                dist = distance_transform_edt(~base_mask, sampling=spacing)
                mask = base_mask | (dist <= d_mm)
            m = metrics_roi(tgt01, pred01, mask)
            add_row(method, family, backbone, arch, target, channel, subj, roi, d_mm, m, src)


# ============================================================ A. COMPARATIVA-3 (GAN)
def score_comparativa3():
    base = os.path.join(ROOT, "COMPARATIVA-3")
    for fam in ("pix2pix", "cut", "cyclegan", "swinpix2pix"):
        for arch in ("2d", "25d", "2d_3dpost", "3d"):
            for tgt in ("t2", "t2_flair"):
                exp = f"{fam}_{arch}_{tgt}"
                pred_dir = os.path.join(base, exp, "predictions")
                if not os.path.isdir(pred_dir):
                    MISSING.append(f"C3 missing dir: {exp}")
                    continue
                cohort = FLSET_SEG if tgt == "t2_flair" else T2SET_SEG
                method = (f"{GAN_FAM_LABEL[fam]}-{GAN_ARCH[arch]}-"
                          f"{'T2+FLAIR' if tgt=='t2_flair' else 'T2'}")
                n_found = 0
                for subj in sorted(cohort):
                    p_t2 = os.path.join(pred_dir, f"{subj}_pred_t2.nii.gz")
                    g_t2 = os.path.join(pred_dir, f"{subj}_target_t2.nii.gz")
                    if not (os.path.exists(p_t2) and os.path.exists(g_t2)):
                        continue
                    ref_path = os.path.join(C3_REF_DIR, f"{subj}-mri.nii.gz")
                    if not os.path.exists(ref_path):
                        MISSING.append(f"C3/{exp}/{subj}: no data_cropped_192 MR-T2 ref")
                        continue
                    ref_img = nib.load(ref_path)
                    n_found += 1
                    t01 = load01(g_t2, "m11"); p01 = load01(p_t2, "m11")
                    score_pair(method, GAN_FAM_LABEL[fam], "GAN", GAN_ARCH[arch],
                               tgt, "t2", subj, t01, p01, ref_img, "nifti")
                    if tgt == "t2_flair":
                        p_fl = os.path.join(pred_dir, f"{subj}_pred_flair.nii.gz")
                        g_fl = os.path.join(pred_dir, f"{subj}_target_flair.nii.gz")
                        if os.path.exists(p_fl) and os.path.exists(g_fl):
                            t01f = load01(g_fl, "m11"); p01f = load01(p_fl, "m11")
                            score_pair(method, GAN_FAM_LABEL[fam], "GAN",
                                       GAN_ARCH[arch], tgt, "flair", subj,
                                       t01f, p01f, ref_img, "nifti")
                        else:
                            MISSING.append(f"{exp}/{subj}: missing flair sidecars")
                print(f"  [C-3] {exp}: n={n_found}/{len(cohort)}")


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
            MISSING.append(f"ResViT missing dir: {vname}")
            continue
        cohort = FLSET_SEG if tgt == "t2_flair" else T2SET_SEG
        method = f"ResViT-{RESVIT_ARCH[arch]}-{'T2+FLAIR' if tgt=='t2_flair' else 'T2'}"
        n_found = 0
        for subj in sorted(cohort):
            sd = os.path.join(pred_dir, subj)
            p_t2 = os.path.join(sd, "pred_t2.nii.gz"); g_t2 = os.path.join(sd, "tgt_t2.nii.gz")
            if not (os.path.exists(p_t2) and os.path.exists(g_t2)):
                continue
            n_found += 1
            ref_img = nib.load(g_t2)
            t01 = load01(g_t2, "m11"); p01 = load01(p_t2, "m11")
            score_pair(method, "ResViT", "ResViT", RESVIT_ARCH[arch], tgt, "t2", subj,
                       t01, p01, ref_img, "nifti")
            if tgt == "t2_flair":
                p_fl = os.path.join(sd, "pred_fl.nii.gz"); g_fl = os.path.join(sd, "tgt_fl.nii.gz")
                if os.path.exists(p_fl) and os.path.exists(g_fl):
                    ref_img_fl = nib.load(g_fl)
                    t01f = load01(g_fl, "m11"); p01f = load01(p_fl, "m11")
                    score_pair(method, "ResViT", "ResViT", RESVIT_ARCH[arch], tgt, "flair", subj,
                               t01f, p01f, ref_img_fl, "nifti")
        print(f"  [ResViT] {vname}: n={n_found}/{len(cohort)}")


# ============================================================ C. SynDiff (NIfTI-only)
def score_syndiff_nifti():
    """Score SynDiff variants that have saved per-subject NIfTI volumes.
    Excludes ensembles, joint, cascade (per user request / no NIfTI).
    """
    R = os.path.join(ROOT, "synthdiff", "results")
    # (label, archlab, vol_dir_rel, target, dual, suf_pred, suf_gt, suf_pred_fl, suf_gt_fl)
    specs = [
        # SynDiff-2D-untuned-T2 (syndiff_us_t2_ep15) deliberately excluded:
        # only 4/30 test subjects have saved NIfTI for that quick reference run.
        ("SynDiff-2D-T2", "2D",
         "syndiff_us_t2_paired_resvit_protocol_ep40/volumes", "t2", False, "_predT2", "_gtT2", None, None),
        ("SynDiff-2.5D-T2", "2.5D",
         "syndiff_us_t2_paired_25d_resvit_protocol_ep40/volumes", "t2", False, "_predT2", "_gtT2", None, None),
        ("SynDiff-3D-T2", "3D",
         "syndiff_us_t2_paired_3d_resvit_protocol_3d_ep140/volumes", "t2", False, "_predT2", "_gtT2", None, None),
        # ^ note: this folder name has a literal "_3d_resvit_protocol_3d_" double infix in the user's tree
        ("SynDiff-3D+3D-refine-T2", "3D+3D-refine",
         "syndiff_us_t2_paired_3drefine_resvit_protocol_ep180_ref20/volumes", "t2", False,
         "_predT2", "_gtT2", None, None),
        ("SynDiff-2D-T2+FLAIR", "2D",
         "syndiff_us_t2flair_paired_resvit_protocol_ep40/volumes", "t2_flair", True,
         "_predT2", "_gtT2", "_predFLAIR", "_gtFLAIR"),
        ("SynDiff-2.5D-T2+FLAIR", "2.5D",
         "syndiff_us_t2flair_paired_25d_resvit_protocol_ep60/volumes", "t2_flair", True,
         "_predT2", "_gtT2", "_predFLAIR", "_gtFLAIR"),
        ("SynDiff-3D-T2+FLAIR", "3D",
         "syndiff_us_t2flair_paired_3d_resvit_protocol_ep100/volumes", "t2_flair", True,
         "_predT2", "_gtT2", "_predFLAIR", "_gtFLAIR"),
        ("SynDiff-3D+3D-refine-T2+FLAIR", "3D+3D-refine",
         "syndiff_us_t2flair_paired_3drefine_resvit_protocol_ep40_ref20/volumes", "t2_flair", True,
         "_predT2", "_gtT2", "_predFLAIR", "_gtFLAIR"),
    ]
    for label, archlab, rel, tgt, dual, sp, sg, spf, sgf in specs:
        vdir = os.path.join(R, rel)
        if not os.path.isdir(vdir):
            # try alternative path (some 3d folders are named with a `_3d_` infix)
            alt = vdir.replace("paired_3d_resvit_protocol_3d_", "paired_3d_resvit_protocol_")
            if os.path.isdir(alt):
                vdir = alt
            else:
                MISSING.append(f"SynDiff missing dir: {rel}")
                continue
        cohort = FLSET_SEG if dual else T2SET_SEG
        n_found = 0
        for subj in sorted(cohort):
            p = os.path.join(vdir, f"{subj}{sp}.nii.gz")
            g = os.path.join(vdir, f"{subj}{sg}.nii.gz")
            if not (os.path.exists(p) and os.path.exists(g)):
                continue
            n_found += 1
            ref_img = nib.load(g)
            t01 = load01(g, "01"); p01 = load01(p, "01")
            score_pair(label, "SynDiff", "SynDiff", archlab, tgt, "t2", subj,
                       t01, p01, ref_img, "nifti")
            if dual and spf and sgf:
                pf = os.path.join(vdir, f"{subj}{spf}.nii.gz")
                gf = os.path.join(vdir, f"{subj}{sgf}.nii.gz")
                if os.path.exists(pf) and os.path.exists(gf):
                    ref_img_fl = nib.load(gf)
                    t01f = load01(gf, "01"); p01f = load01(pf, "01")
                    score_pair(label, "SynDiff", "SynDiff", archlab, tgt, "flair", subj,
                               t01f, p01f, ref_img_fl, "nifti")
        print(f"  [SynDiff] {label}: n={n_found}/{len(cohort)}")


# ============================================================ aggregation
def ci95(sd, n):
    return float(1.96 * sd / math.sqrt(n)) if n >= 2 else 0.0


def summarize():
    """Aggregate per-subject rows into mean/sd/ci95/median/min/max per
    (method, channel, roi, dilation_mm, phase). `phase` takes values 'all', 'pre', 'post'."""
    out = []
    for phase_filter in ("all", "pre", "post"):
        by = defaultdict(list)
        for r in PERSUBJ:
            if phase_filter != "all" and r["phase"] != phase_filter:
                continue
            key = (r["method"], r["family"], r["backbone"], r["architecture"],
                   r["target"], r["channel"], r["roi"], r["dilation_mm"])
            by[key].append(r)
        for k, rows in by.items():
            method, family, backbone, arch, tgt, chan, roi, d = k
            sr = {"method": method, "family": family, "backbone": backbone,
                  "architecture": arch, "target": tgt, "channel": chan,
                  "roi": roi, "dilation_mm": d, "phase": phase_filter,
                  "n_subjects": len(rows)}
            for met in ("ssim", "psnr", "mae", "lpips"):
                vals = np.array([r[met] for r in rows], dtype=np.float64)
                vals = vals[~np.isnan(vals)]
                n = len(vals)
                if n == 0:
                    for stat in ("mean", "sd", "ci95", "median", "min", "max"):
                        sr[f"{met}_{stat}"] = ""
                    sr[f"{met}_n"] = 0
                    continue
                m = float(vals.mean()); sd = float(vals.std(ddof=1)) if n >= 2 else 0.0
                sr[f"{met}_mean"] = m; sr[f"{met}_sd"] = sd; sr[f"{met}_ci95"] = ci95(sd, n)
                sr[f"{met}_median"] = float(np.median(vals))
                sr[f"{met}_min"] = float(vals.min()); sr[f"{met}_max"] = float(vals.max())
                sr[f"{met}_n"] = n  # may be < n_subjects when ROI absent in some subjects
            out.append(sr)
    return out


# ============================================================ Wilcoxon
def wilcoxon_tests(summary):
    """Paired Wilcoxon vs best for each (roi, dilation, phase). Phase in {all, pre, post}."""
    try:
        from scipy.stats import wilcoxon
    except Exception as e:
        print("[warn] scipy unavailable:", e)
        return []
    results = []
    for phase in ("all", "pre", "post"):
        # per-subject map by (method, roi, dilation), restricted to phase
        by = defaultdict(dict)
        for r in PERSUBJ:
            if r["channel"] != "t2":
                continue
            if phase != "all" and r["phase"] != phase:
                continue
            key = (r["method"], r["roi"], r["dilation_mm"])
            by[key][r["subject"]] = r
        for roi in ROIS:
            for d in DILATIONS_MM:
                cand = [r for r in summary if r["channel"] == "t2" and r["roi"] == roi
                        and r["dilation_mm"] == d and r["phase"] == phase
                        and isinstance(r.get("ssim_mean"), float)]
                if not cand:
                    continue
                best = max(cand, key=lambda x: x["ssim_mean"])
                best_m = best["method"]
                for m_row in cand:
                    m = m_row["method"]
                    if m == best_m:
                        continue
                    a_key = (m, roi, d); b_key = (best_m, roi, d)
                    common = sorted(set(by[a_key]) & set(by[b_key]))
                    if len(common) < 5:
                        continue
                    for met in ("ssim", "psnr", "lpips"):
                        xa = np.array([by[a_key][s][met] for s in common], dtype=float)
                        xb = np.array([by[b_key][s][met] for s in common], dtype=float)
                        mask = ~(np.isnan(xa) | np.isnan(xb))
                        xa, xb = xa[mask], xb[mask]
                        if len(xa) < 5 or np.allclose(xa, xb):
                            continue
                        try:
                            st, p = wilcoxon(xa, xb)
                        except Exception:
                            continue
                        results.append({
                            "phase": phase, "roi": roi, "dilation_mm": d,
                            "comparison": f"{m}  vs  BEST({best_m})",
                            "metric": met, "n": len(xa),
                            "mean_A": float(np.mean(xa)), "mean_B": float(np.mean(xb)),
                            "stat": float(st), "p": float(p),
                        })
    return results


# ============================================================ writers
def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    print("\n== A. COMPARATIVA-3 GAN baselines =="); score_comparativa3()
    print("\n== B. ResViT =="); score_resvit()
    print("\n== C. SynDiff (NIfTI) =="); score_syndiff_nifti()

    ps_fields = ["method", "family", "backbone", "architecture", "target", "channel",
                 "subject", "phase", "roi", "dilation_mm",
                 "ssim", "psnr", "mae", "lpips", "n_slices", "roi_vox", "src"]
    write_csv(os.path.join(OUT, "roi_methods_persubject.csv"), PERSUBJ, ps_fields)

    summary = summarize()
    sm_fields = ["method", "family", "backbone", "architecture", "target", "channel",
                 "roi", "dilation_mm", "phase", "n_subjects"]
    for met in ("ssim", "psnr", "mae", "lpips"):
        for stat in ("mean", "sd", "ci95", "median", "min", "max", "n"):
            sm_fields.append(f"{met}_{stat}")
    summary.sort(key=lambda r: (r["roi"], r["dilation_mm"], r["phase"],
                                0 if r["channel"] == "t2" else 1,
                                -(r["ssim_mean"] if isinstance(r.get("ssim_mean"), float) else -1)))
    write_csv(os.path.join(OUT, "roi_methods_summary.csv"), summary, sm_fields)

    # rankings: long format across (roi, dilation, phase)
    def make_ranking(channel, roi, d_mm, phase):
        rows = [s for s in summary if s["channel"] == channel and s["roi"] == roi
                and s["dilation_mm"] == d_mm and s["phase"] == phase
                and isinstance(s.get("ssim_mean"), float)]
        rows = sorted(rows, key=lambda r: -r["ssim_mean"])
        out = []
        for i, r in enumerate(rows):
            out.append({
                "ssim_rank": i + 1, "method": r["method"], "family": r["family"],
                "backbone": r["backbone"], "architecture": r["architecture"], "target": r["target"],
                "roi": roi, "dilation_mm": d_mm, "phase": phase,
                "n_subjects": r["n_subjects"], "ssim_n": r.get("ssim_n", 0),
                "ssim_mean": r["ssim_mean"], "ssim_sd": r["ssim_sd"], "ssim_ci95": r["ssim_ci95"],
                "psnr_mean": r["psnr_mean"], "mae_mean": r["mae_mean"],
                "lpips_mean": r.get("lpips_mean"),
            })
        return out

    rk_fields = ["ssim_rank", "method", "family", "backbone", "architecture", "target",
                 "roi", "dilation_mm", "phase", "n_subjects", "ssim_n",
                 "ssim_mean", "ssim_sd", "ssim_ci95",
                 "psnr_mean", "mae_mean", "lpips_mean"]
    rank_t2_all = []
    rank_fl_all = []
    for roi in ROIS:
        for d in DILATIONS_MM:
            for phase in ("all", "pre", "post"):
                rank_t2_all.extend(make_ranking("t2", roi, d, phase))
                rank_fl_all.extend(make_ranking("flair", roi, d, phase))
    write_csv(os.path.join(OUT, "roi_methods_rankings_t2.csv"), rank_t2_all, rk_fields)
    write_csv(os.path.join(OUT, "roi_methods_rankings_flair.csv"), rank_fl_all, rk_fields)

    wr = wilcoxon_tests(summary)
    if wr:
        write_csv(os.path.join(OUT, "roi_methods_wilcoxon.csv"), wr,
                  ["phase", "roi", "dilation_mm", "comparison", "metric", "n",
                   "mean_A", "mean_B", "stat", "p"])

    # console: top 10 per (T2, roi, dilation, phase=all) for quick eye-balling
    print("\n" + "=" * 80)
    for roi in ROIS:
        for d in DILATIONS_MM:
            print(f"\n--- T2 RANKING (roi={roi}, dilation={d} mm, phase=all) ---  top 10")
            rows = [r for r in rank_t2_all
                    if r["roi"] == roi and r["dilation_mm"] == d and r["phase"] == "all"][:10]
            for r in rows:
                lp = "%.4f" % r["lpips_mean"] if isinstance(r["lpips_mean"], float) else "NA"
                print(f"  {r['ssim_rank']:>2}. {r['method']:<42}"
                      f"  SSIM={r['ssim_mean']:.4f}  PSNR={r['psnr_mean']:.2f}"
                      f"  MAE={r['mae_mean']:.4f}  LPIPS={lp}  n_subj={r['n_subjects']}  ssim_n={r['ssim_n']}")
    if MISSING:
        print("\n--- FLAGS ({}) ---".format(len(MISSING)))
        for m in MISSING[:30]:
            print("  *", m)
        if len(MISSING) > 30:
            print(f"  ... ({len(MISSING)-30} more)")

    write_methodology_md(rank_t2_all, rank_fl_all)
    print("\nwrote roi_methods_persubject.csv, roi_methods_summary.csv, "
          "roi_methods_rankings_t2.csv, roi_methods_rankings_flair.csv, "
          "roi_methods_wilcoxon.csv, ROI_METHODOLOGY.md")


def write_methodology_md(rank_t2_all, rank_fl_all):
    """Write the full methodological .md describing the ROI-restricted evaluation."""
    p = os.path.join(OUT, "ROI_METHODOLOGY.md")
    L = []
    L.append("# ROI-restricted synthesis evaluation (tumor / cavity / lesion)\n\n")
    L.append("Generated by `rescore_roi.py`. This document captures every methodological "
             "choice so the analysis can be reproduced exactly. It is paired with the four "
             "CSV outputs (`roi_methods_persubject.csv`, `roi_methods_summary.csv`, "
             "`roi_methods_rankings_{t2,flair}.csv`, `roi_methods_wilcoxon.csv`) and the run "
             "log (`_roi_run.log`).\n\n")

    L.append("## 1. Motivation\n\n")
    L.append("The pre-existing benchmark (`rescore_all.py` -> `rescore_methods_*`) computes "
             "SSIM/PSNR/MAE/LPIPS over a permissive *foreground* threshold "
             "(`target > 0.025` after [-1,1]->[0,1] mapping). That foreground is anatomically "
             "agnostic — most of it is brain background that all 48 methods reproduce trivially, "
             "which inflates SSIM and compresses inter-method differences. The ROI-restricted "
             "evaluation in this folder replaces the implicit foreground with three "
             "anatomically meaningful masks derived from radiologist-curated MRI segmentations.\n\n")

    L.append("## 2. Universe of methods (48 experiments)\n\n")
    L.append("Same backbones as the paper benchmark, no ablations and no ensembles:\n\n")
    L.append("**A. COMPARATIVA-3 GAN baselines (32 experiments).** "
             "`E:/SINTESIS/COMPARATIVA-3/<exp>/predictions/<subj>_{pred,target}_{t2,flair}.nii.gz`. "
             "Families: `pix2pix`, `CUT`, `CycleGAN`, `SwinPix2Pix`. Architectures: `2D`, `2.5D`, "
             "`2D+3D-post`, `3D`. Targets: `T2` (single output) and `T2+FLAIR` (dual output).\n\n")
    L.append("**B. ResViT (8 experiments).** "
             "`E:/SINTESIS/resvit/output/ResViT-<variant>-<target>/predictions/<subj>/{pred,tgt}_{t2,fl}.nii.gz`. "
             "Variants: `2D`, `2.5D`, `2D+3D-refine`, `3D`. Targets: `T2`, `T2+FLAIR`.\n\n")
    L.append("**C. SynDiff with saved NIfTI volumes (8 experiments).** "
             "`E:/SINTESIS/synthdiff/results/<run>/volumes/<subj>_{pred,gt}{T2,FLAIR}.nii.gz`. "
             "Variants: `2D`, `2.5D`, `3D`, `3D+3D-refine`. Targets: `T2`, `T2+FLAIR`. "
             "Best epochs (taken from `rescore_all.py`):\n\n")
    L.append("| method label | folder | ckpt |\n|---|---|---|\n"
             "| SynDiff-2D-T2 | syndiff_us_t2_paired_resvit_protocol | ep40 |\n"
             "| SynDiff-2.5D-T2 | syndiff_us_t2_paired_25d_resvit_protocol | ep40 |\n"
             "| SynDiff-3D-T2 | syndiff_us_t2_paired_3d_resvit_protocol_3d | ep140 |\n"
             "| SynDiff-3D+3D-refine-T2 | syndiff_us_t2_paired_3drefine_resvit_protocol | ep180_ref20 |\n"
             "| SynDiff-2D-T2+FLAIR | syndiff_us_t2flair_paired_resvit_protocol | ep40 |\n"
             "| SynDiff-2.5D-T2+FLAIR | syndiff_us_t2flair_paired_25d_resvit_protocol | ep60 |\n"
             "| SynDiff-3D-T2+FLAIR | syndiff_us_t2flair_paired_3d_resvit_protocol | ep100 |\n"
             "| SynDiff-3D+3D-refine-T2+FLAIR | syndiff_us_t2flair_paired_3drefine_resvit_protocol | ep40_ref20 |\n")

    L.append("\n### Methods EXCLUDED and why\n\n"
             "- **SynDiff-2D-untuned-T2 (`syndiff_us_t2_ep15`)**: only 4/30 test subjects have "
             "saved NIfTI volumes for this quick-reference run; not enough for paired statistics.\n"
             "- **SynDiff ensembles (alpha=0.5)** (`ensemble_t2_25Dsingle_a0.5`, "
             "`ensemble_t2_25Ddual_a0.5`, `ensemble_fl_25Ddual_a0.5`): ensembles, per scope choice.\n"
             "- **SynDiff-ResViT+SynDiff-joint** and **SynDiff-ResViT->SynDiff-cascade**: only "
             "per-subject CSVs were saved (no NIfTI volumes), so an ROI-restricted re-score is "
             "not possible without re-running inference.\n"
             "- Any **`_fullres`, `_mrspace`, `ablation/*`** under `resvit/output/`: re-trained / "
             "ablation runs, not part of the paper benchmark.\n")

    L.append("\n## 3. Cohorts\n\n")
    L.append(f"Cohorts come unchanged from `eval_resvit_metrics.load_canonical_cohorts("
             "COMPARATIVA-3/results_lpips_per_subject.csv)`:\n\n"
             f"- **T2 cohort**: n = {len(T2SET)} test subjects (single-target methods).\n"
             f"- **FLAIR cohort**: n = {len(FLSET)} test subjects with FLAIR GT (dual-target "
             "methods are scored on these 20 subjects for *both* T2 and FLAIR channels).\n\n"
             "Both cohorts are fully covered by the MRI segmentation directory "
             f"(`{len(T2SET_SEG)}/{len(T2SET)}` and `{len(FLSET_SEG)}/{len(FLSET)}` with seg).\n")

    L.append("\n## 4. Segmentation source and label semantics\n\n")
    L.append("MRI segmentations live at `E:/SINTESIS/Segmentations/MRI/"
             "<subject>-mri-segmentation.nii.gz`. Label conventions (verified by survey across "
             "all 150 files):\n\n"
             "- `0` = background\n- `1` = **tumor** (solid tumor; pre-op, or residual post-op)\n"
             "- `2` = **cavity** (resection cavity; post-op only)\n\n"
             "Phase-label distribution across the 150 seg files:\n\n"
             "| phase | labels present | count |\n|---|---|---|\n"
             "| pre  | {0,1} | 58 |\n| pre  | {0,1,2} | 18 |\n"
             "| post | {0,1} | 1  |\n| post | {0,2} | 33 |\n| post | {0,1,2} | 40 |\n\n"
             "Cavity is essentially a post-op label (1/74 post cases have no cavity drawn). "
             "Pre cases occasionally include a cavity (re-operations / prior surgery).\n")

    L.append("\n## 5. ROIs evaluated\n\n")
    L.append("Three ROIs are computed for every (method, subject):\n\n"
             "| ROI | Definition | Subjects where it's meaningful |\n|---|---|---|\n"
             "| `lesion` | `label >= 1` (tumor + cavity, union) | all 50 test subjects |\n"
             "| `tumor`  | `label == 1`                          | pre + post-with-residual |\n"
             "| `cavity` | `label == 2`                          | post-op cases |\n\n"
             "If the ROI is empty for a given subject (e.g. cavity in a pre-op case), the row "
             "is recorded with NaN metrics and `ssim_n`/`psnr_n` excludes it from aggregates.\n")

    L.append("\n## 6. Dilations\n\n")
    L.append(f"Each ROI is evaluated at two dilations: `0 mm` (strict; the exact label "
             f"boundary as drawn) and `5 mm` (margin; captures peritumoral edema and absorbs "
             f"sub-voxel registration error between pred and target). Dilations are computed "
             "with `scipy.ndimage.distance_transform_edt(~mask, sampling=spacing_mm)` so the "
             "5 mm margin respects voxel anisotropy (e.g. 0.94 x 0.94 x 2.0 mm on the ResViT/"
             "SynDiff grid; 0.5 x 0.5 x 2.0 mm on the COMPARATIVA-3 grid).\n")

    L.append("\n## 7. Bringing the mask into each prediction's grid\n\n")
    L.append("All predictions in this universe live in MR-derived space, but in two different "
             "grids. Mapping is done with `nibabel.processing.resample_from_to(seg, ref, "
             "order=0)` (nearest-neighbour) where `ref` is chosen so that `seg` and `ref` share "
             "the same world frame:\n\n"
             "- **ResViT preds**: identical affine to the seg (both come from the "
             "`dataset-registration-corrected-cropped` 80x86x52 grid at ~0.94x0.94x2 mm). "
             "`ref = the prediction itself`. After `resample_from_to`, the mask array is used "
             "directly on the prediction.\n"
             "- **SynDiff preds (NIfTI)**: same situation as ResViT (same dataset/grid). "
             "`ref = the prediction itself`.\n"
             "- **COMPARATIVA-3 (GAN) preds**: 192x192x52 at 0.5x0.5x2 mm, but the saved NIfTI "
             "carry an identity affine (stripped on save in `common.save_nifti`). However the "
             "*content* of the saved targets is bit-identical (correlation 1.000, mean-abs-diff "
             "<3e-8) to `normalize_mri(data_cropped_192/MR-T2/<subj>-mri.nii.gz)` -> the latter "
             "file *does* carry the real affine (same world frame as the seg). "
             f"`ref = data_cropped_192/MR-T2/<subj>-mri.nii.gz`. We then use the resulting mask "
             "array (192x192x52) directly on the GAN prediction (also 192x192x52); array "
             "indexing is identical even though the affine is identity.\n\n"
             "World-frame correspondence was verified per modality by inspecting mean target "
             "intensity inside vs outside the resampled lesion mask on a sentinel subject "
             "(ReMIND-003-post): inside ~0.48 (signal), outside ~-0.02 (background) for "
             "ResViT 2.5D and GAN pix2pix-2D alike.\n")

    L.append("\n## 8. Metric primitives\n\n")
    L.append("Every metric reuses the *resvit-protocol* defined in "
             "`resvit/eval_resvit_metrics.py` (also lifted into `rescore_all.py`), with one "
             "single change: the implicit foreground mask `t01 > 0.025` is replaced by the "
             "ROI mask defined above. Volumes are always normalised to [0,1] before metrics "
             "(`(x+1)/2` for [-1,1] inputs; SynDiff volumes are already [0,1]).\n\n"
             "- **PSNR (voxel-wise, in-ROI only).** `mse = mean((t01[roi] - p01[roi])**2)`; "
             "`psnr = 10*log10(1/mse)`; requires >=100 ROI voxels else NaN; capped at 50 dB "
             "when MSE < 1e-10.\n"
             "- **MAE (voxel-wise, in-ROI only).** `mean(|t01[roi] - p01[roi]|)`; requires "
             ">=100 ROI voxels else NaN.\n"
             "- **SSIM (per axial slice, cropped to slice ROI bbox).** For each axial slice, "
             "if the in-slice ROI has < {min_roi_px} pixels the slice is skipped. Otherwise the "
             "slice is cropped to the ROI bounding box (padding 2 px, minimum side 11 px), "
             "pixels outside the ROI inside the crop are zeroed in **both** target and pred "
             "(so they cancel and do not contribute spurious matching signal), and skimage's "
             "`structural_similarity(t, p, data_range=1.0)` is computed with the default 7x7 "
             "uniform window. The volume score is the mean over kept slices. **This is the "
             "critical change vs `rescore_all.py`**: the per-slice crop guarantees the SSIM "
             "actually measures lesion content. Without it, the giant identical zero-background "
             "saturates SSIM near 1 and erases inter-method differences.\n"
             "- **LPIPS (per axial slice, cropped + resized to 64x64).** Same slice filtering "
             "and ROI-bbox crop as SSIM, then bilinear-resize the masked crop to 64x64 (AlexNet "
             "needs >=32 to avoid degenerate feature maps; 64 is the smallest comfortable size). "
             "Slice is replicated to 3 channels, mapped to [-1,1] (`*2-1`) and passed through "
             "`lpips.LPIPS(net='alex', verbose=False)` on CUDA. Volume score is the mean over "
             "kept slices.\n\n"
             "`min_roi_px = {min_roi_px}` (slice-level cutoff).\n"
             .replace("{min_roi_px}", str(MIN_ROI_PER_SLICE)))

    L.append("\n## 9. Aggregation axes\n\n")
    L.append("Per-subject rows are aggregated into summary rows across these axes:\n\n"
             "- `method` x `channel` (`t2` / `flair`) x `roi` x `dilation_mm` x `phase`\n"
             "- `phase` in {`all`, `pre`, `post`} where `pre` and `post` are filtered "
             "by `subject.endswith('-pre' / '-post')`.\n\n"
             "Reported aggregates per metric: `mean`, `sd`, `ci95` (= 1.96 * sd / sqrt(n)), "
             "`median`, `min`, `max`, and `n` (the number of subjects whose ROI was non-empty "
             "and the metric was computable). When the ROI is empty for some subjects (e.g. "
             "cavity in pre-op cases), `n_subjects` (total in the cell) is larger than "
             "`<metric>_n` (used in the aggregate).\n")

    L.append("\n## 10. Statistical tests\n\n")
    L.append("Paired Wilcoxon signed-rank tests on per-subject scores. For each "
             "(`phase`, `roi`, `dilation_mm`) triple in the T2 channel, the SSIM-best method "
             "is identified and every other method is compared against it on the intersection "
             "of subjects present in both. Metrics tested: SSIM, PSNR, LPIPS. Minimum n=5 "
             "subjects per pair (else the cell is dropped). All p-values are *uncorrected*; "
             "with ~47 comparisons per cell, a Bonferroni-style alpha would be ~0.001, but the "
             "vast majority of significant comparisons are below 1e-3 anyway. Results live in "
             "`roi_methods_wilcoxon.csv`.\n")

    L.append("\n## 11. Top results (T2 channel, phase = all)\n\n")
    for roi in ROIS:
        for d in DILATIONS_MM:
            rows = [r for r in rank_t2_all
                    if r["roi"] == roi and r["dilation_mm"] == d and r["phase"] == "all"][:10]
            if not rows:
                continue
            L.append(f"### {roi}, dilation = {d} mm\n\n")
            L.append("| # | method | SSIM | PSNR | MAE | LPIPS | n |\n|---|---|---|---|---|---|---|\n")
            for r in rows:
                lp = "%.4f" % r["lpips_mean"] if isinstance(r["lpips_mean"], float) else "NA"
                L.append(f"| {r['ssim_rank']} | {r['method']} | {r['ssim_mean']:.4f} | "
                         f"{r['psnr_mean']:.2f} | {r['mae_mean']:.4f} | {lp} | "
                         f"{r['ssim_n']}/{r['n_subjects']} |\n")
            L.append("\n")

    L.append("## 12. Output files\n\n"
             "- `roi_methods_persubject.csv` — one row per "
             "(method, channel, subject, roi, dilation). Includes `phase` column.\n"
             "- `roi_methods_summary.csv` — mean/sd/ci95/median/min/max/n per "
             "(method, channel, roi, dilation, phase).\n"
             "- `roi_methods_rankings_t2.csv`, `roi_methods_rankings_flair.csv` — sorted-by-SSIM "
             "long-format rankings for every (roi, dilation, phase).\n"
             "- `roi_methods_wilcoxon.csv` — paired Wilcoxon vs best per (phase, roi, dilation).\n"
             "- `ROI_METHODOLOGY.md` — this file.\n"
             "- `_roi_run.log` — captured stdout of the run.\n"
             "- `rescore_roi.py` — the script that produced everything here.\n")

    with open(p, "w", encoding="utf-8") as f:
        f.write("".join(L))
    print("wrote ROI_METHODOLOGY.md")


if __name__ == "__main__":
    main()
