"""Stage test inputs for all methods into nnU-Net inference layout.

For every method that produces T2 (50 sets) -> test_inputs_T2/<set>/<study>_0000.nii.gz
For every method that produces FLAIR (24 sets) -> test_inputs_FLAIR/<set>/<study>_0000.nii.gz

The "_0000" suffix is the channel marker required by nnU-Net.

Spatial handling:
- ResViT / SynDiff predictions are already in the native seg/MRI grid (shape + affine match).
- GAN predictions live on a padded (192, 192, Z) grid with identity affine. Z matches
  the native depth but H,W are center-padded. We center-crop to the native H,W and
  re-attach the segmentation's affine + header so nnU-Net sees the same world space.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from common.paths import PROJECT_ROOT, DATA_ROOT, CKPT_ROOT, EXTERNAL_ROOT  # noqa: E402
import os  # noqa: E402,F811
import json, os, glob, sys
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom as ndzoom

ROOT = str(PROJECT_ROOT)
DS = os.path.join(ROOT, "downstream_seg")
SEG_DIR = os.path.join(ROOT, "Segmentations")
T2_DIR = os.path.join(ROOT, "dataset-registration-corrected-cropped", "MR-T2")
FL_DIR = os.path.join(ROOT, "dataset-registration-corrected-cropped", "MR-FLAIR")
SPLIT_JSON = os.path.join(ROOT, "resvit", "subject_split.json")

OUT_T2 = os.path.join(DS, "test_inputs_T2")
OUT_FL = os.path.join(DS, "test_inputs_FLAIR")


def seg_path(s):
    p = os.path.join(SEG_DIR, f"{s}-mri-segmentation.nii.gz")
    if os.path.exists(p):
        return p
    p2 = os.path.join(SEG_DIR, f"{s}-mir-segmentation.nii.gz")  # ReMIND-034-post typo
    return p2 if os.path.exists(p2) else None


def save_aligned(src_path, dst_path, ref_path, src_kind):
    """Save src as NIfTI aligned to ref's grid.

    src_kind:
        "native": src already on ref grid -> copy data, reaffix ref affine.
        "gan_padded": src is (192, 192, Z), identity affine -> center-crop H,W to ref shape, reaffix ref affine.
    """
    ref = nib.load(ref_path)
    img = nib.load(src_path)
    data = img.get_fdata().astype(np.float32)
    rH, rW, rZ = ref.shape

    if src_kind == "native":
        if data.shape != (rH, rW, rZ):
            print(f"  WARN native shape mismatch {data.shape} vs ref {ref.shape}: {src_path}", file=sys.stderr)
            return False
        out_data = data
    elif src_kind == "gan_padded":
        # GAN volumes live on a (192, 192, Zg) padded grid with identity affine.
        # The (H,W) axes are padded around the native (rH, rW), so we center-crop
        # them when 192 >= ref, or zoom up when ref > 192. Z can be doubled (2x
        # the native, from triplanar oversampling) -> resample by Zg/rZ along
        # depth. Background = -1 (the synth domain), bilinear interp.
        H, W, Z = data.shape
        # Step 1: handle H, W
        if H >= rH and W >= rW:
            h0 = (H - rH) // 2
            w0 = (W - rW) // 2
            cropped = data[h0:h0 + rH, w0:w0 + rW, :]
        else:
            zoom_h = rH / H
            zoom_w = rW / W
            cropped = ndzoom(data, (zoom_h, zoom_w, 1.0), order=1, cval=-1.0)
            # safety: in case rounding leaves off-by-one
            cropped = cropped[:rH, :rW, :]
            if cropped.shape[:2] != (rH, rW):
                pad_h = rH - cropped.shape[0]; pad_w = rW - cropped.shape[1]
                cropped = np.pad(cropped, ((0, max(0, pad_h)), (0, max(0, pad_w)), (0, 0)),
                                 constant_values=-1.0)
        # Step 2: handle Z
        if cropped.shape[2] == rZ:
            out_data = cropped
        else:
            zfac = rZ / cropped.shape[2]
            out_data = ndzoom(cropped, (1.0, 1.0, zfac), order=1, cval=-1.0)
            out_data = out_data[:, :, :rZ]
            if out_data.shape[2] < rZ:
                pad_z = rZ - out_data.shape[2]
                out_data = np.pad(out_data, ((0, 0), (0, 0), (0, pad_z)), constant_values=-1.0)
    elif src_kind == "real_t2_int":
        # Real MRI from disk: just reaffix its own affine (already correct), force float32
        out_data = data
    else:
        raise ValueError(src_kind)

    out = nib.Nifti1Image(out_data.astype(np.float32), ref.affine, ref.header)
    out.set_data_dtype(np.float32)
    nib.save(out, dst_path)
    return True


# --------------------------------------------------- method registry
RESVIT_VARIANTS = {
    "2D": "2d", "2.5D": "2.5d", "2D+3D-refine": "2d_3d_refine", "3D": "full_3d",
}

GAN_FAMS = ["pix2pix", "cut", "cyclegan", "swinpix2pix"]
GAN_ARCHS = ["2d", "25d", "2d_3dpost", "3d"]
GAN_ARCH_LABEL = {"2d": "2D", "25d": "2.5D", "2d_3dpost": "2D+3D-post", "3d": "3D"}

SYNDIFF_T2 = {
    "SynDiff-2D-T2":              "syndiff_us_t2_paired_resvit_protocol_ep40",
    "SynDiff-2.5D-T2":            "syndiff_us_t2_paired_25d_resvit_protocol_ep40",
    "SynDiff-3D-T2":              "syndiff_us_t2_paired_3d_resvit_protocol_ep140",
    "SynDiff-3D+3D-refine-T2":    "syndiff_us_t2_paired_3drefine_resvit_protocol_ep180_ref20",
    "SynDiff-cascade-T2":         "syndiff_us_t2_paired_resvit_refine_resvit_protocol_ep10",
    "SynDiff-joint-T2":           "syndiff_us_t2_joint_finetune_resvit_protocol_ep30",
}
SYNDIFF_DUAL = {
    "SynDiff-2D-T2+FL":           "syndiff_us_t2flair_paired_resvit_protocol_ep40",
    "SynDiff-2.5D-T2+FL":         "syndiff_us_t2flair_paired_25d_resvit_protocol_ep60",
    "SynDiff-3D-T2+FL":           "syndiff_us_t2flair_paired_3d_resvit_protocol_ep100",
    "SynDiff-3D+3D-refine-T2+FL": "syndiff_us_t2flair_paired_3drefine_resvit_protocol_ep40_ref20",
}


def main():
    split = json.load(open(SPLIT_JSON))
    test_studies = split["test"]

    # T2 cohort: any test study with seg
    t2_cohort = [s for s in test_studies if seg_path(s) is not None and os.path.exists(os.path.join(T2_DIR, f"{s}-mri.nii.gz"))]
    # FLAIR cohort: those with seg AND a real FLAIR (used as REAL reference and for size matching)
    fl_cohort = [s for s in t2_cohort if os.path.exists(os.path.join(FL_DIR, f"{s}-mri.nii.gz"))]
    print(f"T2 cohort: {len(t2_cohort)} -> {t2_cohort}")
    print(f"FLAIR cohort: {len(fl_cohort)} -> {fl_cohort}")

    # Save cohort lists for the eval step
    with open(os.path.join(DS, "test_cohort.json"), "w") as f:
        json.dump({"t2": t2_cohort, "flair": fl_cohort}, f, indent=2)

    log = {"T2": {}, "FLAIR": {}}

    def stage(set_name, kind, get_src, cohort, out_root, ref_dir):
        out_dir = os.path.join(out_root, set_name)
        os.makedirs(out_dir, exist_ok=True)
        n_ok = 0; n_miss = 0; n_skip = 0
        for s in cohort:
            src = get_src(s)
            if src is None or not os.path.exists(src):
                n_miss += 1
                continue
            ref = seg_path(s)
            dst = os.path.join(out_dir, f"{s}_0000.nii.gz")
            if os.path.exists(dst):
                n_skip += 1
                n_ok += 1
                continue
            if save_aligned(src, dst, ref, kind):
                n_ok += 1
            else:
                n_miss += 1
        return n_ok, n_miss

    # --- T2 sets ---
    print("\n=== Staging T2 sets ===")

    # REAL_T2
    n, m = stage("REAL_T2", "real_t2_int",
                 lambda s: os.path.join(T2_DIR, f"{s}-mri.nii.gz"),
                 t2_cohort, OUT_T2, T2_DIR)
    log["T2"]["REAL_T2"] = (n, m); print(f"  REAL_T2  ok={n} miss={m}")

    # GAN (T2 single + T2 channel of dual)
    for fam in GAN_FAMS:
        for arch in GAN_ARCHS:
            for tgt in ["t2", "t2_flair"]:
                base = os.path.join(ROOT, "COMPARATIVA-3", f"{fam}_{arch}_{tgt}", "predictions")
                if not os.path.isdir(base):
                    continue
                set_name = f"GAN-{fam}-{GAN_ARCH_LABEL[arch]}-T2-from-{'single' if tgt=='t2' else 'dual'}"
                n, m = stage(set_name, "gan_padded",
                             lambda s, b=base: os.path.join(b, f"{s}_pred_t2.nii.gz"),
                             t2_cohort, OUT_T2, T2_DIR)
                log["T2"][set_name] = (n, m); print(f"  {set_name}  ok={n} miss={m}")

    # ResViT (T2 single + T2 channel of dual)
    for arch_label, arch_dir in RESVIT_VARIANTS.items():
        for tgt in ["t2", "t2_flair"]:
            base = os.path.join(ROOT, "resvit", "output", f"ResViT-{arch_dir}-{tgt}", "predictions")
            if not os.path.isdir(base):
                continue
            set_name = f"ResViT-{arch_label}-T2-from-{'single' if tgt=='t2' else 'dual'}"
            n, m = stage(set_name, "native",
                         lambda s, b=base: os.path.join(b, s, "pred_t2.nii.gz"),
                         t2_cohort, OUT_T2, T2_DIR)
            log["T2"][set_name] = (n, m); print(f"  {set_name}  ok={n} miss={m}")

    # SynDiff T2 single + T2 channel of dual
    for set_name, dirn in SYNDIFF_T2.items():
        base = os.path.join(ROOT, "synthdiff", "results", dirn, "volumes")
        if not os.path.isdir(base):
            print(f"  SKIP {set_name} (missing dir)"); continue
        n, m = stage(set_name, "native",
                     lambda s, b=base: os.path.join(b, f"{s}_predT2.nii.gz"),
                     t2_cohort, OUT_T2, T2_DIR)
        log["T2"][set_name] = (n, m); print(f"  {set_name}  ok={n} miss={m}")
    for set_name, dirn in SYNDIFF_DUAL.items():
        base = os.path.join(ROOT, "synthdiff", "results", dirn, "volumes")
        if not os.path.isdir(base):
            continue
        set_t2 = set_name.replace("-T2+FL", "-T2-from-dual")
        n, m = stage(set_t2, "native",
                     lambda s, b=base: os.path.join(b, f"{s}_predT2.nii.gz"),
                     t2_cohort, OUT_T2, T2_DIR)
        log["T2"][set_t2] = (n, m); print(f"  {set_t2}  ok={n} miss={m}")

    # --- FLAIR sets ---
    print("\n=== Staging FLAIR sets ===")
    n, m = stage("REAL_FLAIR", "real_t2_int",
                 lambda s: os.path.join(FL_DIR, f"{s}-mri.nii.gz"),
                 fl_cohort, OUT_FL, FL_DIR)
    log["FLAIR"]["REAL_FLAIR"] = (n, m); print(f"  REAL_FLAIR  ok={n} miss={m}")

    # GAN dual FLAIR channel
    for fam in GAN_FAMS:
        for arch in GAN_ARCHS:
            base = os.path.join(ROOT, "COMPARATIVA-3", f"{fam}_{arch}_t2_flair", "predictions")
            if not os.path.isdir(base):
                continue
            set_name = f"GAN-{fam}-{GAN_ARCH_LABEL[arch]}-FLAIR"
            n, m = stage(set_name, "gan_padded",
                         lambda s, b=base: os.path.join(b, f"{s}_pred_flair.nii.gz"),
                         fl_cohort, OUT_FL, FL_DIR)
            log["FLAIR"][set_name] = (n, m); print(f"  {set_name}  ok={n} miss={m}")

    # ResViT dual FLAIR channel
    for arch_label, arch_dir in RESVIT_VARIANTS.items():
        base = os.path.join(ROOT, "resvit", "output", f"ResViT-{arch_dir}-t2_flair", "predictions")
        if not os.path.isdir(base):
            continue
        set_name = f"ResViT-{arch_label}-FLAIR"
        n, m = stage(set_name, "native",
                     lambda s, b=base: os.path.join(b, s, "pred_fl.nii.gz"),
                     fl_cohort, OUT_FL, FL_DIR)
        log["FLAIR"][set_name] = (n, m); print(f"  {set_name}  ok={n} miss={m}")

    # SynDiff dual FLAIR channel
    for set_name, dirn in SYNDIFF_DUAL.items():
        base = os.path.join(ROOT, "synthdiff", "results", dirn, "volumes")
        if not os.path.isdir(base):
            continue
        sn = set_name.replace("-T2+FL", "-FLAIR")
        n, m = stage(sn, "native",
                     lambda s, b=base: os.path.join(b, f"{s}_predFLAIR.nii.gz"),
                     fl_cohort, OUT_FL, FL_DIR)
        log["FLAIR"][sn] = (n, m); print(f"  {sn}  ok={n} miss={m}")

    with open(os.path.join(DS, "test_staging_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nDONE. {len(log['T2'])} T2 sets, {len(log['FLAIR'])} FLAIR sets.")


if __name__ == "__main__":
    main()
