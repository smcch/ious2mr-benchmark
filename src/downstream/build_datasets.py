"""Build nnU-Net v2 raw datasets for downstream segmentation.

Dataset501_T2:    real T2 -> tumor/cavity seg, 120 train studies, 61 subj
Dataset502_FLAIR: real FLAIR -> tumor/cavity seg, 81 train studies, 54 subj

Also writes splits_final.json with 5-fold by-subject stratification
(no patient appears in both train and val of the same fold).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from common.paths import PROJECT_ROOT, DATA_ROOT, CKPT_ROOT, EXTERNAL_ROOT  # noqa: E402
import os  # noqa: E402,F811
import json, os, shutil, sys, glob
import nibabel as nib
import numpy as np
from collections import defaultdict

ROOT = str(PROJECT_ROOT)
DS_ROOT = os.path.join(ROOT, "downstream_seg")
RAW = os.path.join(DS_ROOT, "nnUNet_raw")
PREP = os.path.join(DS_ROOT, "nnUNet_preprocessed")

SEG_DIR = os.path.join(ROOT, "Segmentations")
T2_DIR = os.path.join(ROOT, "dataset-registration-corrected-cropped", "MR-T2")
FL_DIR = os.path.join(ROOT, "dataset-registration-corrected-cropped", "MR-FLAIR")
SPLIT_JSON = os.path.join(ROOT, "resvit", "subject_split.json")


def seg_path(study):
    """Handle the ReMIND-034-post-mir typo."""
    p = os.path.join(SEG_DIR, f"{study}-mri-segmentation.nii.gz")
    if os.path.exists(p):
        return p
    p2 = os.path.join(SEG_DIR, f"{study}-mir-segmentation.nii.gz")
    if os.path.exists(p2):
        return p2
    return None


def t2_path(study):
    p = os.path.join(T2_DIR, f"{study}-mri.nii.gz")
    return p if os.path.exists(p) else None


def fl_path(study):
    p = os.path.join(FL_DIR, f"{study}-mri.nii.gz")
    return p if os.path.exists(p) else None


def copy_image(src, dst):
    """Copy NIfTI preserving header; if labels, cast to uint8 and round."""
    img = nib.load(src)
    data = img.get_fdata()
    if "label" in os.path.basename(dst).lower() or "seg" in os.path.basename(dst).lower():
        data = np.rint(data).astype(np.uint8)
        out = nib.Nifti1Image(data, img.affine, img.header)
        out.set_data_dtype(np.uint8)
    else:
        out = nib.Nifti1Image(data.astype(np.float32), img.affine, img.header)
        out.set_data_dtype(np.float32)
    nib.save(out, dst)


def make_5fold_splits_by_subject(studies, seed=42):
    """Return list of 5 {train: [studies], val: [studies]} dicts.

    Stratify by unique subject (patient ID) so pre/post of same patient
    never leak across train/val of a fold.
    """
    rng = np.random.default_rng(seed)
    subjs = sorted({s.rsplit("-", 1)[0] for s in studies})
    subjs_arr = np.array(subjs)
    rng.shuffle(subjs_arr)
    # 5 folds
    folds = np.array_split(subjs_arr, 5)
    out = []
    for k in range(5):
        val_subj = set(folds[k].tolist())
        train_subj = set(subjs) - val_subj
        out.append({
            "train": sorted([s for s in studies if s.rsplit("-", 1)[0] in train_subj]),
            "val":   sorted([s for s in studies if s.rsplit("-", 1)[0] in val_subj]),
        })
    return out


def build_dataset(dataset_id, name, modality, src_fn, train_studies, label_alias_check=True):
    out_root = os.path.join(RAW, f"Dataset{dataset_id:03d}_{name}")
    imgs = os.path.join(out_root, "imagesTr")
    lbls = os.path.join(out_root, "labelsTr")
    os.makedirs(imgs, exist_ok=True)
    os.makedirs(lbls, exist_ok=True)

    actually_used = []
    skipped = []
    for s in train_studies:
        src_img = src_fn(s)
        src_lbl = seg_path(s)
        if src_img is None or src_lbl is None:
            skipped.append((s, "img" if src_img is None else "seg"))
            continue
        # quick affine sanity check
        a, b = nib.load(src_img), nib.load(src_lbl)
        if a.shape != b.shape:
            skipped.append((s, f"shape mismatch {a.shape} vs {b.shape}"))
            continue
        dst_img = os.path.join(imgs, f"{s}_0000.nii.gz")
        dst_lbl = os.path.join(lbls, f"{s}.nii.gz")
        copy_image(src_img, dst_img)
        copy_image(src_lbl, dst_lbl)
        actually_used.append(s)

    # dataset.json
    dj = {
        "channel_names": {"0": modality},
        "labels": {"background": 0, "tumor": 1, "cavity": 2},
        "numTraining": len(actually_used),
        "file_ending": ".nii.gz",
        "name": name,
        "description": f"ReMIND {modality} -> tumor/cavity segmentation (downstream synthesis benchmark)",
    }
    with open(os.path.join(out_root, "dataset.json"), "w") as f:
        json.dump(dj, f, indent=2)

    # splits_final.json (must live in nnUNet_preprocessed/<DatasetXXX_NAME>/)
    splits = make_5fold_splits_by_subject(actually_used)
    prep_ds = os.path.join(PREP, f"Dataset{dataset_id:03d}_{name}")
    os.makedirs(prep_ds, exist_ok=True)
    with open(os.path.join(prep_ds, "splits_final.json"), "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\n=== Dataset{dataset_id:03d}_{name} ===")
    print(f"  used: {len(actually_used)}  skipped: {len(skipped)}")
    if skipped:
        print(f"  skipped detail: {skipped}")
    for k, sp in enumerate(splits):
        unique_train_subj = len({s.rsplit('-',1)[0] for s in sp['train']})
        unique_val_subj   = len({s.rsplit('-',1)[0] for s in sp['val']})
        print(f"  fold {k}: train={len(sp['train'])} ({unique_train_subj} subj) "
              f"val={len(sp['val'])} ({unique_val_subj} subj)")
    return actually_used


def main():
    split = json.load(open(SPLIT_JSON))
    train_studies = split["train"]

    # Dataset501_T2: 120/122
    print("Building Dataset501_T2 ...")
    used_t2 = build_dataset(501, "T2", "T2", t2_path, train_studies)

    # Dataset502_FLAIR: 81/122 (only studies with both FLAIR and seg)
    print("\nBuilding Dataset502_FLAIR ...")
    used_fl = build_dataset(502, "FLAIR", "FLAIR", fl_path, train_studies)

    print(f"\nFinal: T2 train={len(used_t2)}  FLAIR train={len(used_fl)}")


if __name__ == "__main__":
    main()
