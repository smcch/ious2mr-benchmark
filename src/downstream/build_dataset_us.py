"""Build the direct-ioUS segmentation datasets (Seg-US control for the synthesis benchmark).

Dataset503_US     ioUS volume  -> MR-drawn tumour/cavity label   (primary: same supervision
                                                                  target as the evaluation)
Dataset504_USlab  ioUS volume  -> US-drawn (MR-assisted) label    (sensitivity variant)

Both use EXACTLY the 117 training studies and the 5 subject-level folds of Dataset501_T2
(Seg-T2), so the only differences from Seg-T2 are the input modality (and, for 504, the label
source).  The ioUS and the MR labels share the FOV-cropped grid, so nothing is resampled.

Also stages the 29 evaluable test ioUS volumes into test_inputs_US/REAL_US/ for inference.
"""
import json, os, shutil, sys
from pathlib import Path
import nibabel as nib
import numpy as np

ROOT = Path(os.environ.get("IOUS2MR_SOURCE_TREE", r"E:\SINTESIS"))
DS = ROOT / "downstream_seg"
RAW, PREP = DS / "nnUNet_raw", DS / "nnUNet_preprocessed"
US_DIR = ROOT / "dataset-registration-corrected-cropped" / "US"
USLAB_DIR = ROOT / "dataset-registration-corrected-cropped" / "US-seg"
SEG_DIR = ROOT / "Segmentations"
REF501_IMG = RAW / "Dataset501_T2" / "imagesTr"
REF501_SPLITS = PREP / "Dataset501_T2" / "splits_final.json"
TEST_REF = DS / "test_inputs_T2" / "REAL_T2"


def mr_label(s):
    for n in (f"{s}-mri-segmentation.nii.gz", f"{s}-mir-segmentation.nii.gz"):
        if (SEG_DIR / n).exists():
            return SEG_DIR / n
    return None


def save_like(src, dst, is_label):
    im = nib.load(str(src))
    d = np.asarray(im.dataobj)
    if is_label:
        d = np.rint(d).astype(np.uint8)
        out = nib.Nifti1Image(d, im.affine, im.header); out.set_data_dtype(np.uint8)
    else:
        d = d.astype(np.float32)
        out = nib.Nifti1Image(d, im.affine, im.header); out.set_data_dtype(np.float32)
    nib.save(out, str(dst))
    return d.shape


def build(ds_id, name, label_fn, studies):
    root = RAW / f"Dataset{ds_id}_{name}"
    imgs, lbls = root / "imagesTr", root / "labelsTr"
    imgs.mkdir(parents=True, exist_ok=True); lbls.mkdir(parents=True, exist_ok=True)
    used = []
    for s in studies:
        us = US_DIR / f"{s}-us.nii.gz"
        lb = label_fn(s)
        assert us.exists(), f"missing ioUS {s}"
        assert lb is not None and lb.exists(), f"missing label {s} for {name}"
        a, b = nib.load(str(us)), nib.load(str(lb))
        assert a.shape == b.shape, f"shape mismatch {s}: {a.shape} vs {b.shape}"
        assert np.allclose(a.affine, b.affine, atol=1e-3), f"affine mismatch {s}"
        save_like(us, imgs / f"{s}_0000.nii.gz", False)
        save_like(lb, lbls / f"{s}.nii.gz", True)
        used.append(s)
    dj = {"channel_names": {"0": "US"},
          "labels": {"background": 0, "tumor": 1, "cavity": 2},
          "numTraining": len(used), "file_ending": ".nii.gz", "name": name,
          "description": f"ReMIND ioUS -> tumour/cavity ({name}); direct-US control for the "
                         f"synthesis benchmark; same 117 studies + folds as Dataset501_T2"}
    (root / "dataset.json").write_text(json.dumps(dj, indent=2), encoding="utf-8")
    prep = PREP / f"Dataset{ds_id}_{name}"; prep.mkdir(parents=True, exist_ok=True)
    shutil.copy(REF501_SPLITS, prep / "splits_final.json")
    splits = json.loads(REF501_SPLITS.read_text())
    listed = {s for f in splits for s in f["train"] + f["val"]}
    assert listed == set(used), f"splits/studies mismatch: {listed ^ set(used)}"
    print(f"Dataset{ds_id}_{name}: {len(used)} studies, splits copied from Dataset501_T2 (5 folds)")


def main():
    studies = sorted(f.name[:-len("_0000.nii.gz")] for f in REF501_IMG.glob("*_0000.nii.gz"))
    print(f"{len(studies)} training studies from Dataset501_T2")
    build(503, "US", mr_label, studies)
    build(504, "USlab", lambda s: USLAB_DIR / f"{s}-us.nii.gz", studies)

    # test inputs (same 29 studies as REAL_T2)
    out = DS / "test_inputs_US" / "REAL_US"; out.mkdir(parents=True, exist_ok=True)
    tests = sorted(f.name[:-len("_0000.nii.gz")] for f in TEST_REF.glob("*_0000.nii.gz"))
    for s in tests:
        us = US_DIR / f"{s}-us.nii.gz"; ref = TEST_REF / f"{s}_0000.nii.gz"
        assert us.exists(), s
        assert nib.load(str(us)).shape == nib.load(str(ref)).shape, s
        save_like(us, out / f"{s}_0000.nii.gz", False)
    print(f"test_inputs_US/REAL_US: {len(tests)} studies")


if __name__ == "__main__":
    main()
