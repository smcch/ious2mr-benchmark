# -*- coding: utf-8 -*-
"""Multiplanar figure (axial, coronal, sagittal) for one held-out test study.

Columns: input ioUS | synthetic T2w from four configurations (Pix2Pix 2D+3D-refine, the
top-SSIM GAN; ResViT 2D, 2D+3D-refine and full-3D, the three regimes of the best downstream
family) | real T2w. Each plane passes through the centroid of the reference lesion. Overlays:
reference lesion (yellow) on every panel; the lesion predicted by the frozen MRI-trained Seg-T2
nnU-Net (cyan) on every MR panel, synthetic or real. All panels are restricted to the
ultrasound cone; every MR panel (synthetic and real) uses the same display rule (window on the
2nd-99.5th percentile inside the cone). Volumes are reoriented to RAS and shown with their
physical aspect ratio (in-plane ~0.94 mm, slices 2 mm).

Usage:  python make_fig_multiplanar.py [STUDY] [--out PATH] [--dpi N]
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import center_of_mass

SRC = Path(os.environ.get("IOUS2MR_SOURCE_TREE", r"E:\SINTESIS"))
DS = SRC / "downstream_seg"
COLS = [("ioUS input", None),
        ("Pix2Pix 2D+3D", "GAN-pix2pix-2D+3D-post-T2-from-single"),
        ("ResViT 2D", "ResViT-2D-T2-from-single"),
        ("ResViT 2D+3D", "ResViT-2D+3D-refine-T2-from-single"),
        ("ResViT full-3D", "ResViT-3D-T2-from-single"),
        ("real T2w", "REAL_T2")]
REF_C, PRED_C = "#ffd400", "#00e5ff"


def canon(p):
    img = nib.as_closest_canonical(nib.load(str(p)))
    return np.asarray(img.dataobj, dtype=np.float32), img.header.get_zooms()[:3]


def gt_path(s):
    for n in (f"{s}-mri-segmentation.nii.gz", f"{s}-mir-segmentation.nii.gz"):
        if (SRC / "Segmentations" / n).exists():
            return SRC / "Segmentations" / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study", nargs="?", default="ReMIND-045-pre")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dpi", type=int, default=110)
    a = ap.parse_args()
    s = a.study

    us, zooms = canon(SRC / "dataset-registration-corrected-cropped" / "US" / f"{s}-us.nii.gz")
    cone = us > 0
    gt = np.isin(canon(gt_path(s))[0], (1, 2))
    cx, cy, cz = (int(round(v)) for v in center_of_mass(gt))
    dx, dy, dz = zooms

    vols, preds = [], []
    for _, key in COLS:
        if key is None:
            vols.append(us); preds.append(None); continue
        v, _ = canon(DS / "test_inputs_T2" / key / f"{s}_0000.nii.gz")
        p, _ = canon(DS / "predictions_T2" / key / f"{s}.nii.gz")
        assert v.shape == us.shape == p.shape, (key, v.shape, us.shape)
        vols.append(v); preds.append(np.isin(p, (1, 2)))

    def disp(v):
        lo, hi = np.percentile(v[cone], (2, 99.5))
        out = np.clip((v - lo) / (hi - lo + 1e-6), 0, 1)
        out[~cone] = 0
        return out

    planes = [("axial", lambda v: np.rot90(v[:, :, cz]), dy / dx),
              ("coronal", lambda v: np.rot90(v[:, cy, :]), dz / dx),
              ("sagittal", lambda v: np.rot90(v[cx, :, :]), dz / dy)]
    # crop each plane to the cone's bounding box so panels are not mostly black
    fig, ax = plt.subplots(3, len(COLS), figsize=(2.05 * len(COLS), 2.05 * 3))
    for r, (pname, cut, aspect) in enumerate(planes):
        m = cut(cone)
        rows, cols_ = np.nonzero(m)
        r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols_.min(), cols_.max() + 1
        for c, ((title, key), v, pr) in enumerate(zip(COLS, vols, preds)):
            A = ax[r, c]
            img = cut(disp(v))[r0:r1, c0:c1]
            A.imshow(img, cmap="gray", vmin=0, vmax=1, aspect=aspect, interpolation="bilinear")
            A.contour(cut(gt.astype(float))[r0:r1, c0:c1], levels=[0.5], colors=[REF_C], linewidths=1.6)
            if pr is not None:
                A.contour(cut(pr.astype(float))[r0:r1, c0:c1], levels=[0.5], colors=[PRED_C],
                          linewidths=1.6)
            A.set_xticks([]); A.set_yticks([])
            for sp in A.spines.values():
                sp.set_visible(False)
            if r == 0:
                A.set_title(title, fontsize=15)
            if c == 0:
                A.set_ylabel(pname, fontsize=15)
    fig.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.5)
    out = Path(a.out) if a.out else Path(__file__).with_name(f"multiplanar_{s}.png")
    fig.savefig(out, dpi=a.dpi, facecolor="white")
    print("written", out, "| centroid (RAS voxel)", (cx, cy, cz), "| zooms", tuple(round(z, 2) for z in zooms))


if __name__ == "__main__":
    main()
