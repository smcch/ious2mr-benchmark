"""Fig_overview -- benchmark overview, drawn entirely in matplotlib.

Replaces the AI-generated fig1.png (Elsevier does not allow generative-AI artwork, and the
old file carried outdated cohort numbers baked into its pixels). Every number here matches
the manuscript: 114 -> 37 excluded -> 77 subjects; 152 paired studies (102 FLAIR); split
61/16 subjects; test 30 = 16 pre + 14 post; 48 experiments; floor 0.251 / ceiling 0.662.
Thumbnails are real ReMIND test-subject data (ReMIND-003), rendered from the released
pre-processed volumes.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BASE = r"$IOUS2MR_ROOT"
DATA = os.path.join(BASE, "dataset-registration-corrected-cropped")
ASSETS = os.path.join(BASE, "paper_assets")
OUTDIR = os.path.join(ASSETS, "figures_journal_v2")
MANUS = os.path.join(BASE, "latex_manuscript", "us_sintesis", "figures")

plt.rcParams.update({"font.family": "serif", "font.size": 11})

C_GAN, C_RES, C_SYN = "#4477aa", "#d62728", "#333333"
C_BOX = "#f4f4f4"          # neutral boxes
C_EDGE = "#555555"
C_BAND = "#e9edf3"         # header band
C_DARK = "#2f2f2f"


def slc(path, frac=0.5, us=False):
    """Mid-axial slice of a pre-processed volume, display-oriented."""
    d = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    s = d[:, :, int(d.shape[2] * frac)]
    s = np.rot90(s)                       # (x,y) -> row/col display
    lo, hi = np.percentile(s[s != 0], [1, 99]) if (s != 0).any() else (0, 1)
    s = np.clip((s - lo) / max(1e-6, hi - lo), 0, 1)
    return s


def box(ax, x, y, w, h, text, fc=C_BOX, ec=C_EDGE, lw=1.1, fs=10.5, weight="normal",
        style="round,pad=0.008,rounding_size=0.012", tc="black", ls="-", title=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw,
                                ls=ls, mutation_aspect=1, zorder=2))
    ty = y + h / 2
    if title:
        ax.text(x + w / 2, y + h - 0.021, title, ha="center", va="top", fontsize=fs + 0.5,
                fontweight="bold", color=tc, zorder=3)
        ty = y + (h - 0.035) / 2
    ax.text(x + w / 2, ty, text, ha="center", va="center", fontsize=fs,
            fontweight=weight, color=tc, zorder=3, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, lw=1.6, color="#666666", style="-|>", shrink=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, lw=lw,
                                 color=color, mutation_scale=14, shrinkA=shrink,
                                 shrinkB=shrink, zorder=4))


def main():
    fig = plt.figure(figsize=(13.6, 9.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---------------- band 1 : cohort ----------------
    box(ax, 0.015, 0.855, 0.97, 0.135, "", fc=C_BAND, ec="#3b4a5f", lw=1.4)
    ax.text(0.5, 0.968, "ReMIND benchmark cohort (public, TCIA)", ha="center",
            fontsize=14.5, fontweight="bold")
    ax.text(0.5, 0.928,
            "114 screened  $\\rightarrow$  37 excluded (ioUS quality)  "
            "$\\rightarrow$  77 patients  $\\cdot$  152 paired ioUS / T2w studies "
            "(102 with FLAIR)", ha="center", fontsize=11)
    ax.text(0.5, 0.885,
            "subject-level split:  61 patients / 122 studies training  $\\cdot$  "
            "16 patients / 30 studies held-out test (16 pre- + 14 post-resection)",
            ha="center", fontsize=11)
    # corner thumbnails
    us = slc(os.path.join(DATA, "US", "ReMIND-003-pre-us.nii.gz"))
    t2 = slc(os.path.join(DATA, "MR-T2", "ReMIND-003-pre-mri.nii.gz"))
    for img, x0, lab in ((us, 0.024, "ioUS"), (t2, 0.898, "MR (T2w / FLAIR)")):
        a = fig.add_axes([x0, 0.862, 0.078, 0.100])
        a.imshow(img, cmap="gray", aspect="auto")
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_color("white"); s.set_lw(2)
        a.set_title(lab, fontsize=9.5, pad=2)

    arrow(ax, 0.5, 0.853, 0.5, 0.826)

    # ---------------- band 2 : pre-processing ----------------
    ax.text(0.015, 0.812, "Pre-processing (Section 2.2)", fontsize=11.5,
            style="italic")
    steps = [
        ("DICOM $\\rightarrow$ NIfTI", 0.015, 0.115),
        ("Rigid co-registration\n(LC$^2$; ioUS as reference)", 0.148, 0.175),
        ("ioUS resampled to\nthe MR voxel grid", 0.341, 0.16),
        ("FOV crop to the\nioUS cone", 0.519, 0.145),
        ("Intensity normalisation\nUS $\\rightarrow$ [-1,1] $\\cdot$ MR z-score", 0.682, 0.20),
    ]
    for i, (txt, x, w) in enumerate(steps):
        box(ax, x, 0.735, w, 0.062, txt, fs=10)
        if i:
            xp, wp = steps[i - 1][1], steps[i - 1][2]
            arrow(ax, xp + wp + 0.002, 0.766, x - 0.002, 0.766, lw=1.4)
    ax.text(0.941, 0.766, "256$\\times$256\naxial slices /\nnative 3D grid",
            ha="center", va="center", fontsize=9)
    arrow(ax, 0.888, 0.766, 0.906, 0.766, lw=1.4)

    # ---------------- band 3 : examples + families ----------------
    # example 2x2 grid (real data)
    ax.text(0.015, 0.700, "Paired training data (test subject ReMIND-003)",
            fontsize=11.5, style="italic")
    ims = [
        slc(os.path.join(DATA, "MR-T2", "ReMIND-003-pre-mri.nii.gz")),
        slc(os.path.join(DATA, "US", "ReMIND-003-pre-us.nii.gz")),
        slc(os.path.join(DATA, "MR-T2", "ReMIND-003-post-mri.nii.gz")),
        slc(os.path.join(DATA, "US", "ReMIND-003-post-us.nii.gz")),
    ]
    x0s, y0s = [0.075, 0.205], [0.520, 0.372]
    for k, im in enumerate(ims):
        a = fig.add_axes([x0s[k % 2], y0s[k // 2], 0.125, 0.140])
        a.imshow(im, cmap="gray", aspect="auto")
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_color("#888888")
    fig.text(0.1375, 0.667, "T2w (FOV crop)", ha="center", fontsize=10)
    fig.text(0.2675, 0.667, "ioUS", ha="center", fontsize=10)
    fig.text(0.066, 0.590, "pre-\nresection", ha="right", va="center", fontsize=10)
    fig.text(0.066, 0.442, "post-\nresection", ha="right", va="center", fontsize=10)

    arrow(ax, 0.345, 0.520, 0.395, 0.520, lw=2.0)

    # family boxes
    ax.text(0.41, 0.700, "Six generators, three paradigms (Section 2.4)",
            fontsize=11.5, style="italic")
    box(ax, 0.41, 0.575, 0.24, 0.105,
        "Pix2Pix $\\cdot$ SwinPix2Pix\nCycleGAN $\\cdot$ CUT\n"
        "attention-gated U-Net / ResNet + PatchGAN",
        fc="#e8eef6", ec=C_GAN, lw=1.6, fs=9.5, title="GAN baselines")
    box(ax, 0.665, 0.575, 0.155, 0.105,
        "transformer-augmented\nresidual generator\n(9 ART blocks)",
        fc="#fbeaea", ec=C_RES, lw=1.6, fs=9.5, title="ResViT")
    box(ax, 0.835, 0.575, 0.15, 0.105,
        "few-step adversarial\ndiffusion ($T=4$),\npaired, single-direction",
        fc="#ececec", ec=C_SYN, lw=1.6, fs=9.5, title="SynDiff")
    for xm in (0.53, 0.7425, 0.91):
        arrow(ax, xm, 0.573, xm, 0.545, lw=1.5)

    # factorial band
    box(ax, 0.41, 0.462, 0.575, 0.075,
        "$\\times$ 4 inference regimes:   2D   |   2.5D   |   2D + 3D-refine   |   Full-3D\n"
        "$\\times$ 2 targets:   T2w only   |   T2w + FLAIR (multi-task)",
        fc="#f7f7f7", ec="#3b4a5f", lw=1.3, fs=10.5)
    arrow(ax, 0.6975, 0.460, 0.6975, 0.432, lw=1.8)
    box(ax, 0.598, 0.376, 0.20, 0.052, "48 experiments", fc="#3b4a5f", ec="#3b4a5f",
        fs=13, weight="bold", tc="white")
    arrow(ax, 0.6975, 0.374, 0.6975, 0.342, lw=1.8)
    arrow(ax, 0.2, 0.368, 0.2, 0.342, lw=1.8)
    ax.text(0.212, 0.356, "real MR of the same 30 test studies (ceiling),  raw ioUS (floor)",
            fontsize=9, va="center")

    # ---------------- band 4 : evaluation ----------------
    ax.text(0.015, 0.328, "Evaluation on the held-out test set (Sections 2.6-2.7)",
            fontsize=11.5, style="italic")
    box(ax, 0.015, 0.155, 0.30, 0.15,
        "SSIM $\\cdot$ PSNR $\\cdot$ MAE $\\cdot$ LPIPS\n"
        "global (foreground) and ROI-restricted:\nlesion / tumour / cavity,\n"
        "0 mm strict + 5 mm margin",
        fs=10, title="Image fidelity")
    box(ax, 0.33, 0.155, 0.36, 0.15,
        "frozen nnU-Net: Seg-T2 ($n=29$) $\\cdot$ Seg-FLAIR ($n=19$)\n"
        "primary endpoint: lesion = tumour $\\cup$ cavity\n"
        "retention vs real-MR ceiling (Dice 0.662)\n"
        "training-free floor: raw ioUS (Dice 0.251)",
        fs=10, title="Downstream segmentation utility")
    box(ax, 0.705, 0.155, 0.28, 0.15,
        "3 patients $\\cdot$ 9 pre-resection sweeps\n"
        "second centre, different scanner\n"
        "all 48 configurations applied frozen\n"
        "ranking transfer + sweep stability",
        fs=10, title="External pilot (inference only)")

    # ---------------- band 5 : statistics strip ----------------
    box(ax, 0.015, 0.075, 0.97, 0.052,
        "statistics:  patient-level bootstrap (20 000 resamples)  $\\cdot$  paired "
        "pre/post Wilcoxon  $\\cdot$  Holm correction within test families  $\\cdot$  "
        "family-adjusted fidelity-utility associations",
        fc="#efefef", ec="#999999", fs=10)
    ax.text(0.5, 0.038,
            "code, weights, splits, reference segmentations and per-study metric tables "
            "released openly (Section: Code and data availability)",
            ha="center", fontsize=9.5, style="italic", color="#444444")

    for ext, dpi in ((".jpg", 300), (".tiff", 300)):
        fig.savefig(os.path.join(OUTDIR, "Fig_overview" + ext), dpi=dpi,
                    facecolor="white")
    fig.savefig(os.path.join(MANUS, "Fig_overview.jpg"), dpi=300, facecolor="white")
    print("written: Fig_overview")


if __name__ == "__main__":
    main()
