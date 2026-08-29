"""Fig_overview -- benchmark overview, drawn entirely in matplotlib.

Replaces the AI-generated fig1.png (Elsevier does not allow generative-AI artwork, and the
old file carried outdated cohort numbers baked into its pixels). Every number here matches
the manuscript: 114 -> 37 excluded -> 77 subjects; 152 paired studies (102 FLAIR); split
61/16 subjects; test 30 = 16 pre + 14 post; 48 experiments; floor 0.251 / ceiling 0.662.
Thumbnails are real ReMIND test-subject data (ReMIND-003), rendered from the released
pre-processed volumes.

Layout note: FancyBboxPatch "round,pad=0.008" draws each box 0.008 beyond its nominal
coordinates on every side; all gaps/arrows below account for that visual extent.
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

plt.rcParams.update({"font.family": "serif", "font.size": 12})

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


def box(ax, x, y, w, h, text, fc=C_BOX, ec=C_EDGE, lw=1.1, fs=12, weight="normal",
        style="round,pad=0.008,rounding_size=0.012", tc="black", ls="-", title=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw,
                                ls=ls, mutation_aspect=1, zorder=2))
    ty = y + h / 2
    if title:
        ax.text(x + w / 2, y + h - 0.013, title, ha="center", va="top", fontsize=13,
                fontweight="bold", color=tc, zorder=3)
        ty = y + (h - 0.036) / 2
    ax.text(x + w / 2, ty, text, ha="center", va="center", fontsize=fs,
            fontweight=weight, color=tc, zorder=3, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, lw=1.6, color="#666666", style="-|>", shrink=0.0,
          ms=14):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, lw=lw,
                                 color=color, mutation_scale=ms, shrinkA=shrink,
                                 shrinkB=shrink, zorder=4))


def main():
    fig = plt.figure(figsize=(12.3, 10.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---------------- band 1 : cohort ----------------
    box(ax, 0.015, 0.868, 0.97, 0.116, "", fc=C_BAND, ec="#3b4a5f", lw=1.4)
    ax.text(0.5, 0.960, "ReMIND benchmark cohort (public, TCIA)", ha="center",
            fontsize=16, fontweight="bold")
    ax.text(0.5, 0.932,
            "114 screened  $\\rightarrow$  37 excluded (ioUS quality)  "
            "$\\rightarrow$  77 patients  $\\cdot$  152 paired ioUS / T2w studies "
            "(102 with FLAIR)", ha="center", fontsize=12)
    ax.text(0.5, 0.906,
            "subject-level split:  61 patients / 122 studies training",
            ha="center", fontsize=12)
    ax.text(0.5, 0.8805,
            "16 patients / 30 studies held-out test (16 pre- + 14 post-resection)",
            ha="center", fontsize=12)
    # corner thumbnails (labels below the images)
    us = slc(os.path.join(DATA, "US", "ReMIND-003-pre-us.nii.gz"))
    t2 = slc(os.path.join(DATA, "MR-T2", "ReMIND-003-pre-mri.nii.gz"))
    for img, x0, lab in ((us, 0.030, "ioUS"), (t2, 0.895, "MR (T2w / FLAIR)")):
        a = fig.add_axes([x0, 0.885, 0.066, 0.073])
        a.imshow(img, cmap="gray", aspect="auto")
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_color("white"); s.set_lw(2)
        ax.text(x0 + 0.033, 0.869, lab, ha="center", fontsize=11, zorder=3)

    arrow(ax, 0.5, 0.857, 0.5, 0.840)

    # ---------------- band 2 : pre-processing ----------------
    ax.text(0.015, 0.824, "Pre-processing", fontsize=14,
            style="italic")
    steps = [
        ("DICOM $\\rightarrow$ NIfTI", 0.015, 0.118),
        ("Rigid co-registration\n(LC$^2$; ioUS as reference)", 0.170, 0.173),
        ("ioUS resampled to\nthe MR voxel grid", 0.380, 0.131),
        ("FOV crop to the\nioUS cone", 0.548, 0.114),
        ("Intensity normalisation\nUS $\\rightarrow$ [-1,1] $\\cdot$ MR z-score", 0.699, 0.177),
    ]
    for i, (txt, x, w) in enumerate(steps):
        box(ax, x, 0.736, w, 0.066, txt, fs=12)
        if i:
            xp, wp = steps[i - 1][1], steps[i - 1][2]
            arrow(ax, xp + wp + 0.010, 0.769, x - 0.009, 0.769, lw=1.4, ms=12)
    ax.text(0.947, 0.769, "256$\\times$256\naxial slices /\nnative 3D grid",
            ha="center", va="center", fontsize=11)
    arrow(ax, 0.886, 0.769, 0.898, 0.769, lw=1.4, ms=12)

    # ---------------- band 3 : examples + families ----------------
    # example 2x2 grid (real data)
    ax.text(0.015, 0.700, "Paired training data",
            fontsize=14, style="italic")
    ims = [
        slc(os.path.join(DATA, "MR-T2", "ReMIND-003-pre-mri.nii.gz")),
        slc(os.path.join(DATA, "US", "ReMIND-003-pre-us.nii.gz")),
        slc(os.path.join(DATA, "MR-T2", "ReMIND-003-post-mri.nii.gz")),
        slc(os.path.join(DATA, "US", "ReMIND-003-post-us.nii.gz")),
    ]
    x0s, y0s = [0.080, 0.203], [0.528, 0.392]
    for k, im in enumerate(ims):
        a = fig.add_axes([x0s[k % 2], y0s[k // 2], 0.110, 0.125])
        a.imshow(im, cmap="gray", aspect="auto")
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_color("#888888")
    fig.text(0.135, 0.660, "T2w (FOV crop)", ha="center", fontsize=11)
    fig.text(0.258, 0.660, "ioUS", ha="center", fontsize=11)
    fig.text(0.073, 0.5905, "pre-\nresection", ha="right", va="center", fontsize=11)
    fig.text(0.073, 0.4545, "post-\nresection", ha="right", va="center", fontsize=11)

    arrow(ax, 0.323, 0.600, 0.391, 0.600, lw=2.0)

    # family boxes
    ax.text(0.44, 0.700, "Six generators, three paradigms",
            fontsize=14, style="italic")
    box(ax, 0.400, 0.548, 0.162, 0.130,
        "Pix2Pix $\\cdot$ SwinPix2Pix\nCycleGAN $\\cdot$ CUT",
        fc="#e8eef6", ec=C_GAN, lw=1.6, fs=11.5, title="GAN baselines")
    box(ax, 0.6085, 0.548, 0.167, 0.130,
        "transformer-augmented\nresidual generator\n(9 ART blocks)",
        fc="#fbeaea", ec=C_RES, lw=1.6, fs=11.5, title="ResViT")
    box(ax, 0.822, 0.548, 0.163, 0.130,
        "few-step adversarial\ndiffusion ($T=4$),\npaired, single-direction",
        fc="#ececec", ec=C_SYN, lw=1.6, fs=11.5, title="SynDiff")
    for xm in (0.481, 0.692, 0.9035):
        arrow(ax, xm, 0.538, xm, 0.493, lw=1.5)

    # factorial band
    box(ax, 0.40, 0.412, 0.585, 0.072,
        "$\\times$ 4 inference regimes:   2D   |   2.5D   |   2D + 3D-refine   |   Full-3D\n"
        "$\\times$ 2 targets:   T2w only   |   T2w + FLAIR (multi-task)",
        fc="#f7f7f7", ec="#3b4a5f", lw=1.3, fs=12)
    arrow(ax, 0.6925, 0.402, 0.6925, 0.379, lw=1.8)
    box(ax, 0.6125, 0.324, 0.16, 0.046, "48 experiments", fc="#3b4a5f", ec="#3b4a5f",
        fs=14, weight="bold", tc="white")
    arrow(ax, 0.6925, 0.314, 0.6925, 0.248, lw=1.8)
    box(ax, 0.015, 0.324, 0.302, 0.048,
        "real MR of the same 30 test studies (ceiling),\nraw ioUS (floor)",
        fc="#fbfbfb", ec="#888888", lw=1.1, fs=11, ls=(0, (4, 3)))
    arrow(ax, 0.166, 0.314, 0.166, 0.248, lw=1.8)

    # ---------------- band 4 : evaluation ----------------
    ax.text(0.015, 0.218, "Evaluation on the held-out test set",
            fontsize=14, style="italic")
    box(ax, 0.015, 0.048, 0.270, 0.140,
        "SSIM $\\cdot$ PSNR $\\cdot$ MAE $\\cdot$ LPIPS\n"
        "global (foreground) and ROI-restricted:\nlesion / tumour / cavity,\n"
        "0 mm strict + 5 mm margin",
        fs=11.5, title="Image fidelity")
    box(ax, 0.3245, 0.048, 0.3594, 0.140,
        "frozen nnU-Net: Seg-T2 ($n=29$) $\\cdot$ Seg-FLAIR ($n=19$)\n"
        "primary endpoint: lesion = tumour $\\cup$ cavity\n"
        "retention vs real-MR ceiling\n"
        "training-free floor: raw ioUS",
        fs=11.5, title="Downstream segmentation utility")
    box(ax, 0.7235, 0.048, 0.262, 0.140,
        "3 patients $\\cdot$ 9 pre-resection sweeps\n"
        "external centre, different scanner\n"
        "all 48 configurations applied frozen\n"
        "ranking transfer + sweep stability",
        fs=11.5, title="External pilot (inference only)")

    for ext, dpi in ((".jpg", 300), (".tiff", 300)):
        fig.savefig(os.path.join(OUTDIR, "Fig_overview" + ext), dpi=dpi,
                    facecolor="white")
    fig.savefig(os.path.join(MANUS, "Fig_overview.jpg"), dpi=300, facecolor="white")
    print("written: Fig_overview")


if __name__ == "__main__":
    main()
