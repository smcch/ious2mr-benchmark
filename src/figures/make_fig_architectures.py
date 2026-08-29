"""Fig_architectures -- the three generator paradigms, drawn entirely in matplotlib.

Replaces the AI-generated architectures_real_c4.jpg (Elsevier policy). Panel letters and
content match the existing caption: (A) GAN baselines in their shared U-Net/ResNet form,
(B) ResViT with the transformer branch on ART blocks 4-5, (C) SynDiff in the paired
single-direction formulation with T = 4 reverse steps. Loss weights match Section 2.4.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

BASE = r"$IOUS2MR_ROOT"
OUTDIR = os.path.join(BASE, "paper_assets", "figures_journal_v2")
MANUS = os.path.join(BASE, "latex_manuscript", "us_sintesis", "figures")

plt.rcParams.update({"font.family": "serif", "font.size": 11, "mathtext.fontset": "dejavuserif"})

TEAL, TEALD = "#7fb8b1", "#4e8f87"
CORAL, CORALD = "#e8a598", "#c96f5e"
BLUE, BLUED = "#b8c9de", "#5b7b9e"
GREY = "#c9c9c9"


def box(ax, x, y, w, h, text, fc="white", ec="#555555", lw=1.2, fs=9.5, weight="normal",
        ls="-", rot=0, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight=weight, rotation=rot, color=tc, zorder=3, linespacing=1.3)


def arr(ax, x0, y0, x1, y1, lw=1.3, color="#666666", style="-|>", con=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, lw=lw, color=color,
                                 mutation_scale=11, connectionstyle=con, zorder=4))


def panel(fig, rect, letter, title):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.002, 0.01), 0.996, 0.97,
                                boxstyle="round,pad=0.002,rounding_size=0.012",
                                fc="#f7f7f7", ec="#888888", lw=1.2, zorder=0))
    ax.text(0.012, 0.955, letter, fontsize=17, fontweight="bold", va="top")
    ax.text(0.045, 0.952, title, fontsize=13, fontweight="bold", va="top")
    return ax


def main():
    fig = plt.figure(figsize=(13.2, 10.6))

    # ================= A : GAN baselines =================
    ax = panel(fig, [0.005, 0.655, 0.99, 0.34], "A",
               "GAN baselines  (Pix2Pix $\\cdot$ SwinPix2Pix $\\cdot$ CycleGAN $\\cdot$ CUT)")
    ymid = 0.47
    box(ax, 0.022, ymid - 0.13, 0.088, 0.26,
        "ioUS slice /\n3-slice stack\n(256 $\\times$ 256)", fc=BLUE, ec=BLUED, fs=9.5)
    # Swin stem (dashed, optional)
    box(ax, 0.055, 0.77, 0.15, 0.15, "Swin-Transformer stem\n(SwinPix2Pix only)",
        fc="#fdeee9", ec=CORALD, ls="--", fs=9)
    arr(ax, 0.13, 0.77, 0.158, ymid + 0.20, con="arc3,rad=-0.2")
    arr(ax, 0.112, ymid, 0.148, ymid)

    # encoder bars
    enc_x = [0.155, 0.198, 0.241, 0.284, 0.327]
    heights = [0.42, 0.36, 0.30, 0.24, 0.20]
    chans = ["64", "128", "256", "512", "512"]
    for x, h, c in zip(enc_x, heights, chans):
        box(ax, x, ymid - h / 2, 0.026, h, "", fc=TEAL, ec=TEALD)
        ax.text(x + 0.013, ymid - heights[0] / 2 - 0.05, c, ha="center", fontsize=8.5)
    box(ax, 0.368, ymid - 0.10, 0.075, 0.20, "residual\nbottleneck", fc=TEALD, ec=TEALD,
        fs=9, tc="white")
    dec_x = [0.458, 0.501, 0.544, 0.587]
    for x, h, c in zip(dec_x, heights[3::-1], chans[3::-1]):
        box(ax, x, ymid - h / 2, 0.026, h, "", fc=TEAL, ec=TEALD)
        ax.text(x + 0.013, ymid - heights[0] / 2 - 0.05, c, ha="center", fontsize=8.5)
    # skip connections with attention gates (AG chip sits at the decoder end of each skip)
    for (xe, xd, h) in ((0.168, 0.600, 0.42), (0.211, 0.557, 0.36), (0.254, 0.514, 0.30),
                        (0.297, 0.471, 0.24)):
        y = ymid + h / 2 + 0.012
        arr(ax, xe, y, xd, y, lw=0.9, color="#999999", con="arc3,rad=-0.13")
        box(ax, xd - 0.045, y + 0.015, 0.032, 0.062, "AG", fc="#d9ead3", ec="#6aa84f",
            lw=1.0, fs=7)
    box(ax, 0.628, ymid - 0.12, 0.098, 0.24,
        "synthetic T2w\n(tanh)\n+ FLAIR head\n(multi-task)", fc=BLUE, ec=BLUED, fs=9)
    arr(ax, 0.615, ymid, 0.626, ymid)
    # notes
    box(ax, 0.155, 0.06, 0.20, 0.10, "PatchNCE projection heads\n(CUT only)",
        fc="white", ec="#888888", ls="--", fs=8.5)
    box(ax, 0.395, 0.06, 0.28, 0.10,
        "CycleGAN: 9-block ResNet generator\n($G_{AB}$, $G_{BA}$) + cycle / identity terms",
        fc="white", ec="#888888", ls="--", fs=8.5)
    # discriminator + loss
    box(ax, 0.76, 0.60, 0.105, 0.11, "PatchGAN\nscale 1", fc=GREY, ec="#555555", fs=9)
    box(ax, 0.76, 0.46, 0.105, 0.11, "PatchGAN\nscale 2", fc=GREY, ec="#555555", fs=9)
    ax.text(0.878, 0.585, "multi-scale\ndiscriminator", fontsize=9, va="center")
    arr(ax, 0.728, ymid + 0.04, 0.758, 0.61, con="arc3,rad=-0.15")
    arr(ax, 0.728, ymid - 0.04, 0.758, 0.53, con="arc3,rad=0.1")
    box(ax, 0.76, 0.10, 0.225, 0.28,
        "$\\mathcal{L}_{adv} + 10\\,L_1 + 8\\,(1-\\mathrm{SSIM})$\n"
        "$+\\ 5\\,L_{\\nabla} + 2\\,L_{fm}$\n"
        "CUT adds PatchNCE\nCycleGAN adds cycle + identity", fc="white", ec="#555555",
        fs=9.5)
    ax.text(0.8725, 0.395, "loss", fontsize=9.5, fontweight="bold", ha="center")
    ax.text(0.43, 0.022, "TensorFlow / Keras codebase $\\cdot$ NVIDIA RTX 3090 (24 GB)",
            fontsize=9, style="italic", ha="center")

    # ================= B : ResViT =================
    ax = panel(fig, [0.005, 0.335, 0.99, 0.315], "B",
               "ResViT  (Dalmaz et al., 2022; re-implementation, transformer branch on ART blocks 4-5)")
    ym = 0.44
    box(ax, 0.022, ym - 0.13, 0.082, 0.26, "ioUS slice /\n3-slice stack", fc=BLUE, ec=BLUED,
        fs=9)
    box(ax, 0.118, ym - 0.13, 0.075, 0.26, "ReflectionPad\nConv7\nIN + ReLU", fc=TEAL,
        ec=TEALD, fs=8.5)
    box(ax, 0.205, ym - 0.13, 0.062, 0.26, "stride-2\nconv", fc=TEAL, ec=TEALD, fs=8.5)
    box(ax, 0.279, ym - 0.13, 0.062, 0.26, "stride-2\nconv", fc=TEAL, ec=TEALD, fs=8.5)
    for x0, x1 in ((0.104, 0.116), (0.193, 0.203), (0.267, 0.277), (0.341, 0.352)):
        arr(ax, x0, ym, x1, ym)
    # 9 ART blocks
    bx = 0.354
    for i in range(1, 10):
        accent = i in (4, 5)
        box(ax, bx, ym - 0.115, 0.030, 0.23, f"{i}", fc=CORAL if accent else "#f3d1c9",
            ec=CORALD, fs=8, weight="bold" if accent else "normal")
        bx += 0.036
    ax.text(0.516, ym - 0.185, "ART blocks 1-9 (residual; independent weights)",
            ha="center", fontsize=8.5)
    box(ax, 0.40, 0.76, 0.28, 0.17,
        "transformer branch (windowed attention)\n+ 1 $\\times$ 1 conv aggregation",
        fc="#fdeee9", ec=CORALD, fs=9)
    arr(ax, 0.478, ym + 0.115, 0.50, 0.755, con="arc3,rad=-0.15")
    arr(ax, 0.514, ym + 0.115, 0.53, 0.755, con="arc3,rad=0.1")
    box(ax, 0.70, ym - 0.13, 0.062, 0.26, "ConvT\n$\\times$2", fc=TEAL, ec=TEALD, fs=8.5)
    box(ax, 0.774, ym - 0.13, 0.075, 0.26, "ReflectionPad\nConv7\ntanh", fc=TEAL, ec=TEALD,
        fs=8.5)
    for x0, x1 in ((0.686, 0.698), (0.762, 0.772), (0.849, 0.861)):
        arr(ax, x0, ym, x1, ym)
    box(ax, 0.863, ym - 0.13, 0.082, 0.26, "synthetic\nT2w (+ FLAIR)", fc=BLUE, ec=BLUED,
        fs=9)
    box(ax, 0.80, 0.76, 0.185, 0.17, "PatchGAN discriminator\nLSGAN $+\\ 100\\,L_1 + 10\\,L_{fm}$",
        fc=GREY, ec="#555555", fs=9)
    arr(ax, 0.904, ym + 0.132, 0.895, 0.755, con="arc3,rad=0.15")
    ax.text(0.5, 0.035,
            "Phase 1: 100 epochs, CNN-only, lr $2\\times10^{-4}$   $\\rightarrow$   "
            "Phase 2: transformer branch enabled, 100 epochs, lr $1\\times10^{-4}$",
            fontsize=9, style="italic", ha="center")

    # ================= C : SynDiff =================
    ax = panel(fig, [0.005, 0.005, 0.99, 0.325], "C",
               "SynDiff  (Ozbey et al., 2023; adapted to the paired single-direction formulation, $T=4$)")
    ym = 0.47
    box(ax, 0.022, ym + 0.045, 0.07, 0.15, "$x_T$", fc=BLUE, ec=BLUED, fs=11)
    box(ax, 0.022, ym - 0.24, 0.10, 0.19, "conditioning $c$\n(paired ioUS)", fc=BLUE,
        ec=BLUED, fs=9)
    chain_x = [0.155, 0.305, 0.455, 0.605]
    steps = ["$t=4$", "$t=3$", "$t=2$", "$t=1$"]
    outs = ["$x_3$", "$x_2$", "$x_1$"]
    for i, (cx, s) in enumerate(zip(chain_x, steps)):
        box(ax, cx - 0.035, ym + 0.03, 0.07, 0.18, "$G_\\theta$\n" + s, fc=CORAL,
            ec=CORALD, lw=1.4, fs=9.5)
        if i < 3:
            box(ax, cx + 0.05, ym + 0.065, 0.05, 0.11, outs[i], fc="white", ec=BLUED,
                fs=10)
            arr(ax, cx + 0.037, ym + 0.12, cx + 0.048, ym + 0.12)
            arr(ax, cx + 0.102, ym + 0.12, chain_x[i + 1] - 0.037, ym + 0.12)
    arr(ax, 0.092, ym + 0.12, 0.118, ym + 0.12)
    box(ax, 0.655, ym + 0.045, 0.072, 0.15, "$x_0$", fc="#3b4a5f", ec="#3b4a5f", fs=11,
        tc="white")
    arr(ax, 0.642, ym + 0.12, 0.653, ym + 0.12)
    # conditioning bus
    for cx in chain_x:
        arr(ax, cx, ym - 0.145, cx, ym + 0.025, lw=0.9, color="#999999")
    ax.plot([0.124, chain_x[-1]], [ym - 0.145, ym - 0.145], color="#999999", lw=0.9,
            zorder=1)
    ax.text(0.38, ym - 0.21,
            "re-noising between steps:  $x_t \\sim q\\,(x_t \\mid \\hat{x}_0,\\, x_{t+1})$"
            "  $\\cdot$  generator: NCSNpp time-conditional U-Net",
            ha="center", fontsize=9)
    box(ax, 0.76, ym - 0.02, 0.225, 0.24,
        "time-conditional BigGAN-style D\nnon-saturating GAN loss\n"
        "$+\\ R_1$ penalty (lazy) $+\\ 10\\,L_1$", fc=GREY, ec="#555555", fs=9.5)
    ax.text(0.5, 0.045,
            "single-direction paired formulation (no cycle translator) $\\cdot$ "
            "Adam ($\\beta_1=0.5$, $\\beta_2=0.9$) $\\cdot$ EMA 0.999 $\\cdot$ bf16",
            fontsize=9, style="italic", ha="center")

    for ext in (".jpg", ".tiff"):
        fig.savefig(os.path.join(OUTDIR, "Fig_architectures_v2" + ext), dpi=300,
                    facecolor="white")
    fig.savefig(os.path.join(MANUS, "Fig_architectures_v2.jpg"), dpi=300, facecolor="white")
    print("written: Fig_architectures_v2")


if __name__ == "__main__":
    main()
