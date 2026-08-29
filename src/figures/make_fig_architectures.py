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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BASE = r"$IOUS2MR_ROOT"
OUTDIR = os.path.join(BASE, "paper_assets", "figures_journal_v2")
MANUS = os.path.join(BASE, "latex_manuscript", "us_sintesis", "figures")

plt.rcParams.update({"font.family": "serif", "font.size": 12, "mathtext.fontset": "dejavuserif"})

TEAL, TEALD = "#7fb8b1", "#4e8f87"
CORAL, CORALD = "#e8a598", "#c96f5e"
BLUE, BLUED = "#b8c9de", "#5b7b9e"
GREY = "#c9c9c9"


def box(ax, x, y, w, h, text, fc="white", ec="#555555", lw=1.2, fs=12, weight="normal",
        ls="-", rot=0, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight=weight, rotation=rot, color=tc, zorder=3, linespacing=1.25)


def arr(ax, x0, y0, x1, y1, lw=1.3, color="#666666", style="-|>", con=None, ms=12):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, lw=lw, color=color,
                                 mutation_scale=ms, connectionstyle=con, zorder=4))


def panel(fig, rect, letter, title):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.002, 0.01), 0.996, 0.97,
                                boxstyle="round,pad=0.002,rounding_size=0.012",
                                fc="#f7f7f7", ec="#888888", lw=1.2, zorder=0))
    ax.text(0.012, 0.968, letter, fontsize=16, fontweight="bold", va="top")
    ax.text(0.042, 0.965, title, fontsize=14, fontweight="bold", va="top")
    return ax


def main():
    fig = plt.figure(figsize=(12.2, 11.2))

    # ================= A : GAN baselines =================
    ax = panel(fig, [0.005, 0.665, 0.99, 0.33], "A",
               "GAN baselines  (Pix2Pix $\\cdot$ SwinPix2Pix $\\cdot$ CycleGAN $\\cdot$ CUT)")
    ymid = 0.45
    box(ax, 0.016, ymid - 0.14, 0.106, 0.28,
        "ioUS slice /\n3-slice stack\n(256 $\\times$ 256)", fc=BLUE, ec=BLUED, fs=12)
    # Swin stem (dashed, optional) -- arrow enters the first encoder bar at its left edge,
    # left of the skip-connection risers, so it crosses nothing.
    box(ax, 0.030, 0.750, 0.165, 0.13, "Swin-Transformer stem\n(SwinPix2Pix only)",
        fc="#fdeee9", ec=CORALD, ls="--", fs=11)
    arr(ax, 0.112, 0.743, 0.149, 0.655, con="arc3,rad=-0.15")
    arr(ax, 0.128, ymid, 0.149, ymid)

    # encoder bars
    enc_x = [0.155, 0.198, 0.241, 0.284, 0.327]
    heights = [0.46, 0.35, 0.24, 0.13, 0.11]
    chans = ["64", "128", "256", "512", "512"]
    for x, h, c in zip(enc_x, heights, chans):
        box(ax, x, ymid - h / 2, 0.026, h, "", fc=TEAL, ec=TEALD)
        ax.text(x + 0.013, ymid - h / 2 - 0.022, c, ha="center", va="top", fontsize=11)
    box(ax, 0.366, ymid - 0.077, 0.080, 0.13, "residual\nbottleneck", fc=TEALD, ec=TEALD,
        fs=11, tc="white")
    dec_x = [0.458, 0.501, 0.544, 0.587]
    for x, h, c in zip(dec_x, heights[3::-1], chans[3::-1]):
        box(ax, x, ymid - h / 2, 0.026, h, "", fc=TEAL, ec=TEALD)
        ax.text(x + 0.013, ymid - h / 2 - 0.022, c, ha="center", va="top", fontsize=11)
    # skip connections: rectilinear path broken by an AG (attention gate) chip that sits
    # on the horizontal run, near the decoder end; the path never touches the chip text.
    for (xe, xd, h) in ((0.168, 0.600, 0.46), (0.211, 0.557, 0.35), (0.254, 0.514, 0.24),
                        (0.297, 0.471, 0.13)):
        y = ymid + h / 2 + 0.045
        cx0 = xd - 0.058  # chip left edge
        arr(ax, xe, ymid + h / 2 + 0.005, cx0 - 0.006, y, lw=1.0, color="#8f8f8f",
            con="angle,angleA=90,angleB=0,rad=0", ms=10)
        box(ax, cx0, y - 0.024, 0.036, 0.048, "AG", fc="#d9ead3", ec="#6aa84f",
            lw=1.0, fs=11)
        arr(ax, cx0 + 0.041, y, xd, ymid + h / 2 + 0.007, lw=1.0, color="#8f8f8f",
            con="angle,angleA=0,angleB=90,rad=0", ms=10)
    box(ax, 0.632, ymid - 0.14, 0.106, 0.28,
        "synthetic T2w\n(tanh)\n+ FLAIR head\n(multi-task)", fc=BLUE, ec=BLUED, fs=12)
    arr(ax, 0.618, ymid, 0.626, ymid, ms=10)
    # notes (top band, clear of the skip chips)
    box(ax, 0.220, 0.750, 0.215, 0.13, "PatchNCE projection heads\n(CUT only)",
        fc="white", ec="#888888", ls="--", fs=11)
    box(ax, 0.600, 0.750, 0.255, 0.13,
        "CycleGAN: 9-block ResNet generator\n($G_{AB}$, $G_{BA}$) + cycle / identity terms",
        fc="white", ec="#888888", ls="--", fs=11)
    # discriminator + loss
    box(ax, 0.775, 0.60, 0.10, 0.13, "PatchGAN\nscale 1", fc=GREY, ec="#555555", fs=11.5)
    box(ax, 0.775, 0.44, 0.10, 0.13, "PatchGAN\nscale 2", fc=GREY, ec="#555555", fs=11.5)
    ax.text(0.885, 0.585, "multi-scale\ndiscriminator", fontsize=11, va="center")
    arr(ax, 0.715, 0.597, 0.769, 0.663, con="arc3,rad=-0.1")
    arr(ax, 0.744, 0.47, 0.769, 0.503, con="arc3,rad=0.05")
    box(ax, 0.752, 0.08, 0.235, 0.28,
        "$\\mathcal{L}_{adv} + 10\\,L_1 + 8\\,(1-\\mathrm{SSIM})$\n"
        "$+\\ 5\\,L_{\\nabla} + 2\\,L_{fm}$\n"
        "CUT adds PatchNCE\nCycleGAN adds cycle + identity", fc="white", ec="#555555",
        fs=11.5)
    ax.text(0.8695, 0.375, "loss", fontsize=12, fontweight="bold", ha="center")
    ax.text(0.43, 0.028, "TensorFlow / Keras codebase $\\cdot$ NVIDIA RTX 3090 (24 GB)",
            fontsize=11, style="italic", ha="center")

    # ================= B : ResViT =================
    ax = panel(fig, [0.005, 0.340, 0.99, 0.318], "B",
               "ResViT  (Dalmaz et al., 2022; re-implementation, transformer branch on ART blocks 4-5)")
    ym = 0.42
    box(ax, 0.018, ym - 0.13, 0.098, 0.26, "ioUS slice /\n3-slice stack", fc=BLUE, ec=BLUED,
        fs=11)
    box(ax, 0.136, ym - 0.13, 0.098, 0.26, "ReflectionPad\nConv7\nIN + ReLU", fc=TEAL,
        ec=TEALD, fs=11)
    box(ax, 0.254, ym - 0.13, 0.054, 0.26, "stride-2\nconv", fc=TEAL, ec=TEALD, fs=11)
    box(ax, 0.328, ym - 0.13, 0.054, 0.26, "stride-2\nconv", fc=TEAL, ec=TEALD, fs=11)
    for x0, x1 in ((0.121, 0.131), (0.239, 0.249), (0.313, 0.323), (0.387, 0.397)):
        arr(ax, x0, ym, x1, ym, ms=10)
    # 9 ART blocks
    bx = 0.402
    for i in range(1, 10):
        accent = i in (4, 5)
        box(ax, bx, ym - 0.115, 0.025, 0.23, f"{i}", fc=CORAL if accent else "#f3d1c9",
            ec=CORALD, fs=11, weight="bold" if accent else "normal")
        bx += 0.030
    ax.text(0.5345, 0.245, "ART blocks 1-9 (residual; independent weights)",
            ha="center", va="center", fontsize=11)
    box(ax, 0.375, 0.72, 0.30, 0.16,
        "transformer branch (windowed attention)\n+ 1 $\\times$ 1 conv aggregation",
        fc="#fdeee9", ec=CORALD, fs=11.5)
    arr(ax, 0.5045, 0.552, 0.505, 0.714, con="arc3,rad=-0.05")
    arr(ax, 0.5345, 0.552, 0.558, 0.714, con="arc3,rad=0.08")
    box(ax, 0.687, ym - 0.13, 0.054, 0.26, "ConvT\n$\\times$2", fc=TEAL, ec=TEALD, fs=11)
    box(ax, 0.761, ym - 0.13, 0.098, 0.26, "ReflectionPad\nConv7\ntanh", fc=TEAL, ec=TEALD,
        fs=11)
    for x0, x1 in ((0.672, 0.682), (0.746, 0.756), (0.864, 0.874)):
        arr(ax, x0, ym, x1, ym, ms=10)
    box(ax, 0.879, ym - 0.13, 0.099, 0.26, "synthetic\nT2w (+ FLAIR)", fc=BLUE, ec=BLUED,
        fs=11)
    box(ax, 0.775, 0.72, 0.210, 0.16, "PatchGAN discriminator\nLSGAN $+\\ 100\\,L_1 + 10\\,L_{fm}$",
        fc=GREY, ec="#555555", fs=11.5)
    arr(ax, 0.9285, 0.557, 0.885, 0.714, con="arc3,rad=0.12")
    ax.text(0.5, 0.045,
            "Phase 1: 100 epochs, CNN-only, lr $2\\times10^{-4}$   $\\rightarrow$   "
            "Phase 2: transformer branch enabled, 100 epochs, lr $1\\times10^{-4}$",
            fontsize=11, style="italic", ha="center")

    # ================= C : SynDiff =================
    ax = panel(fig, [0.005, 0.004, 0.99, 0.33], "C",
               "SynDiff  (Ozbey et al., 2023; adapted to the paired single-direction formulation, $T=4$)")
    ym = 0.47
    box(ax, 0.020, ym + 0.07, 0.066, 0.12, "$x_T$", fc=BLUE, ec=BLUED, fs=12)
    box(ax, 0.016, ym - 0.26, 0.110, 0.20, "conditioning $c$\n(paired ioUS)", fc=BLUE,
        ec=BLUED, fs=11.5)
    chain_x = [0.147, 0.313, 0.479, 0.645]
    steps = ["$t=4$", "$t=3$", "$t=2$", "$t=1$"]
    outs = ["$x_3$", "$x_2$", "$x_1$"]
    for i, (cx, s) in enumerate(zip(chain_x, steps)):
        box(ax, cx - 0.035, ym + 0.03, 0.070, 0.20, "$G_\\theta$\n" + s, fc=CORAL,
            ec=CORALD, lw=1.4, fs=11.5)
        if i < 3:
            box(ax, cx + 0.060, ym + 0.07, 0.046, 0.12, outs[i], fc="white", ec=BLUED,
                fs=12)
            arr(ax, cx + 0.040, ym + 0.13, cx + 0.054, ym + 0.13, ms=10)
            arr(ax, cx + 0.111, ym + 0.13, cx + 0.125, ym + 0.13, ms=10)
    arr(ax, 0.091, ym + 0.13, 0.106, ym + 0.13, ms=10)
    box(ax, 0.702, ym + 0.055, 0.064, 0.15, "$x_0$", fc="#3b4a5f", ec="#3b4a5f", fs=12,
        tc="white")
    arr(ax, 0.685, ym + 0.13, 0.696, ym + 0.13, ms=10)
    # conditioning bus -- vertical arrows stop clearly short of the G box bottoms
    for cx in chain_x:
        arr(ax, cx, ym - 0.157, cx, ym + 0.005, lw=1.0, color="#999999")
    ax.plot([0.130, chain_x[-1]], [ym - 0.16, ym - 0.16], color="#999999", lw=1.0,
            zorder=1)
    ax.text(0.47, 0.19,
            "re-noising between steps:  $x_t \\sim q\\,(x_t \\mid \\hat{x}_0,\\, x_{t+1})$"
            "  $\\cdot$  generator: NCSNpp time-conditional U-Net",
            ha="center", fontsize=11)
    box(ax, 0.790, ym - 0.05, 0.192, 0.26,
        "time-conditional\nBigGAN-style D\nnon-saturating GAN loss\n"
        "$+\\ R_1$ penalty (lazy) $+\\ 10\\,L_1$", fc=GREY, ec="#555555", fs=11)
    ax.text(0.5, 0.045,
            "single-direction paired formulation (no cycle translator) $\\cdot$ "
            "Adam ($\\beta_1=0.5$, $\\beta_2=0.9$) $\\cdot$ EMA 0.999 $\\cdot$ bf16",
            fontsize=11, style="italic", ha="center")

    for ext in (".jpg", ".tiff"):
        fig.savefig(os.path.join(OUTDIR, "Fig_architectures_v2" + ext), dpi=300,
                    facecolor="white")
    fig.savefig(os.path.join(MANUS, "Fig_architectures_v2.jpg"), dpi=300, facecolor="white")
    print("written: Fig_architectures_v2")


if __name__ == "__main__":
    main()
