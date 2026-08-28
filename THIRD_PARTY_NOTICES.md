# Third-party components and licences

This repository re-implements and adapts several published methods. Below is the provenance
of each component, the licence that governs it, and what that means for reuse. Licence terms
were verified against the upstream repositories on 2026-08-28; **always re-check upstream
before relying on this summary.**

## Summary table

| Component (this repo) | Upstream project | Upstream licence | Restrictions |
|---|---|---|---|
| `src/gan/` — Pix2Pix generator/discriminator | [pix2pix](https://github.com/phillipi/pix2pix), [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) (Isola et al., 2017) | BSD | Retain copyright notice |
| `src/gan/` — CycleGAN generator | [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) (Zhu et al., 2017) | BSD | Retain copyright notice |
| `src/gan/` — CUT / PatchNCE objective | [contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation) (Park et al., 2020) | BSD | Retain copyright notice |
| `src/gan/` — Swin Transformer stem | Swin Transformer (Liu et al., 2021) | MIT | Retain copyright notice |
| `src/resvit/` — ART blocks, two-phase training | [ResViT](https://github.com/icon-lab/ResViT) (Dalmaz et al., 2022) | **MIT** | Retain copyright notice |
| `src/syndiff/` — adversarial-diffusion training loop | [SynDiff](https://github.com/icon-lab/SynDiff) (Özbey et al., 2023) | **MIT** | Retain copyright notice |
| `src/syndiff/` — NCSNpp backbone, DDGAN few-step formulation | [denoising-diffusion-gan](https://github.com/NVlabs/denoising-diffusion-gan) (Xiao et al., 2022) | **NVIDIA Source Code License — NON-COMMERCIAL** | ⚠ See below |
| `src/downstream/` — segmentation | [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) (Isensee et al., 2021) | Apache-2.0 | Used as a dependency, not vendored |
| Evaluation — LPIPS | [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity) (Zhang et al., 2018) | BSD-2-Clause | Used as a dependency |
| Evaluation — SSIM/PSNR | scikit-image | BSD-3-Clause | Dependency |

## ⚠ Non-commercial restriction on the diffusion components

The SynDiff implementation in this repository follows the few-step adversarial-diffusion
formulation of **DDGAN (NVlabs/denoising-diffusion-gan)**, which is distributed under the
**NVIDIA Source Code License**. That licence restricts the work *and any derivative works*
to **non-commercial use** ("research or evaluation purposes only") and requires derivative
distributions to propagate the same use limitation.

Accordingly:

* Everything under `src/syndiff/`, **and the released SynDiff model weights**, are made
  available for **non-commercial research use only**, under the terms of the NVIDIA Source
  Code License (reproduced in `licenses/NVIDIA_SOURCE_CODE_LICENSE.txt`).
* All other code in this repository is released under the licence in `LICENSE`.
* If you require the benchmark for commercial purposes, exclude the SynDiff family or obtain
  the appropriate permissions from the upstream rights-holders.

## Data

The **ReMIND** dataset (Juvekar et al., 2024) is distributed by The Cancer Imaging Archive
under **CC BY 4.0**, which permits redistribution of derived works with attribution. Any
ReMIND-derived artefact in this repository or in the accompanying data release is therefore
provided under CC BY 4.0 and must be cited as:

> Juvekar, P., Dorent, R., Kögl, F., et al. (2023). *The Brain Resection Multimodal Imaging
> Database (ReMIND)* (Version 1) [Data set]. The Cancer Imaging Archive.
> https://doi.org/10.7937/3RAG-D070

The external validation cohort used in the paper's external pilot is **not** included in this
release and cannot be shared (see `DATA.md`).
