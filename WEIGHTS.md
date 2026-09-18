# Released model weights

All trained weights are hosted outside GitHub (they exceed its 100 MB per-file limit) in a
single archived record:

> **Zenodo record: [10.5281/zenodo.22213625](https://doi.org/10.5281/zenodo.22213625)**

The record holds **one RAR archive**, `2-zenodo-pesos.rar`, containing all 66 files, so the
weights are downloaded in one piece and extracted with an external tool; individual files and
families cannot be fetched separately. The helper downloads the archive and then verifies the
extracted tree against the SHA-256 checksums in `configs/weights_manifest.json`:

```bash
python scripts/fetch_weights.py --list            # inventory, sizes and checksums
python scripts/fetch_weights.py --download        # the archive, 4.55 GiB
unrar x weights/2-zenodo-pesos.rar weights/       # or: 7z x ... -oweights/  |  bsdtar -xf ... -C weights/
python scripts/fetch_weights.py --verify          # every file, or --family resvit
export IOUS2MR_CKPT=$PWD/weights
```

Extract so that the family directories sit directly under the weights root
(`weights/gan/`, `weights/resvit/`, `weights/syndiff/`, `weights/nnunet/`); `--verify` says
what it found and where, so a wrong nesting level is immediately visible. Until the record's
files are switched to open access, `--download` will report an HTTP error and point you at the
DOI landing page for a manual download.

## Contents

| Family | Files | Size | What it is |
|---|---:|---:|---|
| GAN baselines | 32 | 1.87 GiB | 24 × `generator_ema.weights.h5` (2D / 2.5D / full-3D) + 8 × `refiner_ema.weights.h5` (the 2D + 3D-refine variants, which reuse the corresponding 2D generator) |
| ResViT | 8 | 1.14 GiB | `p2_best.pth` per experiment — the phase-2 best checkpoint, the one used at inference — plus the 3D refinement heads |
| SynDiff | 10 | 0.31 GiB | the selected diffusive generator per experiment (`gen_diffusive_2_<epoch>.pth`) plus refiners |
| nnU-Net downstream | 16 | 1.23 GiB | Seg-T2 and Seg-FLAIR, 5 folds each (`checkpoint_best.pth`), with the plans/dataset JSONs already in `configs/nnunet/` |
| **Total** | **66** | **4.55 GiB** | |

Intermediate training checkpoints are **not** released: the SynDiff tree alone holds ~17 GB of
per-epoch snapshots, of which only the checkpoints listed in
[`configs/weights_manifest.json`](configs/weights_manifest.json) are needed to reproduce the paper.

## Archive layout

```
weights/
  gan/<architecture>_<regime>_<target>/generator_ema.weights.h5
  gan/<architecture>_2d_3dpost_<target>/refiner_ema.weights.h5
  resvit/ResViT-<regime>-<target>/p2_best.pth
  syndiff/<experiment>/gen_diffusive_2_<epoch>.pth
  nnunet/Dataset501_T2/nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres/fold_<k>/checkpoint_best.pth
  nnunet/Dataset502_FLAIR/...
```

`configs/weights_manifest.json` maps every experiment name in the paper to its file, size and
SHA-256 checksum.

## ⚠ Licence of the diffusion weights

The SynDiff implementation derives from **NVIDIA's DDGAN**, distributed under the NVIDIA Source
Code License, which limits the work *and its derivatives* to **non-commercial research use**.
The `syndiff/` weights in this archive are therefore released for non-commercial research only.
All other weights follow the repository licence. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Reproducibility note

Weights were trained on a single workstation (NVIDIA RTX 4080 SUPER, 16 GB; a few ResViT runs on
an RTX 3090). Exact software versions are pinned in [`envs/`](envs/). Re-training from scratch
reproduces the reported trends but not bit-identical numbers: the pipelines use non-deterministic
GPU kernels and, for the GAN family, TensorFlow data-pipeline shuffling.
