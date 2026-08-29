# ioUS2MR-benchmark

Code, configuration and trained weights for **"A Systematic Benchmark of Intraoperative
Ultrasound-to-MR Synthesis for Brain Tumour Surgery"**.

Six generative architectures × four inference regimes × two synthesis targets = **48 controlled
experiments** on the public [ReMIND](https://doi.org/10.7937/3RAG-D070) cohort, evaluated with
image-fidelity metrics, ROI-restricted metrics and a frozen nnU-Net downstream segmentation
protocol.

| | |
|---|---|
| **Paper** | [TODO — link / DOI once published] |
| **Weights** | Zenodo [TODO — DOI] · `python scripts/fetch_weights.py --list` |
| **Data** | ReMIND (public, CC BY 4.0) — see [`DATA.md`](DATA.md) |
| **Licence** | Apache-2.0, *except* the diffusion family — see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) |

---

## What this benchmark contains

| Axis | Values |
|---|---|
| Architecture | Pix2Pix · SwinPix2Pix · CycleGAN · CUT · ResViT · SynDiff |
| Inference regime | 2D · 2.5D · 2D + 3D-refinement · full-3D |
| Target | T2w · T2w + FLAIR (multi-task) |
| Fidelity metrics | SSIM · PSNR · MAE · LPIPS (global and ROI-restricted at 0/5 mm) |
| Downstream | nnU-Net v2 Seg-T2 / Seg-FLAIR — lesion (primary), tumour and cavity (secondary) |

### Headline findings

* No architecture dominates every axis. The best downstream configuration
  (**ResViT-2D + 3D-refine**) retains **61 %** of the real-T2w lesion Dice; **no GAN baseline
  ranks in the top eight**.
* **Higher global SSIM predicts *worse* downstream utility** (*r* = −0.82). Perceptual LPIPS
  tracks utility equally strongly and is the only fidelity metric whose association survives
  within-study adjustment.
* The **resection cavity** is the failure mode of every model (best retention 41 %).
* In an external three-patient pilot the ranking transfers (SSIM ρ = 0.82–0.88) and synthesis
  behaves as a **domain normaliser** for MRI-trained downstream tools.

---

## Repository layout

```
configs/            frozen subject split, nnU-Net dataset/plans/folds, weights manifest
envs/               two conda environments (TensorFlow for GANs, PyTorch for the rest)
src/
  common/paths.py   filesystem roots, resolved from environment variables
  gan/              Pix2Pix · SwinPix2Pix · CycleGAN · CUT (TensorFlow/Keras)
  resvit/           ResViT re-implementation + ablations (PyTorch)
  syndiff/          adversarial diffusion (PyTorch); vendor/ holds upstream code
  downstream/       nnU-Net dataset build, staging, metrics, robustness analyses
  scoring/          unified fidelity harness (global + ROI-restricted) and paper tables
  figures/          every figure in the paper
results/            all derived metric tables (CSV) — regenerate paper numbers with no GPU
docs/               methodology write-ups and protocol notes
scripts/            weights download helper, repository provenance
```

## Quick start

```bash
git clone https://github.com/smcch/ious2mr-benchmark && cd ious2mr-benchmark
conda env create -f envs/environment-pytorch.yml && conda activate ious2mr-torch

export IOUS2MR_ROOT=/path/to/working/tree     # where the pipeline reads/writes
export IOUS2MR_DATA=/path/to/preprocessed     # ReMIND-derived volumes
export IOUS2MR_CKPT=/path/to/weights

python scripts/fetch_weights.py --family resvit     # ~1.2 GB
```

**Reproduce every table and figure without a GPU** — the released CSVs are enough:

```bash
python src/downstream/make_lesion_primary_tables.py
python src/figures/make_figs_quality.py
```

Full instructions, including training from scratch, are in
[`REPRODUCING.md`](REPRODUCING.md).

## Data

The benchmark runs on **ReMIND** (public, CC BY 4.0). The subject-level split, the manual
tumour / resection-cavity reference segmentations and all per-subject metric tables are
released here or in the weights archive. The external pilot cohort is **private clinical data
and cannot be shared** — only its code and anonymised aggregate metrics are published. See
[`DATA.md`](DATA.md).

## Licence

Original code: **Apache-2.0** ([`LICENSE`](LICENSE)).

⚠ The SynDiff family (`src/syndiff/`) derives from NVIDIA's DDGAN, which restricts use to
**non-commercial research**; that limitation propagates to the SynDiff code *and its released
weights*. Every upstream component and its licence is listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) — read it before reuse.

## Citing

If you use this code, the weights or the derived results, please cite the paper and the
software ([`CITATION.cff`](CITATION.cff)), and — as required by its CC BY 4.0 licence — the
ReMIND dataset (see [`DATA.md`](DATA.md)).
