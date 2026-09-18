# ioUS2MR-benchmark

Code, configuration and trained weights for **"A Systematic Benchmark of Intraoperative
Ultrasound-to-MR Synthesis for Brain Tumour Surgery"**.

Six generative architectures × four inference regimes × two synthesis targets = **48 controlled
experiments** on the public [ReMIND](https://doi.org/10.7937/3RAG-D070) cohort, evaluated with
image-fidelity metrics, ROI-restricted metrics and a frozen nnU-Net downstream segmentation
protocol.

| | |
|---|---|
| **Paper** | *A Systematic Benchmark of Intraoperative Ultrasound-to-MR Synthesis for Brain Tumour Surgery* — submitted to *Medical Image Analysis*; preprint of an earlier version: [arXiv:2606.00630](https://arxiv.org/abs/2606.00630) |
| **Weights** | Zenodo [10.5281/zenodo.22213625](https://doi.org/10.5281/zenodo.22213625) — one 4.55 GiB archive · `python scripts/fetch_weights.py --download` then `--verify` |
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

* **MRI-trained tools can read the synthetic images without retraining, but not yet as well as
  real MRI.** On the best synthetic T2w (**ResViT, 2D + 3D-refine**), the frozen nnU-Net trained
  on real T2w retains **61 %** of its real-MRI lesion Dice (80 % on tumour), and the FLAIR model
  retains 73 % on synthetic FLAIR; every configuration remains significantly below real MRI.
* The **resection cavity** is the failure mode of every model (at most 41 % retention).
* **Global SSIM misranks the generative paradigms.** The GAN baselines lead every global SSIM
  ranking, yet none reaches the downstream top eight; pooled across experiments, higher global
  SSIM goes with *worse* utility (*r* = −0.82), a between-paradigm contrast that reverses within
  families. Measured over the lesion and a 5 mm margin, SSIM becomes a positive predictor.
* In an external three-patient pilot the fidelity ranking transfers (SSIM ρ = 0.82–0.88).

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

python scripts/fetch_weights.py --download          # one 4.55 GiB archive
unrar x weights/2-zenodo-pesos.rar weights/         # or 7z / bsdtar, see WEIGHTS.md
python scripts/fetch_weights.py --verify            # SHA-256 of every extracted file
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
