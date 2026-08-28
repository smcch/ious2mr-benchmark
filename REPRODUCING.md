# Reproducing the benchmark

Three levels of reproduction, from cheapest to most expensive. Pick the one you need.

| Level | Needs | Time | What you get |
|---|---|---|---|
| **1. Tables & figures** | CPU only | minutes | Every number and figure in the paper, from the released CSVs |
| **2. Inference & scoring** | 1 GPU, ReMIND, released weights | hours | The synthetic volumes and all metrics, recomputed |
| **3. Full training** | 1 GPU | ~2 weeks | All 48 experiments trained from scratch |

Set the environment roots first (see [`src/common/paths.py`](src/common/paths.py)):

```bash
export IOUS2MR_ROOT=/path/to/working/tree
export IOUS2MR_DATA=$IOUS2MR_ROOT/data
export IOUS2MR_CKPT=$IOUS2MR_ROOT/weights
export IOUS2MR_PY_TORCH=$(which python)   # used by the shell orchestrators
export IOUS2MR_PY_TF=$(which python)      # the TensorFlow env, for the GAN family
```

---

## Level 1 — tables and figures from released results

No data download, no GPU. The `results/` tree holds every per-subject metric.

```bash
conda env create -f envs/environment-pytorch.yml && conda activate ious2mr-torch

python src/downstream/make_lesion_primary_tables.py   # downstream tables + correlations
python src/downstream/correlation_robustness.py       # cluster bootstrap + mixed effects
python src/figures/make_figs_quality.py               # fidelity overview, pre/post, subgroups
python src/figures/make_fig_multitask.py              # single- vs multi-task
python src/figures/make_fig_downstream.py             # downstream leaderboards + correlations
```

`src/figures/make_fig_external.py` is the only script that needs non-public data (see
[`DATA.md`](DATA.md)); skip it or point `IOUS2MR_EXTERNAL` at your own cohort.

---

## Level 2 — re-run inference and scoring

### 2.1 Get the data

Download ReMIND from TCIA (see [`DATA.md`](DATA.md)) and pre-process it as described in
[`docs/methodology_training.md`](docs/methodology_training.md):

1. DICOM → NIfTI, preserving the navigation reference frame;
2. rigid ioUS/MR co-registration (LC² similarity; the paper used ImFusion Suite);
3. resample ioUS to the MR grid;
4. crop to the ioUS foreground bounding box;
5. intensity normalisation (US: 2nd–98th percentile → [−1, 1]; MR: z-score, ±3σ → [−1, 1]).

The frozen subject split is `configs/subject_split.json` — use it to reproduce the exact
partition (61 training / 16 held-out subjects).

### 2.2 Fetch weights

```bash
python scripts/fetch_weights.py --all      # ≈ 4.9 GB, see WEIGHTS.md
```

### 2.3 Inference

```bash
# GAN family (TensorFlow env)
conda activate ious2mr-tf
python src/gan/infer_single_axis.py                 # axial-only pass
# ResViT / SynDiff (PyTorch env)
conda activate ious2mr-torch
python src/resvit/resvit_final.py --variant 2.5d --target t2 --eval_only
python src/syndiff/eval_resvit_protocol.py
```

### 2.4 Scoring

```bash
python src/scoring/rescore_all.py     # global fidelity, one harness for all 48 experiments
python src/scoring/rescore_roi.py     # ROI-restricted (lesion/tumour/cavity × 0, 5 mm)
python src/scoring/build_paper_tables.py
```

The metric definitions are frozen in [`docs/scoring_protocol.md`](docs/scoring_protocol.md) and
[`docs/roi_methodology.md`](docs/roi_methodology.md).

### 2.5 Downstream segmentation

The two nnU-Net models are released trained; you only need to run inference and metrics.

```bash
export nnUNet_raw=$IOUS2MR_ROOT/nnUNet_raw
export nnUNet_preprocessed=$IOUS2MR_ROOT/nnUNet_preprocessed
export nnUNet_results=$IOUS2MR_CKPT/nnunet

python src/downstream/stage_test_inputs.py    # synthetic volumes → nnU-Net layout
bash   src/downstream/run_inference.sh        # 5-fold ensemble over every set
python src/downstream/compute_seg_metrics.py  # Dice / HD95 / NSD, present-class protocol
python src/downstream/make_lesion_primary_tables.py
```

To retrain the segmentation models instead, use `src/downstream/build_datasets.py` followed by
`src/downstream/train_one_fold.sh` (500 epochs per fold, `configs/nnunet/*/splits_final.json`
holds the exact folds).

---

## Level 3 — train all 48 experiments

Budget roughly two weeks on a single 16 GB GPU. Per-family wall-clock and hyper-parameters are
tabulated in [`docs/methodology_training.md`](docs/methodology_training.md).

```bash
# GAN family — 32 experiments (TensorFlow). Knobs are module-level constants.
conda activate ious2mr-tf
python src/gan/train_all_experiments.py

# ResViT — 8 experiments
conda activate ious2mr-torch
python src/resvit/resvit_final.py --variant {2d|2.5d|2d_3d_refine|full_3d} --target {t2|t2_flair}

# SynDiff — 8 experiments (see docs/methodology_training.md for the epoch selection)
python src/syndiff/eval_all_epochs.py     # after training, to pick the evaluated epoch
```

**Checkpoint selection.** ResViT uses the best phase-2 validation checkpoint on an internal
15 % validation split. For SynDiff the paper's evaluated epoch was chosen by scoring saved
epochs — the selected epochs are documented in `docs/scoring_protocol.md`; if you re-train,
select on a validation split rather than on the test set.

---

## Known deviations to expect

* **Non-determinism.** GPU kernels and data shuffling are not seeded end-to-end; trends
  reproduce, exact decimals do not.
* **Two environments.** The GAN family runs on TensorFlow, everything else on PyTorch; the
  authors ran the TensorFlow half inside WSL2 on Windows.
* **Registration.** The paper used a commercial tool (ImFusion Suite) for the rigid ioUS/MR
  step. Any LC²-based rigid registration should be equivalent; results are mildly sensitive to
  this step, so report which one you used.
