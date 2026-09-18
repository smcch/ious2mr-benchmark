# Downstream segmentation methodology

*Validating the utility of synthetic US→MRI for tumor/cavity segmentation.*
*Last updated: 2026-05-15.*

---

## 1. Question and design

The synthesis benchmark (see `paper_assets/METHODOLOGY.md`) measures **image quality** of synthetic MRI (SSIM, PSNR, MAE, LPIPS). The downstream test answers a different question:

> Is a synthetic MRI good enough that a tumor/cavity segmentation model trained on real MRI still works when given the synthetic image at test time?

Two independent nnU-Net v2 segmentation models, one per modality:

| | **Seg-T2** | **Seg-FLAIR** |
|---|---|---|
| Train input | 1 ch, real T2 | 1 ch, real FLAIR |
| Train studies | 117 (61 subj, ReMIND) | 81 (54 subj) |
| Test studies | **29** (16 unique subj) | **19** (13 unique subj) |
| Configuration | `3d_fullres` only | `3d_fullres` only |
| Trainer | `nnUNetTrainer_500epochs` | `nnUNetTrainer_500epochs` |
| CV | 5-fold by **subject** | 5-fold by **subject** |
| Inference | Ensemble of 5 folds | Ensemble of 5 folds |
| Test sets | **51**: REAL + 32 GAN (T2 single + T2 channel of dual) + 8 ResViT (T2 single + dual) + 10 SynDiff (T2 single + dual) | **25**: REAL + 16 GAN dual (FLAIR channel) + 4 ResViT dual + 4 SynDiff dual |

The same 16 test subjects (31 studies) from `resvit/subject_split.json` are used; the segmentation cohort shrinks to whatever subset has a `*-mri-segmentation.nii.gz` in `$IOUS2MR_ROOT\Segmentations\MRI\`. Effective cohorts: **T2 n=29** (16 subj), **FLAIR n=19** (13 subj). The two studies that drop out are `ReMIND-004-post` (no seg + no T2) and `ReMIND-023-post` (no T2 in cropped dataset).

A second observer added the 3 previously-missing MRI segs (ReMIND-004-pre, 079-post, 109-pre) on 2026-05-15, which expanded the cohort from the initial n=26/17 to the final n=29/19. Results are computed on the expanded cohort.

## 2. Data

Source folders:
- Real T2: `$IOUS2MR_ROOT\dataset-registration-corrected-cropped\MR-T2\<study>-mri.nii.gz`
- Real FLAIR: `$IOUS2MR_ROOT\dataset-registration-corrected-cropped\MR-FLAIR\<study>-mri.nii.gz`
- Segmentations (MRI space): `$IOUS2MR_ROOT\Segmentations\<study>-mri-segmentation.nii.gz`
  - Note: `ReMIND-034-post` is saved as `…-mir-segmentation.nii.gz` (typo upstream); handled as an alias.

Label values (verified across the cohort):
- 0 = background
- 1 = tumor (most pre-op cases; some post-op)
- 2 = cavity (most post-op cases; some pre-op cases also have it)

Voxel counts in train cohort: bg ≈ 38 M, tumor ≈ 1.1 M (~2.7 %), cavity ≈ 0.2 M (~0.5 %).

## 3. nnU-Net pipeline

### 3.1 No manual preprocessing of intensities
nnU-Net v2 detects modality `"MR"` in `dataset.json` and applies its own *per-volume foreground-mean z-score* normalization to both training and inference inputs. Real MRI (raw intensities 0-1000) and synthetic outputs ([-1, 1] or [0, 1]) are therefore mapped to comparable distributions internally; no manual rescaling is applied. The only manual step is **spatial alignment** of GAN predictions (see §3.4).

### 3.2 Dataset layout
```
nnUNet_raw/Dataset501_T2/
  imagesTr/<study>_0000.nii.gz      117 files
  labelsTr/<study>.nii.gz           117 files
  dataset.json                      channels={0:"T2"}, labels={bg,tumor,cavity}
nnUNet_raw/Dataset502_FLAIR/        same shape with 81 files, channels={0:"FLAIR"}
```

### 3.3 Splits
5-fold by subject, stratified so a patient never appears in train and val of the same fold. Written explicitly to `nnUNet_preprocessed/Dataset5{01,02}_*/splits_final.json` before training — nnU-Net respects existing splits (no shuffling). Per-fold sizes:

| | T2 train per fold | T2 val per fold | FLAIR train per fold | FLAIR val per fold |
|---|---|---|---|---|
| fold 0 | 92 (48 s) | 25 (13 s) | 66 (43 s) | 15 (11 s) |
| fold 1 | 94 (49 s) | 23 (12 s) | 62 (43 s) | 19 (11 s) |
| fold 2 | 94 (49 s) | 23 (12 s) | 67 (43 s) | 14 (11 s) |
| fold 3 | 95 (49 s) | 22 (12 s) | 64 (43 s) | 17 (11 s) |
| fold 4 | 93 (49 s) | 24 (12 s) | 65 (44 s) | 16 (10 s) |

### 3.4 Plans (from nnUNetv2_plan_and_preprocess)
Both datasets, `3d_fullres`:
- Median image size: 46 × 102 × 99 (T2) / 46 × 102 × 100 (FLAIR)
- Spacing: 2.0 × 0.86 × 0.86 mm
- Patch size: 48 × 112 × 112
- Batch size: 5 (T2) / 3 (FLAIR)
- Architecture: PlainConvUNet, 5 stages, features (32, 64, 128, 256, 320), InstanceNorm3d, LeakyReLU, batch_dice=False
- Normalization: ZScoreNormalization, no foreground mask required (use_mask_for_norm=False)
- `3d_lowres` dropped automatically (median size equals fullres).

### 3.5 Training
- Trainer: `nnUNetTrainer_500epochs`
- Loss: Dice + CE, deep supervision
- Optimizer: SGD with Nesterov momentum, lr 0.01, poly LR decay
- Augmentation: nnU-Net default heavy 3D augmentation (rotation, scaling, mirror, gamma, contrast, brightness, Gaussian noise/blur, low-resolution simulation, sometimes elastic)
- Wall-clock per fold (RTX 3090, 24 GB): ~45 s/epoch → ~6.3 h / 500 ep
- Total: 5 folds × 2 datasets ≈ 60-63 h serial
- Checkpoint used for inference: `checkpoint_best.pth` (best EMA pseudo-Dice on val)
- Overfitting watch: training_log.txt + progress.png; if val Dice plateaus or worsens for ~100 ep we stop early. Default 500 ep already a 2× reduction from nnU-Net's default 1000.

### 3.6 Test-input staging
Each test set lives at `test_inputs_<MOD>/<set_name>/<study>_0000.nii.gz`.

- **REAL**: copied from MR-T2 / MR-FLAIR.
- **ResViT** (`resvit\output\ResViT-<arch>-<tgt>\predictions\<study>\pred_{t2,fl}.nii.gz`): already on the seg grid (verified), just reaffix seg affine.
- **SynDiff** (`synthdiff\results\syndiff_*\volumes\<study>_pred{T2,FLAIR}.nii.gz`): same, drop-in.
- **GAN** (`COMPARATIVA-3\<exp>\predictions\<study>_pred_{t2,flair}.nii.gz`): stored on a padded (192, 192, Zg) grid with identity affine. Realigned by resampling in image-index space (scipy.ndimage.zoom, linear, cval=-1) to match the seg's (H, W, Z), then reaffix the seg affine. Handles the three pathological cases (e.g. ReMIND-003-pre with Zg=177 vs Z=89; ReMIND-034-pre with native H,W > 192).

### 3.7 Inference
```bash
nnUNetv2_predict -i test_inputs_<MOD>/<set>/ -o predictions_<MOD>/<set>/ \
  -d 50{1,2} -c 3d_fullres -tr nnUNetTrainer_500epochs -f 0 1 2 3 4 \
  --save_probabilities
```
5-fold soft ensemble, sliding window with tile_step_size=0.5, mirror TTA, predicted argmax saved alongside class probabilities.

## 4. Metrics

For each (set × study × class ∈ {tumor, cavity, lesion}):
- **Dice** = 2|gt ∩ pr| / (|gt| + |pr|); reported NaN when both empty.
- **HD95 (mm)** = 95th-percentile of bidirectional surface-to-surface Euclidean distances (`scipy.ndimage.distance_transform_edt` with spacing). Reported NaN if either gt or pr is empty.
- **NSD@2mm** = Normalized Surface Dice (Nikolov 2018) at 2 mm tolerance.
- **Voxel counts** for gt and pr (sanity).
- **Phase**: `preop` if study endswith `-pre`, else `postop`.

The **`lesion`** class is the binary union of labels {1, 2} on both GT and prediction (i.e. tumor ∪ cavity treated as a single foreground class). It answers the question "did the model find the lesion at all, regardless of whether it labeled the right sub-component?" — useful because cavity ↔ tumor confusion at the boundary can drag down per-class Dice without reflecting a real localization failure. It is computed alongside the per-class metrics; aggregation, Wilcoxon, and the dual cross-modal table all include `lesion`.

Aggregations (per set × class × phase ∈ {all, preop, postop}):
- n_studies, n_dice (not NaN)
- Dice: mean, SD, median, min, max
- HD95: mean, median
- NSD: mean, median

Statistical test:
- Per (class, phase) pick the **set with the highest mean Dice** as reference.
- Paired Wilcoxon signed-rank between every other set and the reference, paired by study (only studies in both).
- Outputs `seg_wilcoxon_{T2,FLAIR}.csv`.

Cross-modal table `seg_dual_T2_vs_FLAIR.csv`:
- For each of the 24 dual-target methods, pair Dice-T2 (from its T2 channel, evaluated by Seg-T2) with Dice-FLAIR (evaluated by Seg-FLAIR) on the same 17 test studies.
- Lets us ask: does synthesising FLAIR in addition to T2w improve downstream segmentation, and which methods benefit most?

## 4.1 Headline results (expanded cohort)

> The per-class values in this section score every study (Dice = 0 where a class is predicted but
> absent), which is the sensitivity-analysis rule of the paper. The paper's primary per-class values
> aggregate only over the studies in which the class is present (real T2w: tumour 0.517, n = 22;
> cavity 0.583, n = 16). The lesion endpoint is identical under both rules.

Seg-T2 (n=29) | REAL Dice | Best synth Dice | Best synth |
|---|---|---|---|
| Tumor | 0.455 | **0.324** (71 %) | ResViT-2D+3D-refine-T2-from-single |
| Cavity | 0.388 | **0.191** (49 %) | SynDiff-3D+3D-refine-T2-from-dual |
| **Lesion** (tumor ∪ cavity) | **0.662** | **0.407** (61 %) | ResViT-2D+3D-refine-T2-from-single |

Seg-FLAIR (n=19) | REAL Dice | Best synth Dice | Best synth |
|---|---|---|---|
| Tumor | 0.389 | **0.323** (83 %) | ResViT-3D-FLAIR |
| Cavity | 0.264 | **0.130** (49 %) | ResViT-2D+3D-refine-FLAIR |
| **Lesion** (tumor ∪ cavity) | **0.511** | **0.373** (73 %) | ResViT-2.5D-FLAIR |

By phase, REAL upper-bounds:
- T2 tumor preop n=16: 0.659; T2 tumor postop n=13: 0.092 (tumor mostly resected)
- T2 cavity postop n=13: 0.646; T2 cavity preop n=16: 0.131
- T2 **lesion** preop n=16: 0.736; T2 **lesion** postop n=13: 0.572
- FLAIR tumor preop n=12: 0.522; FLAIR cavity postop n=7: 0.468
- FLAIR **lesion** preop n=12: 0.575; FLAIR **lesion** postop n=7: 0.400

Best-synth lesion by phase:
- T2 lesion preop: ResViT-2D+3D-refine-T2-from-single 0.546 (74 % of REAL 0.736)
- T2 lesion postop: ResViT-3D-T2-from-single 0.300 (52 % of REAL 0.572)
- FLAIR lesion preop: ResViT-2.5D-FLAIR 0.483 (84 % of REAL 0.575)
- FLAIR lesion postop: ResViT-3D-FLAIR 0.231 (58 % of REAL 0.400)

Takeaways: (1) **synthetic FLAIR retains more downstream utility than synthetic T2w** in every view — tumour 83 % vs 71 %, lesion 73 % vs 61 %; (2) **the resection cavity is the bottleneck** in both modalities (~49 % retention), being the class with ~0.5 % of the voxels; (3) the **lesion** metric confirms that the model does localise the diseased region even when it gets the sub-label wrong — the real-versus-synthetic gap is smaller than for tumour or cavity separately, so part of the per-class loss comes from tumour↔cavity confusion at their shared border rather than from missed detection; (4) **ResViT and SynDiff dominate the top five** (the LPIPS leaders outperform the SSIM leaders downstream); (5) outlier: `GAN-cut-3D-FLAIR` collapses to Dice 0.004 on the cavity but recovers Dice 0.190 on the post-resection lesion.

## 5. File map

```
$IOUS2MR_ROOT\downstream_seg\
  build_datasets.py             # builds Dataset501/502 raw + splits
  stage_test_inputs.py          # builds test_inputs_{T2,FLAIR}
  train_one_fold.sh             # single-fold trainer wrapper
  train_remaining_folds.sh      # waits for fold 0, chains folds 1-4 of 501 + 0-4 of 502
  run_inference.sh              # 5-fold ensemble predict for every test set
  compute_seg_metrics.py        # Dice/HD95/NSD + aggregates + Wilcoxon
  nnUNet_raw/                   # dataset NIfTI (training)
  nnUNet_preprocessed/          # preprocessed cases + plans + splits_final.json
  nnUNet_results/Dataset5{01,02}_*/nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres/
                                # checkpoints, training_log, progress.png, val
  test_inputs_T2/<set>/<study>_0000.nii.gz       51 sets × ≤26 studies
  test_inputs_FLAIR/<set>/<study>_0000.nii.gz    25 sets × ≤17 studies
  predictions_T2/<set>/<study>.nii.gz
  predictions_FLAIR/<set>/<study>.nii.gz
  seg_metrics_T2_per_study.csv
  seg_metrics_FLAIR_per_study.csv
  seg_results_T2.csv  / _FLAIR.csv                # aggregated for paper
  seg_wilcoxon_T2.csv / _FLAIR.csv
  seg_dual_T2_vs_FLAIR.csv
  logs/                                            # training stdout per fold
```

## 6. Environment

- Python env: `$IOUS2MR_PY_TORCH (see envs/environment-pytorch.yml)`
  - torch 2.5.1+cu121, nibabel 5.3.2, SimpleITK 2.4.0, scikit-image 0.25.2, scipy 1.14.1, sklearn 1.7.2
  - nnunetv2 2.7.0 (pip-installed on 2026-05-13)
- GPU: RTX 3090 24 GB (single-GPU serial training)
- OS: Windows 10 Pro; running under PowerShell + Git Bash

Env-vars required for every nnU-Net invocation:
```bash
export nnUNet_raw="$IOUS2MR_ROOT/downstream_seg/nnUNet_raw"
export nnUNet_preprocessed="$IOUS2MR_ROOT/downstream_seg/nnUNet_preprocessed"
export nnUNet_results="$IOUS2MR_ROOT/downstream_seg/nnUNet_results"
```

## 7. Reproduce

```bash
# 1) build datasets + splits (~1 min)
python build_datasets.py

# 2) plan + preprocess (~30 s per dataset)
nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 502 -c 3d_fullres --verify_dataset_integrity

# 3) train (~62 h serial)
for f in 0 1 2 3 4; do bash train_one_fold.sh 501 $f ; done
for f in 0 1 2 3 4; do bash train_one_fold.sh 502 $f ; done

# 4) stage test inputs (~5 min)
python stage_test_inputs.py

# 5) inference ensemble (~12 h)
bash run_inference.sh

# 6) metrics + tables (~10 min)
python compute_seg_metrics.py
```