
## Holm correction and the FLAIR significance claim

The configuration-versus-real Wilcoxon tests are run once per configuration, so they form two
families: T2w lesion (48 tests) and FLAIR lesion (24 tests). Holm-corrected within each family
at alpha = 0.05:

  Seg-T2   lesion: 48/48 remain significant (max adjusted p = 1.1e-4)
  Seg-FLAIR lesion: 10/24 remain significant (max adjusted p = 0.113)

The manuscript's earlier claim that every FLAIR configuration is significantly below real
FLAIR held only uncorrected; it is now reported as 10 of 24.

## Cohort matching (corrected 2026-09-03)

`floor_baseline.py` originally compared each configuration's mean on its own evaluation cohort
against the null control's mean over all 29 studies. The 16 multi-task configurations are scored
on the 19-study subset that carries a real FLAIR, and that subset is the easier one (null control
0.281 there, against 0.251 over the full cohort; real T2w 0.694 against 0.662, because it holds
proportionally more pre-resection studies: 12/7 against 16/13).

Comparing each configuration with the control recomputed on the studies they share:

  below FLOOR_US        17 of 48   (was reported as 12)
  below FLOOR_US_HISTM  19 of 48   (was reported as 9)

The manuscript now reports the cohort-matched counts, in line with the common-cohort rule of
Section 2.6.

## Direct-ioUS reference (Seg-US)

`build_dataset_us.py` -> `run_seg_us_chain.sh` -> `score_seg_us.py` ->
`compare_seg_us_vs_synthesis.py` train and score an nnU-Net on the ioUS volumes themselves
(Dataset503_US), with the same recipe, the same 117 training studies, the same five
subject-level folds and the same MR-drawn labels as Seg-T2; only the input modality differs.

  real T2w (upper reference)   0.662   100 %
  best synthesis               0.407    61 %
  Seg-US (direct ioUS)         0.401    61 %   (difference -0.006, 95 % CI -0.083 to +0.077,
                                                paired Wilcoxon p = 0.81)
  null control (raw ioUS)      0.251    38 %

Per-study scores are in `seg_us_reference.csv`.
