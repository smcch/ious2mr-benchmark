# External pilot: running the benchmark on your own cohort

The external validation in the paper used private clinical data that cannot be shared
(see [`../DATA.md`](../DATA.md)). The code is published so the analysis can be repeated on any
comparable cohort. This note describes the data contract it expects.

## What the analysis needs

Per patient, one or more **pre-resection** navigated 3D ioUS sweeps, each already resampled
into the space of that patient's preoperative MR (the paper's cohort was exported already
co-registered by the neuronavigation system), plus:

* `t2.nii.gz` — preoperative T2w, native grid;
* `flair.nii.gz` — preoperative 3D FLAIR (optional; needed for the FLAIR channel);
* `preop-<k>_in_t2.nii.gz` — sweep *k* resampled into the T2w grid;
* `preop-<k>_in_flair.nii.gz` — the same sweep resampled into the FLAIR grid;
* `tumor_seg_t2.nii.gz` — a tumour label drawn once per patient on the native T2w
  (label 1 = tumour; propagated automatically to every sweep grid).

Layout:

```
$IOUS2MR_EXTERNAL/
  subject-1/{t2,flair,tumor_seg_t2}.nii.gz
  subject-1/preop-1_in_t2.nii.gz  preop-1_in_flair.nii.gz
  subject-1/preop-2_in_t2.nii.gz  ...
  subject-2/...
```

Only pre-resection sweeps are valid: after resection the preoperative MR is no longer a
correct reference for the ultrasound field, so post-resection sweeps cannot be scored this way
without intraoperative MR.

## What the analysis reports

1. **Fidelity ranking transfer** — Spearman correlation between the internal (ReMIND) ranking
   and the external ranking of the 48 experiments, per metric. Absolute values are *not*
   comparable across cohorts (different acquisition, different scoring mask); the ranking is.
2. **Within-patient stability** — the per-patient standard deviation of each metric across that
   patient's sweeps, which measures robustness to probe placement.
3. **Downstream retention** — the frozen Seg-T2 / Seg-FLAIR models applied to the real MR and
   to every synthetic volume, scored against the propagated tumour label.

Interpret downstream retention with care: the segmentation models were trained on ReMIND, so
the *real*-MR reference of an external centre is itself domain-shifted. Retention above 100 %
means the models read the synthetic image better than that centre's real MR, not that synthesis
beats real anatomy.

## Scoring mask

External scoring is restricted to the ultrasound cone (the region where synthesis is defined);
outside it the generators only ever saw background. This differs from the internal protocol
(target-intensity foreground), which is another reason absolute values are not comparable
between cohorts.
