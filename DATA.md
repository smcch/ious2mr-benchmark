# Data availability and access

This repository contains **code, configuration and derived numerical results only**. No imaging
data is committed here. This document explains how to obtain the data needed to reproduce the
benchmark, and what cannot be shared.

---

## 1. ReMIND — the benchmark cohort (public)

All models were trained and evaluated on the **ReMIND** database (Brain Resection Multimodal
Imaging Database, Brigham and Women's Hospital), which is publicly available from The Cancer
Imaging Archive (TCIA) under a **CC BY 4.0** licence.

* Collection page: <https://www.cancerimagingarchive.net/collection/remind/>
* DOI: [10.7937/3RAG-D070](https://doi.org/10.7937/3RAG-D070)
* Reference publication: Juvekar et al., *Scientific Data* 11, 494 (2024).
  <https://doi.org/10.1038/s41597-024-03295-z>

**Required citation** when using ReMIND or anything derived from it:

> Juvekar, P., Dorent, R., Kögl, F., et al. (2023). *The Brain Resection Multimodal Imaging
> Database (ReMIND)* (Version 1) [Data set]. The Cancer Imaging Archive.
> https://doi.org/10.7937/3RAG-D070

### Cohort used here

After quality-driven exclusion (poor acoustic coupling, large shadowing artefacts, near-empty
foreground, marked probe-induced distortion), the working cohort was:

| | Subjects | Paired studies |
|---|---|---|
| Total (ioUS / T2w pairs) | 77 | 152 |
| — of which also have FLAIR | — | 102 |
| Training split | 61 | 122 |
| Held-out test split | 16 | 30 (16 pre-resection, 14 post-resection) |

The exact subject-level split is committed in this repository (see `configs/`) so that the
partition can be reproduced without re-running the split logic.

### Derived ReMIND artefacts released with this repository

Because ReMIND is CC BY 4.0, derived works may be redistributed with attribution. The
accompanying data release therefore includes:

* the **manual reference segmentations** (tumour and resection cavity) created for the
  downstream evaluation, drawn on the pre-processed T2w and FLAIR volumes;
* the **per-subject metric tables** for all 48 experiments (fidelity, ROI-restricted and
  downstream endpoints).

These allow every table and figure in the paper to be regenerated without a GPU and without
re-downloading ReMIND. See `WEIGHTS.md` for the release location.

---

## 2. External validation cohort (private — not shareable)

The paper's external pilot uses navigated intraoperative ultrasound sweeps and preoperative MR
examinations from three patients operated at an external institution (University Hospital
Bratislava, Slovakia). **These data are not publicly available and are not included in this
release.** Their retrospective use was approved by the Ethics Committee of the University
Hospital Bratislava, Academician L. Dérer Hospital (approval no. 13/2026), and all patients gave
written informed consent; the approval does not permit redistribution of the images, of
patient-level metadata or of analysis artefacts that could be traced to an individual sweep.

What *is* provided for that analysis is its complete protocol, in `docs/external_pilot.md` (data
organisation, frozen-model inference, cone-masked scoring, label propagation and downstream
evaluation), so that it can be repeated on any comparable cohort of navigated pre-resection
ioUS sweeps with paired preoperative T2w/FLAIR, together with the script that draws the
external-pilot figure from such an analysis (`src/figures/make_fig_external.py`). Neither the
data nor any per-sweep result is distributed.

---

## 3. Trained model weights

The trained generators for all 48 experiments, and the two nnU-Net downstream segmentation
models, are released separately (they exceed GitHub's file-size limits). See **`WEIGHTS.md`**
for the archive location, contents and the licence note that applies to the diffusion models.

## Reference segmentations

The 297 manual tumour and resection-cavity label maps created for the downstream
evaluation (150 on the pre-processed MR grid, 147 on the ultrasound grid) are archived
separately at Zenodo, [10.5281/zenodo.22214974](https://doi.org/10.5281/zenodo.22214974),
under the same CC BY 4.0 licence as ReMIND.
