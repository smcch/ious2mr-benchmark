# Publishing checklist

What is done, and what only you can decide or provide, before this repository and the
weights archive go public. Ordered by what blocks publication.

## ⛔ Before anything is pushed anywhere

- [ ] **Purge the PHI that lives elsewhere in the working tree.** The audit found, *outside*
      this repository but inside `$IOUS2MR_SOURCE_TREE`, identifiable data from a third
      cohort that is not mentioned in the paper: full patient names and 6-digit medical-record
      numbers used as filenames under `registration_prototype/realtime_video/low_grade_outputs/`,
      the same identifiers inside a committed JSON, and family names written into
      `registration_prototype/realtime_video/METHODOLOGY.md`.
      Nothing of this was copied here (the assembly script blocks those paths and `.gitignore`
      denies them), but **never publish, zip or share that tree as-is**, and consider whether
      those files should exist at all in their current form.
- [x] **External-pilot code: excluded, decided.** Nothing from that cohort ships — no data, no
      per-sweep artefacts, no scripts carrying its label scheme (the `BRA-*` labels are not
      one-per-patient, so they would leak the patient↔sweep crosswalk).
      `docs/external_pilot.md` documents the data organisation and protocol so the analysis
      can be repeated on another cohort, and the manuscript's availability statement was
      reworded to match.

## 📄 Placeholders to fill

- [x] GitHub repository created (private): https://github.com/smcch/ious2mr-benchmark — URLs updated in `README.md` and `CITATION.cff`. Flip to public when the paper is submitted/accepted.
- [x] Zenodo DOI and record id → filled in `README.md`, `WEIGHTS.md`,
      `configs/weights_manifest.json` (`base_url`) and the paper:
      weights **10.5281/zenodo.22213625** (record 22213626),
      reference segmentations **10.5281/zenodo.22214974** (record 22214975).
      The code itself is distributed through GitHub only, with no Zenodo archive or DOI
      (author's decision, 2026-09-03); the manuscript no longer claims one.
- [x] `[TODO — journal]` → `Medical Image Analysis` in `CITATION.cff`. ORCIDs and
      `date-released` still commented out there.
- [ ] `[IRB-PENDING]` in the manuscript → the external centre's approval number.
- [ ] **Open the Zenodo files at submission.** Both records are published with *restricted*
      files, while the manuscript states that the weights and the segmentations are openly
      available. Set both to Open Access before the paper is sent out.
- [ ] **Upload the weights as the 66 individual files, not as one archive.** The record
      currently holds a single 4.5 GB `.rar`, but `scripts/fetch_weights.py` builds one URL
      per file from `base_url` (`.../records/22213626/files/gan/<experiment>/<file>`), so
      `--family` / `--experiment` and the SHA-256 verification only work once the files are
      uploaded with their directory structure. The record's own description also promises
      per-family download. Same for the segmentations record (single `.rar`, and its listed
      size of 1.3 MB does not match the ~297 label maps it should contain — check the upload).

## 📦 Weights archive

- [x] Curated and staged: **66 files, 4.55 GiB**, at `$IOUS2MR_RELEASE_DIR`
      (default `./weights-release`), with SHA-256 in `configs/weights_manifest.json`.
- [ ] Upload to Zenodo (50 GB per record — this fits comfortably), then paste the record id
      into `configs/weights_manifest.json` and re-commit.
- [ ] State on the Zenodo record that the `syndiff/` files are **non-commercial research use
      only** (NVIDIA Source Code License), and that everything else follows the repository
      licence.
- [ ] Verify the download path end to end: `python scripts/fetch_weights.py --family resvit`.

## 🔁 Reproducibility gaps worth closing (not blocking)

- [ ] **Stage 0 registration.** `src/data/preprocess_remind.py` implements resampling,
      cropping and normalisation, but the rigid ioUS/MR step used a commercial tool
      (ImFusion Suite). Either publish the transforms as a JSON, or document an open
      alternative (e.g. an LC²/MIND-SSC rigid registration) so the chain is fully open.
- [ ] **Scoring harness fixture.** `src/scoring/rescore_all.py` reads prediction volumes,
      which are not released (~14 GB). Consider shipping one method's test-split predictions
      so the metric code can be verified end to end, or state in `REPRODUCING.md` that only
      the table-building step is runnable from the shipped CSVs.
- [ ] **Qualitative figure.** No script generates the qualitative montage; those panels are
      ReMIND-derived and therefore publishable if you add one.
- [ ] **Reference segmentations.** The manual tumour/cavity labels are ReMIND-derived
      (CC BY 4.0) and are the most valuable artefact for others: add them to the Zenodo record
      and reference them from `DATA.md`.
- [ ] Optional polish: a `pyproject.toml` so `src/` is importable as a package instead of via
      `sys.path`, a smoke test, and CI that at least byte-compiles everything.

## ✅ Already verified

- No secrets, credentials or personal identifiers in the published tree.
- No absolute paths: everything resolves through `IOUS2MR_*` environment variables.
- Every Python file byte-compiles.
- `.gitignore` blocks imaging data, checkpoints, private cohorts, virtualenvs, PDFs and
  secrets — verified with `git check-ignore` on representative paths.
- Upstream licences reproduced verbatim in `licenses/`, with the non-commercial restriction
  on the diffusion arm documented in `THIRD_PARTY_NOTICES.md`, `README.md` and `WEIGHTS.md`.
- ReMIND's CC BY 4.0 terms verified against TCIA; attribution recorded in `DATA.md` and
  `results/LICENSE`.
