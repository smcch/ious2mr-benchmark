# Zenodo record descriptions

The exact text published on the two Zenodo records, kept here so the records and the repository
can be checked against each other.

---

## Weights — 10.5281/zenodo.22213625 (record 22213626)

> **Needs updating on Zenodo.** The published description still tells users to run
> `fetch_weights.py --all` and `--family gan` to download subsets. The record holds one RAR
> archive, so subsets cannot be downloaded; the paragraph below replaces the "How to use them"
> paragraph of the live record.

**Title:** Benchmark of Intraoperative Ultrasound-to-MR Synthesis for Brain Tumor Surgery

**How to use them.** The code is at https://github.com/smcch/ious2mr-benchmark. All 66 files
ship as a single archive, `2-zenodo-pesos.rar` (4.55 GiB). Clone the repository, then run
`python scripts/fetch_weights.py --list` for the inventory, sizes and SHA-256 checksums, and
`python scripts/fetch_weights.py --download` to fetch the archive (or download it from this page
directly). Extract it so that the family directories sit at the top of the weights directory —
`unrar x 2-zenodo-pesos.rar weights/`, or `7z x 2-zenodo-pesos.rar -oweights/`, or
`bsdtar -xf 2-zenodo-pesos.rar -C weights/` — and then run
`python scripts/fetch_weights.py --verify` to check every extracted file against its recorded
checksum (`--family gan|resvit|syndiff|nnunet` verifies one family). Finally set `IOUS2MR_CKPT`
to the weights directory. The file `configs/weights_manifest.json` in the repository maps each
experiment name used in the paper to its file, size and SHA-256 checksum.

The rest of the live description (contents, licence, provenance, data, citation) is accurate and
needs no change.

---

## Reference segmentations — 10.5281/zenodo.22214974 (record 22214975)

Published description is accurate. The archive `3-zenodo-segmentaciones.rar` contains 297 label
maps totalling **1.27 MB** — they are sparse binary masks stored gzipped, averaging 4.2 KB each,
so the small size is expected.

---

## Before the paper goes out

Both records are published with **restricted** files while the manuscript states that the
weights and the segmentations are openly available. Switch both to Open Access at submission,
together with making the GitHub repository public.
