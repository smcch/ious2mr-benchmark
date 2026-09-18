#!/usr/bin/env python3
"""Stage the curated model weights for the public archive and write the manifest.

Copies only the checkpoints actually used to produce the paper's numbers (66 files, ~4.6 GB) out of the ~40 GB of training snapshots, into

    E:\\SINTESIS\\ious2mr-weights-release\\

with the layout documented in WEIGHTS.md, and writes configs/weights_manifest.json
(size + SHA-256 per file) so scripts/fetch_weights.py can verify downloads.

Run once before uploading the archive:  python scripts/_stage_weights_release.py
"""
from __future__ import annotations

import os

import hashlib
import json
import shutil
import sys
from pathlib import Path

SRC = Path(os.environ.get("IOUS2MR_SOURCE_TREE", "."))
OUT = Path(os.environ.get("IOUS2MR_RELEASE_DIR", "./weights-release"))
MANIFEST = Path(__file__).resolve().parents[1] / "configs" / "weights_manifest.json"

GAN_FAMILIES = ["pix2pix", "swinpix2pix", "cyclegan", "cut"]
GAN_TARGETS = ["t2", "t2_flair"]
RESVIT_REGIMES = ["2d", "2.5d", "2d_3d_refine", "full_3d"]

# SynDiff: experiment -> (directory holding the diffusive generator, chosen epoch),
# (the evaluated checkpoints, also listed in configs/weights_manifest.json).
# The "3drefine" experiments are a frozen 2D diffusion Stage-1 (taken from the parent
# experiment at the stated epoch) plus a 3D ResNet refiner trained on top, so their
# generator lives in the parent directory and only refiner_20.pth is theirs.
SYNDIFF_CHOSEN = {
    "syndiff_us_t2_paired": ("syndiff_us_t2_paired", 40, None),
    "syndiff_us_t2_paired_25d": ("syndiff_us_t2_paired_25d", 40, None),
    "syndiff_us_t2_paired_3d": ("syndiff_us_t2_paired_3d", 140, None),
    "syndiff_us_t2_paired_3drefine": ("syndiff_us_t2_paired", 180, "refiner_20.pth"),
    "syndiff_us_t2flair_paired": ("syndiff_us_t2flair_paired", 40, None),
    "syndiff_us_t2flair_paired_25d": ("syndiff_us_t2flair_paired_25d", 60, None),
    "syndiff_us_t2flair_paired_3d": ("syndiff_us_t2flair_paired_3d", 100, None),
    "syndiff_us_t2flair_paired_3drefine": ("syndiff_us_t2flair_paired", 40, "refiner_20.pth"),
}


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def plan() -> list[tuple[str, str, str, Path]]:
    """Return (family, experiment, dest_rel, src_path)."""
    items: list[tuple[str, str, str, Path]] = []

    # --- GAN: 24 generators (2d / 25d / 3d) + 8 refiners (2d_3dpost) ---
    for fam in GAN_FAMILIES:
        for tgt in GAN_TARGETS:
            for regime in ["2d", "25d", "3d"]:
                exp = f"{fam}_{regime}_{tgt}"
                p = SRC / "COMPARATIVA-3" / exp / "checkpoints" / "generator_ema.weights.h5"
                items.append(("gan", exp, f"gan/{exp}/generator_ema.weights.h5", p))
            exp = f"{fam}_2d_3dpost_{tgt}"
            p = SRC / "COMPARATIVA-3" / exp / "checkpoints" / "refiner_ema.weights.h5"
            items.append(("gan", exp, f"gan/{exp}/refiner_ema.weights.h5", p))

    # --- ResViT: phase-2 best per experiment (+ any refinement head) ---
    for regime in RESVIT_REGIMES:
        for tgt in ["t2", "t2_flair"]:
            exp = f"ResViT-{regime}-{tgt}"
            base = SRC / "resvit" / "output" / exp / "checkpoints"
            items.append(("resvit", exp, f"resvit/{exp}/p2_best.pth", base / "p2_best.pth"))
            ref = base / "refine3d_best.pth"
            if ref.exists():
                items.append(("resvit", exp, f"resvit/{exp}/refine3d_best.pth", ref))

    # --- SynDiff: the evaluated checkpoint per experiment ---
    for exp, (gen_dir, ep, refiner) in SYNDIFF_CHOSEN.items():
        gen = SRC / "synthdiff" / "output" / gen_dir / f"gen_diffusive_2_{ep}.pth"
        items.append(("syndiff", exp, f"syndiff/{exp}/gen_diffusive_2_{ep}.pth", gen))
        if refiner:
            items.append(("syndiff", exp, f"syndiff/{exp}/{refiner}",
                          SRC / "synthdiff" / "output" / exp / refiner))

    # --- nnU-Net downstream models: 5 folds each ---
    for ds in ["Dataset501_T2", "Dataset502_FLAIR"]:
        base = SRC / "downstream_seg" / "nnUNet_results" / ds / \
            "nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres"
        for fold in range(5):
            p = base / f"fold_{fold}" / "checkpoint_best.pth"
            items.append(("nnunet", f"{ds}/fold_{fold}",
                          f"nnunet/{ds}/nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres/"
                          f"fold_{fold}/checkpoint_best.pth", p))
        for extra in ["dataset.json", "plans.json", "dataset_fingerprint.json"]:
            p = base / extra
            if p.exists():
                items.append(("nnunet", f"{ds}/{extra}",
                              f"nnunet/{ds}/nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres/{extra}", p))
    return items


def main() -> int:
    items = plan()
    missing = [(f, e, str(p)) for f, e, _d, p in items if not p.exists()]
    present = [(f, e, d, p) for f, e, d, p in items if p.exists()]

    total = sum(p.stat().st_size for _f, _e, _d, p in present)
    print(f"staging {len(present)} files ({total/2**30:.2f} GiB) -> {OUT}")
    if missing:
        print(f"\nMISSING ({len(missing)}):")
        for f, e, p in missing:
            print(f"   [{f}] {e}: {p}")

    files = []
    for i, (fam, exp, dest, src) in enumerate(present, 1):
        d = OUT / dest
        d.parent.mkdir(parents=True, exist_ok=True)
        if not d.exists() or d.stat().st_size != src.stat().st_size:
            shutil.copy2(src, d)
        digest = sha256(d)
        files.append({
            "family": fam,
            "experiment": exp,
            "path": dest,
            "size_bytes": d.stat().st_size,
            "sha256": digest,
        })
        if i % 10 == 0 or i == len(present):
            print(f"  {i}/{len(present)}")

    manifest = {
        "description": "Trained weights for the ioUS->MR synthesis benchmark. "
                       "See WEIGHTS.md; the syndiff family is non-commercial research use only.",
        "base_url": "https://zenodo.org/records/[TODO-RECORD]/files",
        "total_bytes": sum(f["size_bytes"] for f in files),
        "n_files": len(files),
        "files": files,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nmanifest -> {MANIFEST}  ({len(files)} files, "
          f"{manifest['total_bytes']/2**30:.2f} GiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
