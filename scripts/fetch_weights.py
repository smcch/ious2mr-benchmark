#!/usr/bin/env python3
"""Download the released model weights for this benchmark.

The trained generators (48 experiments) and the two nnU-Net downstream
segmentation models are hosted outside GitHub because of their size. This script
fetches them from the archive described in WEIGHTS.md, verifies their checksums
against ``configs/weights_manifest.json`` and lays them out under ``$CKPT_ROOT``
(default: ``./weights``) in the structure the pipeline expects.

Usage
-----
    python scripts/fetch_weights.py --list
    python scripts/fetch_weights.py --family resvit
    python scripts/fetch_weights.py --all
    python scripts/fetch_weights.py --experiment pix2pix_2d_t2

Licence note
------------
The SynDiff-family weights derive from an implementation governed by the NVIDIA
Source Code License and are released for NON-COMMERCIAL research use only. The
script prints this reminder when those files are selected. See
THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "configs" / "weights_manifest.json"
DEFAULT_ROOT = Path(os.environ.get("CKPT_ROOT", REPO_ROOT / "weights"))
NONCOMMERCIAL_FAMILIES = {"syndiff"}


def load_manifest() -> dict:
    if not MANIFEST.exists():
        sys.exit(
            f"Manifest not found: {MANIFEST}\n"
            "This file ships with the repository; re-clone or restore it."
        )
    with MANIFEST.open(encoding="utf-8") as fh:
        return json.load(fh)


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def select(entries: list[dict], args) -> list[dict]:
    if args.all:
        return entries
    out = entries
    if args.family:
        out = [e for e in out if e["family"] == args.family]
    if args.experiment:
        out = [e for e in out if e["experiment"] == args.experiment]
    if not args.family and not args.experiment:
        sys.exit("Select what to download: --all, --family NAME or --experiment NAME "
                 "(use --list to see the options).")
    return out


def download(entry: dict, base_url: str, root: Path, force: bool) -> None:
    dest = root / entry["path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        if entry.get("sha256") and sha256(dest) == entry["sha256"]:
            print(f"  [ok]   {entry['path']} (already present)")
            return
        print(f"  [warn] {entry['path']} exists with unexpected checksum; re-downloading")
    url = entry.get("url") or f"{base_url.rstrip('/')}/{entry['path']}"
    print(f"  [get]  {entry['path']}  ({human(entry.get('size_bytes', 0))})")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 — fixed, documented archive URL
    if entry.get("sha256"):
        got = sha256(tmp)
        if got != entry["sha256"]:
            tmp.unlink(missing_ok=True)
            sys.exit(f"Checksum mismatch for {entry['path']}:\n  expected {entry['sha256']}\n  got      {got}")
    tmp.replace(dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list available weights and exit")
    ap.add_argument("--all", action="store_true", help="download everything")
    ap.add_argument("--family", help="gan | resvit | syndiff | nnunet")
    ap.add_argument("--experiment", help="a single experiment name")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help=f"destination directory (default: {DEFAULT_ROOT})")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    manifest = load_manifest()
    entries = manifest["files"]

    if args.list:
        by_family: dict[str, list[dict]] = {}
        for e in entries:
            by_family.setdefault(e["family"], []).append(e)
        for fam, items in sorted(by_family.items()):
            total = sum(i.get("size_bytes", 0) for i in items)
            flag = "  [NON-COMMERCIAL USE ONLY]" if fam in NONCOMMERCIAL_FAMILIES else ""
            print(f"\n{fam}: {len(items)} files, {human(total)}{flag}")
            for i in sorted(items, key=lambda x: x["experiment"]):
                print(f"    {i['experiment']:38s} {human(i.get('size_bytes', 0)):>10s}  {i['path']}")
        return

    chosen = select(entries, args)
    if not chosen:
        sys.exit("Nothing matched that selection (try --list).")

    if any(e["family"] in NONCOMMERCIAL_FAMILIES for e in chosen):
        print("\n*** Licence notice ***\n"
              "The SynDiff-family weights derive from code governed by the NVIDIA Source Code\n"
              "License and are provided for NON-COMMERCIAL research use only.\n"
              "See THIRD_PARTY_NOTICES.md before using them.\n")

    total = sum(e.get("size_bytes", 0) for e in chosen)
    print(f"Downloading {len(chosen)} file(s), {human(total)} → {args.root}")
    for entry in chosen:
        download(entry, manifest["base_url"], args.root, args.force)
    print("\nDone. Point the pipeline at these weights with:\n"
          f"    export CKPT_ROOT={args.root}")


if __name__ == "__main__":
    main()
