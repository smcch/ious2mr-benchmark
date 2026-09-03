#!/usr/bin/env python3
"""Download and verify the released model weights for this benchmark.

The trained generators (48 experiments) and the two nnU-Net downstream segmentation models are
hosted outside GitHub because of their size, in a single Zenodo record:

    https://doi.org/10.5281/zenodo.22213625

The record holds **one RAR archive** containing all 66 files, so the weights are fetched in one
piece and extracted with an external tool; per-family downloading is not possible. What this
script does is make that process checkable: it downloads the archive, tells you how to extract
it, and then verifies every extracted file against the SHA-256 checksums recorded in
``configs/weights_manifest.json``.

Usage
-----
    python scripts/fetch_weights.py --list                # inventory, sizes, checksums
    python scripts/fetch_weights.py --download            # fetch the archive (4.55 GiB)
    python scripts/fetch_weights.py --verify              # check an extracted tree
    python scripts/fetch_weights.py --verify --family resvit
    export IOUS2MR_CKPT=$PWD/weights                      # point the pipeline at them

Extraction (the archive is RAR; any of these works)::

    unrar x 2-zenodo-pesos.rar weights/         # Linux/macOS: apt install unrar / brew install unrar
    7z x 2-zenodo-pesos.rar -oweights/          # 7-Zip, all platforms
    bsdtar -xf 2-zenodo-pesos.rar -C weights/   # libarchive

Extract it so that the family directories land directly under the weights root, i.e.
``weights/gan/...``, ``weights/resvit/...``, ``weights/syndiff/...``, ``weights/nnunet/...``.
``--verify`` reports what it finds and where, so a wrong nesting level is easy to spot.

Licence note
------------
The SynDiff-family weights derive from an implementation governed by the NVIDIA Source Code
License and are released for NON-COMMERCIAL research use only. The script prints this reminder.
See THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "configs" / "weights_manifest.json"
DEFAULT_ROOT = Path(os.environ.get("CKPT_ROOT", REPO_ROOT / "weights"))
NONCOMMERCIAL_FAMILIES = {"syndiff"}

RECORD_DOI = "https://doi.org/10.5281/zenodo.22213625"
RECORD_URL = "https://zenodo.org/records/22213626"
ARCHIVE_NAME = "2-zenodo-pesos.rar"
ARCHIVE_URL = f"{RECORD_URL}/files/{ARCHIVE_NAME}?download=1"


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


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def select(entries: list[dict], args) -> list[dict]:
    out = entries
    if args.family:
        out = [e for e in out if e["family"] == args.family]
    if args.experiment:
        out = [e for e in out if e["experiment"] == args.experiment]
    return out


def licence_notice(entries: list[dict]) -> None:
    if any(e["family"] in NONCOMMERCIAL_FAMILIES for e in entries):
        print("\n*** Licence notice ***\n"
              "The SynDiff-family weights derive from code governed by the NVIDIA Source Code\n"
              "License and are provided for NON-COMMERCIAL research use only.\n"
              "See THIRD_PARTY_NOTICES.md before using them.\n")


def do_list(entries: list[dict]) -> None:
    by_family: dict[str, list[dict]] = {}
    for e in entries:
        by_family.setdefault(e["family"], []).append(e)
    for fam, items in sorted(by_family.items()):
        total = sum(i.get("size_bytes", 0) for i in items)
        flag = "  [NON-COMMERCIAL USE ONLY]" if fam in NONCOMMERCIAL_FAMILIES else ""
        print(f"\n{fam}: {len(items)} files, {human(total)}{flag}")
        for i in sorted(items, key=lambda x: x["experiment"]):
            print(f"    {i['experiment']:38s} {human(i.get('size_bytes', 0)):>10s}  {i['path']}")
    print(f"\nAll of these ship as one archive, {ARCHIVE_NAME}, in {RECORD_DOI}")


def do_download(dest_dir: Path, force: bool) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / ARCHIVE_NAME
    if dest.exists() and not force:
        print(f"Already downloaded: {dest} ({human(dest.stat().st_size)})\n"
              "Re-download with --force, or extract it and run --verify.")
        return 0
    print(f"Downloading {ARCHIVE_NAME} (4.55 GiB) from {RECORD_URL}\n"
          "This is a single large file; expect a long transfer.")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(ARCHIVE_URL, tmp)  # noqa: S310 — fixed, documented URL
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        print(f"\nDownload failed: HTTP {exc.code}.", file=sys.stderr)
        if exc.code in (401, 403, 404):
            print(
                "The record's files may still be under restricted access, or the file name may\n"
                f"have changed. Open {RECORD_DOI} in a browser, download the archive manually\n"
                f"into {dest_dir}, then extract it and run --verify.",
                file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        print(f"\nDownload failed: {exc.reason}", file=sys.stderr)
        return 1
    tmp.replace(dest)
    print(f"\nSaved {dest} ({human(dest.stat().st_size)})\n\n"
          "Now extract it so the family directories sit directly under the weights root:\n"
          f"    unrar x {dest} {dest_dir}/     # or: 7z x {dest} -o{dest_dir}/\n"
          f"    python {Path(__file__).name} --verify --root {dest_dir}")
    return 0


def do_verify(entries: list[dict], root: Path, quick: bool) -> int:
    if not root.exists():
        sys.exit(f"Weights root not found: {root}\n"
                 "Extract the archive there first (see --download).")
    ok = missing = bad = 0
    for e in sorted(entries, key=lambda x: x["path"]):
        p = root / e["path"]
        if not p.exists():
            print(f"  [missing] {e['path']}")
            missing += 1
            continue
        size = p.stat().st_size
        if e.get("size_bytes") and size != e["size_bytes"]:
            print(f"  [size]    {e['path']}: {human(size)}, expected {human(e['size_bytes'])}")
            bad += 1
            continue
        if quick or not e.get("sha256"):
            ok += 1
            continue
        if sha256(p) != e["sha256"]:
            print(f"  [sha256]  {e['path']}: checksum mismatch")
            bad += 1
        else:
            ok += 1
    print(f"\n{ok} verified, {missing} missing, {bad} corrupt (of {len(entries)} expected) in {root}")
    if missing == len(entries):
        print("Nothing found at all: check the extraction nesting: the archive should yield\n"
              f"{root}/gan, {root}/resvit, {root}/syndiff and {root}/nnunet.")
    if missing or bad:
        return 1
    print(f"\nAll good. Point the pipeline at these weights with:\n    export IOUS2MR_CKPT={root}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="list the weights the archive contains, with sizes")
    ap.add_argument("--download", action="store_true",
                    help=f"download {ARCHIVE_NAME} (4.55 GiB) into --root")
    ap.add_argument("--verify", action="store_true",
                    help="verify an extracted tree against the manifest checksums")
    ap.add_argument("--quick", action="store_true",
                    help="with --verify, check presence and size but not SHA-256")
    ap.add_argument("--family", help="gan | resvit | syndiff | nnunet (filters --list/--verify)")
    ap.add_argument("--experiment", help="a single experiment name (filters --list/--verify)")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help=f"weights directory (default: {DEFAULT_ROOT})")
    ap.add_argument("--force", action="store_true", help="re-download an existing archive")
    args = ap.parse_args()

    entries = select(load_manifest()["files"], args)
    if not entries:
        sys.exit("Nothing matched that selection (try --list).")

    if args.list:
        do_list(entries)
        licence_notice(entries)
        return
    if args.download:
        licence_notice(load_manifest()["files"])
        sys.exit(do_download(args.root, args.force))
    if args.verify:
        licence_notice(entries)
        sys.exit(do_verify(entries, args.root, args.quick))

    ap.print_help()
    print(f"\nThe weights live in one archive at {RECORD_DOI}; "
          "use --download, then extract, then --verify.")


if __name__ == "__main__":
    main()
