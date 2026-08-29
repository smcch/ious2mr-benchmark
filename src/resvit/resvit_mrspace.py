#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resvit_mrspace.py  —  ResViT 2.5d  (target T2)  retrained on the dataset-registration-MRspace "cropped" set: common space = the MR's native grid
(the MR is kept verbatim), the US is resampled into it and refined with the Learn2Reg MIND-SSC
baseline (reg_aladin + reg_f3d), cropped to the US-FOV cone bbox, MR masked to the cone. The
US therefore sits at the MR's resolution (~1mm) -> smooth, no speckle-grain in the synthesis.

Why a separate script (not resvit_final.py): so the original runs / configs stay
untouched, and the data path + the new output name + the memory-efficient loader are
all isolated here.

Output:  $IOUS2MR_ROOT\\resvit\\output\\ResViT-2.5d-t2_fullres\\
Split:   the SAME subject_split.json (122 train / 31 test; ~5 pairs simply absent from
         the new dataset -> ResViT skips them -> ~118 train / ~30 test actually loaded).

RAM:  the full-res cropped volumes are ~700x570x150 ~= 61M voxels each (the cone bbox is
barely smaller than the full grid) -> loading all of them at native size would need ~100 GB.
Fix: ResViT 2.5d resizes every axial slice to cfg.image_size (256) anyway, so we resize
each volume to (256, 256, Z) right after loading and DISCARD the native-size array (and
drop FLAIR for the t2 target).  Result: ~88 MB/vol in RAM -> ~13 GB for the whole dataset,
fits.  The model sees exactly the same 256x256 slices it would have seen otherwise; the
full-res cropped/whole-brain volumes are kept on disk in the dataset / Test_data_full_res.

Usage:
  cd $IOUS2MR_ROOT\\resvit
  python resvit_mrspace.py                       # 2.5d / t2, defaults from resvit_final.py
  python resvit_mrspace.py --phase1_epochs 100 --phase2_epochs 100   # pass-through args ok
  python resvit_mrspace.py --no_resume           # train from scratch (ignore checkpoints)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from common.paths import PROJECT_ROOT, DATA_ROOT, CKPT_ROOT, EXTERNAL_ROOT  # noqa: E402
import os  # noqa: E402,F811
import os, sys
import numpy as np
from scipy.ndimage import zoom as _zoom

# --- import the faithful ResViT implementation (model, train loop, eval) ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resvit_final as RF

# ============================================================================
# CONFIG OVERRIDES for the _fullres run
# ============================================================================
DATA_DIR   = os.path.join(str(PROJECT_ROOT), *r"dataset-registration-MRspace\cropped".split(chr(92)))   # common space = MR native grid; US registered in (baseline L2R); cropped to US-FOV cone, MR masked
SPLIT_JSON = os.path.join(str(PROJECT_ROOT), "resvit", "subject_split.json")
RUN_SUFFIX = "_mrspace"
VARIANT    = "2.5d"
TARGET     = "t2"

RF.cfg.arch_variant    = VARIANT
RF.cfg.target_mode     = TARGET
RF.cfg.base_data_dir   = DATA_DIR
RF.cfg.us_dir          = os.path.join(DATA_DIR, "US")
RF.cfg.t2_dir          = os.path.join(DATA_DIR, "MR-T2")
RF.cfg.fl_dir          = os.path.join(DATA_DIR, "MR-FLAIR")
RF.cfg.split_json_path = SPLIT_JSON

# IMPORTANT (Windows): the in-RAM `samples` list is now ~13 GB (full-res cropped volumes).
# With num_workers>0 on Windows (spawn), the DataLoader pickles a COPY of that list into
# every worker process -> 13 GB x (1 + num_workers) -> blows past the 64 GB RAM ceiling ->
# memory corruption -> segfault (seen at ~epoch 18). Use num_workers=0: data loading runs
# in the main process (no duplication); the per-item cost is just an in-RAM slice + a no-op
# resize + light aug, so the GPU is barely starved.
RF.cfg.num_workers        = 0
RF.cfg.persistent_workers = False

# append "_fullres" to the output run name -> output/ResViT-2.5d-t2_fullres/
_base_run_name = RF.Config.run_name           # the original property descriptor
RF.Config.run_name = property(lambda self: f"ResViT-{self.arch_variant}-{self.target_mode}{RUN_SUFFIX}")

# ============================================================================
# MEMORY-EFFICIENT VOLUME LOADING
# ============================================================================
_orig_load_volume = RF.load_volume

def _resize_xy(v, sz, order):
    """Resize a (H, W, D) volume to (sz, sz, D) (axial-plane resize only)."""
    H, W = v.shape[0], v.shape[1]
    if H == sz and W == sz:
        return v.astype(np.float32) if order != 0 else v
    out = _zoom(v.astype(np.float32), [sz / H, sz / W, 1.0], order=order, mode="nearest")
    return out.astype(np.float32) if order != 0 else (out > 0.5)

def load_volume_fullres(name, cfg_obj):
    """Load as resvit_final does (incl. normalization & fg union), then immediately
    downsize the axial plane to cfg_obj.image_size and free the native-size arrays.
    Also drops FLAIR when the target is t2 only (saves ~1/3 of the RAM)."""
    s = _orig_load_volume(name, cfg_obj)
    if s is None:
        return None
    sz = int(cfg_obj.image_size)
    s["us"]    = _resize_xy(s["us"],    sz, order=1)
    s["t2"]    = _resize_xy(s["t2"],    sz, order=1)
    s["fg"]    = _resize_xy(s["fg"].astype(np.float32), sz, order=0)   # -> bool
    if cfg_obj.target_mode == "t2":
        # FLAIR not needed -> replace with a tiny placeholder so the dict still has the key
        s["flair"] = np.full((sz, sz, 1), -1.0, dtype=np.float32)
        s["has_fl"] = False
        s["tmask"] = np.array([1.0], dtype=np.float32)
    else:
        s["flair"] = _resize_xy(s["flair"], sz, order=1)
    return s

RF.load_volume = load_volume_fullres

# ============================================================================
if __name__ == "__main__":
    # Make sure the variant/target are not accidentally overridden by argv:
    # resvit_final.main() only overrides cfg if --variant/--target are explicitly passed,
    # so passing nothing keeps the values we set above. Pass-through args (--phase1_epochs,
    # --phase2_epochs, --no_resume, --val_fraction) still work.
    print("=" * 70)
    print(f"resvit_fullres | {VARIANT} / {TARGET} -> output/ResViT-{VARIANT}-{TARGET}{RUN_SUFFIX}")
    print(f"  data : {DATA_DIR}")
    print(f"  split: {SPLIT_JSON}")
    print(f"  (volumes resized to cfg.image_size on load; FLAIR dropped for t2 target)")
    print("=" * 70)
    RF.main()
