"""
run_single_axis_inference.py
============================
Regenerate predictions for all non-3D experiments using only axial (axis=2)
2D/2.5D inference — no triplanar averaging. Axial matches the training axis
(`prepare_2d_slice_dataset` default axis=2), which is the anatomically
high-resolution plane.

For 2d_3dpost variants: runs axial-only 2D → 3D refiner.

Outputs to {exp_dir}/predictions_one/{sid}_pred_t2.nii.gz (and _pred_flair
for multitask).

Usage:
    python run_single_axis_inference.py                 # all 24 non-3D exps
    python run_single_axis_inference.py pix2pix_2d_t2   # subset
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from common.paths import PROJECT_ROOT, DATA_ROOT, CKPT_ROOT, EXTERNAL_ROOT  # noqa: E402
import os  # noqa: E402,F811
import gc
import json
import os
import sys
import traceback

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import load_all_data, save_nifti  # noqa: E402
from architectures_2d import (  # noqa: E402
    build_pix2pix_generator_2d,
    build_swin_generator_2d,
    build_cyclegan_generator_2d,
    build_cut_generator_2d,
)
from run_all_experiments import (  # noqa: E402
    build_3d_refiner,
    _sliding_window_inference_3d,
    PATCH_3D,
    PATCH_3D_OVERLAP,
)

BASE_DIR = HERE
DATA_DIR = os.path.join(str(PROJECT_ROOT), "data_cropped_192")
SPLIT_FILE = os.path.join(BASE_DIR, "subject_split.json")
PATCH = 192

ARCHS = ["pix2pix", "swinpix2pix", "cyclegan", "cut"]
VARIANTS = ["2d", "25d", "2d_3dpost"]
TARGETS = ["t2", "t2_flair"]


def build_generator(arch, is_25d, is_multitask):
    input_ch = 3 if is_25d else 1
    output_ch = 2 if is_multitask else 1
    if arch == "pix2pix":
        G = build_pix2pix_generator_2d(input_ch, output_ch)
    elif arch == "swinpix2pix":
        G = build_swin_generator_2d(input_ch, output_ch)
    elif arch == "cyclegan":
        G = build_cyclegan_generator_2d(input_ch, output_ch)
    elif arch == "cut":
        G = build_cut_generator_2d(input_ch, output_ch)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    _ = G(tf.zeros((1, PATCH, PATCH, input_ch)), training=False)
    return G, input_ch, output_ch


def _run_model(G, inp):
    out = G(inp, training=False)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out.numpy()


def _predict_axial_slice(G, sl, input_ch, output_ch):
    """sl: (H, W) or (H, W, input_ch). Axial on 192x192 volumes fits PATCH."""
    if sl.ndim == 2:
        sl = sl[..., np.newaxis]
    H, W, _ = sl.shape
    if H > PATCH or W > PATCH:
        raise RuntimeError(
            f"Axial slice {H}x{W} larger than PATCH={PATCH}; "
            f"volumes are expected to be 192x192xD.")
    patch = np.full((PATCH, PATCH, input_ch), -1.0, dtype=np.float32)
    patch[:H, :W, :] = sl.astype(np.float32)
    pred = _run_model(G, patch[np.newaxis])[0]
    return pred[:H, :W, :]


def axial_inference(G, us_volume, is_25d, output_channels):
    """Single-axis (axial, axis=2) inference."""
    input_ch = 3 if is_25d else 1
    H, W, D = us_volume.shape
    if output_channels == 1:
        pred_vol = np.zeros((H, W, D), dtype=np.float32)
    else:
        pred_vol = np.zeros((H, W, D, output_channels), dtype=np.float32)
    for i in range(D):
        if is_25d:
            ip = max(0, i - 1)
            inext = min(D - 1, i + 1)
            sl = np.stack([
                us_volume[:, :, ip],
                us_volume[:, :, i],
                us_volume[:, :, inext],
            ], axis=-1)
        else:
            sl = us_volume[:, :, i]
        pred_hw = _predict_axial_slice(G, sl, input_ch, output_channels)
        if output_channels == 1:
            pred_vol[:, :, i] = pred_hw[..., 0]
        else:
            pred_vol[:, :, i, :] = pred_hw
    return np.clip(pred_vol, -1, 1)


def run_experiment(exp_name, arch, variant, target, test_data):
    exp_dir = os.path.join(BASE_DIR, exp_name)
    pred_dir = os.path.join(exp_dir, "predictions_one")
    os.makedirs(pred_dir, exist_ok=True)

    is_25d = (variant == "25d")
    is_3dpost = (variant == "2d_3dpost")
    is_multitask = (target == "t2_flair")
    output_ch = 2 if is_multitask else 1

    # The 2D generator inside 2d_3dpost was trained as "2d" (not 2.5D).
    gen_is_25d = is_25d and not is_3dpost
    G, input_ch, _ = build_generator(arch, gen_is_25d, is_multitask)

    if is_3dpost:
        gen_ckpt = os.path.join(
            BASE_DIR, f"{arch}_2d_{target}", "checkpoints",
            "generator_ema.weights.h5")
    else:
        gen_ckpt = os.path.join(exp_dir, "checkpoints",
                                "generator_ema.weights.h5")
    if not os.path.exists(gen_ckpt):
        print(f"[SKIP] {exp_name}: no generator checkpoint at {gen_ckpt}")
        return
    print(f"  Loading generator: {gen_ckpt}")
    G.load_weights(gen_ckpt)

    refiner = None
    if is_3dpost:
        ref_ckpt = os.path.join(exp_dir, "checkpoints",
                                "refiner_ema.weights.h5")
        if not os.path.exists(ref_ckpt):
            print(f"[SKIP] {exp_name}: no refiner at {ref_ckpt}")
            return
        refiner = build_3d_refiner(input_channels=output_ch)
        pH, pW, pD = PATCH_3D
        _ = refiner(tf.zeros((1, pH, pW, pD, output_ch)), training=False)
        print(f"  Loading refiner: {ref_ckpt}")
        refiner.load_weights(ref_ckpt)

    valid = {k: v for k, v in test_data.items()
             if (not is_multitask) or (v.get("flair") is not None)}

    print(f"=== {exp_name} ({len(valid)} subjects) ===")
    for sid in sorted(valid.keys()):
        us_vol = valid[sid]["us"]
        pred = axial_inference(G, us_vol, gen_is_25d, output_ch)

        if is_3dpost:
            feed = pred  # (H,W,D) if output_ch=1 else (H,W,D,C)
            refined = _sliding_window_inference_3d(
                refiner, feed, PATCH_3D, PATCH_3D_OVERLAP,
                is_cut=False, output_channels=output_ch)
            pred = refined

        if is_multitask:
            pred_t2 = pred[..., 0] if pred.ndim == 4 else pred
            save_nifti(pred_t2,
                       os.path.join(pred_dir, f"{sid}_pred_t2.nii.gz"))
            if pred.ndim == 4 and pred.shape[-1] >= 2:
                pred_fl = pred[..., 1]
                save_nifti(pred_fl,
                           os.path.join(pred_dir, f"{sid}_pred_flair.nii.gz"))
        else:
            pv = pred[..., 0] if pred.ndim == 4 else pred
            save_nifti(pv, os.path.join(pred_dir, f"{sid}_pred_t2.nii.gz"))
        print(f"  {sid}: saved")

    del G
    if refiner is not None:
        del refiner
    tf.keras.backend.clear_session()
    gc.collect()


def main():
    print("Loading data...")
    all_data = load_all_data(DATA_DIR)
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    test_ids = set(split["test"])
    test_data = {k: v for k, v in all_data.items() if k in test_ids}
    print(f"  {len(test_data)} test studies")

    experiments = []
    for arch in ARCHS:
        for variant in VARIANTS:
            for target in TARGETS:
                experiments.append(
                    (f"{arch}_{variant}_{target}", arch, variant, target))

    if len(sys.argv) > 1:
        wanted = set(sys.argv[1:])
        experiments = [e for e in experiments if e[0] in wanted]
        print(f"Filtering to {len(experiments)} experiment(s): "
              f"{[e[0] for e in experiments]}")

    for exp_name, arch, variant, target in experiments:
        try:
            run_experiment(exp_name, arch, variant, target, test_data)
        except Exception as ex:
            traceback.print_exc()
            print(f"[FAIL] {exp_name}: {ex}")

    print("\nDONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
