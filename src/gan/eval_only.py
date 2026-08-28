"""
run_eval_only.py
================
Evaluate 2D / 2.5D experiments that finished training but whose final
multi-subject evaluation failed because test volumes have depth D > 192.

Fix: monkey-patches `triplanar_inference_2d/25d/multitask` in
`run_all_experiments` with versions that tile 192x192 patches (sliding
window) over oversized slices, then reuses `_eval_all_test(_cut)` to
write `results.csv`, comparison figures, and NIfTI predictions.

Usage:
    python run_eval_only.py                       # evaluate all pending
    python run_eval_only.py pix2pix_25d_t2 ...    # evaluate specific ones
"""
import os
import sys
import json
import traceback

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common  # noqa: E402
import run_all_experiments as rae  # noqa: E402
from common import (  # noqa: E402
    load_all_data,
    get_or_create_split,
    get_slice_along_axis,
    set_slice_along_axis,
    log_progress,
    append_results_csv,
)
from architectures_2d import (  # noqa: E402
    build_pix2pix_generator_2d,
    build_swin_generator_2d,
    build_cyclegan_generator_2d,
    build_cut_generator_2d,
)

BASE_DIR = rae.BASE_DIR
DATA_DIR = rae.DATA_DIR
SPLIT_FILE = rae.SPLIT_FILE

PATCH = 192
STRIDE = 96  # 50% overlap


# =============================================================================
# Patched triplanar inference: handles slices where H or W > 192
# =============================================================================
def _patch_starts(length, patch=PATCH, stride=STRIDE):
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def _run_model(generator, inp):
    out = generator(inp, training=False)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out.numpy()


def _predict_slice(generator, sl, input_ch, output_ch):
    """Predict a slice of arbitrary (H, W) shape.

    Args:
        sl: (H, W) for 2D or (H, W, input_ch) for 2.5D.
    Returns:
        (H, W, output_ch) float32 prediction.
    """
    if sl.ndim == 2:
        sl = sl[..., np.newaxis]
    H, W, _ = sl.shape

    if H <= PATCH and W <= PATCH:
        patch = np.full((PATCH, PATCH, input_ch), -1.0, dtype=np.float32)
        patch[:H, :W, :] = sl
        pred = _run_model(generator, patch[np.newaxis].astype(np.float32))[0]
        return pred[:H, :W, :]

    out = np.zeros((H, W, output_ch), dtype=np.float32)
    cnt = np.zeros((H, W, 1), dtype=np.float32)
    for y in _patch_starts(H):
        for x in _patch_starts(W):
            patch = np.full((PATCH, PATCH, input_ch), -1.0, dtype=np.float32)
            ph = min(PATCH, H - y)
            pw = min(PATCH, W - x)
            patch[:ph, :pw, :] = sl[y:y + ph, x:x + pw, :]
            pred = _run_model(generator, patch[np.newaxis].astype(np.float32))[0]
            out[y:y + ph, x:x + pw, :] += pred[:ph, :pw, :]
            cnt[y:y + ph, x:x + pw, :] += 1.0
    return out / np.maximum(cnt, 1e-6)


def _alloc_pred_vol(shape, output_channels):
    H, W, D = shape
    if output_channels == 1:
        return np.zeros((H, W, D), dtype=np.float32)
    return np.zeros((H, W, D, output_channels), dtype=np.float32)


def _write_slice(pred_vol, axis, i, pred_hw, output_channels):
    if output_channels == 1:
        set_slice_along_axis(pred_vol, i, axis, pred_hw[..., 0])
    else:
        if axis == 0:
            pred_vol[i, :, :, :] = pred_hw
        elif axis == 1:
            pred_vol[:, i, :, :] = pred_hw
        else:
            pred_vol[:, :, i, :] = pred_hw


def triplanar_inference_2d_fixed(generator, us_volume, output_channels=1):
    predictions = []
    for axis in range(3):
        pred_vol = _alloc_pred_vol(us_volume.shape[:3], output_channels)
        n = us_volume.shape[axis]
        for i in range(n):
            sl = get_slice_along_axis(us_volume, i, axis)
            pred_hw = _predict_slice(generator, sl, 1, output_channels)
            _write_slice(pred_vol, axis, i, pred_hw, output_channels)
        predictions.append(pred_vol)
    return np.clip(np.mean(predictions, axis=0), -1, 1)


def triplanar_inference_25d_fixed(generator, us_volume, output_channels=1):
    predictions = []
    for axis in range(3):
        pred_vol = _alloc_pred_vol(us_volume.shape[:3], output_channels)
        n = us_volume.shape[axis]
        for i in range(n):
            ip = max(0, i - 1)
            inext = min(n - 1, i + 1)
            sl_stack = np.stack([
                get_slice_along_axis(us_volume, ip, axis),
                get_slice_along_axis(us_volume, i, axis),
                get_slice_along_axis(us_volume, inext, axis),
            ], axis=-1)
            pred_hw = _predict_slice(generator, sl_stack, 3, output_channels)
            _write_slice(pred_vol, axis, i, pred_hw, output_channels)
        predictions.append(pred_vol)
    return np.clip(np.mean(predictions, axis=0), -1, 1)


def triplanar_inference_multitask_fixed(generator, us_volume, is_25d=False):
    if is_25d:
        result = triplanar_inference_25d_fixed(generator, us_volume, output_channels=2)
    else:
        result = triplanar_inference_2d_fixed(generator, us_volume, output_channels=2)
    if result.ndim == 4:
        return result[..., 0], result[..., 1]
    return result, result


# Monkey-patch into run_all_experiments so _eval_all_test uses the fixes
rae.triplanar_inference_2d = triplanar_inference_2d_fixed
rae.triplanar_inference_25d = triplanar_inference_25d_fixed
rae.triplanar_inference_multitask = triplanar_inference_multitask_fixed


# =============================================================================
# Pending experiments: have generator_ema.weights.h5 but no results.csv
# =============================================================================
PENDING_EVAL = [
    ("pix2pix_25d_t2",          "pix2pix",     "25d", "t2"),
    ("pix2pix_25d_t2_flair",    "pix2pix",     "25d", "t2_flair"),
    ("pix2pix_2d_t2",           "pix2pix",     "2d",  "t2"),
    ("pix2pix_2d_t2_flair",     "pix2pix",     "2d",  "t2_flair"),
    ("swinpix2pix_25d_t2",      "swinpix2pix", "25d", "t2"),
    ("swinpix2pix_25d_t2_flair","swinpix2pix", "25d", "t2_flair"),
    ("swinpix2pix_2d_t2",       "swinpix2pix", "2d",  "t2"),
    ("swinpix2pix_2d_t2_flair", "swinpix2pix", "2d",  "t2_flair"),
    ("cyclegan_25d_t2",         "cyclegan",    "25d", "t2"),
    ("cyclegan_25d_t2_flair",   "cyclegan",    "25d", "t2_flair"),
    ("cyclegan_2d_t2_flair",    "cyclegan",    "2d",  "t2_flair"),
    ("cut_25d_t2",              "cut",         "25d", "t2"),
    ("cut_25d_t2_flair",        "cut",         "25d", "t2_flair"),
    ("cut_2d_t2",               "cut",         "2d",  "t2"),
    ("cut_2d_t2_flair",         "cut",         "2d",  "t2_flair"),
]


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


def eval_experiment(exp_name, arch, variant, target, all_data, split):
    exp_dir = os.path.join(BASE_DIR, exp_name)
    ckpt = os.path.join(exp_dir, "checkpoints", "generator_ema.weights.h5")
    results_csv = os.path.join(exp_dir, "results.csv")

    if not os.path.exists(ckpt):
        print(f"[SKIP] {exp_name}: no checkpoint at {ckpt}")
        return None
    if os.path.exists(results_csv):
        print(f"[SKIP] {exp_name}: results.csv already exists")
        return None

    print(f"\n{'='*70}\n  EVALUATING {exp_name}\n{'='*70}")
    is_25d = (variant == "25d")
    is_multitask = (target == "t2_flair")

    if is_multitask:
        valid = {k: v for k, v in all_data.items() if v.get("flair") is not None}
    else:
        valid = all_data
    test_data = {k: v for k, v in valid.items() if k in set(split["test"])}
    print(f"  Test studies: {len(test_data)} | is_25d={is_25d} "
          f"is_multitask={is_multitask}")

    G, input_ch, output_ch = build_generator(arch, is_25d, is_multitask)
    print(f"  Loading weights: {ckpt}")
    G.load_weights(ckpt)
    print(f"  Generator params: {G.count_params():,}")

    img_dir = os.path.join(exp_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "predictions"), exist_ok=True)

    log_progress(exp_name, "EVAL_STARTED", base_dir=BASE_DIR)
    results = None
    try:
        if arch == "cut":
            results = rae._eval_all_test_cut(
                G, test_data, exp_name, img_dir, exp_dir,
                is_25d, is_multitask, output_ch)
        else:
            results = rae._eval_all_test(
                G, test_data, exp_name, img_dir, exp_dir,
                is_25d, is_multitask, output_ch)

        if results:
            mean_ssim = float(np.mean([r["ssim"] for r in results]))
            mean_psnr = float(np.mean([r["psnr"] for r in results]))
            log_progress(
                exp_name, "EVAL_COMPLETED",
                metrics={"ssim": mean_ssim, "psnr": mean_psnr},
                base_dir=BASE_DIR,
            )
            append_results_csv(exp_name, results, base_dir=BASE_DIR)
            print(f"  MEAN SSIM={mean_ssim:.4f} PSNR={mean_psnr:.2f}")
        else:
            log_progress(exp_name, "EVAL_NO_RESULTS", base_dir=BASE_DIR)
            print("  [WARN] no results produced")
    except Exception as e:
        traceback.print_exc()
        log_progress(
            exp_name, f"EVAL_FAILED: {str(e)[:120]}", base_dir=BASE_DIR)
        results = None
    finally:
        del G
        tf.keras.backend.clear_session()
        import gc
        gc.collect()
    return results


def main():
    print("Loading data...")
    all_data = load_all_data(DATA_DIR)
    print(f"  {len(all_data)} studies loaded")

    split = get_or_create_split(SPLIT_FILE, os.path.join(DATA_DIR, "US"))

    targets = PENDING_EVAL
    if len(sys.argv) > 1:
        wanted = set(sys.argv[1:])
        targets = [t for t in PENDING_EVAL if t[0] in wanted]
        if not targets:
            print(f"[ERROR] No matching experiments in PENDING_EVAL for {wanted}")
            print(f"        Available: {[t[0] for t in PENDING_EVAL]}")
            return 1

    print(f"\n{len(targets)} experiment(s) to evaluate:")
    for t in targets:
        print(f"  - {t[0]}")

    for exp_name, arch, variant, target in targets:
        eval_experiment(exp_name, arch, variant, target, all_data, split)

    print("\n" + "=" * 60)
    print("  EVAL-ONLY PASS COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
