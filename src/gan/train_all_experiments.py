"""
Comprehensive 2D/2.5D/3D US->MRI Synthesis Comparison
======================================================
32 experiments: 4 architectures x 4 variants x 2 targets

Architectures: Pix2Pix, SwinPix2Pix, CycleGAN, CUT
Variants: 2D (triplanar), 2.5D (3-slice), 2D+3D post, 3D (patch-based)
Targets: T2-only, T2+FLAIR (multi-task)

Subject-level train/test split to prevent data leakage.

Author: Santiago Cepeda / BrainUS-AI
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from common.paths import PROJECT_ROOT, DATA_ROOT, CKPT_ROOT, EXTERNAL_ROOT  # noqa: E402
import os  # noqa: E402,F811
import sys
import os
import json
import time
import traceback
import numpy as np
import tensorflow as tf
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (
    load_all_data, get_or_create_split, create_subject_split,
    prepare_2d_slice_dataset, make_2d_dataset, make_3d_refiner_dataset,
    extract_random_patch_3d, _pad_volume_for_patch,
    augment_2d_pair, EMA, CosineWarmup, SN, GN,
    lsgan_g, lsgan_d, fm_loss, ssim_loss_2d, edge_loss_2d,
    r1_penalty_2d, r1_penalty_uncond_2d, cycle_loss, identity_loss,
    ssim_loss_3d, edge_loss_3d, ssim_loss_3d_refiner,
    ssim_3d, psnr_3d, mae_3d,
    triplanar_inference_2d, triplanar_inference_25d, triplanar_inference_multitask,
    save_comparison_figure, save_training_slice_comparison,
    log_progress, append_results_csv, save_nifti,
    get_slice_along_axis, set_slice_along_axis,
    PatchNCEMLP, PatchNCELoss, sample_patches_2d, gather_at_2d,
    sample_patches_3d, gather_at_3d,
    VOLUME_SHAPE, BASELINE_3D_RESULTS,
)
from architectures_2d import (
    build_pix2pix_generator_2d, build_swin_generator_2d,
    build_cyclegan_generator_2d, build_cut_generator_2d,
    build_conditional_disc_2d, build_unconditional_disc_2d,
    MultiScaleCondDisc2D, MultiScaleUncondDisc2D,
    build_3d_refiner,
    build_pix2pix_generator_3d, build_cyclegan_generator_3d,
    build_cut_generator_3d,
    MultiScaleCondDisc3D, MultiScaleUncondDisc3D,
)

# =============================================================================
# GLOBAL CONFIG
# =============================================================================
BASE_DIR = os.path.join(str(PROJECT_ROOT), "COMPARATIVA-3")
DATA_DIR = os.path.join(str(PROJECT_ROOT), "data_cropped_192")
SPLIT_FILE = os.path.join(BASE_DIR, "subject_split.json")

# Training configs per architecture
CONFIGS = {
    "pix2pix": {
        "lr_g": 2e-4, "lr_d": 5e-5,
        "lambda_l1": 10, "lambda_ssim": 8, "lambda_fm": 2, "lambda_edge": 5,
        "lambda_r1": 10, "r1_interval": 16,
    },
    "swinpix2pix": {
        "lr_g": 2e-4, "lr_d": 5e-5,
        "lambda_l1": 10, "lambda_ssim": 8, "lambda_fm": 2, "lambda_edge": 5,
        "lambda_r1": 10, "r1_interval": 16,
    },
    "cyclegan": {
        "lr_g": 2e-4, "lr_d": 2e-4,
        "lambda_cycle": 10, "lambda_idt": 5, "lambda_sup": 10, "lambda_ssim": 8,
        "lambda_r1": 10, "r1_interval": 16,
    },
    "cut": {
        "lr_g": 2e-4, "lr_d": 2e-4,
        "lambda_nce": 1, "lambda_idt": 0.5, "lambda_sup": 10, "lambda_ssim": 8,
        "lambda_edge": 2, "lambda_r1": 10, "r1_interval": 16,
    },
}

# Training params
STEPS_2D = 20000
STEPS_3D = 50000
STEPS_REFINER = 10000
BATCH_2D = 16
BATCH_2D_CYCLEGAN = 4  # may be lowered (e.g. 2) to avoid OOM via monkey-patch
BATCH_3D = 1
# 3D patches: asymmetric (64,64,32) due to variable depth (min=11, median=51)
# Depth=32 fits most volumes; smaller ones get padded
PATCH_3D = (64, 64, 32)
PATCH_3D_OVERLAP = (32, 32, 16)
WARMUP = 1000
EMA_DECAY = 0.999
EVAL_EVERY = 1000
SAVE_IMG_EVERY = 500

# Build experiment matrix
EXPERIMENT_MATRIX = []
for variant in ["25d", "2d", "2d_3dpost", "3d"]:  # 2.5D first to validate quickly
    for arch in ["pix2pix", "swinpix2pix", "cyclegan", "cut"]:
        for target in ["t2", "t2_flair"]:
            EXPERIMENT_MATRIX.append({
                "arch": arch, "variant": variant, "target": target
            })


# =============================================================================
# TRAINING STEP: Pix2Pix / SwinPix2Pix
# =============================================================================
def make_pix2pix_train_fns(G, D, opt_g, opt_d, cfg, is_multitask=False,
                            log_var_t2=None, log_var_flair=None):
    """Create training functions for Pix2Pix-style models."""
    L_L1 = cfg["lambda_l1"]
    L_SSIM = cfg["lambda_ssim"]
    L_FM = cfg["lambda_fm"]
    L_EDGE = cfg["lambda_edge"]
    L_R1 = cfg["lambda_r1"]

    if is_multitask:
        @tf.function
        def train_step_g(us_batch, us_disc, t2_batch, flair_batch):
            target = tf.concat([t2_batch, flair_batch], axis=-1)
            with tf.GradientTape() as tape:
                fake = G(us_batch, training=True)
                if isinstance(fake, (list, tuple)):
                    fake = fake[0]
                dr = D([us_disc, target], training=False)
                df = D([us_disc, fake], training=False)

                gan_loss = lsgan_g(df)
                fm_l = fm_loss(dr, df)

                fake_t2 = fake[:, :, :, 0:1]
                fake_flair = fake[:, :, :, 1:2]
                loss_t2 = (L_L1 * tf.reduce_mean(tf.abs(t2_batch - fake_t2))
                           + L_SSIM * ssim_loss_2d(t2_batch, fake_t2)
                           + L_EDGE * edge_loss_2d(t2_batch, fake_t2))
                loss_flair = (L_L1 * tf.reduce_mean(tf.abs(flair_batch - fake_flair))
                              + L_SSIM * ssim_loss_2d(flair_batch, fake_flair)
                              + L_EDGE * edge_loss_2d(flair_batch, fake_flair))

                prec_t2 = tf.exp(-log_var_t2)
                prec_fl = tf.exp(-log_var_flair)
                weighted = (prec_t2 * loss_t2 + log_var_t2
                            + prec_fl * loss_flair + log_var_flair)
                total_g = gan_loss + weighted + L_FM * fm_l

            g_vars = G.trainable_variables + [log_var_t2, log_var_flair]
            grads = tape.gradient(total_g, g_vars)
            opt_g.apply_gradients(zip(grads, g_vars))
            return total_g, fake

        @tf.function
        def train_step_d(us_disc, t2_batch, flair_batch, fake):
            target = tf.concat([t2_batch, flair_batch], axis=-1)
            with tf.GradientTape() as tape:
                dr = D([us_disc, target], training=True)
                df = D([us_disc, tf.stop_gradient(fake)], training=True)
                d_loss = lsgan_d(dr, df)
            grads = tape.gradient(d_loss, D.trainable_variables)
            opt_d.apply_gradients(zip(grads, D.trainable_variables))
            return d_loss

        @tf.function
        def train_step_r1(us_disc, t2_batch, flair_batch):
            target = tf.concat([t2_batch, flair_batch], axis=-1)
            with tf.GradientTape() as tape:
                r1 = r1_penalty_2d(D, us_disc, target)
                r1_loss = L_R1 * r1
            grads = tape.gradient(r1_loss, D.trainable_variables)
            opt_d.apply_gradients(zip(grads, D.trainable_variables))
            return r1_loss

        return train_step_g, train_step_d, train_step_r1

    else:
        @tf.function
        def train_step_g(us_batch, us_disc, mri_batch):
            with tf.GradientTape() as tape:
                fake = G(us_batch, training=True)
                if isinstance(fake, (list, tuple)):
                    fake = fake[0]
                dr = D([us_disc, mri_batch], training=False)
                df = D([us_disc, fake], training=False)

                g_loss = (lsgan_g(df)
                          + L_L1 * tf.reduce_mean(tf.abs(mri_batch - fake))
                          + L_SSIM * ssim_loss_2d(mri_batch, fake)
                          + L_FM * fm_loss(dr, df)
                          + L_EDGE * edge_loss_2d(mri_batch, fake))
            grads = tape.gradient(g_loss, G.trainable_variables)
            opt_g.apply_gradients(zip(grads, G.trainable_variables))
            return g_loss, fake

        @tf.function
        def train_step_d(us_disc, mri_batch, fake):
            with tf.GradientTape() as tape:
                dr = D([us_disc, mri_batch], training=True)
                df = D([us_disc, tf.stop_gradient(fake)], training=True)
                d_loss = lsgan_d(dr, df)
            grads = tape.gradient(d_loss, D.trainable_variables)
            opt_d.apply_gradients(zip(grads, D.trainable_variables))
            return d_loss

        @tf.function
        def train_step_r1(us_disc, mri_batch):
            with tf.GradientTape() as tape:
                r1 = r1_penalty_2d(D, us_disc, mri_batch)
                r1_loss = L_R1 * r1
            grads = tape.gradient(r1_loss, D.trainable_variables)
            opt_d.apply_gradients(zip(grads, D.trainable_variables))
            return r1_loss

        return train_step_g, train_step_d, train_step_r1


# =============================================================================
# TRAINING STEP: CycleGAN
# =============================================================================
def make_cyclegan_train_fns(G_AB, G_BA, D_A, D_B, opt_g, opt_d, cfg,
                             is_multitask=False, is_25d=False):
    """Create training functions for CycleGAN."""
    L_CYC = cfg["lambda_cycle"]
    # Skip identity for 2.5D or multi-task (channel mismatch between generators)
    L_IDT = 0.0 if (is_25d or is_multitask) else cfg["lambda_idt"]
    L_SUP = cfg["lambda_sup"]
    L_SSIM = cfg["lambda_ssim"]
    L_R1 = cfg["lambda_r1"]

    # No @tf.function for CycleGAN — avoids graph compilation issues with InstanceNorm
    def train_step_g(us_batch, mri_batch):
        with tf.GradientTape() as tape:
            fake_mri = G_AB(us_batch, training=True)
            fake_us = G_BA(mri_batch, training=True)
            cycled_us = G_BA(fake_mri, training=True)
            cycled_mri = G_AB(fake_us, training=True)

            # Adversarial
            d_fake_mri = D_B(fake_mri, training=False)
            d_fake_us = D_A(fake_us, training=False)
            g_adv = (lsgan_g(d_fake_mri) + lsgan_g(d_fake_us)) / 2

            # Cycle
            g_cyc = (cycle_loss(us_batch, cycled_us) +
                     cycle_loss(mri_batch, cycled_mri))

            # Identity (skip for 2.5D — generators have mismatched channels)
            g_idt = tf.constant(0.0)
            if L_IDT > 0:
                idt_mri = G_AB(mri_batch, training=True)
                idt_us = G_BA(us_batch, training=True)
                g_idt = (identity_loss(mri_batch, idt_mri) +
                         identity_loss(us_batch, idt_us))

            # Supervised (paired)
            g_sup = tf.reduce_mean(tf.abs(mri_batch - fake_mri))
            g_ssim = ssim_loss_2d(mri_batch, fake_mri)

            total_g = (g_adv + L_CYC * g_cyc + L_IDT * g_idt
                       + L_SUP * g_sup + L_SSIM * g_ssim)

        g_vars = G_AB.trainable_variables + G_BA.trainable_variables
        grads = tape.gradient(total_g, g_vars)
        opt_g.apply_gradients(zip(grads, g_vars))
        return total_g, fake_mri

    def train_step_d(us_batch, mri_batch, fake_mri):
        fake_us = G_BA(mri_batch, training=False)
        with tf.GradientTape() as tape:
            dr_mri = D_B(mri_batch, training=True)
            df_mri = D_B(tf.stop_gradient(fake_mri), training=True)
            dr_us = D_A(us_batch, training=True)
            df_us = D_A(tf.stop_gradient(fake_us), training=True)
            d_loss = (lsgan_d(dr_mri, df_mri) + lsgan_d(dr_us, df_us)) / 2

        d_vars = D_A.trainable_variables + D_B.trainable_variables
        grads = tape.gradient(d_loss, d_vars)
        opt_d.apply_gradients(zip(grads, d_vars))
        return d_loss

    def train_step_r1(us_batch, mri_batch):
        with tf.GradientTape() as tape:
            r1_a = r1_penalty_uncond_2d(D_A, us_batch)
            r1_b = r1_penalty_uncond_2d(D_B, mri_batch)
            r1_loss = L_R1 * (r1_a + r1_b) / 2
        d_vars = D_A.trainable_variables + D_B.trainable_variables
        grads = tape.gradient(r1_loss, d_vars)
        opt_d.apply_gradients(zip(grads, d_vars))
        return r1_loss

    return train_step_g, train_step_d, train_step_r1


# =============================================================================
# TRAINING STEP: CUT
# =============================================================================
def make_cut_train_fns(G, D, nce_mlps, nce_loss_fn, opt_g, opt_d, cfg,
                        is_multitask=False, is_25d=False):
    """Create training functions for CUT."""
    L_NCE = cfg["lambda_nce"]
    # Skip identity for 2.5D (channel mismatch) or multi-task (G expects 1ch, target is 2ch)
    L_IDT = 0.0 if (is_25d or is_multitask) else cfg["lambda_idt"]
    L_SUP = cfg["lambda_sup"]
    L_SSIM = cfg["lambda_ssim"]
    L_EDGE = cfg["lambda_edge"]
    L_R1 = cfg["lambda_r1"]

    @tf.function
    def train_step_d(us_batch, mri_batch):
        fake_out = G(us_batch, training=False)
        fake = fake_out[0] if isinstance(fake_out, (list, tuple)) else fake_out
        with tf.GradientTape() as tape:
            dr = D(mri_batch, training=True)
            df = D(tf.stop_gradient(fake), training=True)
            d_loss = lsgan_d(dr, df)
        grads = tape.gradient(d_loss, D.trainable_variables)
        opt_d.apply_gradients(zip(grads, D.trainable_variables))
        return d_loss

    @tf.function
    def train_step_g(us_batch, mri_batch):
        with tf.GradientTape() as tape:
            # Forward through generator
            g_out = G(us_batch, training=True)
            fake = g_out[0]
            src_feats = g_out[1:]  # encoder features (query)

            # Forward source again for NCE keys (stopped gradient)
            g_out_key = G(us_batch, training=False)
            src_feats_key = g_out_key[1:]

            # Adversarial
            df = D(fake, training=False)
            g_adv = lsgan_g(df)

            # PatchNCE loss: compare features from training vs stopped-gradient pass
            nce_total = tf.constant(0.0)
            for i, (qf, kf, mlp) in enumerate(
                    zip(src_feats, src_feats_key, nce_mlps)):
                q_sampled, idx = sample_patches_2d(qf, 256)
                k_sampled = gather_at_2d(kf, idx)
                q_proj = mlp(q_sampled)
                k_proj = tf.stop_gradient(mlp(k_sampled))
                nce_total += nce_loss_fn(q_proj, k_proj)
            nce_total /= float(len(nce_mlps))

            # Identity NCE (skip for 2.5D — channel mismatch)
            idt_nce = tf.constant(0.0)
            if L_IDT > 0:
                g_out_idt = G(mri_batch, training=True)
                idt_feats = g_out_idt[1:]
                g_out_idt_key = G(mri_batch, training=False)
                idt_feats_key = g_out_idt_key[1:]

                for i, (qf, kf, mlp) in enumerate(
                        zip(idt_feats, idt_feats_key, nce_mlps)):
                    q_sampled, idx = sample_patches_2d(qf, 256)
                    k_sampled = gather_at_2d(kf, idx)
                    q_proj = mlp(q_sampled)
                    k_proj = tf.stop_gradient(mlp(k_sampled))
                    idt_nce += nce_loss_fn(q_proj, k_proj)
                idt_nce /= float(len(nce_mlps))

            # Supervised
            g_sup = tf.reduce_mean(tf.abs(mri_batch - fake))
            g_ssim = ssim_loss_2d(mri_batch, fake)
            g_edge = edge_loss_2d(mri_batch, fake)

            total_g = (g_adv + L_NCE * nce_total + L_IDT * idt_nce
                       + L_SUP * g_sup + L_SSIM * g_ssim + L_EDGE * g_edge)

        g_vars = G.trainable_variables
        for mlp in nce_mlps:
            g_vars = g_vars + mlp.trainable_variables
        grads = tape.gradient(total_g, g_vars)
        opt_g.apply_gradients(zip(grads, g_vars))
        return total_g, fake

    @tf.function
    def train_step_r1(mri_batch):
        with tf.GradientTape() as tape:
            r1 = r1_penalty_uncond_2d(D, mri_batch)
            r1_loss = L_R1 * r1
        grads = tape.gradient(r1_loss, D.trainable_variables)
        opt_d.apply_gradients(zip(grads, D.trainable_variables))
        return r1_loss

    return train_step_g, train_step_d, train_step_r1


# =============================================================================
# 2D / 2.5D EXPERIMENT RUNNER
# =============================================================================
def run_2d_experiment(exp_name, arch, variant, target, train_data, test_data,
                      exp_dir, base_dir, cfg, all_data=None, split=None):
    """Run a 2D or 2.5D experiment."""
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    img_dir = os.path.join(exp_dir, "images")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    is_25d = (variant == "25d")
    is_multitask = (target == "t2_flair")
    input_ch = 3 if is_25d else 1
    output_ch = 2 if is_multitask else 1
    total_steps = STEPS_2D

    # Prepare slice data
    # Build a mock split for prepare_2d_slice_dataset
    mock_split = {
        "train": list(train_data.keys()),
        "test": list(test_data.keys()),
    }
    # Remap data keys to match expected format
    data_remapped = {}
    for k, v in {**train_data, **test_data}.items():
        data_remapped[k] = v

    print(f"[INFO] Extracting {'2.5D' if is_25d else '2D'} slices...")
    train_slices = prepare_2d_slice_dataset(
        data_remapped, mock_split, axis=2, is_25d=is_25d,
        include_flair=is_multitask, is_training=True)
    print(f"  Training slices: {len(train_slices)}")

    if len(train_slices) == 0:
        print(f"[WARN] No training slices found for {exp_name}. Skipping.")
        return []

    # CycleGAN needs much smaller batch (2 generators + 2 discs = OOM with batch=8)
    batch = BATCH_2D_CYCLEGAN if arch == "cyclegan" else (8 if arch == "cut" else BATCH_2D)
    train_ds = make_2d_dataset(
        train_slices, batch, is_training=True,
        include_flair=is_multitask, is_25d=is_25d)
    train_iter = iter(train_ds)

    # Build models
    log_var_t2 = None
    log_var_flair = None

    if arch in ("pix2pix", "swinpix2pix"):
        if arch == "pix2pix":
            G = build_pix2pix_generator_2d(input_ch, output_ch)
        else:
            G = build_swin_generator_2d(input_ch, output_ch)

        disc_inp_ch = 1  # Disc always gets center US slice (1ch), even for 2.5D
        disc_tar_ch = output_ch
        D = MultiScaleCondDisc2D(disc_inp_ch, disc_tar_ch, num_scales=2)

        # Build
        dummy_us = tf.zeros((1, 192, 192, input_ch))
        dummy_out = G(dummy_us, training=False)
        if isinstance(dummy_out, (list, tuple)):
            dummy_pred = dummy_out[0]
        else:
            dummy_pred = dummy_out
        dummy_us_disc = tf.zeros((1, 192, 192, 1))  # Disc always 1ch input
        dummy_tar = tf.zeros((1, 192, 192, output_ch))
        _ = D([dummy_us_disc, dummy_tar], training=False)

        print(f"  Generator params: {G.count_params():,}")

        ema = EMA(G, EMA_DECAY)
        lr_g = CosineWarmup(cfg["lr_g"], WARMUP, total_steps)
        lr_d = CosineWarmup(cfg["lr_d"], WARMUP, total_steps)
        opt_g = tf.keras.optimizers.Adam(lr_g, beta_1=0.0, beta_2=0.999, clipnorm=1.0)
        opt_d = tf.keras.optimizers.Adam(lr_d, beta_1=0.0, beta_2=0.999, clipnorm=1.0)

        if is_multitask:
            log_var_t2 = tf.Variable(0.0, trainable=True, dtype=tf.float32)
            log_var_flair = tf.Variable(0.0, trainable=True, dtype=tf.float32)

        ts_g, ts_d, ts_r1 = make_pix2pix_train_fns(
            G, D, opt_g, opt_d, cfg, is_multitask, log_var_t2, log_var_flair)

        # Training loop
        print(f"[INFO] Training {exp_name}...")
        t0 = time.time()
        history = {"g_loss": [], "d_loss": [], "step": []}

        for step in range(1, total_steps + 1):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_ds)
                batch = next(train_iter)

            if is_multitask:
                us_b, t2_b, fl_b = batch
                us_disc = us_b[:, :, :, 1:2] if is_25d else us_b
                g_loss, fake = ts_g(us_b, us_disc, t2_b, fl_b)
                d_loss = ts_d(us_disc, t2_b, fl_b, fake)
                if step % cfg["r1_interval"] == 0:
                    ts_r1(us_disc, t2_b, fl_b)
            else:
                us_b, mri_b = batch
                us_disc = us_b[:, :, :, 1:2] if is_25d else us_b
                g_loss, fake = ts_g(us_b, us_disc, mri_b)
                d_loss = ts_d(us_disc, mri_b, fake)
                if step % cfg["r1_interval"] == 0:
                    ts_r1(us_disc, mri_b)

            ema.update()

            if step % 100 == 0:
                elapsed = time.time() - t0
                print(f"  Step {step}/{total_steps} ({elapsed:.0f}s) "
                      f"G={float(g_loss):.4f} D={float(d_loss):.4f}")
                history["g_loss"].append(float(g_loss))
                history["d_loss"].append(float(d_loss))
                history["step"].append(step)

            if step % SAVE_IMG_EVERY == 0:
                if isinstance(fake, tf.Tensor):
                    us_vis = us_b.numpy()
                    if is_25d:
                        us_vis = us_vis[:, :, :, 1:2]
                    save_training_slice_comparison(
                        us_vis, fake.numpy()[:, :, :, 0:1],
                        (mri_b if not is_multitask else t2_b).numpy(),
                        os.path.join(img_dir, f"train_step_{step}.png"), step)

            if step % EVAL_EVERY == 0:
                ema.apply_shadow()
                _eval_one_subject(G, test_data, exp_name, step, img_dir,
                                  is_25d, is_multitask, output_ch)
                ema.restore()

        # Save final
        ema.apply_shadow()
        G.save_weights(os.path.join(ckpt_dir, "generator_ema.weights.h5"))
        ema.restore()

        # Final eval
        ema.apply_shadow()
        results = _eval_all_test(G, test_data, exp_name, img_dir, exp_dir,
                                  is_25d, is_multitask, output_ch)
        ema.restore()

        with open(os.path.join(exp_dir, "history.json"), "w") as f:
            json.dump(history, f)

        # Cleanup
        del G, D, ema, opt_g, opt_d
        tf.keras.backend.clear_session()
        return results

    elif arch == "cyclegan":
        G_AB = build_cyclegan_generator_2d(input_ch, output_ch)
        G_BA = build_cyclegan_generator_2d(output_ch, input_ch)
        D_A = MultiScaleUncondDisc2D(input_ch, num_scales=2)
        D_B = MultiScaleUncondDisc2D(output_ch, num_scales=2)

        # Build
        dummy_us = tf.zeros((1, 192, 192, input_ch))
        dummy_mri = tf.zeros((1, 192, 192, output_ch))
        _ = G_AB(dummy_us, training=False)
        _ = G_BA(dummy_mri, training=False)
        _ = D_A(dummy_us, training=False)
        _ = D_B(dummy_mri, training=False)

        print(f"  G_AB params: {G_AB.count_params():,}")
        print(f"  G_BA params: {G_BA.count_params():,}")

        ema = EMA(G_AB, EMA_DECAY)
        lr_g = CosineWarmup(cfg["lr_g"], WARMUP, total_steps)
        lr_d = CosineWarmup(cfg["lr_d"], WARMUP, total_steps)
        opt_g = tf.keras.optimizers.Adam(lr_g, beta_1=0.0, beta_2=0.999, clipnorm=1.0)
        opt_d = tf.keras.optimizers.Adam(lr_d, beta_1=0.0, beta_2=0.999, clipnorm=1.0)

        ts_g, ts_d, ts_r1 = make_cyclegan_train_fns(
            G_AB, G_BA, D_A, D_B, opt_g, opt_d, cfg, is_multitask, is_25d)

        print(f"[INFO] Training {exp_name}...")
        t0 = time.time()
        history = {"g_loss": [], "d_loss": [], "step": []}

        for step in range(1, total_steps + 1):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_ds)
                batch = next(train_iter)

            if is_multitask:
                us_b, t2_b, fl_b = batch
                mri_b = tf.concat([t2_b, fl_b], axis=-1)
            else:
                us_b, mri_b = batch

            g_loss, fake_mri = ts_g(us_b, mri_b)
            d_loss = ts_d(us_b, mri_b, fake_mri)
            if step % cfg["r1_interval"] == 0:
                ts_r1(us_b, mri_b)

            ema.update()

            if step % 100 == 0:
                elapsed = time.time() - t0
                print(f"  Step {step}/{total_steps} ({elapsed:.0f}s) "
                      f"G={float(g_loss):.4f} D={float(d_loss):.4f}")
                history["g_loss"].append(float(g_loss))
                history["d_loss"].append(float(d_loss))
                history["step"].append(step)

            if step % SAVE_IMG_EVERY == 0:
                us_vis = us_b.numpy()
                if is_25d:
                    us_vis = us_vis[:, :, :, 1:2]
                target_vis = (mri_b if not is_multitask else t2_b).numpy()
                save_training_slice_comparison(
                    us_vis, fake_mri.numpy()[:, :, :, 0:1], target_vis,
                    os.path.join(img_dir, f"train_step_{step}.png"), step)

            if step % EVAL_EVERY == 0:
                ema.apply_shadow()
                _eval_one_subject(G_AB, test_data, exp_name, step, img_dir,
                                  is_25d, is_multitask, output_ch)
                ema.restore()

        ema.apply_shadow()
        G_AB.save_weights(os.path.join(ckpt_dir, "generator_ema.weights.h5"))
        ema.restore()

        ema.apply_shadow()
        results = _eval_all_test(G_AB, test_data, exp_name, img_dir, exp_dir,
                                  is_25d, is_multitask, output_ch)
        ema.restore()

        with open(os.path.join(exp_dir, "history.json"), "w") as f:
            json.dump(history, f)

        del G_AB, G_BA, D_A, D_B, ema, opt_g, opt_d
        tf.keras.backend.clear_session()
        return results

    elif arch == "cut":
        G = build_cut_generator_2d(input_ch, output_ch)
        D = MultiScaleUncondDisc2D(output_ch, num_scales=2)

        # Build
        dummy_us = tf.zeros((1, 192, 192, input_ch))
        dummy_mri = tf.zeros((1, 192, 192, output_ch))
        g_out = G(dummy_us, training=False)
        _ = D(dummy_mri, training=False)

        print(f"  Generator params: {G.count_params():,}")

        # NCE MLPs -- one per feature level
        n_feat_levels = len(g_out) - 1
        nce_mlps = [PatchNCEMLP(out_dim=128, name=f"nce_mlp_{i}")
                    for i in range(n_feat_levels)]
        nce_loss_fn = PatchNCELoss(num_patches=256, temperature=0.07)

        # Build NCE MLPs
        for i, feat in enumerate(g_out[1:]):
            dummy_flat = tf.zeros((1, 256, feat.shape[-1]))
            _ = nce_mlps[i](dummy_flat)

        ema = EMA(G, EMA_DECAY)
        lr_g = CosineWarmup(cfg["lr_g"], WARMUP, total_steps)
        lr_d = CosineWarmup(cfg["lr_d"], WARMUP, total_steps)
        opt_g = tf.keras.optimizers.Adam(lr_g, beta_1=0.0, beta_2=0.999, clipnorm=1.0)
        opt_d = tf.keras.optimizers.Adam(lr_d, beta_1=0.0, beta_2=0.999, clipnorm=1.0)

        ts_g, ts_d, ts_r1 = make_cut_train_fns(
            G, D, nce_mlps, nce_loss_fn, opt_g, opt_d, cfg, is_multitask, is_25d)

        print(f"[INFO] Training {exp_name}...")
        t0 = time.time()
        history = {"g_loss": [], "d_loss": [], "step": []}

        for step in range(1, total_steps + 1):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_ds)
                batch = next(train_iter)

            if is_multitask:
                us_b, t2_b, fl_b = batch
                mri_b = tf.concat([t2_b, fl_b], axis=-1)
            else:
                us_b, mri_b = batch

            d_loss = ts_d(us_b, mri_b)
            g_loss, fake = ts_g(us_b, mri_b)

            if step % cfg["r1_interval"] == 0:
                ts_r1(mri_b)

            ema.update()

            if step % 100 == 0:
                elapsed = time.time() - t0
                print(f"  Step {step}/{total_steps} ({elapsed:.0f}s) "
                      f"G={float(g_loss):.4f} D={float(d_loss):.4f}")
                history["g_loss"].append(float(g_loss))
                history["d_loss"].append(float(d_loss))
                history["step"].append(step)

            if step % SAVE_IMG_EVERY == 0:
                us_vis = us_b.numpy()
                if is_25d:
                    us_vis = us_vis[:, :, :, 1:2]
                target_vis = (mri_b if not is_multitask else t2_b).numpy()
                save_training_slice_comparison(
                    us_vis, fake.numpy()[:, :, :, 0:1], target_vis,
                    os.path.join(img_dir, f"train_step_{step}.png"), step)

            if step % EVAL_EVERY == 0:
                ema.apply_shadow()
                # CUT generator returns list -- wrap for eval
                _eval_one_subject_cut(G, test_data, exp_name, step, img_dir,
                                       is_25d, is_multitask, output_ch)
                ema.restore()

        ema.apply_shadow()
        G.save_weights(os.path.join(ckpt_dir, "generator_ema.weights.h5"))
        ema.restore()

        ema.apply_shadow()
        results = _eval_all_test_cut(G, test_data, exp_name, img_dir, exp_dir,
                                      is_25d, is_multitask, output_ch)
        ema.restore()

        with open(os.path.join(exp_dir, "history.json"), "w") as f:
            json.dump(history, f)

        del G, D, ema, opt_g, opt_d, nce_mlps, nce_loss_fn
        tf.keras.backend.clear_session()
        return results


# =============================================================================
# EVALUATION HELPERS
# =============================================================================
def _cut_predict_wrapper(G, inp):
    """Wrapper to extract just the output from CUT generator."""
    out = G(inp, training=False)
    if isinstance(out, (list, tuple)):
        return out[0]
    return out


def _eval_one_subject(G, test_data, exp_name, step, img_dir,
                       is_25d, is_multitask, output_ch):
    """Evaluate on first test subject for progress tracking."""
    test_studies = list(test_data.keys())
    if not test_studies:
        return
    study_id = test_studies[0]
    us_vol = test_data[study_id]["us"]
    target_vol = test_data[study_id]["t2"]

    if is_multitask:
        pred_t2, pred_fl = triplanar_inference_multitask(G, us_vol, is_25d)
        pred_vol = pred_t2
    elif is_25d:
        pred_vol = triplanar_inference_25d(G, us_vol, output_ch)
    else:
        pred_vol = triplanar_inference_2d(G, us_vol, output_ch)

    if pred_vol.ndim == 4:
        pred_vol = pred_vol[..., 0]

    ssim_val = ssim_3d(target_vol, pred_vol)
    psnr_val = psnr_3d(target_vol, pred_vol)
    print(f"    [EVAL] {study_id}: SSIM={ssim_val:.4f} PSNR={psnr_val:.2f}")
    save_comparison_figure(
        us_vol, pred_vol, target_vol,
        os.path.join(img_dir, f"eval_step_{step}_{study_id}.png"),
        f"{exp_name} - Step {step} - {study_id}")


def _eval_one_subject_cut(G, test_data, exp_name, step, img_dir,
                            is_25d, is_multitask, output_ch):
    """Eval helper for CUT (generator returns list)."""
    class CUTWrapper:
        def __init__(self, model):
            self.model = model
        def __call__(self, x, training=False):
            out = self.model(x, training=training)
            return out[0] if isinstance(out, (list, tuple)) else out
    wrapper = CUTWrapper(G)
    _eval_one_subject(wrapper, test_data, exp_name, step, img_dir,
                       is_25d, is_multitask, output_ch)


def _eval_all_test(G, test_data, exp_name, img_dir, exp_dir,
                    is_25d, is_multitask, output_ch):
    """Full evaluation on all test subjects."""
    from common import save_nifti
    pred_dir = os.path.join(exp_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    results = []
    for study_id, vols in test_data.items():
        us_vol = vols["us"]
        target_t2 = vols["t2"]

        if is_multitask:
            pred_t2, pred_fl = triplanar_inference_multitask(G, us_vol, is_25d)
            target_flair = vols.get("flair")

            # Metrics for T2
            ssim_t2 = ssim_3d(target_t2, pred_t2)
            psnr_t2 = psnr_3d(target_t2, pred_t2)
            mae_t2 = mae_3d(target_t2, pred_t2)

            # Metrics for FLAIR
            if target_flair is not None:
                ssim_fl = ssim_3d(target_flair, pred_fl)
                psnr_fl = psnr_3d(target_flair, pred_fl)
                mae_fl = mae_3d(target_flair, pred_fl)
            else:
                ssim_fl, psnr_fl, mae_fl = 0.0, 0.0, 1.0

            results.append({
                "subject": study_id,
                "ssim_t2": ssim_t2, "psnr_t2": psnr_t2, "mae_t2": mae_t2,
                "ssim_flair": ssim_fl, "psnr_flair": psnr_fl, "mae_flair": mae_fl,
                # Keep ssim/psnr/mae as T2 for backward compat
                "ssim": ssim_t2, "psnr": psnr_t2, "mae": mae_t2,
            })
            print(f"  {study_id}: T2 SSIM={ssim_t2:.4f} PSNR={psnr_t2:.2f} | "
                  f"FLAIR SSIM={ssim_fl:.4f} PSNR={psnr_fl:.2f}")
            save_comparison_figure(
                us_vol, pred_t2, target_t2,
                os.path.join(img_dir, f"final_{study_id}_t2.png"),
                f"{exp_name} - {study_id} - T2")
            if target_flair is not None:
                save_comparison_figure(
                    us_vol, pred_fl, target_flair,
                    os.path.join(img_dir, f"final_{study_id}_flair.png"),
                    f"{exp_name} - {study_id} - FLAIR")

            # Save NIfTI predictions
            save_nifti(pred_t2, os.path.join(pred_dir, f"{study_id}_pred_t2.nii.gz"))
            save_nifti(pred_fl, os.path.join(pred_dir, f"{study_id}_pred_flair.nii.gz"))
            save_nifti(target_t2, os.path.join(pred_dir, f"{study_id}_target_t2.nii.gz"))
            if target_flair is not None:
                save_nifti(target_flair, os.path.join(pred_dir, f"{study_id}_target_flair.nii.gz"))
            save_nifti(us_vol, os.path.join(pred_dir, f"{study_id}_input_us.nii.gz"))
            np.save(os.path.join(exp_dir, f"pred_{study_id}_t2.npy"), pred_t2)
            np.save(os.path.join(exp_dir, f"pred_{study_id}_flair.npy"), pred_fl)

        else:
            if is_25d:
                pred_vol = triplanar_inference_25d(G, us_vol, output_ch)
            else:
                pred_vol = triplanar_inference_2d(G, us_vol, output_ch)

            if pred_vol.ndim == 4:
                pred_vol = pred_vol[..., 0]

            ssim_val = ssim_3d(target_t2, pred_vol)
            psnr_val = psnr_3d(target_t2, pred_vol)
            mae_val = mae_3d(target_t2, pred_vol)
            results.append({
                "subject": study_id, "ssim": ssim_val,
                "psnr": psnr_val, "mae": mae_val})
            print(f"  {study_id}: SSIM={ssim_val:.4f} PSNR={psnr_val:.2f} "
                  f"MAE={mae_val:.4f}")
            save_comparison_figure(
                us_vol, pred_vol, target_t2,
                os.path.join(img_dir, f"final_{study_id}.png"),
                f"{exp_name} - {study_id}")

            # Save NIfTI predictions
            save_nifti(pred_vol, os.path.join(pred_dir, f"{study_id}_pred_t2.nii.gz"))
            save_nifti(target_t2, os.path.join(pred_dir, f"{study_id}_target_t2.nii.gz"))
            save_nifti(us_vol, os.path.join(pred_dir, f"{study_id}_input_us.nii.gz"))
            np.save(os.path.join(exp_dir, f"pred_{study_id}.npy"), pred_vol)

    if results:
        mean_ssim = np.mean([r["ssim"] for r in results])
        mean_psnr = np.mean([r["psnr"] for r in results])
        print(f"  MEAN: SSIM={mean_ssim:.4f} PSNR={mean_psnr:.2f}")

    # Save CSV
    import csv as csv_mod
    csv_path = os.path.join(exp_dir, "results.csv")
    if is_multitask:
        fieldnames = ["subject", "ssim_t2", "psnr_t2", "mae_t2",
                       "ssim_flair", "psnr_flair", "mae_flair"]
    else:
        fieldnames = ["subject", "ssim", "psnr", "mae"]
    with open(csv_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    return results


def _eval_all_test_cut(G, test_data, exp_name, img_dir, exp_dir,
                        is_25d, is_multitask, output_ch):
    """Full evaluation for CUT."""
    class CUTWrapper:
        def __init__(self, model):
            self.model = model
        def __call__(self, x, training=False):
            out = self.model(x, training=training)
            return out[0] if isinstance(out, (list, tuple)) else out
    wrapper = CUTWrapper(G)
    return _eval_all_test(wrapper, test_data, exp_name, img_dir, exp_dir,
                           is_25d, is_multitask, output_ch)


# =============================================================================
# 2D + 3D POST-PROCESSING
# =============================================================================
def run_2d_3dpost_experiment(exp_name, arch, target, train_data, test_data,
                              exp_dir, base_dir, cfg, all_data, split):
    """Run 2D model + 3D refiner post-processing."""
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    img_dir = os.path.join(exp_dir, "images")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    is_multitask = (target == "t2_flair")
    output_ch = 2 if is_multitask else 1

    # Phase 1: check if base 2D model exists, if not train it
    base_2d_name = f"{arch}_2d_{target}"
    base_2d_dir = os.path.join(base_dir, base_2d_name)
    base_2d_ckpt = os.path.join(base_2d_dir, "checkpoints", "generator_ema.weights.h5")

    input_ch = 1
    if arch == "pix2pix":
        G = build_pix2pix_generator_2d(input_ch, output_ch)
    elif arch == "swinpix2pix":
        G = build_swin_generator_2d(input_ch, output_ch)
    elif arch == "cyclegan":
        G = build_cyclegan_generator_2d(input_ch, output_ch)
    elif arch == "cut":
        G = build_cut_generator_2d(input_ch, output_ch)

    dummy = tf.zeros((1, 192, 192, input_ch))
    _ = G(dummy, training=False)

    if os.path.exists(base_2d_ckpt):
        print(f"[INFO] Loading 2D weights from {base_2d_ckpt}")
        G.load_weights(base_2d_ckpt)
    else:
        print(f"[INFO] No base 2D weights found at {base_2d_ckpt}")
        print(f"[INFO] Training 2D model first...")
        # Run the base 2D experiment
        base_results = run_2d_experiment(
            base_2d_name, arch, "2d", target, train_data, test_data,
            base_2d_dir, base_dir, cfg, all_data, split)
        # Reload
        if arch == "pix2pix":
            G = build_pix2pix_generator_2d(input_ch, output_ch)
        elif arch == "swinpix2pix":
            G = build_swin_generator_2d(input_ch, output_ch)
        elif arch == "cyclegan":
            G = build_cyclegan_generator_2d(input_ch, output_ch)
        elif arch == "cut":
            G = build_cut_generator_2d(input_ch, output_ch)
        _ = G(tf.zeros((1, 192, 192, input_ch)), training=False)
        if os.path.exists(base_2d_ckpt):
            G.load_weights(base_2d_ckpt)

    is_cut = (arch == "cut")

    # Phase 2: Generate triplanar predictions for training set
    # For multi-task: refiner works on output_ch channels (T2 + FLAIR)
    refiner_ch = output_ch  # 1 for T2-only, 2 for T2+FLAIR
    print(f"[INFO] Generating triplanar predictions for training set (ch={refiner_ch})...")
    train_preds = []
    train_targets = []
    for study_id, vols in train_data.items():
        us_vol = vols["us"]
        if is_multitask:
            pred = triplanar_inference_multitask(G, us_vol, is_25d=False) if not is_cut \
                else _cut_triplanar_multitask(G, us_vol)
            # pred is (pred_t2, pred_flair) tuple
            pred_stack = np.stack(pred, axis=-1)  # (H, W, D, 2)
            target_t2 = vols["t2"]
            target_fl = vols.get("flair")
            if target_fl is None:
                target_fl = np.full_like(target_t2, -1.0)
            target_stack = np.stack([target_t2, target_fl], axis=-1)
        else:
            if is_cut:
                pred_vol = _cut_triplanar(G, us_vol, output_ch)
            else:
                pred_vol = triplanar_inference_2d(G, us_vol, output_ch)
            if pred_vol.ndim == 4:
                pred_stack = pred_vol  # keep channels
            else:
                pred_stack = pred_vol[..., np.newaxis]
            target_stack = vols["t2"][..., np.newaxis]
        train_preds.append(pred_stack)
        train_targets.append(target_stack)
        print(f"  {study_id}")

    # Phase 3: Train 3D refiner
    print(f"[INFO] Training 3D refiner (channels={refiner_ch})...")
    refiner = build_3d_refiner(input_channels=refiner_ch)
    pH, pW, pD = PATCH_3D
    _ = refiner(tf.zeros((1, pH, pW, pD, refiner_ch)), training=False)
    print(f"  Refiner params: {refiner.count_params():,}")

    ema_ref = EMA(refiner, EMA_DECAY)
    opt_ref = tf.keras.optimizers.Adam(1e-4, clipnorm=1.0)

    ref_ds = make_3d_refiner_dataset(train_preds, train_targets, batch_size=1,
                                      patch_size=PATCH_3D)
    ref_iter = iter(ref_ds)

    # @tf.function  # eager for 3D
    def train_step_refiner(inp_patch, tar_patch):
        with tf.GradientTape() as tape:
            refined = refiner(inp_patch, training=True)
            l1 = tf.reduce_mean(tf.abs(tar_patch - refined))
            ssim = ssim_loss_3d_refiner(tar_patch, refined)
            loss = 10.0 * l1 + 8.0 * ssim
        grads = tape.gradient(loss, refiner.trainable_variables)
        opt_ref.apply_gradients(zip(grads, refiner.trainable_variables))
        return loss

    t0 = time.time()
    for step in range(1, STEPS_REFINER + 1):
        try:
            batch = next(ref_iter)
        except StopIteration:
            ref_iter = iter(ref_ds)
            batch = next(ref_iter)
        loss = train_step_refiner(batch[0], batch[1])
        ema_ref.update()
        if step % 200 == 0:
            elapsed = time.time() - t0
            print(f"  Refiner Step {step}/{STEPS_REFINER} ({elapsed:.0f}s) "
                  f"Loss={float(loss):.4f}")

    ema_ref.apply_shadow()
    refiner.save_weights(os.path.join(ckpt_dir, "refiner_ema.weights.h5"))

    # Phase 4: Evaluate on all test subjects
    from common import save_nifti
    pred_save_dir = os.path.join(exp_dir, "predictions")
    os.makedirs(pred_save_dir, exist_ok=True)

    print("[INFO] Final evaluation with 3D refinement...")
    results = []
    for study_id, vols in test_data.items():
        us_vol = vols["us"]
        target_t2 = vols["t2"]

        # 2D triplanar prediction
        if is_multitask:
            if is_cut:
                pred_t2_2d, pred_fl_2d = _cut_triplanar_multitask(G, us_vol)
            else:
                pred_t2_2d, pred_fl_2d = triplanar_inference_multitask(G, us_vol, is_25d=False)
            pred_2d = np.stack([pred_t2_2d, pred_fl_2d], axis=-1)  # (H,W,D,2)
        else:
            if is_cut:
                pred_2d = _cut_triplanar(G, us_vol, output_ch)
            else:
                pred_2d = triplanar_inference_2d(G, us_vol, output_ch)
            if pred_2d.ndim == 3:
                pred_2d = pred_2d[..., np.newaxis]

        # 3D refinement
        refined = _sliding_window_inference_3d(
            refiner, pred_2d if pred_2d.ndim == 3 else pred_2d[..., 0] if refiner_ch == 1 else pred_2d,
            PATCH_3D, PATCH_3D_OVERLAP, is_cut=False, output_channels=refiner_ch)

        if is_multitask:
            pred_t2 = refined[..., 0] if refined.ndim == 4 else refined
            pred_fl = refined[..., 1] if refined.ndim == 4 and refined.shape[-1] > 1 else pred_fl_2d
            target_flair = vols.get("flair")

            ssim_t2 = ssim_3d(target_t2, pred_t2)
            psnr_t2 = psnr_3d(target_t2, pred_t2)
            mae_t2 = mae_3d(target_t2, pred_t2)
            if target_flair is not None:
                ssim_fl = ssim_3d(target_flair, pred_fl)
                psnr_fl = psnr_3d(target_flair, pred_fl)
                mae_fl = mae_3d(target_flair, pred_fl)
            else:
                ssim_fl, psnr_fl, mae_fl = 0.0, 0.0, 1.0

            results.append({
                "subject": study_id,
                "ssim_t2": ssim_t2, "psnr_t2": psnr_t2, "mae_t2": mae_t2,
                "ssim_flair": ssim_fl, "psnr_flair": psnr_fl, "mae_flair": mae_fl,
                "ssim": ssim_t2, "psnr": psnr_t2, "mae": mae_t2,
            })
            print(f"  {study_id}: T2 SSIM={ssim_t2:.4f} PSNR={psnr_t2:.2f} | "
                  f"FLAIR SSIM={ssim_fl:.4f} PSNR={psnr_fl:.2f}")
            save_comparison_figure(us_vol, pred_t2, target_t2,
                os.path.join(img_dir, f"final_{study_id}_t2.png"),
                f"{exp_name} - {study_id} - T2")
            save_nifti(pred_t2, os.path.join(pred_save_dir, f"{study_id}_pred_t2.nii.gz"))
            save_nifti(pred_fl, os.path.join(pred_save_dir, f"{study_id}_pred_flair.nii.gz"))
            save_nifti(target_t2, os.path.join(pred_save_dir, f"{study_id}_target_t2.nii.gz"))
            if target_flair is not None:
                save_nifti(target_flair, os.path.join(pred_save_dir, f"{study_id}_target_flair.nii.gz"))
                save_comparison_figure(us_vol, pred_fl, target_flair,
                    os.path.join(img_dir, f"final_{study_id}_flair.png"),
                    f"{exp_name} - {study_id} - FLAIR")
            save_nifti(us_vol, os.path.join(pred_save_dir, f"{study_id}_input_us.nii.gz"))
        else:
            pred_vol = refined[..., 0] if refined.ndim == 4 else refined
            ssim_val = ssim_3d(target_t2, pred_vol)
            psnr_val = psnr_3d(target_t2, pred_vol)
            mae_val = mae_3d(target_t2, pred_vol)
            results.append({"subject": study_id, "ssim": ssim_val,
                            "psnr": psnr_val, "mae": mae_val})
            print(f"  {study_id}: SSIM={ssim_val:.4f} PSNR={psnr_val:.2f}")
            save_comparison_figure(us_vol, pred_vol, target_t2,
                os.path.join(img_dir, f"final_{study_id}.png"),
                f"{exp_name} - {study_id}")
            save_nifti(pred_vol, os.path.join(pred_save_dir, f"{study_id}_pred_t2.nii.gz"))
            save_nifti(target_t2, os.path.join(pred_save_dir, f"{study_id}_target_t2.nii.gz"))
            save_nifti(us_vol, os.path.join(pred_save_dir, f"{study_id}_input_us.nii.gz"))

    ema_ref.restore()

    import csv as csv_mod
    csv_path = os.path.join(exp_dir, "results.csv")
    if is_multitask:
        fieldnames = ["subject", "ssim_t2", "psnr_t2", "mae_t2",
                       "ssim_flair", "psnr_flair", "mae_flair"]
    else:
        fieldnames = ["subject", "ssim", "psnr", "mae"]
    with open(csv_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    if results:
        mean_ssim = np.mean([r["ssim"] for r in results])
        mean_psnr = np.mean([r["psnr"] for r in results])
        print(f"  MEAN: SSIM={mean_ssim:.4f} PSNR={mean_psnr:.2f}")

    del G, refiner, ema_ref, opt_ref
    tf.keras.backend.clear_session()
    return results


def _cut_triplanar(G, us_volume, output_channels):
    """Triplanar inference for CUT generator (returns list)."""
    class CUTWrapper:
        def __init__(self, model):
            self.model = model
        def __call__(self, x, training=False):
            out = self.model(x, training=training)
            return out[0] if isinstance(out, (list, tuple)) else out
    return triplanar_inference_2d(CUTWrapper(G), us_volume, output_channels)


def _cut_triplanar_multitask(G, us_volume):
    """Triplanar multitask inference for CUT generator (returns list).
    Wraps G so its list output is unwrapped, then delegates to the shared
    multitask triplanar inference. Returns (pred_t2, pred_flair).
    """
    class CUTWrapper:
        def __init__(self, model):
            self.model = model

        def __call__(self, x, training=False):
            out = self.model(x, training=training)
            return out[0] if isinstance(out, (list, tuple)) else out

    return triplanar_inference_multitask(CUTWrapper(G), us_volume, is_25d=False)


# =============================================================================
# 3D EXPERIMENT RUNNER
# =============================================================================
def run_3d_experiment(exp_name, arch, target, train_data, test_data,
                      exp_dir, base_dir, cfg):
    """Run a full 3D experiment using patch-based training."""
    from scipy.ndimage import rotate as scipy_rotate
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    img_dir = os.path.join(exp_dir, "images")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    is_multitask = (target == "t2_flair")
    output_ch = 2 if is_multitask else 1
    input_ch = 1
    ps = PATCH_3D  # (64, 64, 32) -- asymmetric tuple
    pH, pW, pD = ps
    total_steps = STEPS_3D

    # Build 3D generator + discriminator
    # Use None,None,None input shape for generators (fully convolutional)
    # Discriminator uses fixed patch size, num_scales=1 (depth=32 too small for multi-scale)
    if arch == "pix2pix":
        G = build_pix2pix_generator_3d(input_ch, output_ch, patch_size=None)
        D = MultiScaleCondDisc3D(patch_size=None, input_channels=input_ch,
                                  target_channels=output_ch, num_scales=1)
    elif arch == "swinpix2pix":
        G = build_pix2pix_generator_3d(input_ch, output_ch, patch_size=None)
        D = MultiScaleCondDisc3D(patch_size=None, input_channels=input_ch,
                                  target_channels=output_ch, num_scales=1)
    elif arch == "cyclegan":
        G = build_cyclegan_generator_3d(input_ch, output_ch, patch_size=None)
        D = MultiScaleUncondDisc3D(patch_size=None, input_channels=output_ch,
                                    num_scales=1)
    elif arch == "cut":
        G = build_cut_generator_3d(input_ch, output_ch, patch_size=None)
        D = MultiScaleUncondDisc3D(patch_size=None, input_channels=output_ch,
                                    num_scales=1)

    # Build models with asymmetric dummy tensors
    dummy_us = tf.zeros((1, pH, pW, pD, input_ch))
    dummy_mri = tf.zeros((1, pH, pW, pD, output_ch))
    g_out = G(dummy_us, training=False)
    if isinstance(g_out, (list, tuple)):
        dummy_pred = g_out[0]
    else:
        dummy_pred = g_out

    if arch in ("pix2pix", "swinpix2pix"):
        _ = D([dummy_us, dummy_mri], training=False)
    else:
        _ = D(dummy_mri, training=False)

    print(f"  Generator params: {G.count_params():,}")

    # Data pipeline -- handle multitask (T2+FLAIR) vs single target (T2)
    if is_multitask:
        train_pairs = []
        for v in train_data.values():
            if v.get("flair") is not None:
                train_pairs.append((v["us"], v["t2"], v["flair"]))
    else:
        train_pairs = [(v["us"], v["t2"]) for v in train_data.values()]

    def patch_gen():
        indices = np.arange(len(train_pairs))
        while True:
            np.random.shuffle(indices)
            for i in indices:
                if is_multitask:
                    us_vol, t2_vol, flair_vol = train_pairs[i]
                    u, t2, fl = us_vol.copy(), t2_vol.copy(), flair_vol.copy()
                    for ax in range(3):
                        if np.random.rand() < 0.5:
                            u = np.flip(u, ax).copy()
                            t2 = np.flip(t2, ax).copy()
                            fl = np.flip(fl, ax).copy()
                    if np.random.rand() < 0.3:
                        angle = np.random.uniform(-10, 10)
                        rot_axes = [(0, 1), (0, 2), (1, 2)][np.random.randint(3)]
                        u = scipy_rotate(u, angle, axes=rot_axes, reshape=False,
                                         order=1, mode='nearest').astype(np.float32)
                        t2 = scipy_rotate(t2, angle, axes=rot_axes, reshape=False,
                                          order=1, mode='nearest').astype(np.float32)
                        fl = scipy_rotate(fl, angle, axes=rot_axes, reshape=False,
                                          order=1, mode='nearest').astype(np.float32)
                    # Pad if needed for asymmetric patch extraction
                    u = _pad_volume_for_patch(u, ps)
                    t2 = _pad_volume_for_patch(t2, ps)
                    fl = _pad_volume_for_patch(fl, ps)
                    # Extract paired patches
                    max_start = [s - p for s, p in zip(u.shape[:3], ps)]
                    start = [np.random.randint(0, mx + 1) for mx in max_start]
                    slc = tuple(slice(s, s + p) for s, p in zip(start, ps))
                    up = u[slc]
                    t2p = t2[slc]
                    flp = fl[slc]
                    # Stack T2 and FLAIR as 2-channel target
                    mp = np.stack([t2p, flp], axis=-1)
                    yield (up[..., np.newaxis].astype(np.float32),
                           mp.astype(np.float32))
                else:
                    us_vol, mri_vol = train_pairs[i]
                    u, m = us_vol.copy(), mri_vol.copy()
                    for ax in range(3):
                        if np.random.rand() < 0.5:
                            u = np.flip(u, ax).copy()
                            m = np.flip(m, ax).copy()
                    if np.random.rand() < 0.3:
                        angle = np.random.uniform(-10, 10)
                        rot_axes = [(0, 1), (0, 2), (1, 2)][np.random.randint(3)]
                        u = scipy_rotate(u, angle, axes=rot_axes, reshape=False,
                                         order=1, mode='nearest').astype(np.float32)
                        m = scipy_rotate(m, angle, axes=rot_axes, reshape=False,
                                         order=1, mode='nearest').astype(np.float32)
                    up, mp = extract_random_patch_3d(u, m, ps)
                    yield (up[..., np.newaxis].astype(np.float32),
                           mp[..., np.newaxis].astype(np.float32))

    train_ds = tf.data.Dataset.from_generator(
        patch_gen,
        output_signature=(
            tf.TensorSpec((pH, pW, pD, input_ch), tf.float32),
            tf.TensorSpec((pH, pW, pD, output_ch), tf.float32),
        )).shuffle(8).batch(BATCH_3D, drop_remainder=True).prefetch(1)
    train_iter = iter(train_ds)

    ema = EMA(G, EMA_DECAY)
    lr_g = CosineWarmup(cfg["lr_g"], WARMUP, total_steps)
    lr_d = CosineWarmup(cfg["lr_d"], WARMUP, total_steps)
    opt_g = tf.keras.optimizers.Adam(lr_g, beta_1=0.0, beta_2=0.999, clipnorm=1.0)
    opt_d = tf.keras.optimizers.Adam(lr_d, beta_1=0.0, beta_2=0.999, clipnorm=1.0)

    # Architecture-specific training steps
    if arch in ("pix2pix", "swinpix2pix"):
        L_L1 = cfg["lambda_l1"]
        L_SSIM = cfg["lambda_ssim"]
        L_FM = cfg["lambda_fm"]
        L_EDGE = cfg["lambda_edge"]
        L_R1 = cfg["lambda_r1"]

        # @tf.function  # eager mode for 3D
        def train_g(us_b, mri_b):
            with tf.GradientTape() as tape:
                fake = G(us_b, training=True)
                if isinstance(fake, (list, tuple)):
                    fake = fake[0]
                dr = D([us_b, mri_b], training=False)
                df = D([us_b, fake], training=False)
                g_loss = (lsgan_g(df)
                          + L_L1 * tf.reduce_mean(tf.abs(mri_b - fake))
                          + L_SSIM * ssim_loss_3d(mri_b, fake)
                          + L_FM * fm_loss(dr, df)
                          + L_EDGE * edge_loss_3d(mri_b, fake))
            grads = tape.gradient(g_loss, G.trainable_variables)
            opt_g.apply_gradients(zip(grads, G.trainable_variables))
            return g_loss, fake

        # @tf.function  # eager for 3D
        def train_d(us_b, mri_b, fake):
            with tf.GradientTape() as tape:
                dr = D([us_b, mri_b], training=True)
                df = D([us_b, tf.stop_gradient(fake)], training=True)
                d_loss = lsgan_d(dr, df)
            grads = tape.gradient(d_loss, D.trainable_variables)
            opt_d.apply_gradients(zip(grads, D.trainable_variables))
            return d_loss

    elif arch == "cyclegan":
        G_BA = build_cyclegan_generator_3d(output_ch, input_ch, patch_size=None)
        _ = G_BA(dummy_mri, training=False)
        D_A = MultiScaleUncondDisc3D(patch_size=None, input_channels=input_ch,
                                      num_scales=1)
        _ = D_A(dummy_us, training=False)

        L_CYC = cfg["lambda_cycle"]
        # Disable identity for multi-task (channel mismatch: G takes 1ch, target is 2ch)
        L_IDT = 0.0 if is_multitask else cfg["lambda_idt"]
        L_SUP = cfg["lambda_sup"]
        L_SSIM = cfg["lambda_ssim"]

        def train_g(us_b, mri_b):  # No @tf.function for CycleGAN (InstanceNorm issues)
            with tf.GradientTape() as tape:
                fake_mri = G(us_b, training=True)
                fake_us = G_BA(mri_b, training=True)
                cyc_us = G_BA(fake_mri, training=True)
                cyc_mri = G(fake_us, training=True)
                d_fake_mri = D(fake_mri, training=False)
                d_fake_us = D_A(fake_us, training=False)
                g_adv = (lsgan_g(d_fake_mri) + lsgan_g(d_fake_us)) / 2
                g_cyc = cycle_loss(us_b, cyc_us) + cycle_loss(mri_b, cyc_mri)
                g_idt = tf.constant(0.0)
                if L_IDT > 0:
                    idt_mri = G(mri_b, training=True)
                    idt_us = G_BA(us_b, training=True)
                    g_idt = (identity_loss(mri_b, idt_mri) +
                             identity_loss(us_b, idt_us))
                g_sup = tf.reduce_mean(tf.abs(mri_b - fake_mri))
                g_ssim = ssim_loss_3d(mri_b, fake_mri)
                total = (g_adv + L_CYC * g_cyc + L_IDT * g_idt
                         + L_SUP * g_sup + L_SSIM * g_ssim)
            g_vars = G.trainable_variables + G_BA.trainable_variables
            grads = tape.gradient(total, g_vars)
            opt_g.apply_gradients(zip(grads, g_vars))
            return total, fake_mri

        # @tf.function  # eager for 3D
        def train_d(us_b, mri_b, fake_mri):
            fake_us = G_BA(mri_b, training=False)
            with tf.GradientTape() as tape:
                dr_mri = D(mri_b, training=True)
                df_mri = D(tf.stop_gradient(fake_mri), training=True)
                dr_us = D_A(us_b, training=True)
                df_us = D_A(tf.stop_gradient(fake_us), training=True)
                d_loss = (lsgan_d(dr_mri, df_mri) + lsgan_d(dr_us, df_us)) / 2
            d_vars = D.trainable_variables + D_A.trainable_variables
            grads = tape.gradient(d_loss, d_vars)
            opt_d.apply_gradients(zip(grads, d_vars))
            return d_loss

    elif arch == "cut":
        n_feat = len(g_out) - 1 if isinstance(g_out, (list, tuple)) else 0
        nce_mlps_3d = [PatchNCEMLP(out_dim=128, name=f"nce3d_{i}")
                       for i in range(max(n_feat, 1))]
        nce_loss_fn = PatchNCELoss(num_patches=128, temperature=0.07)
        if isinstance(g_out, (list, tuple)):
            for i, feat in enumerate(g_out[1:]):
                dummy_flat = tf.zeros((1, 128, feat.shape[-1]))
                _ = nce_mlps_3d[i](dummy_flat)

        L_NCE = cfg["lambda_nce"]
        L_SUP = cfg["lambda_sup"]
        L_SSIM = cfg["lambda_ssim"]
        L_EDGE = cfg["lambda_edge"]

        # @tf.function  # eager for 3D
        def train_g(us_b, mri_b):
            with tf.GradientTape() as tape:
                g_result = G(us_b, training=True)
                fake = g_result[0] if isinstance(g_result, (list, tuple)) else g_result
                df = D(fake, training=False)
                g_adv = lsgan_g(df)
                g_sup = tf.reduce_mean(tf.abs(mri_b - fake))
                g_ssim = ssim_loss_3d(mri_b, fake)
                g_edge = edge_loss_3d(mri_b, fake)
                total = g_adv + L_SUP * g_sup + L_SSIM * g_ssim + L_EDGE * g_edge
            g_vars = G.trainable_variables
            for mlp in nce_mlps_3d:
                g_vars = g_vars + mlp.trainable_variables
            grads = tape.gradient(total, g_vars)
            opt_g.apply_gradients(zip(grads, g_vars))
            return total, fake

        # @tf.function  # eager for 3D
        def train_d(us_b, mri_b, fake):
            with tf.GradientTape() as tape:
                dr = D(mri_b, training=True)
                df = D(tf.stop_gradient(fake), training=True)
                d_loss = lsgan_d(dr, df)
            grads = tape.gradient(d_loss, D.trainable_variables)
            opt_d.apply_gradients(zip(grads, D.trainable_variables))
            return d_loss

    # Training loop
    print(f"[INFO] Training 3D {exp_name}...", flush=True)
    # Force first batch to verify pipeline works
    _first_batch = next(iter(train_ds))
    print(f"  First batch OK: {_first_batch[0].shape}, {_first_batch[1].shape}", flush=True)
    del _first_batch
    t0 = time.time()
    history = {"g_loss": [], "d_loss": [], "step": []}

    for step in range(1, total_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_ds)
            batch = next(train_iter)

        us_b, mri_b = batch
        g_loss, fake = train_g(us_b, mri_b)
        d_loss = train_d(us_b, mri_b, fake)
        ema.update()

        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f"  Step {step}/{total_steps} ({elapsed:.0f}s) "
                  f"G={float(g_loss):.4f} D={float(d_loss):.4f}")
            history["g_loss"].append(float(g_loss))
            history["d_loss"].append(float(d_loss))
            history["step"].append(step)

        if step % EVAL_EVERY == 0:
            ema.apply_shadow()
            _eval_3d_one_subject(G, test_data, exp_name, step, img_dir, arch,
                                 is_multitask, output_ch)
            ema.restore()

    # Save
    ema.apply_shadow()
    G.save_weights(os.path.join(ckpt_dir, "generator_ema.weights.h5"))

    # Full eval with sliding window
    results = _eval_3d_all_test(G, test_data, exp_name, img_dir, exp_dir, arch,
                                 is_multitask, output_ch)
    ema.restore()

    with open(os.path.join(exp_dir, "history.json"), "w") as f:
        json.dump(history, f)

    del G, D, ema, opt_g, opt_d
    if arch == "cyclegan":
        del G_BA, D_A
    tf.keras.backend.clear_session()
    return results


def _sliding_window_inference_3d(G, us_volume, patch_size=None, overlap=None,
                                  is_cut=False, output_channels=1):
    """Sliding window inference for 3D models with asymmetric patches.
    Accepts both 3D (H,W,D) and 4D (H,W,D,C_in) input volumes. Pads along the
    first 3 dims if the volume is smaller than the patch in any dim.
    """
    if patch_size is None:
        patch_size = PATCH_3D
    if overlap is None:
        overlap = PATCH_3D_OVERLAP

    if isinstance(patch_size, (list, tuple)):
        pH, pW, pD = patch_size
    else:
        pH = pW = pD = patch_size
    if isinstance(overlap, (list, tuple)):
        oH, oW, oD = overlap
    else:
        oH = oW = oD = overlap

    sH, sW, sD = pH - oH, pW - oW, pD - oD
    H, W, D = us_volume.shape[:3]
    in_is_4d = (us_volume.ndim == 4)

    pad_H = max(0, pH - H)
    pad_W = max(0, pW - W)
    pad_D = max(0, pD - D)
    if pad_H > 0 or pad_W > 0 or pad_D > 0:
        pads = [(0, pad_H), (0, pad_W), (0, pad_D)]
        if in_is_4d:
            pads.append((0, 0))
        volume = np.pad(us_volume, pads, mode='reflect')
    else:
        volume = us_volume

    H2, W2, D2 = volume.shape[:3]
    if output_channels == 1:
        pred = np.zeros((H2, W2, D2), dtype=np.float32)
    else:
        pred = np.zeros((H2, W2, D2, output_channels), dtype=np.float32)
    count = np.zeros((H2, W2, D2), dtype=np.float32)

    def _starts(total, p, step):
        st = list(range(0, total - p + 1, step))
        if st and st[-1] != total - p:
            st.append(total - p)
        if not st:
            st = [0]
        return st

    starts_h = _starts(H2, pH, sH)
    starts_w = _starts(W2, pW, sW)
    starts_d = _starts(D2, pD, sD)

    for i in starts_h:
        for j in starts_w:
            for k in starts_d:
                if in_is_4d:
                    patch = volume[i:i+pH, j:j+pW, k:k+pD, :]
                    inp = patch[np.newaxis].astype(np.float32)
                else:
                    patch = volume[i:i+pH, j:j+pW, k:k+pD]
                    inp = patch[np.newaxis, ..., np.newaxis].astype(np.float32)
                out = G(inp, training=False)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                out_np = out.numpy()[0]
                if output_channels == 1:
                    pred[i:i+pH, j:j+pW, k:k+pD] += out_np[..., 0]
                else:
                    pred[i:i+pH, j:j+pW, k:k+pD, :] += out_np
                count[i:i+pH, j:j+pW, k:k+pD] += 1.0

    count = np.maximum(count, 1.0)
    if output_channels == 1:
        result = np.clip(pred / count, -1, 1)
    else:
        result = np.clip(pred / count[..., np.newaxis], -1, 1)
    return result[:H, :W, :D] if output_channels == 1 else result[:H, :W, :D, :]


def _eval_3d_one_subject(G, test_data, exp_name, step, img_dir, arch,
                          is_multitask=False, output_ch=1):
    test_studies = list(test_data.keys())
    if not test_studies:
        return
    study_id = test_studies[0]
    us_vol = test_data[study_id]["us"]
    target_vol = test_data[study_id]["t2"]
    is_cut = (arch == "cut")
    pred_vol = _sliding_window_inference_3d(
        G, us_vol, PATCH_3D, PATCH_3D_OVERLAP, is_cut,
        output_channels=output_ch)
    if is_multitask and pred_vol.ndim == 4:
        pred_t2 = pred_vol[..., 0]
        ssim_val = ssim_3d(target_vol, pred_t2)
        psnr_val = psnr_3d(target_vol, pred_t2)
        print(f"    [EVAL] {study_id}: T2 SSIM={ssim_val:.4f} PSNR={psnr_val:.2f}")
        save_comparison_figure(
            us_vol, pred_t2, target_vol,
            os.path.join(img_dir, f"eval_step_{step}_{study_id}.png"),
            f"{exp_name} - Step {step}")
    else:
        if pred_vol.ndim == 4:
            pred_vol = pred_vol[..., 0]
        ssim_val = ssim_3d(target_vol, pred_vol)
        psnr_val = psnr_3d(target_vol, pred_vol)
        print(f"    [EVAL] {study_id}: SSIM={ssim_val:.4f} PSNR={psnr_val:.2f}")
        save_comparison_figure(
            us_vol, pred_vol, target_vol,
            os.path.join(img_dir, f"eval_step_{step}_{study_id}.png"),
            f"{exp_name} - Step {step}")


def _eval_3d_all_test(G, test_data, exp_name, img_dir, exp_dir, arch,
                       is_multitask=False, output_ch=1):
    from common import save_nifti
    pred_dir = os.path.join(exp_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    results = []
    is_cut = (arch == "cut")

    for study_id, vols in test_data.items():
        us_vol = vols["us"]
        target_t2 = vols["t2"]

        if is_multitask:
            pred_vol = _sliding_window_inference_3d(
                G, us_vol, PATCH_3D, PATCH_3D_OVERLAP, is_cut,
                output_channels=output_ch)
            # pred_vol shape: (D, H, W, 2) for multitask
            pred_t2 = pred_vol[..., 0]
            pred_fl = pred_vol[..., 1]
            target_flair = vols.get("flair")

            ssim_t2 = ssim_3d(target_t2, pred_t2)
            psnr_t2 = psnr_3d(target_t2, pred_t2)
            mae_t2 = mae_3d(target_t2, pred_t2)

            if target_flair is not None:
                ssim_fl = ssim_3d(target_flair, pred_fl)
                psnr_fl = psnr_3d(target_flair, pred_fl)
                mae_fl = mae_3d(target_flair, pred_fl)
            else:
                ssim_fl, psnr_fl, mae_fl = 0.0, 0.0, 1.0

            results.append({
                "subject": study_id,
                "ssim_t2": ssim_t2, "psnr_t2": psnr_t2, "mae_t2": mae_t2,
                "ssim_flair": ssim_fl, "psnr_flair": psnr_fl, "mae_flair": mae_fl,
                "ssim": ssim_t2, "psnr": psnr_t2, "mae": mae_t2,
            })
            print(f"  {study_id}: T2 SSIM={ssim_t2:.4f} PSNR={psnr_t2:.2f} | "
                  f"FLAIR SSIM={ssim_fl:.4f} PSNR={psnr_fl:.2f}")
            save_comparison_figure(
                us_vol, pred_t2, target_t2,
                os.path.join(img_dir, f"final_{study_id}_t2.png"),
                f"{exp_name} - {study_id} - T2")
            if target_flair is not None:
                save_comparison_figure(
                    us_vol, pred_fl, target_flair,
                    os.path.join(img_dir, f"final_{study_id}_flair.png"),
                    f"{exp_name} - {study_id} - FLAIR")

            save_nifti(pred_t2, os.path.join(pred_dir, f"{study_id}_pred_t2.nii.gz"))
            save_nifti(pred_fl, os.path.join(pred_dir, f"{study_id}_pred_flair.nii.gz"))
            save_nifti(target_t2, os.path.join(pred_dir, f"{study_id}_target_t2.nii.gz"))
            if target_flair is not None:
                save_nifti(target_flair, os.path.join(pred_dir, f"{study_id}_target_flair.nii.gz"))
            save_nifti(us_vol, os.path.join(pred_dir, f"{study_id}_input_us.nii.gz"))
            np.save(os.path.join(exp_dir, f"pred_{study_id}_t2.npy"), pred_t2)
            np.save(os.path.join(exp_dir, f"pred_{study_id}_flair.npy"), pred_fl)

        else:
            pred_vol = _sliding_window_inference_3d(
                G, us_vol, PATCH_3D, PATCH_3D_OVERLAP, is_cut,
                output_channels=1)
            if pred_vol.ndim == 4:
                pred_vol = pred_vol[..., 0]
            ssim_val = ssim_3d(target_t2, pred_vol)
            psnr_val = psnr_3d(target_t2, pred_vol)
            mae_val = mae_3d(target_t2, pred_vol)
            results.append({
                "subject": study_id, "ssim": ssim_val,
                "psnr": psnr_val, "mae": mae_val})
            print(f"  {study_id}: SSIM={ssim_val:.4f} PSNR={psnr_val:.2f}")
            save_comparison_figure(
                us_vol, pred_vol, target_t2,
                os.path.join(img_dir, f"final_{study_id}.png"),
                f"{exp_name} - {study_id}")
            save_nifti(pred_vol, os.path.join(pred_dir, f"{study_id}_pred_t2.nii.gz"))
            save_nifti(target_t2, os.path.join(pred_dir, f"{study_id}_target_t2.nii.gz"))
            save_nifti(us_vol, os.path.join(pred_dir, f"{study_id}_input_us.nii.gz"))
            np.save(os.path.join(exp_dir, f"pred_{study_id}.npy"), pred_vol)

    if results:
        mean_ssim = np.mean([r["ssim"] for r in results])
        mean_psnr = np.mean([r["psnr"] for r in results])
        print(f"  MEAN: SSIM={mean_ssim:.4f} PSNR={mean_psnr:.2f}")

    import csv as csv_mod
    csv_path = os.path.join(exp_dir, "results.csv")
    if is_multitask:
        fieldnames = ["subject", "ssim_t2", "psnr_t2", "mae_t2",
                       "ssim_flair", "psnr_flair", "mae_flair"]
    else:
        fieldnames = ["subject", "ssim", "psnr", "mae"]
    with open(csv_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    return results


# =============================================================================
# MASTER EXPERIMENT RUNNER
# =============================================================================
def run_experiment(exp_cfg, all_data, split, base_dir):
    """Run a single experiment: build model, train, evaluate, save results."""
    arch = exp_cfg["arch"]
    variant = exp_cfg["variant"]
    target = exp_cfg["target"]
    exp_name = f"{arch}_{variant}_{target}"
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "predictions"), exist_ok=True)

    log_progress(exp_name, "STARTED", base_dir=base_dir)

    cfg = CONFIGS[arch]
    is_multitask = (target == "t2_flair")

    # Filter data
    if is_multitask:
        valid_studies = {k: v for k, v in all_data.items()
                         if v.get("flair") is not None}
    else:
        valid_studies = all_data

    train_data = {k: v for k, v in valid_studies.items()
                  if k in set(split["train"])}
    test_data = {k: v for k, v in valid_studies.items()
                 if k in set(split["test"])}

    print(f"  Data: {len(train_data)} train, {len(test_data)} test studies")

    if len(train_data) == 0:
        print(f"[WARN] No training data for {exp_name}. Skipping.")
        log_progress(exp_name, "SKIPPED: no data", base_dir=base_dir)
        return []

    # Route to appropriate runner
    if variant == "3d":
        results = run_3d_experiment(
            exp_name, arch, target, train_data, test_data,
            exp_dir, base_dir, cfg)
    elif variant == "2d_3dpost":
        results = run_2d_3dpost_experiment(
            exp_name, arch, target, train_data, test_data,
            exp_dir, base_dir, cfg, all_data, split)
    else:
        results = run_2d_experiment(
            exp_name, arch, variant, target, train_data, test_data,
            exp_dir, base_dir, cfg, all_data, split)

    if results:
        mean_ssim = np.mean([r["ssim"] for r in results])
        mean_psnr = np.mean([r["psnr"] for r in results])
        log_progress(exp_name, "COMPLETED",
                     metrics={"ssim": mean_ssim, "psnr": mean_psnr},
                     base_dir=base_dir)
        append_results_csv(exp_name, results, base_dir=base_dir)
    else:
        log_progress(exp_name, "COMPLETED: no results", base_dir=base_dir)

    return results


# =============================================================================
# FINAL COMPARISON
# =============================================================================
def generate_final_comparison(base_dir):
    """Generate final comparison table and figures."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    csv_path = os.path.join(base_dir, "results_live.csv")
    if not os.path.exists(csv_path):
        print("[WARN] No results_live.csv found. Cannot generate comparison.")
        return

    # Read results (new CSV format: ssim_t2, psnr_t2, mae_t2, ssim_flair, ...)
    experiments = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp = row["experiment"]
            if exp not in experiments:
                experiments[exp] = []
            entry = {"subject": row["subject"]}
            # Parse T2 metrics
            entry["ssim_t2"] = float(row.get("ssim_t2", row.get("ssim", 0)))
            entry["psnr_t2"] = float(row.get("psnr_t2", row.get("psnr", 0)))
            entry["mae_t2"] = float(row.get("mae_t2", row.get("mae", 0)))
            # Parse FLAIR metrics (may be NaN)
            try:
                entry["ssim_flair"] = float(row.get("ssim_flair", "NaN"))
            except (ValueError, TypeError):
                entry["ssim_flair"] = float("nan")
            try:
                entry["psnr_flair"] = float(row.get("psnr_flair", "NaN"))
            except (ValueError, TypeError):
                entry["psnr_flair"] = float("nan")
            try:
                entry["mae_flair"] = float(row.get("mae_flair", "NaN"))
            except (ValueError, TypeError):
                entry["mae_flair"] = float("nan")
            experiments[exp].append(entry)

    # Compute means
    summary = []
    for exp_name, results in experiments.items():
        parts = exp_name.split("_")
        arch = parts[0]
        variant = parts[1] if len(parts) > 1 else ""
        target = "_".join(parts[2:]) if len(parts) > 2 else ""

        mean_ssim_t2 = np.mean([r["ssim_t2"] for r in results])
        mean_psnr_t2 = np.mean([r["psnr_t2"] for r in results])
        mean_mae_t2 = np.mean([r["mae_t2"] for r in results])
        std_ssim_t2 = np.std([r["ssim_t2"] for r in results])

        # FLAIR metrics (filter NaN)
        flair_ssim = [r["ssim_flair"] for r in results
                      if not np.isnan(r["ssim_flair"])]
        flair_psnr = [r["psnr_flair"] for r in results
                      if not np.isnan(r["psnr_flair"])]
        flair_mae = [r["mae_flair"] for r in results
                     if not np.isnan(r["mae_flair"])]
        mean_ssim_fl = np.mean(flair_ssim) if flair_ssim else float("nan")
        mean_psnr_fl = np.mean(flair_psnr) if flair_psnr else float("nan")
        mean_mae_fl = np.mean(flair_mae) if flair_mae else float("nan")

        summary.append({
            "experiment": exp_name,
            "arch": arch,
            "variant": variant,
            "target": target,
            "ssim_t2_mean": mean_ssim_t2,
            "ssim_t2_std": std_ssim_t2,
            "psnr_t2_mean": mean_psnr_t2,
            "mae_t2_mean": mean_mae_t2,
            "ssim_flair_mean": mean_ssim_fl,
            "psnr_flair_mean": mean_psnr_fl,
            "mae_flair_mean": mean_mae_fl,
            # Keep backward-compatible keys
            "ssim_mean": mean_ssim_t2,
            "ssim_std": std_ssim_t2,
            "psnr_mean": mean_psnr_t2,
            "mae_mean": mean_mae_t2,
            "n_subjects": len(results),
        })

    # Add 3D baselines
    for name, metrics in BASELINE_3D_RESULTS.items():
        summary.append({
            "experiment": name,
            "arch": name.split()[0].lower(),
            "variant": "3d_baseline",
            "target": "t2",
            "ssim_t2_mean": metrics["SSIM"],
            "ssim_t2_std": 0.0,
            "psnr_t2_mean": metrics["PSNR"],
            "mae_t2_mean": 0.0,
            "ssim_flair_mean": float("nan"),
            "psnr_flair_mean": float("nan"),
            "mae_flair_mean": float("nan"),
            "ssim_mean": metrics["SSIM"],
            "ssim_std": 0.0,
            "psnr_mean": metrics["PSNR"],
            "mae_mean": 0.0,
            "n_subjects": 0,
        })

    # Sort by SSIM
    summary.sort(key=lambda x: x["ssim_mean"], reverse=True)

    # Print table
    print("\n" + "=" * 110)
    print("  FINAL COMPARISON TABLE")
    print("=" * 110)
    print(f"{'Rank':<5} {'Experiment':<35} {'SSIM_T2':>10} {'PSNR_T2':>10} "
          f"{'MAE_T2':>10} {'SSIM_FL':>10} {'PSNR_FL':>10}")
    print("-" * 110)
    for i, s in enumerate(summary):
        fl_ssim = f"{s['ssim_flair_mean']:.4f}" if not np.isnan(s.get('ssim_flair_mean', float('nan'))) else "  ---"
        fl_psnr = f"{s['psnr_flair_mean']:.2f}" if not np.isnan(s.get('psnr_flair_mean', float('nan'))) else "  ---"
        print(f"{i+1:<5} {s['experiment']:<35} "
              f"{s['ssim_t2_mean']:>8.4f}+{s['ssim_t2_std']:.3f} "
              f"{s['psnr_t2_mean']:>8.2f} "
              f"{s['mae_t2_mean']:>8.4f} "
              f"{fl_ssim:>10} {fl_psnr:>10}")

    # Save summary CSV
    summary_csv = os.path.join(base_dir, "final_results.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "experiment", "arch", "variant", "target",
            "ssim_t2_mean", "ssim_t2_std", "psnr_t2_mean", "mae_t2_mean",
            "ssim_flair_mean", "psnr_flair_mean", "mae_flair_mean",
            "n_subjects"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nSaved to {summary_csv}")

    # Bar chart
    try:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        names = [s["experiment"][:30] for s in summary[:20]]
        ssim_vals = [s["ssim_mean"] for s in summary[:20]]
        psnr_vals = [s["psnr_mean"] for s in summary[:20]]

        colors = []
        for s in summary[:20]:
            if "3d_baseline" in s.get("variant", ""):
                colors.append("#ff6b6b")
            elif "3d" in s.get("variant", ""):
                colors.append("#4ecdc4")
            elif "25d" in s.get("variant", ""):
                colors.append("#45b7d1")
            elif "3dpost" in s.get("variant", ""):
                colors.append("#96ceb4")
            else:
                colors.append("#dfe6e9")

        axes[0].barh(range(len(names)), ssim_vals, color=colors)
        axes[0].set_yticks(range(len(names)))
        axes[0].set_yticklabels(names, fontsize=8)
        axes[0].set_xlabel("SSIM")
        axes[0].set_title("SSIM (higher is better)")
        axes[0].invert_yaxis()

        axes[1].barh(range(len(names)), psnr_vals, color=colors)
        axes[1].set_yticks(range(len(names)))
        axes[1].set_yticklabels(names, fontsize=8)
        axes[1].set_xlabel("PSNR (dB)")
        axes[1].set_title("PSNR (higher is better)")
        axes[1].invert_yaxis()

        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "final_comparison.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved comparison figure to {os.path.join(base_dir, 'final_comparison.png')}")
    except Exception as e:
        print(f"[WARN] Could not generate comparison figure: {e}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("  COMPREHENSIVE 2D/3D US->MRI SYNTHESIS COMPARISON")
    print(f"  {len(EXPERIMENT_MATRIX)} experiments")
    print("=" * 60)

    # Load ALL data once
    print("\n[INFO] Loading all data...")
    all_data = load_all_data(DATA_DIR)

    # Create or load subject-level split
    split = get_or_create_split(SPLIT_FILE, os.path.join(DATA_DIR, "US"))

    print(f"\nSplit: {len(split['train'])} train, {len(split['test'])} test studies")
    print(f"       {len(split['train_subjects'])} train, "
          f"{len(split['test_subjects'])} test subjects")

    # Initialize progress
    with open(os.path.join(BASE_DIR, "progress.txt"), "w") as f:
        f.write(f"STARTED: {datetime.now()}\n")
        f.write(f"Total experiments: {len(EXPERIMENT_MATRIX)}\n\n")

    # Run all experiments
    all_results = {}
    for i, exp_cfg in enumerate(EXPERIMENT_MATRIX):
        exp_name = f"{exp_cfg['arch']}_{exp_cfg['variant']}_{exp_cfg['target']}"
        exp_dir = os.path.join(BASE_DIR, exp_name)
        results_csv = os.path.join(exp_dir, "results.csv")

        # Skip if results.csv exists (experiment already completed)
        if os.path.exists(results_csv):
            print(f"\n[SKIP] {exp_name} — results.csv found")
            continue

        print(f"\n{'#' * 60}")
        print(f"  EXPERIMENT {i + 1}/{len(EXPERIMENT_MATRIX)}: {exp_name}")
        print(f"{'#' * 60}")
        try:
            results = run_experiment(exp_cfg, all_data, split, BASE_DIR)
            all_results[exp_name] = results
        except Exception as e:
            print(f"[ERROR] {exp_name}: {e}")
            traceback.print_exc()
            log_progress(exp_name, f"FAILED: {e}", base_dir=BASE_DIR)
        finally:
            # Clear GPU memory between experiments
            import gc
            tf.keras.backend.clear_session()
            gc.collect()

    # Final comparison
    print("\n" + "=" * 60)
    print("  GENERATING FINAL COMPARISON")
    print("=" * 60)
    generate_final_comparison(BASE_DIR)

    print("\n" + "=" * 60)
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"  Results in: {BASE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    DRY_RUN = "--dry-run" in sys.argv
    if DRY_RUN:
        print("\n*** DRY RUN MODE: 5 steps per experiment ***\n")
        STEPS_2D = 5
        STEPS_3D = 5
        STEPS_REFINER = 5
        EVAL_EVERY = 999999  # skip eval in dry run
        SAVE_IMG_EVERY = 999999
    main()
