"""
Common utilities for 2D/2.5D/3D US->MRI synthesis experiments.
Shared data loading, normalization, metrics, visualization, EMA, losses.

Author: Santiago Cepeda / BrainUS-AI
"""
import matplotlib
matplotlib.use("Agg")

import os, json, glob, time, math, csv
import numpy as np
import nibabel as nib
import tensorflow as tf
from scipy.ndimage import zoom, rotate
from matplotlib import pyplot as plt
from datetime import datetime

print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

# Allow memory growth
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

# =============================================================================
# COMPAT: SpectralNormalization (Keras 3.x safe)
# =============================================================================
try:
    SN = tf.keras.layers.SpectralNormalization
    print("[INFO] Using built-in SpectralNormalization")
except AttributeError:
    class SN(tf.keras.layers.Wrapper):
        """Spectral Normalization -- Keras 3.x compatible."""
        def __init__(self, layer, power_iterations=1, **kwargs):
            super().__init__(layer, **kwargs)
            self.power_iterations = power_iterations

        def build(self, input_shape):
            super().build(input_shape)
            k = self.layer.kernel
            self.u = self.add_weight(
                name="sn_u", shape=(1, k.shape[-1]),
                initializer=tf.initializers.TruncatedNormal(0.02),
                trainable=False, dtype=tf.float32)

        def call(self, inputs, training=None):
            k = tf.cast(self.layer.kernel, tf.float32)
            kr = tf.reshape(k, [-1, k.shape[-1]])
            u_hat = tf.cast(self.u, tf.float32)
            for _ in range(self.power_iterations):
                v_hat = tf.nn.l2_normalize(tf.matmul(u_hat, tf.transpose(kr)))
                u_hat = tf.nn.l2_normalize(tf.matmul(v_hat, kr))
            sigma = tf.squeeze(tf.matmul(tf.matmul(v_hat, kr), tf.transpose(u_hat)))
            if training:
                self.u.assign(u_hat)
            nk = tf.cast(k / (sigma + 1e-12), self.layer.kernel.dtype)
            old_kernel = self.layer.kernel
            self.layer.kernel = nk
            out = self.layer(inputs)
            self.layer.kernel = old_kernel
            return out
    print("[COMPAT] Custom SpectralNormalization (Keras 3.x)")


# =============================================================================
# COMPAT: GroupNormalization
# =============================================================================
try:
    _GN = tf.keras.layers.GroupNormalization
except AttributeError:
    class _GN(tf.keras.layers.Layer):
        def __init__(self, groups=32, epsilon=1e-5, **kwargs):
            super().__init__(**kwargs)
            self.groups = groups
            self.epsilon = epsilon
        def build(self, input_shape):
            ch = int(input_shape[-1])
            g = min(self.groups, ch)
            while g > 1 and ch % g != 0:
                g -= 1
            self.groups = g
            self.gamma = self.add_weight(
                name="gamma", shape=(ch,), initializer="ones", trainable=True)
            self.beta = self.add_weight(
                name="beta", shape=(ch,), initializer="zeros", trainable=True)
            super().build(input_shape)
        def call(self, x):
            s = tf.shape(x); nd = len(x.shape); ch = x.shape[-1]
            x = tf.reshape(x, tf.concat([s[:-1], [self.groups, ch // self.groups]], 0))
            m, v = tf.nn.moments(x, axes=[-1], keepdims=True)
            x = (x - m) / tf.sqrt(v + self.epsilon)
            x = tf.reshape(x, s)
            sh = [1] * (nd - 1) + [ch]
            return x * tf.reshape(self.gamma, sh) + tf.reshape(self.beta, sh)
    print("[COMPAT] Custom GroupNormalization")


def _valid_groups(channels, desired=32):
    g = min(desired, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return g


def GN(groups=32, channels=None, **kw):
    if channels is not None:
        groups = _valid_groups(channels, groups)
    return _GN(groups=groups, **kw)


# =============================================================================
# COMPAT: InstanceNormalization2D (normalize over H, W for 2D)
# =============================================================================
class InstanceNormalization2D(tf.keras.layers.Layer):
    """Instance Normalization for 2D: normalizes over spatial dims [1, 2]."""
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.gamma = self.add_weight(
            name="gamma", shape=(channels,), initializer="ones", trainable=True)
        self.beta = self.add_weight(
            name="beta", shape=(channels,), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x):
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_norm = (x - mean) / tf.sqrt(var + self.epsilon)
        ndims = len(x.shape)
        channels = x.shape[-1]
        shape = [1] * (ndims - 1) + [channels]
        return x_norm * tf.reshape(self.gamma, shape) + tf.reshape(self.beta, shape)

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


def InstNorm2D(**kwargs):
    return InstanceNormalization2D(**kwargs)


# =============================================================================
# COMPAT: InstanceNormalization3D (normalize over D, H, W for 3D)
# =============================================================================
class InstanceNormalization3D(tf.keras.layers.Layer):
    """Instance Normalization for 3D: normalizes over spatial dims [1, 2, 3]."""
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.gamma = self.add_weight(
            name="gamma", shape=(channels,), initializer="ones", trainable=True)
        self.beta = self.add_weight(
            name="beta", shape=(channels,), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x):
        mean, var = tf.nn.moments(x, axes=[1, 2, 3], keepdims=True)
        x_norm = (x - mean) / tf.sqrt(var + self.epsilon)
        ndims = len(x.shape)
        channels = x.shape[-1]
        shape = [1] * (ndims - 1) + [channels]
        return x_norm * tf.reshape(self.gamma, shape) + tf.reshape(self.beta, shape)

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


def InstNorm3D(**kwargs):
    return InstanceNormalization3D(**kwargs)


# =============================================================================
# GLOBAL CONFIG
# =============================================================================
VOLUME_SHAPE = (192, 192, None)  # 192x192xN (variable depth)
FG_THRESHOLD_FRAC = 0.01
US_PERCENTILE_LOW = 2
US_PERCENTILE_HIGH = 98
MRI_ZSCORE_CLIP = 3.0

# 3D baseline results for comparison
BASELINE_3D_RESULTS = {
    "Pix2Pix v3 3D":     {"SSIM": 0.773, "PSNR": 19.93},
    "SwinPix2Pix v4 3D": {"SSIM": 0.772, "PSNR": 20.00},
    "CycleGAN v2 3D":    {"SSIM": 0.741, "PSNR": 19.72},
    "CUT v3 3D":         {"SSIM": 0.737, "PSNR": 19.88},
}


# =============================================================================
# 1. SUBJECT-LEVEL SPLIT
# =============================================================================
def create_subject_split(us_dir, seed=42):
    """Split by SUBJECT (ReMIND-XXX), not by study (ReMIND-XXX-pre/post)."""
    us_files = sorted(glob.glob(os.path.join(us_dir, "*.nii.gz")))
    subjects = {}
    for f in us_files:
        fn = os.path.basename(f)
        study = fn.replace("-us.nii.gz", "")
        subj = study.rsplit("-", 1)[0]
        if subj not in subjects:
            subjects[subj] = []
        subjects[subj].append(study)

    subj_list = sorted(subjects.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(subj_list)
    n_train = int(len(subj_list) * 0.8)
    train_subjs = sorted(subj_list[:n_train])
    test_subjs = sorted(subj_list[n_train:])
    train_studies = [s for subj in train_subjs for s in sorted(subjects[subj])]
    test_studies = [s for subj in test_subjs for s in sorted(subjects[subj])]
    return {
        "train_subjects": train_subjs,
        "test_subjects": test_subjs,
        "train": train_studies,
        "test": test_studies,
    }


def get_or_create_split(split_file, us_dir):
    """Load existing split or create new one."""
    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            split = json.load(f)
        print(f"[INFO] Loaded split from {split_file}")
    else:
        split = create_subject_split(us_dir)
        os.makedirs(os.path.dirname(split_file), exist_ok=True)
        with open(split_file, "w") as f:
            json.dump(split, f, indent=2)
        print(f"[INFO] Created and saved split to {split_file}")
    print(f"  Train subjects: {len(split['train_subjects'])}, "
          f"studies: {len(split['train'])}")
    print(f"  Test subjects: {len(split['test_subjects'])}, "
          f"studies: {len(split['test'])}")
    return split


# =============================================================================
# 2. DATA LOADING & NORMALIZATION
# =============================================================================
def load_nifti(file_path):
    img = nib.load(file_path)
    return img.get_fdata().astype(np.float32), img.affine


def resize_volume(volume, target_shape=(128, 128, 128)):
    factors = [t / s for t, s in zip(target_shape, volume.shape[:3])]
    return zoom(volume, factors, order=1).astype(np.float32)


def get_foreground_mask(volume, threshold_frac=0.01):
    return volume > (volume.max() * threshold_frac)


def normalize_us(volume, fg_thr=0.01, plow=2, phigh=98):
    fg = volume > volume.max() * fg_thr
    if fg.sum() < 100:
        return np.full_like(volume, -1.0)
    fv = volume[fg]
    lo, hi = np.percentile(fv, plow), np.percentile(fv, phigh)
    out = np.full_like(volume, -1.0)
    if hi - lo > 1e-8:
        c = np.clip(volume, lo, hi)
        s = (c - lo) / (hi - lo) * 2 - 1
        out[fg] = s[fg]
    return out.astype(np.float32)


def normalize_mri(volume, fg_thr=0.01, clip_std=3.0):
    fg = volume > volume.max() * fg_thr
    if fg.sum() < 100:
        return np.full_like(volume, -1.0)
    fv = volume[fg]
    m, s = np.mean(fv), np.std(fv)
    out = np.full_like(volume, -1.0)
    if s > 1e-8:
        z = np.clip((volume - m) / s, -clip_std, clip_std) / clip_std
        out[fg] = z[fg]
    return out.astype(np.float32)


def load_all_data(base_dir):
    """Load US, T2, and optionally FLAIR for all studies.
    Volumes are loaded at native 192x192xN size (no resizing).
    """
    us_dir = os.path.join(base_dir, "US")
    t2_dir = os.path.join(base_dir, "MR-T2")
    flair_dir = os.path.join(base_dir, "MR-FLAIR")

    data = {}
    for usp in sorted(glob.glob(os.path.join(us_dir, "*.nii.gz"))):
        fn = os.path.basename(usp)
        study = fn.replace("-us.nii.gz", "")
        t2p = os.path.join(t2_dir, fn.replace("-us.nii.gz", "-mri.nii.gz"))
        flp = os.path.join(flair_dir, fn.replace("-us.nii.gz", "-mri.nii.gz"))
        if not os.path.exists(t2p):
            continue
        try:
            uv, _ = load_nifti(usp)
            tv, _ = load_nifti(t2p)
            uv = normalize_us(uv)
            tv = normalize_mri(tv)
            # Check for empty volumes
            if uv.max() <= -0.99 or tv.max() <= -0.99:
                print(f"  [WARN] {study}: empty volume after normalization, skipping")
                continue
            entry = {"us": uv, "t2": tv, "flair": None}
            if os.path.exists(flp):
                fv, _ = load_nifti(flp)
                fv_norm = normalize_mri(fv)
                if fv_norm.max() > -0.99:
                    entry["flair"] = fv_norm
                else:
                    entry["flair"] = None
            data[study] = entry
            print(f"  Loaded {study} shape={uv.shape} "
                  f"(FLAIR={'yes' if entry['flair'] is not None else 'no'})")
        except Exception as e:
            print(f"  [ERROR] {study}: {e}")
            continue
    print(f"[INFO] Total studies: {len(data)}")
    return data


def save_nifti(volume, path, affine=None):
    """Save a volume as NIfTI. Affine defaults to identity if not provided."""
    if affine is None:
        affine = np.eye(4)
    img = nib.Nifti1Image(volume.astype(np.float32), affine)
    nib.save(img, path)


# =============================================================================
# 3. SLICE EXTRACTION
# =============================================================================
def get_slice_along_axis(volume, idx, axis):
    if axis == 0:
        return volume[idx, :, :]
    elif axis == 1:
        return volume[:, idx, :]
    else:
        return volume[:, :, idx]


def set_slice_along_axis(volume, idx, axis, sl):
    if axis == 0:
        volume[idx, :, :] = sl
    elif axis == 1:
        volume[:, idx, :] = sl
    else:
        volume[:, :, idx] = sl


def extract_slices_along_axis(volume, axis=2, min_fg_fraction=0.01):
    """Extract non-empty slices along a given axis."""
    slices = []
    for i in range(volume.shape[axis]):
        sl = get_slice_along_axis(volume, i, axis)
        if np.mean(sl > -0.95) > min_fg_fraction:
            slices.append((i, sl))
    return slices


def prepare_2d_slice_dataset(data, split, axis=2, is_25d=False,
                              include_flair=False, is_training=True):
    """Extract 2D slices from loaded volumes for training.

    Returns list of tuples: (us_slice, mri_t2_slice[, mri_flair_slice])
    For 2.5D: us_slice is (H, W, 3) with 3 consecutive slices.
    """
    study_ids = split["train"] if is_training else split["test"]
    slices = []

    for study_id in study_ids:
        if study_id not in data:
            continue
        us_vol = data[study_id]["us"]
        t2_vol = data[study_id]["t2"]
        flair_vol = data[study_id].get("flair", None)

        n = us_vol.shape[axis]
        for i in range(n):
            us_sl = get_slice_along_axis(us_vol, i, axis)
            t2_sl = get_slice_along_axis(t2_vol, i, axis)

            if np.mean(us_sl > -0.95) < 0.01:
                continue

            if is_25d:
                idx_prev = max(0, i - 1)
                idx_next = min(n - 1, i + 1)
                us_prev = get_slice_along_axis(us_vol, idx_prev, axis)
                us_next = get_slice_along_axis(us_vol, idx_next, axis)
                us_sl = np.stack([us_prev, us_sl, us_next], axis=-1)
            else:
                us_sl = us_sl[..., np.newaxis]

            if include_flair and flair_vol is not None:
                flair_sl = get_slice_along_axis(flair_vol, i, axis)
                slices.append((us_sl, t2_sl[..., np.newaxis],
                               flair_sl[..., np.newaxis]))
            else:
                slices.append((us_sl, t2_sl[..., np.newaxis]))

    return slices


# =============================================================================
# 4. 2D AUGMENTATION (paired)
# =============================================================================
def augment_2d_pair(us_slice, mri_slice, mri_flair_slice=None,
                     flip_p=0.5, rot_p=0.3, intensity_p=0.4):
    """Apply same spatial augmentation to US and MRI slices.
    Handles multi-channel inputs (2.5D US with shape (H,W,3) or multi-target).
    """
    targets = [mri_slice]
    if mri_flair_slice is not None:
        targets.append(mri_flair_slice)

    # Horizontal flip
    if np.random.rand() < flip_p:
        us_slice = np.flip(us_slice, axis=1).copy()
        targets = [np.flip(t, axis=1).copy() for t in targets]

    # Vertical flip
    if np.random.rand() < flip_p:
        us_slice = np.flip(us_slice, axis=0).copy()
        targets = [np.flip(t, axis=0).copy() for t in targets]

    # Rotation
    if np.random.rand() < rot_p:
        from scipy.ndimage import rotate as rot2d
        angle = np.random.uniform(-15, 15)
        # Handle multi-channel: rotate each channel
        if us_slice.ndim == 3 and us_slice.shape[-1] > 1:
            us_rotated = np.stack([
                rot2d(us_slice[..., c], angle, reshape=False, order=1,
                      mode='nearest').astype(np.float32)
                for c in range(us_slice.shape[-1])
            ], axis=-1)
            us_slice = us_rotated
        else:
            us_2d = us_slice[..., 0] if us_slice.ndim == 3 else us_slice
            us_2d = rot2d(us_2d, angle, reshape=False, order=1,
                          mode='nearest').astype(np.float32)
            us_slice = us_2d[..., np.newaxis] if us_slice.ndim == 3 else us_2d
        targets_rot = []
        for t in targets:
            if t.ndim == 3 and t.shape[-1] > 1:
                t_rot = np.stack([
                    rot2d(t[..., c], angle, reshape=False, order=1,
                          mode='nearest').astype(np.float32)
                    for c in range(t.shape[-1])
                ], axis=-1)
            else:
                t_2d = t[..., 0] if t.ndim == 3 else t
                t_2d = rot2d(t_2d, angle, reshape=False, order=1,
                              mode='nearest').astype(np.float32)
                t_rot = t_2d[..., np.newaxis] if t.ndim == 3 else t_2d
            targets_rot.append(t_rot)
        targets = targets_rot

    # US intensity augmentation (only on US)
    if np.random.rand() < intensity_p:
        if us_slice.ndim == 3:
            for c in range(us_slice.shape[-1]):
                ch = us_slice[..., c]
                fg = ch > -0.95
                gamma = np.random.uniform(0.8, 1.2)
                aug = (np.power(np.clip((ch + 1) / 2, 1e-8, 1), gamma) * 2 - 1)
                noise_std = np.random.uniform(0.0, 0.03)
                aug = aug + np.random.normal(0, noise_std, ch.shape)
                us_slice[..., c] = np.where(fg, np.clip(aug, -1, 1), ch).astype(np.float32)
        else:
            fg = us_slice > -0.95
            gamma = np.random.uniform(0.8, 1.2)
            aug = (np.power(np.clip((us_slice + 1) / 2, 1e-8, 1), gamma) * 2 - 1)
            noise_std = np.random.uniform(0.0, 0.03)
            aug = aug + np.random.normal(0, noise_std, us_slice.shape)
            us_slice = np.where(fg, np.clip(aug, -1, 1), us_slice).astype(np.float32)

    if mri_flair_slice is not None:
        return us_slice, targets[0], targets[1]
    return us_slice, targets[0]


# =============================================================================
# 5. EMA (Exponential Moving Average)
# =============================================================================
class EMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = [tf.Variable(w, trainable=False, name=f"ema_{i}")
                       for i, w in enumerate(model.weights)]
        self.backup = None

    def update(self):
        for shadow, weight in zip(self.shadow, self.model.weights):
            shadow.assign(self.decay * shadow + (1.0 - self.decay) * weight)

    def apply_shadow(self):
        """Swap model weights with EMA weights (for evaluation)."""
        self.backup = [tf.identity(w) for w in self.model.weights]
        for shadow, weight in zip(self.shadow, self.model.weights):
            weight.assign(shadow)

    def restore(self):
        """Restore original model weights after evaluation."""
        if self.backup is not None:
            for backup, weight in zip(self.backup, self.model.weights):
                weight.assign(backup)


# =============================================================================
# 6. LR SCHEDULE: Cosine Annealing with Warmup
# =============================================================================
class CosineWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr, warmup, total):
        super().__init__()
        self.lr = lr
        self.warmup = float(warmup)
        self.total = float(total)

    def __call__(self, step):
        s = tf.cast(step, tf.float32)
        w = self.lr * s / tf.maximum(self.warmup, 1.0)
        p = tf.clip_by_value((s - self.warmup) / (self.total - self.warmup), 0, 1)
        c = self.lr * 0.5 * (1 + tf.cos(math.pi * p))
        return tf.where(s < self.warmup, w, c)

    def get_config(self):
        return {"lr": self.lr, "warmup": self.warmup, "total": self.total}


# =============================================================================
# 7. LOSSES (2D)
# =============================================================================
def lsgan_g(fake_outs):
    return tf.reduce_mean(
        [tf.reduce_mean(tf.square(f[0] - 1)) for f in fake_outs]) / len(fake_outs)


def lsgan_d(real_outs, fake_outs):
    loss = 0.0
    for r, f in zip(real_outs, fake_outs):
        loss += tf.reduce_mean(tf.square(r[0] - 1)) + tf.reduce_mean(tf.square(f[0]))
    return loss / (2 * len(real_outs))


def fm_loss(real_outs, fake_outs):
    loss = 0.0
    c = 0
    for r, f in zip(real_outs, fake_outs):
        for rf, ff in zip(r[1:], f[1:]):
            loss += tf.reduce_mean(tf.abs(rf - ff))
            c += 1
    return loss / max(c, 1)


def ssim_loss_2d(target, pred):
    """SSIM loss for 2D images (batch, H, W, C). Handles multi-channel."""
    t01 = (tf.cast(target, tf.float32) + 1.0) / 2.0
    p01 = (tf.cast(pred, tf.float32) + 1.0) / 2.0
    n_ch = t01.shape[-1] or 1
    if n_ch == 1:
        return 1.0 - tf.reduce_mean(tf.image.ssim(t01, p01, max_val=1.0))
    else:
        ssim_vals = []
        for c in range(n_ch):
            tc = t01[..., c:c + 1]
            gc = p01[..., c:c + 1]
            ssim_vals.append(tf.image.ssim(tc, gc, max_val=1.0))
        return 1.0 - tf.reduce_mean(tf.stack(ssim_vals))


def edge_loss_2d(target, pred):
    """Edge preservation loss for 2D."""
    target = tf.cast(target, tf.float32)
    pred = tf.cast(pred, tf.float32)
    dx_t = target[:, 1:, :, :] - target[:, :-1, :, :]
    dx_p = pred[:, 1:, :, :] - pred[:, :-1, :, :]
    dy_t = target[:, :, 1:, :] - target[:, :, :-1, :]
    dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    return (tf.reduce_mean(tf.abs(dx_t - dx_p)) +
            tf.reduce_mean(tf.abs(dy_t - dy_p))) / 2


def r1_penalty_2d(disc, real_inp, real_tar):
    """R1 gradient penalty for 2D discriminator."""
    real_tar = tf.cast(real_tar, tf.float32)
    with tf.GradientTape() as t:
        t.watch(real_tar)
        pred = disc([real_inp, real_tar], training=True)
        loss = tf.reduce_sum([tf.reduce_sum(p[0]) for p in pred])
    g = t.gradient(loss, real_tar)
    if g is None:
        return tf.constant(0.0)
    return tf.reduce_mean(tf.reduce_sum(tf.square(g), axis=[1, 2, 3]))


def r1_penalty_uncond_2d(disc, real_img):
    """R1 gradient penalty for unconditional 2D discriminator."""
    real_img = tf.cast(real_img, tf.float32)
    with tf.GradientTape() as t:
        t.watch(real_img)
        pred = disc(real_img, training=True)
        loss = tf.reduce_sum([tf.reduce_sum(p[0]) for p in pred])
    g = t.gradient(loss, real_img)
    if g is None:
        return tf.constant(0.0)
    return tf.reduce_mean(tf.reduce_sum(tf.square(g), axis=[1, 2, 3]))


def cycle_loss(real, cycled):
    return tf.reduce_mean(tf.abs(real - cycled))


def identity_loss(real, same):
    return tf.reduce_mean(tf.abs(real - same))


# =============================================================================
# 8. LOSSES (3D)
# =============================================================================
def ssim_loss_3d(target, gen):
    """SSIM loss for 3D patches -- computed on axial slices (axis 3 = depth).
    Input shape: (batch, H, W, D, ch). Slices along D give HxW slices.
    Handles multi-channel and asymmetric patches (e.g. 64x64x32).
    """
    t01 = (tf.cast(target, tf.float32) + 1) / 2
    g01 = (tf.cast(gen, tf.float32) + 1) / 2
    # Subsample along depth for efficiency
    t01 = t01[:, :, :, ::2]
    g01 = g01[:, :, :, ::2]
    s = tf.shape(t01)
    n_ch = t01.shape[-1] or 1
    total = tf.constant(0.0)
    for c in range(n_ch):
        tc = t01[..., c:c+1]
        gc = g01[..., c:c+1]
        # Reshape: merge batch and depth dims, keep H, W as spatial
        # (batch, H, W, D, 1) -> (batch*D, H, W, 1)
        tc = tf.transpose(tc, [0, 3, 1, 2, 4])  # (batch, D, H, W, 1)
        gc = tf.transpose(gc, [0, 3, 1, 2, 4])
        tc = tf.reshape(tc, [s[0] * s[3], s[1], s[2], 1])
        gc = tf.reshape(gc, [s[0] * s[3], s[1], s[2], 1])
        total += 1.0 - tf.reduce_mean(tf.image.ssim(tc, gc, 1.0))
    return total / tf.cast(n_ch, tf.float32)


def edge_loss_3d(target, generated):
    target = tf.cast(target, tf.float32)
    generated = tf.cast(generated, tf.float32)
    loss = 0.0
    for axis in [1, 2, 3]:
        grad_t = target - tf.roll(target, shift=1, axis=axis)
        grad_g = generated - tf.roll(generated, shift=1, axis=axis)
        loss += tf.reduce_mean(tf.abs(grad_t - grad_g))
    return loss / 3.0


def ssim_loss_3d_refiner(target, generated):
    """SSIM loss for 3D refiner patches. Slices along depth axis.
    Supports arbitrary channel count (uses static last-dim size).
    """
    t01 = (tf.cast(target, tf.float32) + 1.0) / 2.0
    g01 = (tf.cast(generated, tf.float32) + 1.0) / 2.0
    C = int(t01.shape[-1]) if t01.shape[-1] is not None else 1
    shape = tf.shape(t01)
    # (batch, H, W, D, C) -> (batch, D, H, W, C)
    t01 = tf.transpose(t01, [0, 3, 1, 2, 4])
    g01 = tf.transpose(g01, [0, 3, 1, 2, 4])
    t2d = tf.reshape(t01, [shape[0] * shape[3], shape[1], shape[2], C])
    g2d = tf.reshape(g01, [shape[0] * shape[3], shape[1], shape[2], C])
    return 1.0 - tf.reduce_mean(tf.image.ssim(t2d, g2d, max_val=1.0))


# =============================================================================
# 9. 3D METRICS (full volumes)
# =============================================================================
def ssim_3d(target, pred):
    """Compute SSIM on full 3D volume (slice-by-slice along axis 2 / depth)."""
    t01 = (target + 1.0) / 2.0
    p01 = (pred + 1.0) / 2.0
    ssim_vals = []
    for z in range(target.shape[2]):
        t_sl = t01[:, :, z][np.newaxis, ..., np.newaxis]
        p_sl = p01[:, :, z][np.newaxis, ..., np.newaxis]
        if np.mean(t_sl > 0.025) > 0.01:
            s = tf.image.ssim(
                tf.constant(t_sl, dtype=tf.float32),
                tf.constant(p_sl, dtype=tf.float32),
                max_val=1.0).numpy()
            ssim_vals.append(float(s))
    return np.mean(ssim_vals) if ssim_vals else 0.0


def psnr_3d(target, pred):
    """Compute PSNR on full 3D volume (foreground only)."""
    t01 = (target + 1.0) / 2.0
    p01 = (pred + 1.0) / 2.0
    fg = t01 > 0.025
    if np.sum(fg) < 100:
        return 0.0
    mse = np.mean((t01[fg] - p01[fg]) ** 2)
    if mse < 1e-10:
        return 50.0
    return 10.0 * np.log10(1.0 / mse)


def mae_3d(target, pred):
    """Compute MAE on full 3D volume (foreground only)."""
    t01 = (target + 1.0) / 2.0
    p01 = (pred + 1.0) / 2.0
    fg = t01 > 0.025
    if np.sum(fg) < 100:
        return 1.0
    return float(np.mean(np.abs(t01[fg] - p01[fg])))


# =============================================================================
# 10. TRIPLANAR INFERENCE (192x192xN volumes)
# =============================================================================
def _pad_to_192(slice_2d):
    """Pad a HxW slice to 192x192 with -1 (background).
    Crops dims that exceed 192 (caller should prefer sliding window for that
    case — this remains here for backward compatibility with callers that
    never receive oversized slices).
    """
    h, w = slice_2d.shape
    if h == 192 and w == 192:
        return slice_2d
    padded = np.full((192, 192), -1.0, dtype=np.float32)
    ch = min(h, 192)
    cw = min(w, 192)
    padded[:ch, :cw] = slice_2d[:ch, :cw]
    return padded


# =============================================================================
# Sliding-window inference helpers for triplanar_inference_2d / _25d
# Used when slices along axis 0 or 1 are larger than 192 (because D > 192).
# =============================================================================
_TRIPLANAR_PATCH = 192
_TRIPLANAR_STRIDE = 96


def _triplanar_patch_starts(length, patch=_TRIPLANAR_PATCH, stride=_TRIPLANAR_STRIDE):
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def _triplanar_predict_slice(generator, sl, input_ch, output_ch):
    """Predict a slice of any HxW size via pad-or-sliding-window.
    sl: 2D (H,W) or 3D (H,W,input_ch); returns (H,W,output_ch) float32.
    """
    if sl.ndim == 2:
        sl = sl[..., np.newaxis]
    H, W, _ = sl.shape

    if H <= _TRIPLANAR_PATCH and W <= _TRIPLANAR_PATCH:
        patch = np.full(
            (_TRIPLANAR_PATCH, _TRIPLANAR_PATCH, input_ch),
            -1.0, dtype=np.float32)
        patch[:H, :W, :] = sl
        inp = patch[np.newaxis].astype(np.float32)
        pred = generator(inp, training=False)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        return pred.numpy()[0][:H, :W, :]

    out = np.zeros((H, W, output_ch), dtype=np.float32)
    cnt = np.zeros((H, W, 1), dtype=np.float32)
    for y in _triplanar_patch_starts(H):
        for x in _triplanar_patch_starts(W):
            patch = np.full(
                (_TRIPLANAR_PATCH, _TRIPLANAR_PATCH, input_ch),
                -1.0, dtype=np.float32)
            ph = min(_TRIPLANAR_PATCH, H - y)
            pw = min(_TRIPLANAR_PATCH, W - x)
            patch[:ph, :pw, :] = sl[y:y + ph, x:x + pw, :]
            inp = patch[np.newaxis].astype(np.float32)
            pred = generator(inp, training=False)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            pred_np = pred.numpy()[0]
            out[y:y + ph, x:x + pw, :] += pred_np[:ph, :pw, :]
            cnt[y:y + ph, x:x + pw, :] += 1.0
    return out / np.maximum(cnt, 1e-6)


def _triplanar_write_slice(pred_vol, axis, i, pred_hw, output_channels):
    if output_channels == 1:
        set_slice_along_axis(pred_vol, i, axis, pred_hw[..., 0])
    else:
        if axis == 0:
            pred_vol[i, :, :, :] = pred_hw
        elif axis == 1:
            pred_vol[:, i, :, :] = pred_hw
        else:
            pred_vol[:, :, i, :] = pred_hw


def triplanar_inference_2d(generator, us_volume, output_channels=1):
    """Predict from 3 axes and average. Works for 2D models.
    Handles 192x192xD volumes with any D: non-axial slices are (192, D), which
    get pad-or-sliding-window predicted to (192, D, output_channels).
    """
    H, W, D = us_volume.shape[:3]
    predictions = []
    for axis in range(3):
        if output_channels == 1:
            pred_vol = np.zeros((H, W, D), dtype=np.float32)
        else:
            pred_vol = np.zeros((H, W, D, output_channels), dtype=np.float32)
        n = us_volume.shape[axis]
        for i in range(n):
            sl = get_slice_along_axis(us_volume, i, axis)
            pred_hw = _triplanar_predict_slice(generator, sl, 1, output_channels)
            _triplanar_write_slice(pred_vol, axis, i, pred_hw, output_channels)
        predictions.append(pred_vol)
    return np.clip(np.mean(predictions, axis=0), -1, 1)


def triplanar_inference_25d(generator, us_volume, output_channels=1):
    """Predict from 3 axes using 3-slice windows and average.
    Handles 192x192xD volumes with any D via pad-or-sliding-window inference.
    """
    H, W, D = us_volume.shape[:3]
    predictions = []
    for axis in range(3):
        if output_channels == 1:
            pred_vol = np.zeros((H, W, D), dtype=np.float32)
        else:
            pred_vol = np.zeros((H, W, D, output_channels), dtype=np.float32)
        n = us_volume.shape[axis]
        for i in range(n):
            idx_prev = max(0, i - 1)
            idx_next = min(n - 1, i + 1)
            sl_stack = np.stack([
                get_slice_along_axis(us_volume, idx_prev, axis),
                get_slice_along_axis(us_volume, i, axis),
                get_slice_along_axis(us_volume, idx_next, axis),
            ], axis=-1)
            pred_hw = _triplanar_predict_slice(generator, sl_stack, 3, output_channels)
            _triplanar_write_slice(pred_vol, axis, i, pred_hw, output_channels)
        predictions.append(pred_vol)
    return np.clip(np.mean(predictions, axis=0), -1, 1)


def triplanar_inference_multitask(generator, us_volume, is_25d=False):
    """Predict from 3 axes for multi-task model (2-channel output: T2+FLAIR)."""
    if is_25d:
        result = triplanar_inference_25d(generator, us_volume, output_channels=2)
    else:
        result = triplanar_inference_2d(generator, us_volume, output_channels=2)
    if result.ndim == 4:
        return result[..., 0], result[..., 1]
    return result, result


# =============================================================================
# 11. VISUALIZATION
# =============================================================================
def save_comparison_figure(us_vol, pred_vol, target_vol, save_path, title=""):
    """Save 5-column comparison figure with axial, coronal, sagittal views.
    For 192x192xN volumes: axial=vol[:,:,mid_d], coronal=vol[:,mid_h,:], sagittal=vol[mid_w,:,:].
    """
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    H, W, D = us_vol.shape[:3]
    mid_d = D // 2   # axial (depth axis)
    mid_h = H // 2   # coronal
    mid_w = W // 2   # sagittal

    slices_data = [
        ("Axial", us_vol[:, :, mid_d], pred_vol[:, :, mid_d], target_vol[:, :, mid_d]),
        ("Coronal", us_vol[:, mid_h, :], pred_vol[:, mid_h, :], target_vol[:, mid_h, :]),
        ("Sagittal", us_vol[mid_w, :, :], pred_vol[mid_w, :, :],
         target_vol[mid_w, :, :]),
    ]

    col_titles = ["US Input", "Prediction", "Target MRI", "|Difference|",
                   "Pred vs Target"]

    for row, (view_name, us_sl, pred_sl, tar_sl) in enumerate(slices_data):
        diff = np.abs(pred_sl - tar_sl)
        overlay = np.stack([
            np.clip((tar_sl + 1) / 2, 0, 1),
            np.clip((pred_sl + 1) / 2, 0, 1),
            np.clip((tar_sl + 1) / 2, 0, 1),
        ], axis=-1)

        images = [us_sl, pred_sl, tar_sl, diff, overlay]
        cmaps = ["gray", "gray", "gray", "hot", None]

        for col, (img, cmap) in enumerate(zip(images, cmaps)):
            ax = axes[row, col]
            if cmap is not None:
                vmin = -1.0 if cmap == "gray" else 0.0
                vmax = 1.0 if cmap == "gray" else 2.0
                ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            else:
                ax.imshow(np.clip(img, 0, 1), aspect="auto")
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=12)
            if col == 0:
                ax.set_ylabel(view_name, fontsize=12, rotation=90, labelpad=10)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_training_slice_comparison(us_batch, pred_batch, target_batch,
                                   save_path, step):
    """Save quick comparison of training slices."""
    n = min(4, us_batch.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for i in range(n):
        us_ch = 0 if us_batch.shape[-1] == 1 else 1
        axes[i, 0].imshow(us_batch[i, :, :, us_ch], cmap="gray", vmin=-1, vmax=1)
        axes[i, 0].set_title("US" if i == 0 else "")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(pred_batch[i, :, :, 0], cmap="gray", vmin=-1, vmax=1)
        axes[i, 1].set_title("Prediction" if i == 0 else "")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(target_batch[i, :, :, 0], cmap="gray", vmin=-1, vmax=1)
        axes[i, 2].set_title("Target" if i == 0 else "")
        axes[i, 2].axis("off")
    plt.suptitle(f"Step {step}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 12. PROGRESS TRACKING
# =============================================================================
def log_progress(exp_name, status, metrics=None, base_dir=""):
    """Append to progress.txt and update current_experiment.txt"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(base_dir, "current_experiment.txt"), "w") as f:
        f.write(f"{exp_name} | {status} | {ts}\n")
    with open(os.path.join(base_dir, "progress.txt"), "a") as f:
        line = f"[{ts}] {exp_name}: {status}"
        if metrics:
            line += f" | SSIM={metrics.get('ssim', 0):.4f} PSNR={metrics.get('psnr', 0):.2f}"
        f.write(line + "\n")


def append_results_csv(exp_name, metrics_list, base_dir=""):
    """Append per-subject results to live CSV.
    Supports both single-target (T2) and multi-target (T2+FLAIR) experiments.
    For multi-target: metrics have ssim_t2, psnr_t2, mae_t2, ssim_flair, psnr_flair, mae_flair.
    For single-target: metrics have ssim, psnr, mae (flair columns are NaN).
    """
    csv_path = os.path.join(base_dir, "results_live.csv")
    header = not os.path.exists(csv_path)
    with open(csv_path, "a") as f:
        if header:
            f.write("experiment,subject,ssim_t2,psnr_t2,mae_t2,ssim_flair,psnr_flair,mae_flair\n")
        for m in metrics_list:
            # Multi-target results have separate keys
            if "ssim_t2" in m:
                f.write(f"{exp_name},{m['subject']},"
                        f"{m['ssim_t2']:.4f},{m['psnr_t2']:.2f},{m['mae_t2']:.4f},"
                        f"{m['ssim_flair']:.4f},{m['psnr_flair']:.2f},{m['mae_flair']:.4f}\n")
            else:
                # Single target -- ssim/psnr/mae are T2, flair columns are NaN
                f.write(f"{exp_name},{m['subject']},"
                        f"{m['ssim']:.4f},{m['psnr']:.2f},{m['mae']:.4f},"
                        f"NaN,NaN,NaN\n")


# =============================================================================
# 13. TF.DATA PIPELINES
# =============================================================================
def make_2d_dataset(slices_list, batch_size, is_training=True,
                    include_flair=False, is_25d=False):
    """Create tf.data.Dataset from slice list."""
    input_ch = 3 if is_25d else 1

    if include_flair:
        def gen():
            indices = np.arange(len(slices_list))
            while True:
                if is_training:
                    np.random.shuffle(indices)
                for idx in indices:
                    us_sl, t2_sl, flair_sl = slices_list[idx]
                    if is_training:
                        us_aug = us_sl.copy()
                        t2_aug = t2_sl.copy()
                        flair_aug = flair_sl.copy()
                        if not is_25d:
                            us_2d = us_aug[:, :, 0]
                            us_2d, t2_2d, flair_2d = augment_2d_pair(
                                us_2d, t2_aug[:, :, 0], flair_aug[:, :, 0])
                            us_aug = us_2d[..., np.newaxis]
                            t2_aug = t2_2d[..., np.newaxis]
                            flair_aug = flair_2d[..., np.newaxis]
                        else:
                            us_aug, t2_aug, flair_aug = augment_2d_pair(
                                us_aug, t2_aug, flair_aug)
                        yield (us_aug.astype(np.float32),
                               t2_aug.astype(np.float32),
                               flair_aug.astype(np.float32))
                    else:
                        yield (us_sl.astype(np.float32),
                               t2_sl.astype(np.float32),
                               flair_sl.astype(np.float32))
                if not is_training:
                    break

        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(192, 192, input_ch), dtype=tf.float32),
                tf.TensorSpec(shape=(192, 192, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(192, 192, 1), dtype=tf.float32),
            ))
    else:
        def gen():
            indices = np.arange(len(slices_list))
            while True:
                if is_training:
                    np.random.shuffle(indices)
                for idx in indices:
                    us_sl, t2_sl = slices_list[idx][:2]
                    if is_training:
                        us_aug = us_sl.copy()
                        t2_aug = t2_sl.copy()
                        if not is_25d:
                            us_2d = us_aug[:, :, 0]
                            us_2d, t2_2d = augment_2d_pair(us_2d, t2_aug[:, :, 0])
                            us_aug = us_2d[..., np.newaxis]
                            t2_aug = t2_2d[..., np.newaxis]
                        else:
                            us_aug, t2_aug = augment_2d_pair(us_aug, t2_aug)
                        yield (us_aug.astype(np.float32),
                               t2_aug.astype(np.float32))
                    else:
                        yield (us_sl.astype(np.float32),
                               t2_sl.astype(np.float32))
                if not is_training:
                    break

        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(192, 192, input_ch), dtype=tf.float32),
                tf.TensorSpec(shape=(192, 192, 1), dtype=tf.float32),
            ))

    if is_training:
        ds = ds.shuffle(256)
    return ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


def _pad_volume_for_patch(volume, patch_size):
    """Pad volume with reflect so each of the first 3 dims >= patch_size.
    Extra (channel) dims are left unpadded. Works for 3D and 4D volumes.
    """
    pads = []
    for s, p in zip(volume.shape[:3], patch_size):
        if s < p:
            pads.append((0, p - s))
        else:
            pads.append((0, 0))
    while len(pads) < volume.ndim:
        pads.append((0, 0))
    if any(pad != (0, 0) for pad in pads):
        volume = np.pad(volume, pads, mode='reflect')
    return volume


def extract_random_patch_3d(vol_in, vol_target, patch_size=(64, 64, 32)):
    """Extract a random 3D patch from paired volumes.
    Handles both 3D (H,W,D) and 4D (H,W,D,C) volumes; preserves the channel
    dim when present. Pads along the first 3 dims with reflect if needed.
    """
    vol_in = _pad_volume_for_patch(vol_in, patch_size)
    vol_target = _pad_volume_for_patch(vol_target, patch_size)
    max_start = [s - p for s, p in zip(vol_in.shape[:3], patch_size)]
    start = [np.random.randint(0, m + 1) for m in max_start]
    slices = tuple(slice(s, s + p) for s, p in zip(start, patch_size))
    if vol_in.ndim == 4:
        slices = slices + (slice(None),)
    tar_slices = slices if vol_target.ndim == vol_in.ndim else \
        (tuple(slice(s, s + p) for s, p in zip(start, patch_size))
         + ((slice(None),) if vol_target.ndim == 4 else ()))
    return vol_in[slices], vol_target[tar_slices]


def make_3d_refiner_dataset(pred_volumes, target_volumes, batch_size=1,
                             patch_size=(64, 64, 32)):
    """Create tf.data.Dataset for 3D refiner training.
    Accepts both 3D and 4D volumes transparently. Channel count is inferred
    from the first volume; yielded patches have shape (*patch_size, in_ch)
    and (*patch_size, out_ch) with no extra singleton dims.
    """
    def _ensure_4d(v):
        return v if v.ndim == 4 else v[..., np.newaxis]

    preds4 = [_ensure_4d(v) for v in pred_volumes]
    targets4 = [_ensure_4d(v) for v in target_volumes]
    in_ch = int(preds4[0].shape[-1])
    out_ch = int(targets4[0].shape[-1])

    def gen():
        indices = np.arange(len(preds4))
        while True:
            np.random.shuffle(indices)
            for idx in indices:
                inp, tar = extract_random_patch_3d(
                    preds4[idx], targets4[idx], patch_size)
                for ax in range(3):
                    if np.random.rand() < 0.5:
                        inp = np.flip(inp, axis=ax)
                        tar = np.flip(tar, axis=ax)
                yield (inp.copy().astype(np.float32),
                       tar.copy().astype(np.float32))

    ps = patch_size
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(*ps, in_ch), dtype=tf.float32),
            tf.TensorSpec(shape=(*ps, out_ch), dtype=tf.float32),
        ))
    return ds.shuffle(32).batch(batch_size, drop_remainder=True).prefetch(
        tf.data.AUTOTUNE)


# =============================================================================
# 14. PatchNCE utilities (2D)
# =============================================================================
class PatchNCEMLP(tf.keras.layers.Layer):
    """Per-layer MLP projection head for contrastive embeddings."""
    def __init__(self, out_dim=128, **kw):
        super().__init__(**kw)
        self.out_dim = out_dim

    def build(self, ishape):
        ch = int(ishape[-1])
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(ch, activation="relu"),
            tf.keras.layers.Dense(self.out_dim),
        ])
        super().build(ishape)

    def call(self, x):
        return tf.math.l2_normalize(self.mlp(x), axis=-1)


class PatchNCELoss(tf.keras.layers.Layer):
    """InfoNCE: query patches from G(x) match keys from x at same location."""
    def __init__(self, num_patches=256, temperature=0.07, **kw):
        super().__init__(**kw)
        self.np = num_patches
        self.t = temperature

    def call(self, query, key):
        B = tf.shape(query)[0]
        N = tf.shape(query)[1]
        logits = tf.matmul(query, key, transpose_b=True) / self.t
        labels = tf.tile(tf.expand_dims(tf.range(N), 0), [B, 1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=tf.cast(logits, tf.float32))
        return tf.reduce_mean(loss)


def sample_patches_2d(feats, n=256):
    """Sample n random spatial locations from 2D feature maps."""
    B = tf.shape(feats)[0]
    H = feats.shape[1]
    W = feats.shape[2]
    C = feats.shape[-1]
    flat = tf.reshape(feats, [B, H * W, C])
    idx = tf.random.shuffle(tf.range(H * W))[:n]
    return tf.gather(flat, idx, axis=1), idx


def gather_at_2d(feats, idx):
    """Gather features at specific spatial indices."""
    B = tf.shape(feats)[0]
    H = feats.shape[1]
    W = feats.shape[2]
    C = feats.shape[-1]
    flat = tf.reshape(feats, [B, H * W, C])
    return tf.gather(flat, idx, axis=1)


def sample_patches_3d(feats, n=128):
    """Sample n random spatial locations from 3D feature maps."""
    s = tf.shape(feats)
    B = s[0]
    N = s[1] * s[2] * s[3]
    C = feats.shape[-1]
    flat = tf.reshape(feats, [B, N, C])
    idx = tf.random.shuffle(tf.range(N))[:n]
    return tf.gather(flat, idx, axis=1), idx


def gather_at_3d(feats, idx):
    s = tf.shape(feats)
    B = s[0]
    C = feats.shape[-1]
    flat = tf.reshape(feats, [B, s[1] * s[2] * s[3], C])
    return tf.gather(flat, idx, axis=1)
