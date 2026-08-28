"""
ResViT Ablation v2 (Windows) — based on resvit_ablation_v2.py
================================================
Windows variant for the RTX 3090 rig, created after the Linux NVMe failure.

Differences vs resvit_ablation_v2.py:
  - Default nnU-Net perceptual checkpoint path points to E:/HSA_SEGMENTATION.
  - New --perc_mode flag (only affects ablation B with 2-channel targets such
    as t2_flair). Prior runs showed single-pass multimodal perceptual loss
    degraded BOTH t2 and FLAIR LPIPS vs the single-target ablB-t2 baseline.
    Modes:
      single_pass  : legacy behaviour (default). Keeps t2-only runs identical.
      split        : one encoder forward per target channel (each channel
                     zeroed in the other slots). Isolates gradients per
                     modality; mirrors the working ablB-t2 recipe twice.
      split_ctx    : two forwards where the "other" channel slot is filled
                     with the detached GT (multimodal context). Gradients
                     flow only through the channel being supervised, but the
                     encoder sees in-distribution 4-channel input.
      late_only    : single-pass multimodal but uses encoder stages (3,4)
                     instead of (0,1,2). Tests whether early-stage channel
                     mixing is responsible for the cross-channel degradation.

  - Per-channel perceptual loss is logged (P_t2, P_fl) whenever perc_mode
    is split / split_ctx, to diagnose dominance between channels.

  - Default tag set to `win-<perc_mode>` when perc_mode != single_pass, so
    outputs land in e.g. output/ablation/ResViT-2.5d-t2_flair-ablB-win-split/.

Usage (Windows, mmhvae env):
  python resvit_ablation_v2_win.py --variant 2.5d --target t2_flair --ablation B --perc_mode split
  python resvit_ablation_v2_win.py --variant 2.5d --target t2_flair --ablation B --perc_mode split_ctx
  python resvit_ablation_v2_win.py --variant 2.5d --target t2_flair --ablation B --perc_mode late_only
"""

import matplotlib
matplotlib.use('Agg')

import faulthandler
faulthandler.enable()

import os, sys, math, time, glob, json, random, argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from scipy.ndimage import zoom, rotate, uniform_filter
from matplotlib import pyplot as plt


# ============================================================================
# 0. GPU PROFILES
# ============================================================================
def get_gpu_vram_gb():
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


def get_gpu_profile():
    vram = get_gpu_vram_gb()
    print(f"[GPU] {vram:.1f} GB VRAM detected")
    if vram >= 20:
        return dict(label="24GB", ngf2d=64, bs2d=16, imgsz=256,
                    ngf3d=48, bs3d=1, ps3d=(96,96,96), ov3d=(24,24,24),
                    acc3d=2, ckpt3d=False, ndf2d=64, ndf3d=48)
    elif vram >= 14:
        return dict(label="16GB", ngf2d=64, bs2d=8, imgsz=256,
                    ngf3d=32, bs3d=1, ps3d=(64,64,64), ov3d=(16,16,16),
                    acc3d=4, ckpt3d=True, ndf2d=64, ndf3d=32)
    elif vram >= 8:
        return dict(label="8-12GB", ngf2d=48, bs2d=4, imgsz=224,
                    ngf3d=24, bs3d=1, ps3d=(48,48,48), ov3d=(12,12,12),
                    acc3d=8, ckpt3d=True, ndf2d=48, ndf3d=24)
    else:
        return dict(label="CPU/small", ngf2d=32, bs2d=1, imgsz=128,
                    ngf3d=16, bs3d=1, ps3d=(32,32,32), ov3d=(8,8,8),
                    acc3d=8, ckpt3d=True, ndf2d=32, ndf3d=16)


# ============================================================================
# 0b. CONFIGURATION
# ============================================================================
class Config:
    seed = 42

    # --- Paths ---
    _script_dir     = os.path.dirname(os.path.abspath(__file__))
    base_data_dir   = os.path.join(_script_dir, "dataset-registration-corrected-cropped")
    us_dir          = os.path.join(base_data_dir, "US")
    t2_dir          = os.path.join(base_data_dir, "MR-T2")
    fl_dir          = os.path.join(base_data_dir, "MR-FLAIR")
    split_json_path = os.path.join(_script_dir, "subject_split.json")
    output_root     = os.path.join(_script_dir, "output")

    # --- Architecture ---
    arch_variant = "2.5d"        # 2d | 2.5d | 2d_3d_refine | full_3d
    target_mode  = "t2"          # t2 | flair | t2_flair
    context_slices = 3           # for 2.5d

    # --- Inference ---
    # Triplanar is WRONG for anisotropic US. Only enable for isotropic data.
    triplanar = False
    triplanar_weights = (0.50, 0.25, 0.25)

    # --- Split ---
    # Validation is carved from training subjects (by subject, no leakage)
    val_fraction = 0.15          # fraction of train subjects used for validation

    # --- Training defaults (overridden by GPU profile) ---
    batch_size = 16
    batch_size_3d = 1
    accum_steps = 1
    accum_steps_3d = 2
    num_workers = 4          # Linux is stable with higher worker counts
    persistent_workers = True
    prefetch_factor = 2
    use_amp = True
    use_bf16 = True          # bf16 on Ampere+ — no GradScaler needed
    channels_last = False    # keep disabled: ReflectionPad2d can be flaky with channels_last
    use_compile = False      # keep disabled: Swin dynamic shapes cause recompiles
    image_size = 256
    patches_per_volume = 8

    phase1_epochs = 100;  phase1_lr_g = 2e-4;  phase1_lr_d = 2e-4
    phase2_epochs = 100;  phase2_lr_g = 1e-4;  phase2_lr_d = 1e-4
    refine_epochs = 30;   refine_lr   = 1e-4

    # --- Generator (faithful to paper) ---
    ngf = 64
    n_art_blocks = 9
    transformer_positions = [4, 5]   # which bottleneck blocks get Swin branch
    n_heads = 8
    swin_layer_pairs = 2             # pairs of (W-MSA, SW-MSA) per transformer
    window_size_2d = 8
    window_size_3d = 4
    mlp_ratio = 4.0
    dropout = 0.0

    # --- Discriminator ---
    ndf = 64;  n_layers_d = 3

    # --- Loss ---
    lambda_l1 = 100.0;  lambda_adv = 1.0;  lambda_fm = 10.0

    # --- Ablation flags (cascade) ---
    ablation = "A"               # A | B | C | D | E  (cascading)

    # B: perceptual via frozen nnU-Net encoder
    use_perceptual      = False
    nnunet_ckpt         = r"E:/HSA_SEGMENTATION/my_nnunet/nnUNet_results/Dataset016_RH-GlioSeg_v3/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"
    nnunet_channels     = {"flair": 0, "t2": 3}
    perceptual_layers   = (0, 1, 2)
    lambda_perceptual   = 10.0
    perc_mode           = "single_pass"   # single_pass | split | split_ctx | late_only

    # C: frequency loss (Laplacian pyramid L1, foreground-masked)
    use_frequency       = False
    lambda_frequency    = 5.0
    freq_pyramid_levels = 3

    # D: hierarchical stochastic skips (MMHVAE-style conditional prior)
    use_hierarchical    = False
    lambda_kl           = 0.001
    kl_warmup_epochs    = 20
    infer_samples       = 1          # N>1 averages N stochastic samples at test time
    train_temperature   = 1.0        # stochastic at train
    infer_temperature   = 0.0        # deterministic at infer

    # E: seg guidance (skipped until masks arrive)
    use_seg_guide       = False
    seg_weight_tumor    = 4.0
    seg_weight_cavity   = 2.0
    seg_weight_bg       = 1.0

    # --- Normalization ---
    fg_thr = 0.01;  us_plo = 2;  us_phi = 98;  mri_clip = 3.0

    # --- Augmentation ---
    aug_flip = 0.5;  aug_rot = 0.3;  aug_int = 0.3

    # --- 3D ---
    patch_size_3d = (96,96,96);  patch_overlap_3d = (24,24,24);  grad_ckpt_3d = False

    eval_every = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def apply_gpu_profile(self):
        p = get_gpu_profile()
        print(f"[GPU] Profile: {p['label']}")
        is3d = self.arch_variant == "full_3d"
        if is3d:
            self.ngf = p["ngf3d"]; self.ndf = p["ndf3d"]
            self.batch_size_3d = p["bs3d"]
            self.patch_size_3d = p["ps3d"]; self.patch_overlap_3d = p["ov3d"]
            self.accum_steps_3d = p["acc3d"]; self.grad_ckpt_3d = p["ckpt3d"]
        else:
            self.ngf = p["ngf2d"]; self.ndf = p["ndf2d"]
            self.batch_size = p["bs2d"]; self.image_size = p["imgsz"]
            self.accum_steps = max(1, 4 // p["bs2d"])
        # Summary
        bn = self.ngf * 4
        if is3d:
            r = tuple(x//4 for x in self.patch_size_3d)
            mb = bn * r[0]*r[1]*r[2] * 4 / 1e6
            print(f"  3D: {bn}ch@{r}, ~{mb:.0f}MB/map, ckpt={'ON' if self.grad_ckpt_3d else 'OFF'}")
        else:
            r = self.image_size // 4
            mb = bn * r * r * 4 / 1e6
            print(f"  2D: {bn}ch@{r}², batch={self.batch_size}, accum={self.accum_steps}")

    @property
    def out_ch(self):
        return 2 if self.target_mode == "t2_flair" else 1

    @property
    def in_ch(self):
        return self.context_slices if self.arch_variant == "2.5d" else 1

    @property
    def run_name(self):
        tag = getattr(self, "run_tag", "")
        return f"ResViT-{self.arch_variant}-{self.target_mode}{tag}"

    def _dir(self, sub):
        # v2: everything goes under output/ablation/{run_name}/
        d = os.path.join(self.output_root, "ablation", self.run_name, sub)
        os.makedirs(d, exist_ok=True); return d

    @property
    def ckpt_dir(self): return self._dir("checkpoints")
    @property
    def log_dir(self): return self._dir("logs")
    @property
    def pred_dir(self): return self._dir("predictions")


cfg = Config()


# ============================================================================
# 1. UTILITIES
# ============================================================================
def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = False; torch.backends.cudnn.benchmark = True

def set_grad(m, on):
    for p in m.parameters(): p.requires_grad = on

def load_nii(p):
    img = nib.load(p); return img.get_fdata().astype(np.float32), img.affine

def save_nii(v, p, aff=np.eye(4)):
    nib.save(nib.Nifti1Image(v.astype(np.float32), aff), p)

def fg_mask(v, t=0.01):
    mx = float(np.max(v)); return v > mx * t if mx > 0 else np.zeros_like(v, dtype=bool)

def norm_pct(v, m, lo=2, hi=98):
    fg = v[m]
    if fg.size == 0: return np.full_like(v, -1., dtype=np.float32)
    a, b = np.percentile(fg, lo), np.percentile(fg, hi)
    o = np.full_like(v, -1., dtype=np.float32)
    if b - a > 1e-8:
        o = np.clip((np.clip(v, a, b) - a) / (b - a), 0, 1) * 2 - 1
        o[~m] = -1.
    return o.astype(np.float32)

def norm_z(v, m, c=3.):
    fg = v[m]
    if fg.size == 0: return np.full_like(v, -1., dtype=np.float32)
    mu, s = np.mean(fg), np.std(fg)
    o = np.full_like(v, -1., dtype=np.float32)
    if s > 1e-8:
        o = np.clip((v - mu) / s, -c, c) / c; o[~m] = -1.
    return o.astype(np.float32)

def rsz(img, sz):
    if img.shape[0] == sz and img.shape[1] == sz: return img.astype(np.float32)
    return zoom(img, [sz/img.shape[0], sz/img.shape[1]], order=1).astype(np.float32)

def npar(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def hanning3d(ps):
    w = np.maximum(np.hanning(ps[0]),1e-3)[:,None,None] * \
        np.maximum(np.hanning(ps[1]),1e-3)[None,:,None] * \
        np.maximum(np.hanning(ps[2]),1e-3)[None,None,:]
    return (w / w.max()).astype(np.float32)

def slide_starts(sz, p, step):
    if sz <= p: return [0]
    s = list(range(0, sz - p + 1, step))
    if s[-1] != sz - p: s.append(sz - p)
    return s

def pad_vol(v, mins, pv):
    orig_shape = v.shape
    pads = [(max(m-s,0)//2, max(m-s,0)-max(m-s,0)//2) for s, m in zip(v.shape, mins)]
    if any(a+b > 0 for a, b in pads): v = np.pad(v, pads, "constant", constant_values=pv)
    crop = tuple(slice(a, a+s) for (a,_), s in zip(pads, orig_shape))
    return v, crop


# ============================================================================
# 2. SUBJECT SPLIT + DATA LOADING
# ============================================================================
def load_split(json_path):
    """
    Load the fixed subject split from JSON.
    Returns (train_names, test_names) — lists of volume names like "ReMIND-002-post".
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data["train"], data["test"]


def split_train_val(train_names, val_fraction, seed=42):
    """
    Carve validation from training BY SUBJECT to avoid data leakage.
    e.g., if ReMIND-002-pre is in val, ReMIND-002-post must also be in val.
    """
    # Extract unique subject IDs (e.g., "ReMIND-002" from "ReMIND-002-post")
    subj_to_vols = {}
    for name in train_names:
        # "ReMIND-XXX-pre/post" → subject = "ReMIND-XXX"
        parts = name.rsplit("-", 1)  # ["ReMIND-XXX", "pre"] or ["ReMIND-XXX", "post"]
        subj = parts[0]
        subj_to_vols.setdefault(subj, []).append(name)

    subjects = sorted(subj_to_vols.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    n_val = max(1, int(len(subjects) * val_fraction))
    val_subjects = subjects[:n_val]
    train_subjects = subjects[n_val:]

    val_names = [v for s in val_subjects for v in subj_to_vols[s]]
    tr_names = [v for s in train_subjects for v in subj_to_vols[s]]

    print(f"  Split: {len(train_subjects)} train subj ({len(tr_names)} vols) | "
          f"{len(val_subjects)} val subj ({len(val_names)} vols)")
    return tr_names, val_names


def load_volume(name, cfg_obj):
    """
    Load a single volume with flexible target availability.
    Missing modalities → filled with -1.0, marked in target_mask.
    """
    us_path = os.path.join(cfg_obj.us_dir, f"{name}-us.nii.gz")
    if not os.path.exists(us_path):
        return None

    us, aff = load_nii(us_path)
    uf = fg_mask(us, cfg_obj.fg_thr)
    us = norm_pct(us, uf, cfg_obj.us_plo, cfg_obj.us_phi)

    # T2 — try both -mri.nii.gz and -mr.nii.gz (dataset has mixed naming)
    t2, has_t2 = None, False
    for suf in [f"{name}-mri.nii.gz", f"{name}-mr.nii.gz"]:
        p = os.path.join(cfg_obj.t2_dir, suf)
        if os.path.exists(p):
            raw, _ = load_nii(p)
            t2 = norm_z(raw, fg_mask(raw, cfg_obj.fg_thr), cfg_obj.mri_clip)
            has_t2 = True; break

    # FLAIR — try multiple suffixes (dataset has mixed naming)
    fl, has_fl = None, False
    for suf in [f"{name}-mri.nii.gz", f"{name}-mr.nii.gz", f"{name}-flair.nii.gz"]:
        p = os.path.join(cfg_obj.fl_dir, suf)
        if os.path.exists(p):
            raw, _ = load_nii(p)
            fl = norm_z(raw, fg_mask(raw, cfg_obj.fg_thr), cfg_obj.mri_clip)
            has_fl = True; break

    # Check requirements
    tm = cfg_obj.target_mode
    if tm == "t2" and not has_t2: return None
    if tm == "flair" and not has_fl: return None
    if tm == "t2_flair" and not has_t2 and not has_fl: return None

    # Fill missing with -1
    sh = us.shape
    if t2 is None: t2 = np.full(sh, -1., dtype=np.float32)
    if fl is None: fl = np.full(sh, -1., dtype=np.float32)

    # FG union from available
    fgu = us > -0.95
    if has_t2: fgu |= t2 > -0.95
    if has_fl: fgu |= fl > -0.95

    # Target mask
    if tm == "t2_flair":
        tmask = np.array([float(has_t2), float(has_fl)], dtype=np.float32)
    elif tm == "t2":
        tmask = np.array([1.0], dtype=np.float32)
    else:
        tmask = np.array([1.0], dtype=np.float32)

    return dict(name=name, us=us, t2=t2, flair=fl, fg=fgu, aff=aff,
                has_t2=has_t2, has_fl=has_fl, tmask=tmask)


def load_samples(names, cfg_obj, label=""):
    """Load all volumes for a list of names."""
    samples = []
    for name in names:
        s = load_volume(name, cfg_obj)
        if s is None: continue
        if np.sum(s["fg"]) < 100: continue
        samples.append(s)
        t = []
        if s["has_t2"]: t.append("T2")
        if s["has_fl"]: t.append("FL")
        print(f"  [{label}] {name} | {s['us'].shape} | {'+'.join(t) or 'NONE'}")
    print(f"  [{label}] {len(samples)} volumes loaded")
    return samples


# ============================================================================
# 3. AUGMENTATION
# ============================================================================
def aug2d(inp, tgts):
    if np.random.rand() < cfg.aug_flip:
        inp = np.flip(inp, -1).copy(); tgts = [np.flip(t, -1).copy() for t in tgts]
    if np.random.rand() < cfg.aug_flip:
        inp = np.flip(inp, -2).copy(); tgts = [np.flip(t, -2).copy() for t in tgts]
    if np.random.rand() < cfg.aug_rot:
        a = np.random.uniform(-15, 15)
        inp = rotate(inp, a, axes=(-2,-1), reshape=False, order=1, mode="nearest").astype(np.float32)
        tgts = [rotate(t, a, axes=(-2,-1), reshape=False, order=1, mode="nearest").astype(np.float32) for t in tgts]
    if np.random.rand() < cfg.aug_int:
        g = np.random.uniform(0.8, 1.2)
        inp = np.power(np.clip((inp+1)/2, 0, 1), g)*2 - 1
        inp = np.clip(inp + np.random.normal(0, 0.02, inp.shape).astype(np.float32), -1, 1)
    return inp.astype(np.float32), tgts


# ============================================================================
# 4. DATASETS
# ============================================================================
def _get_targets(s, tm):
    """Return list of target arrays based on target_mode."""
    if tm == "t2":      return [s["t2"]]
    if tm == "flair":   return [s["flair"]]
    return [s["t2"], s["flair"]]  # t2_flair


class DS2D(Dataset):
    def __init__(self, samples, sz, tm, train=True, min_fg=0.05):
        self.S, self.sz, self.tm, self.tr = samples, sz, tm, train
        self.idx = []
        for vi, s in enumerate(samples):
            for si in range(s["us"].shape[2]):
                if np.mean(s["fg"][:,:,si]) >= min_fg:
                    self.idx.append((vi, si))
        print(f"    {'Tr' if train else 'Va'} 2D: {len(self.idx)} slices")

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        vi, si = self.idx[i]; s = self.S[vi]
        us = rsz(s["us"][:,:,si], self.sz)
        tgts = [rsz(t[:,:,si], self.sz) for t in _get_targets(s, self.tm)]
        if self.tr: us, tgts = aug2d(us, tgts)
        return (torch.from_numpy(us[None].copy()).float(),
                torch.from_numpy(np.stack(tgts, 0).copy()).float(),
                torch.from_numpy(s["tmask"].copy()).float())


class DS25D(Dataset):
    def __init__(self, samples, sz, tm, ctx=3, train=True, min_fg=0.05):
        self.S, self.sz, self.tm, self.tr = samples, sz, tm, train
        self.h = ctx // 2; self.idx = []
        for vi, s in enumerate(samples):
            D = s["us"].shape[2]
            for si in range(self.h, D - self.h):
                if np.mean(s["fg"][:,:,si]) >= min_fg:
                    self.idx.append((vi, si))
        print(f"    {'Tr' if train else 'Va'} 2.5D: {len(self.idx)} slices")

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        vi, si = self.idx[i]; s = self.S[vi]
        us = np.stack([rsz(s["us"][:,:,si+o], self.sz) for o in range(-self.h, self.h+1)], 0)
        tgts = [rsz(t[:,:,si], self.sz) for t in _get_targets(s, self.tm)]
        if self.tr: us, tgts = aug2d(us, tgts)
        return (torch.from_numpy(us.copy()).float(),
                torch.from_numpy(np.stack(tgts, 0).copy()).float(),
                torch.from_numpy(s["tmask"].copy()).float())


class DS3D(Dataset):
    def __init__(self, samples, ps, tm, ppv=8, train=True):
        self.S, self.ps, self.tm, self.ppv, self.tr = samples, ps, tm, ppv, train
        print(f"    {'Tr' if train else 'Va'} 3D: {len(samples)*ppv} patches/epoch")

    def __len__(self): return len(self.S) * self.ppv

    def _pad(self, v, pv):
        for s, p in zip(v.shape, self.ps):
            if s < p:
                pads = [(max(p-s,0)//2, max(p-s,0)-max(p-s,0)//2) for s, p in zip(v.shape, self.ps)]
                return np.pad(v, pads, "constant", constant_values=pv)
        return v

    def __getitem__(self, i):
        s = self.S[i % len(self.S)]
        us = self._pad(s["us"], -1.)
        tgts = [self._pad(t, -1.) for t in _get_targets(s, self.tm)]
        st = [np.random.randint(0, max(us.shape[j]-self.ps[j],0)+1) for j in range(3)]
        sl = tuple(slice(st[j], st[j]+self.ps[j]) for j in range(3))
        up = us[sl].copy(); tp = [t[sl].copy() for t in tgts]
        if self.tr:
            for ax in range(3):
                if np.random.rand() < cfg.aug_flip:
                    up = np.flip(up, ax).copy(); tp = [np.flip(t, ax).copy() for t in tp]
        return (torch.from_numpy(up[None]).float(),
                torch.from_numpy(np.stack(tp)).float(),
                torch.from_numpy(s["tmask"].copy()).float())


# ============================================================================
# 5. SWIN TRANSFORMER — 2D (faithful shifted windows)
# ============================================================================
def wp2d(x, ws):
    B,H,W,C = x.shape
    return x.view(B,H//ws,ws,W//ws,ws,C).permute(0,1,3,2,4,5).contiguous().view(-1,ws,ws,C)

def wr2d(w, ws, H, W):
    B = int(w.shape[0]/(H*W/ws/ws))
    return w.view(B,H//ws,W//ws,ws,ws,-1).permute(0,1,3,2,4,5).contiguous().view(B,H,W,-1)

def amask2d(H, W, ws, ss):
    if ss == 0: return None
    m = torch.zeros(1,H,W,1); cnt = 0
    for hs in [slice(0,-ws), slice(-ws,-ss), slice(-ss,None)]:
        for ws_ in [slice(0,-ws), slice(-ws,-ss), slice(-ss,None)]:
            m[:,hs,ws_,:] = cnt; cnt += 1
    mw = wp2d(m, ws).view(-1, ws*ws)
    am = mw.unsqueeze(1) - mw.unsqueeze(2)
    return am.masked_fill(am != 0, -100.).masked_fill(am == 0, 0.)

class WA2D(nn.Module):
    def __init__(self, d, ws, nh, dr=0.):
        super().__init__()
        self.nh, self.sc, self.ws = nh, (d//nh)**-.5, ws
        self.rpb = nn.Parameter(torch.zeros((2*ws-1)*(2*ws-1), nh))
        nn.init.trunc_normal_(self.rpb, std=.02)
        c = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing='ij'))
        cf = c.flatten(1); rc = (cf[:,:,None]-cf[:,None,:]).permute(1,2,0).contiguous()
        rc[:,:,0] += ws-1; rc[:,:,1] += ws-1; rc[:,:,0] *= 2*ws-1
        self.register_buffer("rpi", rc.sum(-1))
        self.qkv = nn.Linear(d, d*3); self.proj = nn.Linear(d, d); self.drop = nn.Dropout(dr)

    def forward(self, x, mask=None):
        B,N,C = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.nh,C//self.nh).permute(2,0,3,1,4)
        q,k,v = qkv.unbind(0)
        a = (q*self.sc) @ k.transpose(-2,-1)
        a = a + self.rpb[self.rpi.view(-1)].view(N,N,-1).permute(2,0,1).unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            a = a.view(B//nW, nW, self.nh, N, N) + mask.unsqueeze(1).unsqueeze(0)
            a = a.view(-1, self.nh, N, N)
        a = self.drop(a.softmax(-1))
        out = (a @ v).transpose(1,2).reshape(B, N, C)
        return self.proj(out)

class SB2D(nn.Module):
    def __init__(self, d, nh, ws, ss=0, mr=4., dr=0.):
        super().__init__()
        self.ss, self.ws = ss, ws
        self.n1 = nn.LayerNorm(d); self.attn = WA2D(d, ws, nh, dr)
        self.n2 = nn.LayerNorm(d)
        h = int(d*mr)
        self.mlp = nn.Sequential(nn.Linear(d,h), nn.GELU(), nn.Dropout(dr), nn.Linear(h,d), nn.Dropout(dr))

    def forward(self, x, H, W, mask=None):
        sc = x; x = self.n1(x).view(-1, H, W, x.shape[-1])
        if self.ss > 0: x = torch.roll(x, (-self.ss,-self.ss), (1,2))
        xw = wp2d(x, self.ws).view(-1, self.ws**2, x.shape[-1])
        xw = self.attn(xw, mask)
        x = wr2d(xw.view(-1, self.ws, self.ws, xw.shape[-1]), self.ws, H, W)
        if self.ss > 0: x = torch.roll(x, (self.ss, self.ss), (1,2))
        x = sc + x.reshape(sc.shape)
        return x + self.mlp(self.n2(x))

class Swin2D(nn.Module):
    def __init__(self, ch, nh, np_, ws, mr=4., dr=0.):
        super().__init__()
        self.ws = ws; self.layers = nn.ModuleList()
        for _ in range(np_):
            self.layers.append(SB2D(ch, nh, ws, 0, mr, dr))
            self.layers.append(SB2D(ch, nh, ws, ws//2, mr, dr))

    def forward(self, x):
        B,C,H,W = x.shape; ws = self.ws
        pH, pW = (ws-H%ws)%ws, (ws-W%ws)%ws
        if pH or pW: x = F.pad(x, (0,pW,0,pH))
        _,_,Hp,Wp = x.shape
        mask = amask2d(Hp, Wp, ws, ws//2)
        if mask is not None: mask = mask.to(x.device)
        x = x.permute(0,2,3,1).contiguous().view(B, Hp*Wp, C)
        for l in self.layers:
            x = l(x, Hp, Wp, mask if l.ss > 0 else None)
        x = x.view(B, Hp, Wp, C).permute(0,3,1,2).contiguous()
        return x[:,:,:H,:W] if pH or pW else x


# ============================================================================
# 6. SWIN TRANSFORMER — 3D
# ============================================================================
def wp3d(x, ws):
    B,D,H,W,C = x.shape
    return x.view(B,D//ws,ws,H//ws,ws,W//ws,ws,C).permute(0,1,3,5,2,4,6,7).contiguous().view(-1,ws**3,C)

def wr3d(w, ws, D, H, W):
    B = int(w.shape[0]/(D*H*W/ws**3))
    return w.view(B,D//ws,H//ws,W//ws,ws,ws,ws,-1).permute(0,1,4,2,5,3,6,7).contiguous().view(B,D,H,W,-1)

class WA3D(nn.Module):
    def __init__(self, d, ws, nh, dr=0.):
        super().__init__()
        self.nh, self.sc = nh, (d//nh)**-.5
        self.rpb = nn.Parameter(torch.zeros((2*ws-1)**3, nh))
        nn.init.trunc_normal_(self.rpb, std=.02)
        c = torch.stack(torch.meshgrid(torch.arange(ws),torch.arange(ws),torch.arange(ws), indexing='ij'))
        cf = c.flatten(1); rc = (cf[:,:,None]-cf[:,None,:]).permute(1,2,0).contiguous()
        rc[:,:,0]+=ws-1; rc[:,:,1]+=ws-1; rc[:,:,2]+=ws-1
        rc[:,:,0]*=(2*ws-1)**2; rc[:,:,1]*=2*ws-1
        self.register_buffer("rpi", rc.sum(-1))
        self.qkv = nn.Linear(d,d*3); self.proj = nn.Linear(d,d); self.drop = nn.Dropout(dr)

    def forward(self, x, mask=None):
        B,N,C = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.nh,C//self.nh).permute(2,0,3,1,4)
        q,k,v = qkv.unbind(0)
        a = (q*self.sc) @ k.transpose(-2,-1)
        a = a + self.rpb[self.rpi.view(-1)].view(N,N,-1).permute(2,0,1).unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            a = a.view(B//nW,nW,self.nh,N,N)+mask.unsqueeze(1).unsqueeze(0)
            a = a.view(-1,self.nh,N,N)
        a = self.drop(a.softmax(-1))
        return self.proj((a@v).transpose(1,2).reshape(B,N,C))

class SB3D(nn.Module):
    def __init__(self, d, nh, ws, ss=0, mr=4., dr=0.):
        super().__init__()
        self.ss, self.ws = ss, ws
        self.n1 = nn.LayerNorm(d); self.attn = WA3D(d, ws, nh, dr)
        self.n2 = nn.LayerNorm(d)
        h = int(d*mr)
        self.mlp = nn.Sequential(nn.Linear(d,h), nn.GELU(), nn.Dropout(dr), nn.Linear(h,d), nn.Dropout(dr))

    def forward(self, x, D, H, W):
        sc = x; x = self.n1(x).view(-1, D, H, W, x.shape[-1])
        if self.ss > 0: x = torch.roll(x, (-self.ss,)*3, (1,2,3))
        xw = wp3d(x, self.ws)
        xw = self.attn(xw)
        x = wr3d(xw, self.ws, D, H, W)
        if self.ss > 0: x = torch.roll(x, (self.ss,)*3, (1,2,3))
        x = sc + x.reshape(sc.shape)
        return x + self.mlp(self.n2(x))

class Swin3D(nn.Module):
    def __init__(self, ch, nh, np_, ws, mr=4., dr=0.):
        super().__init__()
        self.ws = ws; self.layers = nn.ModuleList()
        for _ in range(np_):
            self.layers.append(SB3D(ch, nh, ws, 0, mr, dr))
            self.layers.append(SB3D(ch, nh, ws, ws//2, mr, dr))

    def forward(self, x):
        B,C,D,H,W = x.shape; ws = self.ws
        pD,pH,pW = (ws-D%ws)%ws,(ws-H%ws)%ws,(ws-W%ws)%ws
        if pD or pH or pW: x = F.pad(x, (0,pW,0,pH,0,pD))
        _,_,Dp,Hp,Wp = x.shape
        x = x.permute(0,2,3,4,1).contiguous().view(B, Dp*Hp*Wp, C)
        for l in self.layers: x = l(x, Dp, Hp, Wp)
        x = x.view(B, Dp, Hp, Wp, C).permute(0,4,1,2,3).contiguous()
        return x[:,:,:D,:H,:W] if pD or pH or pW else x


# ============================================================================
# 7. ART BLOCKS — PARALLEL AGGREGATION
# ============================================================================
class RC2D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(c,c,3), nn.InstanceNorm2d(c), nn.ReLU(True),
                               nn.ReflectionPad2d(1), nn.Conv2d(c,c,3), nn.InstanceNorm2d(c))
    def forward(self, x): return x + self.b(x)

class RC3D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.Conv3d(c,c,3,padding=1), nn.InstanceNorm3d(c), nn.ReLU(True),
                               nn.Conv3d(c,c,3,padding=1), nn.InstanceNorm3d(c))
    def forward(self, x): return x + self.b(x)

class ART2D(nn.Module):
    def __init__(self, c, has_t=False, trans=None):
        super().__init__(); self._t = has_t; self.cnn = RC2D(c); self.trans = trans
        if has_t: self.agg = nn.Sequential(nn.Conv2d(c*2,c,1), nn.InstanceNorm2d(c), nn.ReLU(True))
    def forward(self, x):
        c = self.cnn(x)
        return self.agg(torch.cat([c, self.trans(x)], 1)) if self._t and self.trans else c
    def set_t(self, on): self._t = on

class ART3D(nn.Module):
    def __init__(self, c, has_t=False, trans=None, ckpt=False):
        super().__init__(); self._t, self._ck = has_t, ckpt; self.cnn = RC3D(c); self.trans = trans
        if has_t: self.agg = nn.Sequential(nn.Conv3d(c*2,c,1), nn.InstanceNorm3d(c), nn.ReLU(True))
    def _f(self, x):
        c = self.cnn(x)
        return self.agg(torch.cat([c, self.trans(x)], 1)) if self._t and self.trans else c
    def forward(self, x):
        return grad_checkpoint(self._f, x, use_reentrant=False) if self._ck and self.training else self._f(x)
    def set_t(self, on): self._t = on


# ============================================================================
# 8. GENERATORS (baseline = plain enc/ART/dec, no U-Net skips)
#    Optional stochastic bottleneck for ablation D (hierarchical).
# ============================================================================
class StochBN2D(nn.Module):
    """Stochastic bottleneck: prior N(0, 1). Posterior q(z|f) predicted from
    encoder features. At train: z = mu + exp(0.5*lv)*eps*temp, KL(q || N(0,1)).
    At infer (temp=0): z = mu (deterministic). Output = f + z (residual latent)."""
    def __init__(self, c):
        super().__init__()
        self.mu = nn.Conv2d(c, c, 1)
        self.lv = nn.Conv2d(c, c, 1)

    def forward(self, f, temp=1.0):
        mu = self.mu(f); lv = self.lv(f).clamp(-6, 2)
        if temp > 0.0:
            z = mu + temp * torch.exp(0.5 * lv) * torch.randn_like(mu)
        else:
            z = mu
        # KL(N(mu, sigma^2) || N(0,1)) = 0.5*(mu^2 + sigma^2 - 1 - lv)
        kl = 0.5 * (mu.pow(2) + lv.exp() - 1 - lv)
        return f + z, kl.mean()


class StochBN3D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.mu = nn.Conv3d(c, c, 1)
        self.lv = nn.Conv3d(c, c, 1)

    def forward(self, f, temp=1.0):
        mu = self.mu(f); lv = self.lv(f).clamp(-6, 2)
        if temp > 0.0:
            z = mu + temp * torch.exp(0.5 * lv) * torch.randn_like(mu)
        else:
            z = mu
        kl = 0.5 * (mu.pow(2) + lv.exp() - 1 - lv)
        return f + z, kl.mean()


class Gen2D(nn.Module):
    def __init__(self, ic=1, oc=1, ngf=64, nart=9, tpos=(4,5),
                 nh=8, slp=2, ws=8, mr=4., dr=0., hierarchical=False):
        super().__init__()
        bn = ngf*4
        self.hierarchical = hierarchical
        self.enc = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(ic,ngf,7), nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.Conv2d(ngf,ngf*2,3,2,1), nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.Conv2d(ngf*2,bn,3,2,1), nn.InstanceNorm2d(bn), nn.ReLU(True))
        if hierarchical:
            self.stoch = StochBN2D(bn)
        self.arts = nn.ModuleList(); self._tp = list(tpos)
        for i in range(nart):
            if i in tpos:
                self.arts.append(ART2D(bn, True, Swin2D(bn, nh, slp, ws, mr, dr)))
            else:
                self.arts.append(ART2D(bn, False))
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(bn,ngf*2,4,2,1), nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2,ngf,4,2,1), nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.ReflectionPad2d(3), nn.Conv2d(ngf,oc,7), nn.Tanh())

    def forward(self, x, temp=1.0, return_kl=False):
        f = self.enc(x)
        kl = torch.zeros((), device=x.device)
        if self.hierarchical:
            f, kl = self.stoch(f, temp)
        for a in self.arts: f = a(f)
        out = self.dec(f)
        if return_kl: return out, kl
        return out

    def set_trans(self, on):
        for i, a in enumerate(self.arts):
            if i in self._tp: a.set_t(on)
        print(f"  [G] Trans {'ON' if on else 'OFF'}")


class Gen3D(nn.Module):
    def __init__(self, ic=1, oc=1, ngf=48, nart=9, tpos=(4,5),
                 nh=8, slp=2, ws=4, mr=4., dr=0., ckpt=False, hierarchical=False):
        super().__init__()
        bn = ngf*4
        self.hierarchical = hierarchical
        self.enc = nn.Sequential(
            nn.Conv3d(ic,ngf,7,padding=3), nn.InstanceNorm3d(ngf), nn.ReLU(True),
            nn.Conv3d(ngf,ngf*2,3,2,1), nn.InstanceNorm3d(ngf*2), nn.ReLU(True),
            nn.Conv3d(ngf*2,bn,3,2,1), nn.InstanceNorm3d(bn), nn.ReLU(True))
        if hierarchical:
            self.stoch = StochBN3D(bn)
        self.arts = nn.ModuleList(); self._tp = list(tpos)
        for i in range(nart):
            if i in tpos:
                self.arts.append(ART3D(bn, True, Swin3D(bn, nh, slp, ws, mr, dr), ckpt))
            else:
                self.arts.append(ART3D(bn, False, ckpt=ckpt))
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(bn,ngf*2,4,2,1), nn.InstanceNorm3d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose3d(ngf*2,ngf,4,2,1), nn.InstanceNorm3d(ngf), nn.ReLU(True),
            nn.Conv3d(ngf,oc,7,padding=3), nn.Tanh())

    def forward(self, x, temp=1.0, return_kl=False):
        f = self.enc(x)
        kl = torch.zeros((), device=x.device)
        if self.hierarchical:
            f, kl = self.stoch(f, temp)
        for a in self.arts: f = a(f)
        out = self.dec(f)
        if return_kl: return out, kl
        return out

    def set_trans(self, on):
        for i, a in enumerate(self.arts):
            if i in self._tp: a.set_t(on)
        print(f"  [G3D] Trans {'ON' if on else 'OFF'}")


class Refine3D(nn.Module):
    def __init__(self, c=1, nf=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(c,nf,3,padding=1), nn.InstanceNorm3d(nf), nn.ReLU(True),
            nn.Conv3d(nf,nf,3,padding=1), nn.InstanceNorm3d(nf), nn.ReLU(True),
            nn.Conv3d(nf,nf,3,padding=1), nn.InstanceNorm3d(nf), nn.ReLU(True),
            nn.Conv3d(nf,c,3,padding=1), nn.Tanh())
    def forward(self, x): return x + 0.1 * self.net(x)


# ============================================================================
# 9. DISCRIMINATOR
# ============================================================================
class Disc2D(nn.Module):
    def __init__(self, ic=2, ndf=64, nl=3):
        super().__init__()
        blks = [nn.Sequential(nn.Conv2d(ic,ndf,4,2,1), nn.LeakyReLU(.2,True))]
        ch = ndf
        for i in range(1, nl):
            cn = min(ndf*(2**i), 512)
            blks.append(nn.Sequential(nn.Conv2d(ch,cn,4,2,1), nn.InstanceNorm2d(cn), nn.LeakyReLU(.2,True)))
            ch = cn
        cn = min(ndf*(2**nl), 512)
        blks.append(nn.Sequential(nn.Conv2d(ch,cn,4,1,1), nn.InstanceNorm2d(cn), nn.LeakyReLU(.2,True)))
        self.blks = nn.ModuleList(blks); self.fin = nn.Conv2d(cn,1,4,1,1)

    def forward(self, x, rf=False):
        fs = []
        for b in self.blks: x = b(x); fs.append(x)
        o = self.fin(x); return (o, fs) if rf else o


class Disc3D(nn.Module):
    def __init__(self, ic=2, ndf=48, nl=3):
        super().__init__()
        blks = [nn.Sequential(nn.Conv3d(ic,ndf,4,2,1), nn.LeakyReLU(.2,True))]
        ch = ndf
        for i in range(1, nl):
            cn = min(ndf*(2**i), 512)
            blks.append(nn.Sequential(nn.Conv3d(ch,cn,4,2,1), nn.InstanceNorm3d(cn), nn.LeakyReLU(.2,True)))
            ch = cn
        cn = min(ndf*(2**nl), 512)
        blks.append(nn.Sequential(nn.Conv3d(ch,cn,4,1,1), nn.InstanceNorm3d(cn), nn.LeakyReLU(.2,True)))
        self.blks = nn.ModuleList(blks); self.fin = nn.Conv3d(cn,1,4,1,1)

    def forward(self, x, rf=False):
        fs = []
        for b in self.blks: x = b(x); fs.append(x)
        o = self.fin(x); return (o, fs) if rf else o


# ============================================================================
# 10. LOSSES & METRICS
# ============================================================================
def adv_ls(p, real):
    return F.mse_loss(p, torch.ones_like(p) if real else torch.zeros_like(p))

def fm_l(fk, rl):
    return sum(F.l1_loss(f, r.detach()) for f, r in zip(fk, rl)) / max(len(fk), 1)

def masked_l1(pred, tgt, cmask):
    """L1 only on available channels. cmask: (B, C), pred/tgt: (B, C, ...)."""
    m = cmask
    for _ in range(pred.dim() - 2): m = m.unsqueeze(-1)
    diff = torch.abs(pred - tgt) * m
    spatial = 1
    for d in pred.shape[2:]: spatial *= d
    n = m.sum() * spatial
    return diff.sum() / n.clamp(min=1)

def psnr_np(t, p):
    mse = np.mean(((t+1)/2-(p+1)/2)**2); return 10*math.log10(1./max(mse,1e-10))

def ssim_2d(t, p, w=11):
    t0, p0 = np.clip((t+1)/2,0,1), np.clip((p+1)/2,0,1)
    mt, mp = uniform_filter(t0,w), uniform_filter(p0,w)
    st = uniform_filter(t0**2,w)-mt**2; sp = uniform_filter(p0**2,w)-mp**2
    stp = uniform_filter(t0*p0,w)-mt*mp
    c1,c2 = .01**2, .03**2
    return float(np.mean(((2*mt*mp+c1)*(2*stp+c2))/((mt**2+mp**2+c1)*(st+sp+c2))))

def ssim_vol(t, p):
    v = [ssim_2d(t[:,:,i], p[:,:,i]) for i in range(t.shape[2]) if np.max(t[:,:,i]) > -.9]
    return float(np.mean(v)) if v else 0.


# ----------------------------------------------------------------------------
# Ablation B — nnU-Net perceptual (domain-specific, 4-ch FLAIR/T1/T1CE/T2).
# FIX vs v1: our predictions are in [-1, 1] after Tanh with background=-1;
# nnU-Net was trained on Z-score-normalized inputs (roughly N(0, 1), bg=0).
# We undo the /3 clipping (our norm_z divides by c=3) to recover ~z-score
# range, and set background to 0 (NOT -1) before forwarding.
# ----------------------------------------------------------------------------
class NNUNetPerceptual(nn.Module):
    def __init__(self, ckpt_path, layers=(0, 1, 2), device="cuda",
                 n_input_channels=4, mri_clip_c=3.0, fg_thr=-0.9,
                 perc_mode="single_pass"):
        super().__init__()
        self.layers = tuple(layers)
        self.device_str = device
        self.n_in = n_input_channels
        self.mri_clip_c = float(mri_clip_c)
        self.fg_thr = float(fg_thr)
        self.perc_mode = perc_mode
        # last per-channel contributions (populated in forward), for logging
        self.last_per_ch = {}
        self.encoder = self._load_encoder(ckpt_path).to(device)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    def _load_encoder(self, ckpt_path):
        """Load nnU-Net encoder stages only. Uses nnunetv2 API.
        Falls back to a generic encoder if nnunetv2 is unavailable."""
        try:
            from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
            from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
            from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ia = ck["init_args"]
            plans = PlansManager(ia["plans"])
            cfgp = plans.get_configuration(ia["configuration"])
            n_in = determine_num_input_channels(plans, cfgp, ia["dataset_json"])
            n_out = plans.get_label_manager(ia["dataset_json"]).num_segmentation_heads
            net = get_network_from_plans(
                cfgp.network_arch_class_name,
                cfgp.network_arch_init_kwargs,
                cfgp.network_arch_init_kwargs_req_import,
                n_in, n_out, allow_init=True, deep_supervision=False)
            sd = ck["network_weights"]
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            missing, unexpected = net.load_state_dict(sd, strict=False)
            print(f"  [Perceptual] nnU-Net loaded (missing={len(missing)} unexpected={len(unexpected)})")
            self.n_in = n_in
            return net.encoder
        except Exception as e:
            print(f"  [Perceptual] nnunetv2 unavailable ({e}); using generic encoder.")
            return _GenericEncoder3D(in_ch=self.n_in, base=32, stages=max(self.layers)+1)

    def _rescale_to_zscore(self, x):
        """Undo our normalization: our pipeline does z-score -> clip(+/-c) / c,
        yielding [-1, 1] with background=-1. Multiply by c to recover ~z-score
        range, and zero the background (nnU-Net expects 0 outside foreground)."""
        fg = (x > self.fg_thr).float()
        return x * self.mri_clip_c * fg

    def _embed(self, x_single, ch_idx):
        """Place a single-channel MRI tensor into the n_in-channel nnU-Net input."""
        B = x_single.shape[0]
        shp = x_single.shape[2:]
        inp = x_single.new_zeros((B, self.n_in, *shp))
        inp[:, ch_idx:ch_idx+1] = self._rescale_to_zscore(x_single)
        return inp

    def _extract(self, x4):
        """Forward through encoder and collect selected-stage features.
        NO @torch.no_grad here — gradient must flow from pred features.
        Encoder parameters are frozen via requires_grad=False."""
        feats = []
        h = x4
        stages = getattr(self.encoder, "stages", None)
        if stages is not None:
            for i, st in enumerate(stages):
                h = st(h)
                if i in self.layers: feats.append(h)
                if i >= max(self.layers): break
        else:
            feats_all = self.encoder(x4, return_feats=True)
            feats = [feats_all[i] for i in self.layers if i < len(feats_all)]
        return feats

    def _feat_loss(self, f_p, f_t):
        """Average normalized L1 across feature stages."""
        loss = f_p[0].new_tensor(0.0)
        for fp, ft in zip(f_p, f_t):
            scale = ft.detach().abs().mean().clamp(min=1e-3)
            loss = loss + F.l1_loss(fp, ft.detach()) / scale
        return loss / max(len(f_p), 1)

    def forward(self, pred, target, target_mask, channel_map):
        """Dispatcher for the four perc_mode strategies.

        single_pass : build ONE 4-channel input with all target channels placed
                      in their nnU-Net slots simultaneously. Fast. Matches
                      resvit_ablation_v2.py exactly. Known to degrade LPIPS
                      of BOTH channels in t2_flair runs vs single-target B.

        split       : one forward per target channel, each with only its own
                      slot filled. Isolates the gradient of each channel from
                      the other. Replicates the successful ablB-t2 recipe.

        split_ctx   : one forward per target channel; the OTHER channels of
                      channel_map are filled with the detached GT (providing
                      in-distribution multimodal context to the encoder). The
                      context slot is identical in pred/target so its feature
                      contribution cancels — only the supervised slot produces
                      a non-zero gradient.

        late_only   : identical to single_pass but self.layers is set to
                      tardy encoder stages (e.g., (3,4)). Tests whether early
                      cross-channel mixing is the root cause.

        target_mask[b, c] == 1 when subject b has modality c; zero otherwise.
        Missing-modality pathways are neutralized by zeroing both sides.

        Stores per-channel losses in self.last_per_ch (keys = channel indices
        as in pred, values = scalar python floats). For single_pass and
        late_only, only the joint loss is stored under key -1.
        """
        if pred.dim() == 4:
            pred = pred.unsqueeze(2); target = target.unsqueeze(2)
        B, C = pred.shape[:2]

        self.last_per_ch = {}

        if target_mask.sum() < 1:
            return pred.new_tensor(0.0)

        if self.perc_mode in ("single_pass", "late_only", "late4_only") or C == 1:
            return self._forward_single_pass(pred, target, target_mask, channel_map)
        elif self.perc_mode == "split":
            return self._forward_split(pred, target, target_mask, channel_map, use_context=False)
        elif self.perc_mode == "split_ctx":
            return self._forward_split(pred, target, target_mask, channel_map, use_context=True)
        else:
            raise ValueError(f"unknown perc_mode: {self.perc_mode}")

    def _forward_single_pass(self, pred, target, target_mask, channel_map):
        B, C = pred.shape[:2]
        spatial = pred.shape[2:]
        bcast = (B, 1) + (1,) * (pred.dim() - 2)

        p_emb = pred.new_zeros((B, self.n_in, *spatial))
        t_emb = target.new_zeros((B, self.n_in, *spatial))

        for c in range(C):
            nnc = channel_map[c]
            avm = target_mask[:, c:c+1].view(*bcast)
            p_emb[:, nnc:nnc+1] = self._rescale_to_zscore(pred[:, c:c+1])   * avm
            t_emb[:, nnc:nnc+1] = self._rescale_to_zscore(target[:, c:c+1]) * avm

        f_p = self._extract(p_emb.to(self.device_str))
        with torch.no_grad():
            f_t = self._extract(t_emb.to(self.device_str))
        loss = self._feat_loss(f_p, f_t)
        self.last_per_ch = {-1: float(loss.detach().item())}
        return loss

    def _forward_split(self, pred, target, target_mask, channel_map, use_context):
        """Two-pass perceptual: one forward per target channel.

        use_context=False : other slots are zero (OOD but channel-isolated).
        use_context=True  : other slots get the detached GT modality in their
                            nnU-Net slot (in-distribution context, gradient
                            cancels through identical pred/target ctx slots).
        """
        B, C = pred.shape[:2]
        spatial = pred.shape[2:]
        bcast = (B, 1) + (1,) * (pred.dim() - 2)

        total = pred.new_tensor(0.0)
        n_valid = 0
        for c in range(C):
            avm_c = target_mask[:, c:c+1].view(*bcast)
            if target_mask[:, c].sum() < 1:
                # nothing to supervise for this channel in this batch
                self.last_per_ch[c] = 0.0
                continue

            p_emb = pred.new_zeros((B, self.n_in, *spatial))
            t_emb = target.new_zeros((B, self.n_in, *spatial))

            # supervised slot
            nnc = channel_map[c]
            p_emb[:, nnc:nnc+1] = self._rescale_to_zscore(pred[:, c:c+1])   * avm_c
            t_emb[:, nnc:nnc+1] = self._rescale_to_zscore(target[:, c:c+1]) * avm_c

            # context slots (only in split_ctx)
            if use_context:
                for cc in range(C):
                    if cc == c:
                        continue
                    nncc = channel_map[cc]
                    avm_cc = target_mask[:, cc:cc+1].view(*bcast)
                    ctx = self._rescale_to_zscore(target[:, cc:cc+1]).detach() * avm_cc
                    p_emb[:, nncc:nncc+1] = ctx
                    t_emb[:, nncc:nncc+1] = ctx

            f_p = self._extract(p_emb.to(self.device_str))
            with torch.no_grad():
                f_t = self._extract(t_emb.to(self.device_str))
            loss_c = self._feat_loss(f_p, f_t)
            self.last_per_ch[c] = float(loss_c.detach().item())
            total = total + loss_c
            n_valid += 1

        return total / max(n_valid, 1)


class _GenericEncoder3D(nn.Module):
    """Fallback when nnunetv2 is unavailable. Matches nnU-Net schedule."""
    def __init__(self, in_ch=4, base=32, stages=6):
        super().__init__()
        chs = [base, 64, 128, 256, 320, 320][:stages]
        self.stages = nn.ModuleList()
        prev = in_ch
        for i, c in enumerate(chs):
            stride = 1 if i == 0 else 2
            self.stages.append(nn.Sequential(
                nn.Conv3d(prev, c, 3, stride=stride, padding=1),
                nn.InstanceNorm3d(c), nn.LeakyReLU(0.01, True),
                nn.Conv3d(c, c, 3, padding=1),
                nn.InstanceNorm3d(c), nn.LeakyReLU(0.01, True),
            ))
            prev = c

    def forward(self, x, return_feats=False):
        feats = []
        for st in self.stages:
            x = st(x); feats.append(x)
        return feats if return_feats else x


# ----------------------------------------------------------------------------
# Ablation C — Laplacian pyramid L1 frequency loss, foreground-masked.
# FIX vs v1: global FFT over bg=-1 images produced Gibbs ringing at the
# foreground boundary. Laplacian pyramid is multi-scale, stable, and we
# apply it only inside the foreground (via tmask + spatial fg mask).
# ----------------------------------------------------------------------------
def _gauss_kernel(ch, dim, device, dtype, sigma=1.0):
    """Separable Gaussian kernel of fixed size 5."""
    k = torch.tensor([1., 4., 6., 4., 1.], device=device, dtype=dtype) / 16.
    if dim == 2:
        w = torch.einsum("i,j->ij", k, k).view(1, 1, 5, 5).expand(ch, 1, 5, 5).contiguous()
    else:
        w = torch.einsum("i,j,k->ijk", k, k, k).view(1, 1, 5, 5, 5).expand(ch, 1, 5, 5, 5).contiguous()
    return w


def _downsample(x, kernel):
    conv = F.conv2d if x.dim() == 4 else F.conv3d
    pad = [2] * (2 * (x.dim() - 2))
    x = F.pad(x, pad, mode="reflect")
    x = conv(x, kernel, groups=x.shape[1])
    # stride-2 downsample
    if x.dim() == 4:
        return x[:, :, ::2, ::2]
    return x[:, :, ::2, ::2, ::2]


def _laplacian_pyramid(x, levels=3):
    """Returns list of laplacian bands [l0, l1, ...] plus residual low-pass."""
    dim = x.dim() - 2
    ch = x.shape[1]
    kernel = _gauss_kernel(ch, dim, x.device, x.dtype)
    cur = x
    bands = []
    for _ in range(levels):
        down = _downsample(cur, kernel)
        # Upsample back to current size
        up = F.interpolate(down, size=cur.shape[2:], mode=("bilinear" if dim == 2 else "trilinear"),
                           align_corners=False)
        bands.append(cur - up)
        cur = down
    bands.append(cur)  # low-pass residual
    return bands


def laplacian_frequency_loss(pred, target, tmask, fg_mask, levels=3):
    """Multi-scale Laplacian pyramid L1, masked to foreground, averaged over
    available channels. pred/target shape: (B, C, H, W) or (B, C, D, H, W).
    fg_mask shape: (B, H, W) or (B, D, H, W). tmask: (B, C)."""
    if pred.dim() == 4:
        fg = fg_mask.unsqueeze(1).float()      # (B, 1, H, W)
    else:
        fg = fg_mask.unsqueeze(1).float()      # (B, 1, D, H, W)

    # mask out background (both pred and target)
    pred_m   = pred   * fg
    target_m = target * fg

    # cast to float32 for numerical stability (bf16 conv with kernel 5 is fine)
    bp = _laplacian_pyramid(pred_m.float(),   levels=levels)
    bt = _laplacian_pyramid(target_m.float(), levels=levels)

    # per-level L1, averaged; weight by channel availability
    total = pred.new_tensor(0.0, dtype=torch.float32)
    n_lvl = 0
    for lp, lt in zip(bp[:-1], bt[:-1]):   # skip the final low-pass residual
        # (B, C, *) → mean over spatial; weight by tmask per channel
        diff = (lp - lt).abs()
        spatial_dims = tuple(range(2, diff.dim()))
        per_ch = diff.mean(dim=spatial_dims)   # (B, C)
        w = tmask.to(per_ch.dtype)
        num = (per_ch * w).sum()
        den = w.sum().clamp(min=1.0)
        total = total + num / den
        n_lvl += 1
    return total / max(n_lvl, 1)


def ablation_cascade(level):
    """Returns dict of flags for a given cascade letter A..E."""
    f = dict(use_perceptual=False, use_frequency=False,
             use_hierarchical=False, use_seg_guide=False)
    if level >= "B": f["use_perceptual"]   = True
    if level >= "C": f["use_frequency"]    = True
    if level >= "D": f["use_hierarchical"] = True
    if level >= "E": f["use_seg_guide"]    = True
    return f


# ============================================================================
# 11. INFERENCE
# ============================================================================
def _gen_infer(gen, inp):
    """Run generator in inference mode. Honors cfg.infer_temperature and
    cfg.infer_samples (averages K stochastic samples if temp > 0).
    Safe for non-hierarchical generators (temp is ignored)."""
    temp = getattr(cfg, "infer_temperature", 0.0)
    k = max(1, getattr(cfg, "infer_samples", 1)) if temp > 0.0 else 1
    def _once():
        out = gen(inp, temp=temp)
        return out[0] if isinstance(out, tuple) else out
    if k == 1:
        return _once()
    acc = None
    for _ in range(k):
        o = _once()
        acc = o if acc is None else acc + o
    return acc / k


@torch.no_grad()
def infer_plane(gen, vol, axis, sz, oc, dev):
    """2D inference along one axis. Returns (oc, H, W, D)."""
    gen.eval(); orig = vol.shape; n = orig[axis]
    other = [i for i in range(3) if i != axis]
    ss = (orig[other[0]], orig[other[1]])
    preds = []
    for si in range(n):
        sl = vol[si,:,:] if axis==0 else vol[:,si,:] if axis==1 else vol[:,:,si]
        inp = torch.from_numpy(rsz(sl, sz)[None,None]).float().to(dev)
        if cfg.channels_last:
            inp = inp.to(memory_format=torch.channels_last)
        with _amp_ctx(dev):
            pr = _gen_infer(gen, inp)
        pn = pr[0].float().cpu().numpy()
        chs = [zoom(pn[c], [ss[0]/sz, ss[1]/sz], order=1).astype(np.float32) for c in range(oc)]
        preds.append(np.stack(chs, 0))
    stk = np.stack(preds, 0)
    if axis==0: return stk.transpose(1,0,2,3)
    if axis==1: return stk.transpose(1,2,0,3)
    return stk.transpose(1,2,3,0)


@torch.no_grad()
def infer_plane_25d(gen, vol, axis, sz, oc, dev, ctx=3):
    """2.5D inference: stack ctx adjacent slices as input channels. Returns (oc, H, W, D)."""
    gen.eval(); orig = vol.shape; n = orig[axis]; h = ctx // 2
    other = [i for i in range(3) if i != axis]
    ss = (orig[other[0]], orig[other[1]])
    preds = []
    for si in range(n):
        slices = []
        for o in range(-h, h + 1):
            idx = min(max(si + o, 0), n - 1)  # clamp at borders
            sl = vol[idx,:,:] if axis==0 else vol[:,idx,:] if axis==1 else vol[:,:,idx]
            slices.append(rsz(sl, sz))
        inp = torch.from_numpy(np.stack(slices, 0)[None]).float().to(dev)  # (1, ctx, H, W)
        if cfg.channels_last:
            inp = inp.to(memory_format=torch.channels_last)
        with _amp_ctx(dev):
            pr = _gen_infer(gen, inp)
        pn = pr[0].float().cpu().numpy()
        chs = [zoom(pn[c], [ss[0]/sz, ss[1]/sz], order=1).astype(np.float32) for c in range(oc)]
        preds.append(np.stack(chs, 0))
    stk = np.stack(preds, 0)
    if axis==0: return stk.transpose(1,0,2,3)
    if axis==1: return stk.transpose(1,2,0,3)
    return stk.transpose(1,2,3,0)


@torch.no_grad()
def infer_triplanar(gen, vol, sz, oc, dev, w=(.5,.25,.25)):
    """Only for isotropic data. NOT default for US."""
    pa = infer_plane(gen, vol, 2, sz, oc, dev)
    pc = infer_plane(gen, vol, 1, sz, oc, dev)
    ps = infer_plane(gen, vol, 0, sz, oc, dev)
    return (w[0]*pa + w[1]*pc + w[2]*ps).astype(np.float32)


@torch.no_grad()
def infer_sw3d(gen, vol, ps, ov, oc, dev):
    gen.eval(); vp, crop = pad_vol(vol, ps, -1.); sh = vp.shape
    step = [max(p-o,1) for p,o in zip(ps,ov)]
    out = np.zeros((oc,)+sh, np.float32); cnt = np.zeros(sh, np.float32)
    wm = hanning3d(ps)
    sts = [slide_starts(s,p,st) for s,p,st in zip(sh,ps,step)]
    tot = len(sts[0])*len(sts[1])*len(sts[2]); idx = 0
    for i in sts[0]:
        for j in sts[1]:
            for k in sts[2]:
                idx += 1
                sl = (slice(i,i+ps[0]),slice(j,j+ps[1]),slice(k,k+ps[2]))
                pt = torch.from_numpy(vp[sl][None,None]).float().to(dev)
                with _amp_ctx(dev):
                    pr = _gen_infer(gen, pt)
                pn = pr[0].float().cpu().numpy()
                for c in range(oc): out[c][sl] += pn[c]*wm
                cnt[sl] += wm
                if idx % 5 == 0: print(f"\r    Patch {idx}/{tot}", end="")
    print()
    cnt = np.maximum(cnt, 1e-6)
    for c in range(oc): out[c] /= cnt
    return out[(slice(None),) + crop].astype(np.float32)


def run_inference(gen, s, cfg_obj, dev):
    """Run inference for a single sample, respecting variant and settings."""
    gen = _unwrap(gen)  # bypass torch.compile for variable-shape inference
    oc = cfg_obj.out_ch
    v = cfg_obj.arch_variant
    if v in ("2d", "2.5d", "2d_3d_refine"):
        if cfg_obj.triplanar:
            return infer_triplanar(gen, s["us"], cfg_obj.image_size, oc, dev, cfg_obj.triplanar_weights)
        if v == "2.5d":
            return infer_plane_25d(gen, s["us"], 2, cfg_obj.image_size, oc, dev, cfg_obj.context_slices)
        return infer_plane(gen, s["us"], 2, cfg_obj.image_size, oc, dev)
    else:
        return infer_sw3d(gen, s["us"], cfg_obj.patch_size_3d, cfg_obj.patch_overlap_3d, oc, dev)


# ============================================================================
# 12. TRAINING
# ============================================================================
def build_all(cfg_obj, dev):
    v, oc, ic = cfg_obj.arch_variant, cfg_obj.out_ch, cfg_obj.in_ch
    hier = cfg_obj.use_hierarchical
    if v in ("2d","2.5d","2d_3d_refine"):
        G = Gen2D(ic, oc, cfg_obj.ngf, cfg_obj.n_art_blocks, cfg_obj.transformer_positions,
                  cfg_obj.n_heads, cfg_obj.swin_layer_pairs, cfg_obj.window_size_2d,
                  cfg_obj.mlp_ratio, cfg_obj.dropout, hierarchical=hier).to(dev)
        D = Disc2D(ic+oc, cfg_obj.ndf, cfg_obj.n_layers_d).to(dev)
        if cfg_obj.channels_last:
            G = G.to(memory_format=torch.channels_last)
            D = D.to(memory_format=torch.channels_last)
    else:
        G = Gen3D(1, oc, cfg_obj.ngf, cfg_obj.n_art_blocks, cfg_obj.transformer_positions,
                  cfg_obj.n_heads, cfg_obj.swin_layer_pairs, cfg_obj.window_size_3d,
                  cfg_obj.mlp_ratio, cfg_obj.dropout, cfg_obj.grad_ckpt_3d,
                  hierarchical=hier).to(dev)
        D = Disc3D(1+oc, cfg_obj.ndf, cfg_obj.n_layers_d).to(dev)
    R = Refine3D(oc).to(dev) if v == "2d_3d_refine" else None
    return G, D, R


def build_loaders(cfg_obj, tr, va):
    v = cfg_obj.arch_variant
    if v in ("2d","2d_3d_refine"):
        tds = DS2D(tr, cfg_obj.image_size, cfg_obj.target_mode, True)
        vds = DS2D(va, cfg_obj.image_size, cfg_obj.target_mode, False)
        bs = cfg_obj.batch_size
    elif v == "2.5d":
        tds = DS25D(tr, cfg_obj.image_size, cfg_obj.target_mode, cfg_obj.context_slices, True)
        vds = DS25D(va, cfg_obj.image_size, cfg_obj.target_mode, cfg_obj.context_slices, False)
        bs = cfg_obj.batch_size
    else:
        tds = DS3D(tr, cfg_obj.patch_size_3d, cfg_obj.target_mode, cfg_obj.patches_per_volume, True)
        vds = DS3D(va, cfg_obj.patch_size_3d, cfg_obj.target_mode, 2, False)
        bs = cfg_obj.batch_size_3d
    dl_kw = dict(num_workers=cfg_obj.num_workers, pin_memory=False)
    if cfg_obj.num_workers > 0:
        dl_kw["persistent_workers"] = cfg_obj.persistent_workers
        dl_kw["prefetch_factor"] = cfg_obj.prefetch_factor
    tl = DataLoader(tds, bs, True, drop_last=True, **dl_kw)
    vl = DataLoader(vds, bs, False, **dl_kw)
    return tl, vl


def _amp_ctx(dev):
    dtype = torch.bfloat16 if cfg.use_bf16 else torch.float16
    return torch.autocast(device_type=dev.type,
                          dtype=dtype,
                          enabled=(cfg.use_amp and dev.type == "cuda"))


def _unwrap(m):
    """Return the uncompiled underlying nn.Module for variable-shape inference."""
    return m._orig_mod if hasattr(m, "_orig_mod") else m


def _to_device(t, dev, memfmt=None):
    t = t.to(dev, non_blocking=True)
    if memfmt is not None and t.dim() == 4:
        t = t.to(memory_format=memfmt)
    return t


def _target_fg_mask(tar, thr=-0.95):
    """Foreground: any target channel > thr. Returns (B, *spatial) bool->float."""
    # tar: (B, C, ...) — reduce over C
    fg = (tar > thr).any(dim=1).float()
    return fg


def train_ep(G, D, ld, oG, oD, sc, dev, ep, acc, la, ll, lf,
             perceptual_net=None, cfg_obj=None):
    """Training epoch. Supports ablation modules B (perceptual), C (freq),
    D (hierarchical, KL). Baseline (A) runs identically to resvit_final.py."""
    G.train(); D.train()
    r = {"g":0.,"d":0.,"l1":0.,"adv":0.,"fm":0.,"perc":0.,"freq":0.,"kl":0.,
         "perc_t2":0.,"perc_fl":0.}; n = 0
    # Map perc_mode channel indices -> logging keys (t2_flair only).
    tm_cur = cfg_obj.target_mode if cfg_obj is not None else None
    perch_key = {}
    if tm_cur == "t2_flair":
        perch_key = {0: "perc_t2", 1: "perc_fl"}
    elif tm_cur == "t2":
        perch_key = {0: "perc_t2"}
    elif tm_cur == "flair":
        perch_key = {0: "perc_fl"}
    oG.zero_grad(set_to_none=True); oD.zero_grad(set_to_none=True)
    memfmt = torch.channels_last if (cfg.channels_last and cfg.arch_variant != "full_3d") else None

    # Channel map for perceptual (how our out_ch indices map to nnU-Net input channels)
    if cfg_obj is not None and cfg_obj.use_perceptual:
        tm = cfg_obj.target_mode
        if   tm == "t2":    ch_map = [cfg_obj.nnunet_channels["t2"]]
        elif tm == "flair": ch_map = [cfg_obj.nnunet_channels["flair"]]
        else:               ch_map = [cfg_obj.nnunet_channels["t2"],
                                      cfg_obj.nnunet_channels["flair"]]
    else:
        ch_map = None

    # KL annealing
    if cfg_obj is not None and cfg_obj.use_hierarchical:
        kl_w = cfg_obj.lambda_kl * min(1.0, ep / max(1, cfg_obj.kl_warmup_epochs))
    else:
        kl_w = 0.0

    for i, (inp, tar, cm) in enumerate(ld):
        inp = _to_device(inp, dev, memfmt)
        tar = _to_device(tar, dev, memfmt)
        cm  = cm.to(dev, non_blocking=True)
        aw = cm.mean().clamp(min=0.1)

        # ---- G forward (+ KL if hierarchical) ----
        with _amp_ctx(dev):
            if cfg_obj is not None and cfg_obj.use_hierarchical:
                fk_raw, kl_val = G(inp, temp=cfg_obj.train_temperature, return_kl=True)
            else:
                fk_raw = G(inp)
                kl_val = torch.zeros((), device=inp.device)
        fk_det = fk_raw.detach()

        # ---- D update ----
        set_grad(D, True)
        with _amp_ctx(dev):
            ld_ = .5*(adv_ls(D(torch.cat([inp, tar], 1)), True) +
                       adv_ls(D(torch.cat([inp, fk_det], 1)), False)) * aw / acc
        sc.scale(ld_).backward()

        # ---- G update (D frozen) ----
        set_grad(D, False)
        with _amp_ctx(dev):
            pf, ff = D(torch.cat([inp, fk_raw], 1), rf=True)
            _, rf  = D(torch.cat([inp, tar],    1), rf=True)
            la_ = adv_ls(pf, True) * aw
            ll_ = masked_l1(fk_raw, tar, cm)
            lf_ = fm_l(ff, rf)

            # Perceptual (B)
            if cfg_obj is not None and cfg_obj.use_perceptual and perceptual_net is not None:
                lp_ = perceptual_net(fk_raw, tar, cm, ch_map)
            else:
                lp_ = torch.zeros((), device=inp.device)

            # Frequency (C) — Laplacian pyramid over foreground
            if cfg_obj is not None and cfg_obj.use_frequency:
                fg = _target_fg_mask(tar, thr=-0.95)
                lfq_ = laplacian_frequency_loss(fk_raw, tar, cm, fg,
                                                levels=cfg_obj.freq_pyramid_levels)
            else:
                lfq_ = torch.zeros((), device=inp.device)

            lg = (la * la_
                  + ll * ll_
                  + lf * lf_
                  + (cfg_obj.lambda_perceptual if cfg_obj else 0.0) * lp_
                  + (cfg_obj.lambda_frequency  if cfg_obj else 0.0) * lfq_
                  + kl_w * kl_val) / acc
        sc.scale(lg).backward()
        set_grad(D, True)

        if (i+1) % acc == 0:
            if sc.is_enabled():
                sc.unscale_(oD); sc.unscale_(oG)
            nn.utils.clip_grad_norm_(D.parameters(), 1.)
            nn.utils.clip_grad_norm_(G.parameters(), 1.)
            sc.step(oD); sc.step(oG); sc.update()
            oD.zero_grad(set_to_none=True); oG.zero_grad(set_to_none=True)

        r["g"]+=lg.item()*acc; r["d"]+=ld_.item()*acc
        r["l1"]+=ll_.item(); r["adv"]+=la_.item(); r["fm"]+=lf_.item()
        r["perc"]+=lp_.item(); r["freq"]+=lfq_.item(); r["kl"]+=kl_val.item()
        # Per-channel perceptual (split/split_ctx modes populate last_per_ch)
        cur_per_ch = {}
        if (cfg_obj is not None and cfg_obj.use_perceptual
                and perceptual_net is not None and perch_key):
            lp_dict = getattr(perceptual_net, "last_per_ch", {}) or {}
            for c_idx, key in perch_key.items():
                v = lp_dict.get(c_idx, None)
                if v is not None:
                    r[key] += v
                    cur_per_ch[key] = v
        n+=1
        if (i+1) % 50 == 0:
            extra = ""
            if cfg_obj is not None and cfg_obj.use_perceptual:
                extra += f" P:{lp_.item():.4f}"
                if cur_per_ch:
                    extra += " (" + " ".join(f"{k.split('_')[1]}={v:.3f}" for k, v in cur_per_ch.items()) + ")"
            if cfg_obj is not None and cfg_obj.use_frequency:     extra += f" Fq:{lfq_.item():.4f}"
            if cfg_obj is not None and cfg_obj.use_hierarchical:  extra += f" KL:{kl_val.item():.4f}"
            print(f"  E{ep} [{i+1}/{len(ld)}] L1:{ll_.item():.4f} Adv:{la_.item():.4f} "
                  f"FM:{lf_.item():.4f} D:{ld_.item()*acc:.4f}{extra}", flush=True)

    if n%acc!=0:
        if sc.is_enabled():
            sc.unscale_(oD); sc.unscale_(oG)
        nn.utils.clip_grad_norm_(D.parameters(),1.); nn.utils.clip_grad_norm_(G.parameters(),1.)
        sc.step(oD); sc.step(oG); sc.update()
        oD.zero_grad(set_to_none=True); oG.zero_grad(set_to_none=True)

    for k in r: r[k] /= max(n,1)
    return r


@torch.no_grad()
def val_ep(G, ld, dev):
    # Use uncompiled module for val: last batch may be smaller than training bs.
    Gv = _unwrap(G)
    Gv.eval(); s, n = 0., 0
    memfmt = torch.channels_last if (cfg.channels_last and cfg.arch_variant != "full_3d") else None
    for inp, tar, cm in ld:
        inp = _to_device(inp, dev, memfmt)
        tar = _to_device(tar, dev, memfmt)
        cm  = cm.to(dev, non_blocking=True)
        with _amp_ctx(dev):
            # Deterministic in val: temp=0 (ignored for non-hierarchical)
            out = Gv(inp, temp=0.0)
            if isinstance(out, tuple): out = out[0]
            s += masked_l1(out, tar, cm).item()
        n += 1
    return s / max(n,1)


@torch.no_grad()
def eval_vols(G, samples, dev, cfg_obj, save_dir=None, tag=""):
    G.eval(); oc = cfg_obj.out_ch
    m = {"ssim_t2":[], "psnr_t2":[], "mae_t2":[],
         "ssim_fl":[], "psnr_fl":[], "mae_fl":[]}

    for si, s in enumerate(samples):
        print(f"  Eval {si+1}/{len(samples)}: {s['name']} (T2={'Y' if s['has_t2'] else 'N'} FL={'Y' if s['has_fl'] else 'N'})")
        pred = run_inference(G, s, cfg_obj, dev)
        fg = s["fg"].astype(bool)

        if s["has_t2"]:
            ch = 0
            m["ssim_t2"].append(ssim_vol(s["t2"], pred[ch]))
            m["psnr_t2"].append(psnr_np(s["t2"], pred[ch]))
            if np.any(fg): m["mae_t2"].append(float(np.mean(np.abs(s["t2"][fg]-pred[ch][fg]))))

        if s["has_fl"] and oc > 1:
            ch = 1
            m["ssim_fl"].append(ssim_vol(s["flair"], pred[ch]))
            m["psnr_fl"].append(psnr_np(s["flair"], pred[ch]))
            if np.any(fg): m["mae_fl"].append(float(np.mean(np.abs(s["flair"][fg]-pred[ch][fg]))))

        if save_dir and si == 0:
            mid = s["us"].shape[2]//2
            nc = 5 if oc > 1 else 3
            fig, ax = plt.subplots(2, nc, figsize=(4*nc, 8))
            ax[0,0].imshow(s["us"][:,:,mid], cmap="gray", vmin=-1, vmax=1); ax[0,0].set_title("US")
            if s["has_t2"]: ax[0,1].imshow(s["t2"][:,:,mid], cmap="gray", vmin=-1, vmax=1); ax[0,1].set_title("T2 tgt")
            ax[0,2].imshow(pred[0][:,:,mid], cmap="gray", vmin=-1, vmax=1); ax[0,2].set_title("T2 pred")
            mh = s["us"].shape[0]//2
            ax[1,0].imshow(s["us"][mh,:,:], cmap="gray", vmin=-1, vmax=1); ax[1,0].set_title("US cor")
            if s["has_t2"]: ax[1,1].imshow(s["t2"][mh,:,:], cmap="gray", vmin=-1, vmax=1); ax[1,1].set_title("T2 tgt cor")
            ax[1,2].imshow(pred[0][mh,:,:], cmap="gray", vmin=-1, vmax=1); ax[1,2].set_title("T2 pred cor")
            if nc == 5 and s["has_fl"]:
                ax[0,3].imshow(s["flair"][:,:,mid], cmap="gray", vmin=-1, vmax=1); ax[0,3].set_title("FL tgt")
                ax[0,4].imshow(pred[1][:,:,mid], cmap="gray", vmin=-1, vmax=1); ax[0,4].set_title("FL pred")
                ax[1,3].imshow(s["flair"][mh,:,:], cmap="gray", vmin=-1, vmax=1); ax[1,3].set_title("FL tgt cor")
                ax[1,4].imshow(pred[1][mh,:,:], cmap="gray", vmin=-1, vmax=1); ax[1,4].set_title("FL pred cor")
            for a in ax.flat: a.axis("off")
            plt.suptitle(f"{cfg_obj.run_name} | {tag}"); plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"eval_{tag}.png"), dpi=150); plt.close()

    return {k: float(np.mean(v)) for k, v in m.items() if v}


# ============================================================================
# 13. MAIN
# ============================================================================
def run_phase(G, D, tl, vl, vs, dev, sc, nep, lrg, lrd, name, cfg_obj, resume=True,
              perceptual_net=None):
    acc = cfg_obj.accum_steps_3d if cfg_obj.arch_variant == "full_3d" else cfg_obj.accum_steps
    oG = torch.optim.Adam(G.parameters(), lrg, betas=(.5,.999), foreach=False)
    oD = torch.optim.Adam(D.parameters(), lrd, betas=(.5,.999), foreach=False)
    sG = torch.optim.lr_scheduler.CosineAnnealingLR(oG, nep, 1e-6)
    sD = torch.optim.lr_scheduler.CosineAnnealingLR(oD, nep, 1e-6)
    best, bp = float("inf"), os.path.join(cfg_obj.ckpt_dir, f"{name}_best.pth")
    lp = os.path.join(cfg_obj.ckpt_dir, f"{name}_latest.pth")
    start_ep = 1

    # --- Resume from latest checkpoint ---
    if resume and os.path.exists(lp):
        print(f"  [RESUME] Loading {lp} ...")
        ck = torch.load(lp, map_location=dev)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"])
        oG.load_state_dict(ck["oG"]); oD.load_state_dict(ck["oD"])
        sG.load_state_dict(ck["sG"]); sD.load_state_dict(ck["sD"])
        if "scaler" in ck and sc.is_enabled():
            sc.load_state_dict(ck["scaler"])
        best = ck.get("best", float("inf"))
        start_ep = ck["ep"] + 1
        print(f"  [RESUME] Resuming from epoch {start_ep}, best={best:.4f}")

    for ep in range(start_ep, nep+1):
        t0 = time.time()
        st = train_ep(G, D, tl, oG, oD, sc, dev, ep, acc,
                      cfg_obj.lambda_adv, cfg_obj.lambda_l1, cfg_obj.lambda_fm,
                      perceptual_net=perceptual_net, cfg_obj=cfg_obj)
        sG.step(); sD.step()
        extras = ""
        if cfg_obj.use_perceptual:
            extras += f" Perc:{st['perc']:.4f}"
            if cfg_obj.target_mode == "t2_flair" and (st.get("perc_t2",0) or st.get("perc_fl",0)):
                extras += f" (t2={st['perc_t2']:.4f} fl={st['perc_fl']:.4f})"
        if cfg_obj.use_frequency:    extras += f" Freq:{st['freq']:.4f}"
        if cfg_obj.use_hierarchical: extras += f" KL:{st['kl']:.4f}"
        print(f"  [{name}] E{ep}/{nep} G:{st['g']:.4f} D:{st['d']:.4f} L1:{st['l1']:.4f}{extras} | {time.time()-t0:.1f}s")

        if ep % cfg_obj.eval_every == 0:
            vl1 = val_ep(G, vl, dev)
            print(f"    ValL1: {vl1:.4f}")
            if vl1 < best:
                best = vl1
                torch.save({"G":G.state_dict(), "D":D.state_dict(), "ep":ep, "vl1":vl1}, bp)
                print(f"    [BEST] {vl1:.4f}")
            if ep % (cfg_obj.eval_every * 2) == 0:
                vm = eval_vols(G, vs, dev, cfg_obj, cfg_obj.ckpt_dir, f"{name}_e{ep}")
                print(f"    Vol: {vm}")

        # --- Save latest checkpoint every epoch for resume ---
        torch.save({"G":G.state_dict(), "D":D.state_dict(),
                     "oG":oG.state_dict(), "oD":oD.state_dict(),
                     "sG":sG.state_dict(), "sD":sD.state_dict(),
                     "scaler":sc.state_dict(), "ep":ep, "best":best}, lp)

    torch.save({"G":G.state_dict(), "D":D.state_dict()}, os.path.join(cfg_obj.ckpt_dir, f"{name}_final.pth"))
    # Clean up latest checkpoint after phase completes successfully
    if os.path.exists(lp): os.remove(lp)
    return bp


def main():
    pa = argparse.ArgumentParser("ResViT Ablation v2")
    pa.add_argument("--variant", choices=["2d","2.5d","2d_3d_refine","full_3d"])
    pa.add_argument("--target", choices=["t2","flair","t2_flair"])
    pa.add_argument("--phase1_epochs", type=int)
    pa.add_argument("--phase2_epochs", type=int)
    pa.add_argument("--triplanar", action="store_true", help="Enable triplanar (only for isotropic data)")
    pa.add_argument("--split_json", type=str, help="Path to subject_split.json")
    pa.add_argument("--val_fraction", type=float)
    pa.add_argument("--no_resume", action="store_true", help="Disable auto-resume, train from scratch")

    # --- Ablation flags ---
    pa.add_argument("--ablation", choices=["A","B","C","D","E"], default="A",
                    help="A=baseline | B=+perceptual | C=+frequency | D=+hierarchical | E=+seg")
    pa.add_argument("--nnunet_ckpt", type=str, help="Override path to nnU-Net checkpoint_best.pth")
    pa.add_argument("--lambda_perceptual", type=float)
    pa.add_argument("--lambda_frequency", type=float)
    pa.add_argument("--lambda_kl", type=float)
    pa.add_argument("--infer_samples", type=int, default=1, help="Average K samples at inference (for hierarchical)")
    pa.add_argument("--infer_temperature", type=float, default=0.0, help="Stochastic temperature at inference")
    pa.add_argument("--tag", type=str, default="", help="Suffix for run directory")
    pa.add_argument("--perc_mode", choices=["single_pass","split","split_ctx","late_only","late4_only"],
                    default="single_pass",
                    help="How to feed multi-channel targets into the nnU-Net perceptual encoder.")

    args, _ = pa.parse_known_args()

    if args.variant: cfg.arch_variant = args.variant
    if args.target: cfg.target_mode = args.target
    if args.phase1_epochs is not None: cfg.phase1_epochs = args.phase1_epochs
    if args.phase2_epochs is not None: cfg.phase2_epochs = args.phase2_epochs
    if args.triplanar: cfg.triplanar = True
    if args.split_json: cfg.split_json_path = args.split_json
    if args.val_fraction: cfg.val_fraction = args.val_fraction
    if args.nnunet_ckpt: cfg.nnunet_ckpt = args.nnunet_ckpt
    if args.lambda_perceptual is not None: cfg.lambda_perceptual = args.lambda_perceptual
    if args.lambda_frequency  is not None: cfg.lambda_frequency  = args.lambda_frequency
    if args.lambda_kl         is not None: cfg.lambda_kl         = args.lambda_kl
    cfg.infer_samples     = max(1, args.infer_samples)
    cfg.infer_temperature = max(0.0, args.infer_temperature)
    resume = not args.no_resume

    # Apply the ablation cascade (B⇒B, C⇒B+C, D⇒B+C+D, E⇒all)
    cfg.ablation = args.ablation
    for k, v in ablation_cascade(args.ablation).items():
        setattr(cfg, k, v)
    # Skip E unless we actually have seg masks (not used yet)
    if cfg.use_seg_guide:
        print("[WARN] ablation E requested but seg guidance is not yet implemented in v2 — disabling.")
        cfg.use_seg_guide = False

    # Perceptual mode (Windows variant only)
    cfg.perc_mode = args.perc_mode
    if cfg.perc_mode == "late_only":
        # Switch from early (0,1,2) to late semantic (3,4) stages.
        cfg.perceptual_layers = (3, 4)
    elif cfg.perc_mode == "late4_only":
        # Only the deepest semantic stage (4). Maximises abstraction, should
        # minimise cross-channel gradient leakage in multi-target synthesis.
        cfg.perceptual_layers = (4,)

    # Auto-suffix tag by perc_mode when user didn't provide one explicitly.
    auto_tag = args.tag
    if (not auto_tag) and cfg.perc_mode != "single_pass":
        auto_tag = f"win-{cfg.perc_mode}"
    cfg.run_tag = f"-abl{args.ablation}" + (f"-{auto_tag}" if auto_tag else "")

    seed_all(cfg.seed)
    dev = torch.device(cfg.device)

    # --- Global performance flags (torch 2.x) ---
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    print("=" * 70)
    print(f"ResViT Faithful | {cfg.arch_variant} | {cfg.target_mode}")
    print(f"Split: {cfg.split_json_path}")
    print(f"Device: {cfg.device}  torch={torch.__version__}")
    amp_dtype = "bf16" if cfg.use_bf16 else "fp16"
    print(f"AMP: {amp_dtype}  channels_last: {cfg.channels_last}  TF32: on")
    print("=" * 70)

    cfg.apply_gpu_profile()

    # --- Load split ---
    print("\n[1] Loading subject split...")
    train_names, test_names = load_split(cfg.split_json_path)
    print(f"  JSON: {len(train_names)} train volumes, {len(test_names)} test volumes")

    tr_names, va_names = split_train_val(train_names, cfg.val_fraction, cfg.seed)

    # --- Load volumes ---
    print("\n[2] Loading volumes...")
    tr_s = load_samples(tr_names, cfg, "TRAIN")
    va_s = load_samples(va_names, cfg, "VAL")
    te_s = load_samples(test_names, cfg, "TEST")

    if not tr_s:
        print("[ERROR] No training samples."); return

    # --- Build models ---
    print("\n[3] Building models...")
    G, D, R = build_all(cfg, dev)
    print(f"  G: {npar(G):,}  D: {npar(D):,}" + (f"  R: {npar(R):,}" if R else ""))

    # torch.compile (torch 2.x) — try to speed up inner loops.
    # Swin's dynamic mask / set_t toggles may cause recompiles; keep mode='default'.
    if cfg.use_compile and hasattr(torch, "compile") and dev.type == "cuda":
        try:
            G = torch.compile(G, mode="default", dynamic=False)
            D = torch.compile(D, mode="default", dynamic=False)
            print("  [compile] G & D wrapped with torch.compile")
        except Exception as e:
            print(f"  [compile] skipped: {e}")

    # --- Dataloaders ---
    print("\n[4] Dataloaders...")
    tl, vl = build_loaders(cfg, tr_s, va_s)

    # GradScaler only needed for fp16; bf16 has fp32-like dynamic range.
    scaler_enabled = (cfg.use_amp and dev.type == "cuda" and not cfg.use_bf16)
    try:
        sc = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except (TypeError, AttributeError):
        sc = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    # --- Perceptual network (module B) ---
    perceptual_net = None
    if cfg.use_perceptual:
        print(f"\n[4b] Building nnU-Net perceptual extractor:")
        print(f"     {cfg.nnunet_ckpt}")
        if not os.path.exists(cfg.nnunet_ckpt):
            print("  [WARN] nnU-Net checkpoint missing — disabling perceptual loss.")
            cfg.use_perceptual = False
        else:
            try:
                perceptual_net = NNUNetPerceptual(
                    cfg.nnunet_ckpt, layers=cfg.perceptual_layers, device=cfg.device,
                    n_input_channels=4, mri_clip_c=cfg.mri_clip, fg_thr=-0.9,
                    perc_mode=cfg.perc_mode,
                ).to(dev)
                print(f"  Perceptual net ready (layers={cfg.perceptual_layers}, "
                      f"lambda={cfg.lambda_perceptual}, mode={cfg.perc_mode})")
            except Exception as e:
                print(f"  [ERROR] Could not load nnU-Net perceptual net: {e}")
                print("  Disabling perceptual loss.")
                cfg.use_perceptual = False
                perceptual_net = None

    # Summary of active ablation modules
    print(f"\n[ABLATION] Level {cfg.ablation}")
    print(f"  Perceptual (nnU-Net):    {'ON' if cfg.use_perceptual else 'off'}")
    print(f"  Frequency (Laplacian):   {'ON' if cfg.use_frequency else 'off'}")
    print(f"  Hierarchical (StochBN):  {'ON' if cfg.use_hierarchical else 'off'}")
    print(f"  Seg guidance:            {'ON' if cfg.use_seg_guide else 'off (skipped)'}")
    print(f"  Inference temperature:   {cfg.infer_temperature}  samples={cfg.infer_samples}")

    # --- Phase 1 ---
    bp1 = os.path.join(cfg.ckpt_dir, "p1_best.pth")
    p1_final = os.path.join(cfg.ckpt_dir, "p1_final.pth")
    p1_latest = os.path.join(cfg.ckpt_dir, "p1_latest.pth")

    if resume and os.path.exists(p1_final):
        print(f"\n[5] Phase 1 — SKIPPED (p1_final.pth exists)")
    else:
        print(f"\n[5] Phase 1 — CNN only — {cfg.phase1_epochs} ep")
        # If resuming mid-phase, G/D state is restored inside run_phase.
        # If starting fresh, ensure transformer is off.
        if not (resume and os.path.exists(p1_latest)):
            G.set_trans(False)
        else:
            G.set_trans(False)  # Phase 1 is always CNN-only
        bp1 = run_phase(G, D, tl, vl, va_s, dev, sc,
                        cfg.phase1_epochs, cfg.phase1_lr_g, cfg.phase1_lr_d, "p1", cfg, resume,
                        perceptual_net=perceptual_net)

    # --- Phase 2 ---
    bp2 = os.path.join(cfg.ckpt_dir, "p2_best.pth")
    p2_final = os.path.join(cfg.ckpt_dir, "p2_final.pth")
    p2_latest = os.path.join(cfg.ckpt_dir, "p2_latest.pth")

    if resume and os.path.exists(p2_final):
        print(f"\n[6] Phase 2 — SKIPPED (p2_final.pth exists)")
    else:
        print(f"\n[6] Phase 2 — Full ART — {cfg.phase2_epochs} ep")
        # Load P1 best weights only if NOT resuming mid-P2
        if not (resume and os.path.exists(p2_latest)):
            if os.path.exists(bp1):
                ck = torch.load(bp1, map_location=dev)
                G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"])
                print(f"  Loaded P1 best vl1={ck.get('vl1','?')}")
        G.set_trans(True)
        bp2 = run_phase(G, D, tl, vl, va_s, dev, sc,
                        cfg.phase2_epochs, cfg.phase2_lr_g, cfg.phase2_lr_d, "p2", cfg, resume,
                        perceptual_net=perceptual_net)

    # --- Refinement ---
    if cfg.arch_variant == "2d_3d_refine" and R:
        if os.path.exists(bp2):
            G.load_state_dict(torch.load(bp2, map_location=dev)["G"])
        print(f"\n[6b] 3D Refinement — {cfg.refine_epochs} ep")
        G.eval(); opt = torch.optim.Adam(R.parameters(), cfg.refine_lr, foreach=False); oc = cfg.out_ch
        for ep in range(1, cfg.refine_epochs+1):
            R.train(); ls, n = 0., 0
            for s in tr_s:
                pred = run_inference(G, s, cfg, dev)
                tgts = _get_targets(s, cfg.target_mode)
                tgt_np = np.stack(tgts, 0)  # (C, H, W, D)
                pv = torch.from_numpy(pred.transpose(0,3,1,2)[None]).float().to(dev)
                tv = torch.from_numpy(tgt_np.transpose(0,3,1,2)[None]).float().to(dev)
                cm = torch.from_numpy(s["tmask"][None]).float().to(dev)
                loss = masked_l1(R(pv), tv, cm)
                opt.zero_grad(); loss.backward(); opt.step()
                ls += loss.item(); n += 1
            print(f"  [Ref] E{ep} L1={ls/max(n,1):.4f}")
        torch.save(R.state_dict(), os.path.join(cfg.ckpt_dir, "refine.pth"))

    # --- Final eval ---
    print("\n[EVAL] Final on test set...")
    if os.path.exists(bp2):
        G.load_state_dict(torch.load(bp2, map_location=dev)["G"])
    fm = eval_vols(G, te_s, dev, cfg, cfg.pred_dir, "test_final")
    print(f"  TEST: {fm}")

    # Also val metrics
    fmv = eval_vols(G, va_s, dev, cfg, cfg.pred_dir, "val_final")
    print(f"  VAL:  {fmv}")

    # --- Save predictions ---
    print("\n[SAVE] Test predictions...")
    G.eval()
    for s in te_s:
        pred = run_inference(G, s, cfg, dev)
        ob = os.path.join(cfg.pred_dir, s["name"]); os.makedirs(ob, exist_ok=True)
        save_nii(s["us"], os.path.join(ob, "us.nii.gz"), s["aff"])
        if s["has_t2"]:
            save_nii(s["t2"], os.path.join(ob, "tgt_t2.nii.gz"), s["aff"])
            save_nii(pred[0], os.path.join(ob, "pred_t2.nii.gz"), s["aff"])
        if s["has_fl"] and cfg.out_ch > 1:
            save_nii(s["flair"], os.path.join(ob, "tgt_fl.nii.gz"), s["aff"])
            save_nii(pred[1], os.path.join(ob, "pred_fl.nii.gz"), s["aff"])
        print(f"  Saved {s['name']}")

    # --- Save results summary ---
    summary = {"run": cfg.run_name, "variant": cfg.arch_variant, "target": cfg.target_mode,
               "test_metrics": fm, "val_metrics": fmv,
               "n_train": len(tr_s), "n_val": len(va_s), "n_test": len(te_s),
               "ngf": cfg.ngf, "phase1_ep": cfg.phase1_epochs, "phase2_ep": cfg.phase2_epochs}
    with open(os.path.join(cfg.pred_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}\n[DONE] {cfg.run_name}\n  TEST: {fm}\n  {cfg.pred_dir}\n{'='*70}")


if __name__ == "__main__":
    main()

# 2D, solo T2, split del MMHVAE
#python resvit_final.py --variant 2d --target t2

# 2D + refinement 3D, T2+FLAIR con targets parciales
#python resvit_final.py --variant 2d_3d_refine --target t2_flair

# Full 3D en la 4080 SUPER (auto-ajusta a patch 64³, ngf=32, grad ckpt)
#python resvit_final.py --variant full_3d --target t2

#python resvit_final.py --variant full_3d --target t2_flair