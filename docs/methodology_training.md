# Methodology — US→MRI synthesis benchmark (ReMIND)

*Single reference document for the paper: data, models, training/inference compute, metric protocol, and where every number lives. Last updated May 2026.*

---

## 1. Task & data

- **Task:** synthesize MRI (T2-weighted; and, in the dual-target setting, T2-FLAIR) from intra-operative B-mode ultrasound. Single-target experiments output 1 channel (T2); dual-target experiments output 2 channels (T2 + FLAIR), trained with a masked loss for the subjects that lack FLAIR.
- **Dataset:** ReMIND, paired US / MR-T2 / MR-FLAIR, rigidly registered and cropped (`dataset-registration-corrected-cropped` / `3D_pairs_whole_MR_US_resampled_cropped`). Each subject has a pre-resection and a post-resection acquisition (`-pre` / `-post`).
- **Split** (`subject_split.json`): **61 train subjects → 122 train studies, 16 test subjects → 31 test studies** (one study lacks FLAIR/post). SynDiff used a slightly different list (114 train / 16 test subjects). Validation is carved at the slice level (~1 k slices) from the *train* subjects (ResViT: 15 % of train subjects); the test split is never touched for model selection.
- **Test sets reported:** **n = 30 test studies** for T2-only experiments; **n = 20 test studies** for dual T2+FLAIR experiments (the subset with FLAIR ground truth) — for those 20, both the T2 and the FLAIR channel are scored.
- **Pre-processing:** US — foreground percentile normalization (p2/p98 on the foreground) → [-1, 1]; MR — z-score, clip at ±3σ → [-1, 1]; background set to −1.0; foreground mask = `volume > 1 % of max`.
- **Augmentation (training only):** horizontal flip p = 0.5, vertical flip p = 0.5, rotation p = 0.3; US-only intensity jitter (γ + additive Gaussian noise, σ ≤ 0.03) p = 0.3–0.4; the same spatial transform is applied to US and MR.
- **Hardware:** all training and the GAN inference timing on **NVIDIA RTX 4080 SUPER, 16 GB** (the original ResViT 2D/2.5D/3D-refine/full-3D runs were on a Linux box reporting ~15.6 GB, "16 GB profile"; a couple of later ResViT retrains used an RTX 3090 24 GB). The standalone PyTorch compute benchmark (params/FLOPs/latency/VRAM) was re-measured on an **RTX 3090 24 GB** — note this differs from the training GPU (≈ ±20 % on these workloads).
- **Frameworks:** GAN baselines — TensorFlow/Keras 3 (project logs: TF 2.20.0, cuDNN 9). ResViT & SynDiff — PyTorch 2.5.1 + cu121, bf16 AMP, TF32 on.

---

## 2. Methods & architectures (no ablations; ensembles computed but **excluded from final tables**)

Three backbone families, each evaluated across four architecture variants and two targets:

| Variant | Meaning |
|---|---|
| **2D** | slice-wise model; at test time predicted along the axial axis (ResViT/SynDiff) or "triplanar" = predicted along all 3 axes and averaged (GAN baselines) |
| **2.5D** | same as 2D but the input is 3 adjacent axial slices stacked as channels |
| **2D + 3D-post / 2D + 3D-refine** | a small 3D residual network refines the assembled 3D volume produced by the frozen 2D model |
| **3D** | fully volumetric model trained on 3D patches, sliding-window inference |

### 2.1 GAN baselines (`COMPARATIVA-3/`): pix2pix, CUT, CycleGAN, SwinPix2Pix
- **2D generators** (input 192×192): pix2pix = Attention Res-U-Net (enc 64-128-256-512-512 + 1 res block, attention-gated decoder, dropout 0.5); CUT = U-Net with PatchNCE feature hooks (5 feature levels, NCE MLP out-dim 128); CycleGAN = ResNet-9-residual-block generator with InstanceNorm, **two generators** G_AB/G_BA; SwinPix2Pix = `SwinGenerator2D` (PatchEmbed patch 2 / embed 36, 3 Swin stages depths (2,2,2) heads (6,6,12) window 4, conv bottleneck, 3 Swin decoder blocks). **2.5D** = same with `input_channels = 3`.
- **3D generators** (patch 64×64×32, overlap 32×32×16): pix2pix-3D = 3D Attention Res-U-Net (base 32, 4 down + res block); SwinPix2Pix-3D reuses the pix2pix-3D generator (no 3D Swin); CycleGAN-3D = 3D ResNet-6-block gen base 32, two generators; CUT-3D = 3D CUT gen base 32 + 3D PatchNCE MLPs.
- **Discriminators:** PatchGAN, 4×4 convs, LeakyReLU 0.2, spectral norm on every conv, R1 penalty λ = 10 every 16 steps; multi-scale (2 scales for 2D, 1 for 3D). pix2pix/SwinPix2Pix use a *conditional* disc (input = US ⊕ target); CUT/CycleGAN an *unconditional* disc; CycleGAN has two (D_A, D_B).
- **2D+3D-post pipeline:** (1) load the frozen `{family}_2d_{target}` generator (EMA weights); (2) run it with triplanar inference on all 122 train studies → 3D predicted volumes; (3) train `build_3d_refiner` (a ~160 k-param 3D residual U-Net: enc Conv3D 16→32, bottleneck 2× res-block(32), dec, output = input + Conv3D residual clipped to [-1,1]) for 10 000 steps, batch 1, on 64×64×32 patches sampled from (predicted → ground-truth) pairs, loss = 10·L1 + 8·SSIM3D, Adam lr 1e-4, EMA 0.999; (4) test = triplanar 2D prediction → sliding-window refiner pass → metrics + NIfTI.

### 2.2 ResViT (`resvit/`)
Faithful ResViT (ART = aggregated residual transformer blocks). **Two-phase training**: Phase 1 CNN-only, Phase 2 transformer branches enabled. Generator: encoder ReflectionPad+Conv7×7 (→ngf) + 2 stride-2 convs → bottleneck at H/4, 256 ch; **9 ART blocks**, a Swin transformer branch active at ART blocks 4 & 5 (`swin_layer_pairs = 2` → 4 attention blocks each, 8 heads, window 8 [2D] / 4 [3D], mlp ratio 4); decoder 2× ConvTranspose s2 → Conv7×7 → Tanh. `ngf = 64` (2D/2.5D, image 256²), `ngf = 48` (full-3D, 96³ patches, overlap 24³). Discriminator: PatchGAN, ndf = 64 (2D) / 48 (3D), 3 layers, InstanceNorm, used with feature-matching loss. **2D+3D-refine:** after Phase 1+2, freeze the 2D generator, assemble 3D volumes from its 2D predictions, train `Refine3D` (4× Conv3D 3×3, 32 ch, Tanh, `out = x + 0.1·net(x)`, ~57 k params) for 30 epochs, Adam lr 1e-4.

### 2.3 SynDiff (`synthdiff/`) — 4-step adversarial-diffusion
- **Stage-1 generator** = a *shrunk* NCSNpp (`ncsnpp_generator_adagn`): image 256², `nf = ngf = 32` (vs 64 in upstream SynDiff → ≈ 12 % of full size), `ch_mult = [1,1,2,2,4,4]`, 2 res blocks/level, attention @16², BigGAN res blocks, FIR, positional embedding, latent z dim 100, **`num_timesteps = 4`** (4 reverse diffusion steps), VP schedule β∈[0.1, 20]. Discriminator = `Discriminator_large`, time-conditioned, ngf 32 (≈ 7.2 M params). Channel configs: 2D-T2 → gen 2 ch / disc nc 2; 2.5D-T2 → gen 4; 2D-dual → gen 3; 2.5D-dual → gen 5 / disc nc 4.
- **Full-3D**: `NCSNpp3D` — pure Conv3d/GroupNorm, no FIR, 64³ patches, `ngf = 24`, `ch_mult = [1,2,2,2]`, attention @8³, 4 timesteps, gradient checkpointing. Disc = 3D `Discriminator_large`, ngf 24 (≈ 9.1–9.3 M params).
- **3D-refine cascade** (`paired_3drefine`): a 6-block 3D-ResNet refiner (`Refiner3D`, ngf 24, ≈ 0.21 M params) trained on the frozen Stage-1 (2D-diffusion) predictions assembled into volumes; 3D PatchGAN disc (≈ 1.56 M). *(In `rescore_methods_*` this is labelled "3D+3D-refine" but it is in fact 2D-diffusion-Stage-1 + 3D-ResNet-refiner — annotated in `table_compute.csv`.)*
- **ResViT→SynDiff cascade** (`paired_resvit_refine`): a SynDiff diffusion refiner (`num_channels = 3`: US ⊕ coarse-T2 ⊕ noisy-T2) trained on top of a **frozen** ResViT-2.5D-T2 coarse prediction.
- **ResViT+SynDiff joint** (`joint_finetune`): the ResViT-2.5D Stage-1 (≈ 18.3 M) and the SynDiff diffusion refiner (≈ 12.1 M gen / 7.2 M disc) fine-tuned **end-to-end**.
- **Untuned full SynDiff** (`syndiff_us_t2`, upstream code, ngf 64, CycleGAN-style with cycle losses, ≈ 156 M generator-equivalent) — reference baseline; ⚠ only 4 test subjects have saved volumes, so its row carries n = 4.
- **Ensembles** (`ensemble_*`): per-voxel weighted average of a trained ResViT-2.5D and a trained SynDiff-2.5D (α = 0.5). **Computed but excluded from the final paper tables** at the authors' request (to avoid confusion); the raw numbers can be regenerated from `evaluacion-final/rescore_all.py`.

---

## 3. Training configuration & wall-clock per model

| Family | Optimizer / LR | Schedule | Budget | Batch | Loss weights | Train wall-clock |
|---|---|---|---|---|---|---|
| GAN — pix2pix / SwinPix2Pix | Adam β1=0, β2=0.999, clipnorm 1; lr_G 2e-4, lr_D 5e-5 | warmup 1 k, EMA 0.999 | 2D/2.5D **20 k steps**; 3D **50 k steps**; 3D-refiner **10 k steps** | 16 (2D/2.5D); 1 (3D, patch) | L1 10, SSIM 8, FM 2, edge 5, R1 10 (every 16) | pix2pix-2D/2.5D ≈ **0.8 h**; SwinPix2Pix-2D/2.5D ≈ **1.2 h**; both 3D ≈ **4.7–5.0 h**; 2D+3D-post adds ≈ 15 min–1.5 h (mostly train-set prediction generation) |
| GAN — CUT | Adam, lr_G=lr_D 2e-4 | EMA 0.999 | as above | 16 (2D/2.5D); 1 (3D) | NCE 1 (256 patches, τ 0.07), idt 0.5, paired-sup 10, SSIM 8, edge 2, R1 10 | CUT-2D/2.5D ≈ **0.6 h**; CUT-3D ≈ **3.6–4.0 h** |
| GAN — CycleGAN | Adam, lr_G=lr_D 2e-4 | EMA 0.999 | as above | **4** (2D/2.5D, → 2 on OOM); 1 (3D) | cycle 10, idt 5 (off for 2.5D/dual), paired-sup 10, SSIM 8, R1 10 | CycleGAN-2D/2.5D ≈ **4.5–5.0 h** (batch limit + 2 G + 2 D); CycleGAN-3D ≈ **9.6–12.4 h** |
| GAN — 3D-post refiner | Adam lr 1e-4, clipnorm 1 | EMA 0.999 | 10 k steps | 1 | 10·L1 + 8·SSIM3D | ≈ **15 min** (≈ 0.25 h), on top of the frozen 2D model |
| ResViT | Adam β=(0.5,0.999); Phase1 lr 2e-4, Phase2 lr 1e-4 | CosineAnnealing → 1e-6 per phase; grad-clip 1 | Phase1 **100 ep** + Phase2 **100 ep** (+ 30-ep 3D-refine for the refine variant) | 8 (2D/2.5D, 16 GB profile); 1 (3D, grad-accum 2–8, gradient ckpt) | L1 100, LSGAN adv 1, feature-matching 10 | ResViT-2D ≈ **5.0 h** (T2) / 4.3 h (dual); ResViT-2.5D ≈ **3.85 h** / 3.9 h; ResViT-2D+3D-refine ≈ **5.5 h** / 7–8 h; ResViT-3D ≈ **6.0 h** / 6.5 h |
| SynDiff | Adam β1=0.5, β2=0.9; lr_G 1.6e-4, lr_D 1e-4 | CosineAnnealing → 1e-5; EMA G 0.999; R1 γ=1 lazy (every 64 / every 32 for 3D) | 2D/2.5D up to **300 ep** (≈ 200 used); dual **200 ep**; full-3D **200 ep**; 3D-refiner **100 ep** (on frozen Stage-1); ResViT-refine diffusion 200 (≈ 100 used); joint-finetune **50 ep** (lr_resvit 5e-6, lr_diff 2e-4); untuned full SynDiff 80 ep | 8 (2D/2.5D); 1 (full-3D, 64³, 4 patches/vol) | L1 10, adv 1 | SynDiff-2D ≈ **13.1 h** (T2) / 11.2 h (dual); SynDiff-2.5D ≈ **13.2 h** / 6.9 h; SynDiff-full-3D ≈ **5.4 h** / 4.7 h; SynDiff-3D-refine ≈ **+4.0 h** on frozen Stage-1; ResViT→SynDiff cascade ≈ **6.5 h** (+ ResViT 3.85 h prereq); joint-finetune ≈ **4.7 h** (+ prereqs); untuned full SynDiff — not separately timed; ensembles — **0** (combine pretrained models) |

Per-epoch reference (RTX 4080 SUPER, bf16): SynDiff-2D ≈ 236 s/ep × 736 it; SynDiff-2.5D ≈ 237 s; SynDiff-full-3D ≈ 97 s × 416 it; SynDiff-3D-refiner ≈ 145 s; joint-finetune ≈ 337 s (13.5 GB peak). ResViT-2.5D ≈ 46.8 s/ep (Phase 1) / 87.9 s/ep (Phase 2), 694 it/ep @ bs 8.

---

## 4. Inference compute per model

Measured forward-pass latency / FLOPs / peak VRAM for the PyTorch models on an RTX 3090 (10 warm-up + 30 timed iterations, CUDA events, bf16 autocast, FLOPs via `torch.utils.flop_counter.FlopCounterMode`); per-volume time derived from a representative test volume (≈ 50 foreground axial slices; 3D ↔ patch count over a ≈ 128³ volume). GAN inference latency is taken from the original RTX 4080 SUPER eval-log timestamps (includes I/O / resampling — not an isolated forward); GAN params are confirmed from the training logs (generators) + a TF re-instantiation (discriminators, CUT-3D generator, FLOPs — see `gan_compute_tf.csv` / `gan_compute_tf.md`).

| Model | Gen params | Disc params | GFLOPs / fwd (gen ; disc) | Peak VRAM | **Inference / volume** |
|---|---|---|---|---|---|
| pix2pix-2D / 2.5D | 37,022,817 / 37,024,865 | 5,531,012 (cond, 2-scale) | ≈ 22.1–22.2 ; 4.3 | — | ≈ **25 s** (triplanar) |
| CUT-2D / 2.5D | 37,090,593 / 37,096,865 | 1,323,396 (uncond, 2-scale) | ≈ 23.6–24.0 ; 1.5 | — | ≈ **25 s** |
| CycleGAN-2D / 2.5D | 11,370,881 ×2 / 11,377,153 ×2 | 1,323,396 ×2 (1,327,492+1,323,396 at 2.5D) | ≈ 63.1–63.6 ×2 ; 1.5 ×2 | — | ≈ **16 s** |
| SwinPix2Pix-2D / 2.5D | 1,444,120 / 1,445,056 | 5,531,012 (cond, 2-scale) | ≈ 8.4 ; 4.3 | — | ≈ **30 s** |
| pix2pix-2D+3D-post (= 2D gen + 160,289 refiner) | 37,183,106 | 5,531,012 | 22.1 (gen) + 3.4 (refiner) | — | ≈ **25 s + 3D-refiner sliding-window** |
| (CUT / CycleGAN / SwinPix2Pix 2D+3D-post — add the 160,289 refiner, +3.4 GFLOPs) | | | | | |
| pix2pix-3D / SwinPix2Pix-3D | 13,967,377 | 11,053,762 (cond, 1-scale 3D) | 13.0 (64³) ; 6.0 | — | sliding-window 64×64×32 patches |
| **CUT-3D** | **23,629,057** (T2) / 23,640,034 (dual) — *same 3D-ResNet backbone as CycleGAN-3D* | 2,643,138 (uncond, 1-scale 3D) | 46.2 (64³) ; 3.4 | — | sliding-window 64×64×32 patches |
| CycleGAN-3D | 23,629,057 ×2 | 2,643,138 ×2 | 46.2 ×2 (64³) ; 3.4 ×2 | — | sliding-window 64×64×32 patches |
| ResViT-2D | 18,247,489 | 2,763,713 | 157.1 | 0.16 GB | ≈ **0.70 s** (≈ 1.26 s @ ~90 sl) |
| ResViT-2.5D | 18,253,761 | 2,765,761 | 157.9 | 0.17 GB | ≈ **0.71 s** |
| ResViT-2D+3D-refine | 18.30 M (+ 57 k Refine3D) | 2,763,713 | 157.1 + 239.2 (refine 128³) | 0.16 / 0.55 GB | ≈ **0.72 s** (+1 Refine3D pass) |
| ResViT-3D (ngf 48, 96³) | 23,778,113 | 6,224,593 | 800.1 / patch | 0.38 GB | ≈ **0.39 s** (8× 96³ patches) |
| SynDiff-2D | 12,089,442 | 7,216,705 | 286.0 / fwd → **1144 / 4-step sample** | 0.66 GB | ≈ **2.15 s** (b8 slices; ≈ 3.69 s @ ~90 sl) |
| SynDiff-2.5D | 12,091,172 | 7,216,705 | 287.4 / fwd → 1150 / sample | 0.67 GB | ≈ **2.16 s** |
| SynDiff-full-3D (ngf 24, 64³) | 5,421,921 | 9,119,089 | 254.2 / fwd → 1017 / sample / patch | 0.36 GB | ≈ **4.89 s** (27× 64³ patches) |
| SynDiff-3D-refine (2D-diffusion + 0.21 M Refiner3D) | 12.30 M | 8.78 M | 1144 / sample + 886 (refiner 128³) | 0.66 / 1.05 GB | ≈ **2.31 s** (2D coarse + 1 Refiner3D) |
| SynDiff ResViT→SynDiff cascade | 18.25 M (ResViT, frozen) + 12.09 M (diffusion refiner) | 2.77 M + 7.22 M | 157.9 + 1144 / sample | 0.67 GB | ≈ **2.86 s** (ResViT 0.71 s + diffusion refiner ≈ 2.15 s) |
| SynDiff ResViT+SynDiff joint | same ≈ 30.3 M | 9.98 M | 157.9 + 1144 / sample | 0.67 GB | ≈ **2.86 s** |
| SynDiff (full, untuned) | ≈ 156 M | — | not measured | — | not measured (~10–15 s/vol est.) |

(Full numbers, per target and per channel, in `evaluacion-final/master_benchmark_table.csv` and `evaluacion-final/table_compute.csv`; the TF-measured GAN FLOPs and the CUT-3D generator count in `paper_assets/gan_compute_tf.csv`.)

---

## 5. Evaluation protocol (the **single** metric protocol used everywhere)

This is the "resvit-protocol", implemented in `resvit/eval_resvit_metrics.py` and `synthdiff/eval_resvit_protocol*.py`, and re-applied uniformly to every method by `evaluacion-final/rescore_all.py`. It matches the conventions of the cross-modality synthesis literature (pGAN/cGAN — Dar et al., IEEE TMI 2019; ResViT — Dalmaz et al., IEEE TMI 2022; SynDiff — Özbey et al., IEEE TMI 2023).

For each (method, subject), load the saved prediction volume and its ground-truth volume (both stored normalized to [-1, 1]); let `t01 = (gt + 1)/2`, `p01 = (pred + 1)/2` (i.e. renormalized to [0, 1]):
- **SSIM** — for each axial slice z, skip slices with `mean(t01[:,:,z] > 0.025) < 0.01` (background slices); compute `structural_similarity(t01_z, p01_z, data_range = 1.0)` with scikit-image's default 7×7 uniform window; SSIM(subject) = mean over the kept slices. An additional column `ssim_gauss` uses the Wang-2004 settings (`gaussian_weights = True, sigma = 1.5, use_sample_covariance = False`).
- **PSNR** — foreground mask `fg = t01 > 0.025`; if `fg.sum() < 100` the volume is skipped; `MSE = mean((t01[fg] − p01[fg])²)`; `PSNR = 10·log10(1.0 / MSE)` (i.e. `data_range = 1`).
- **MAE** — `mean(|t01[fg] − p01[fg]|)`.
- **LPIPS** — for each kept axial slice, take the [-1, 1] slice, replicate to 3 channels, evaluate `lpips.LPIPS(net='alex')` (AlexNet trunk, v0.1) on GPU; LPIPS(subject) = mean over the kept slices.
- **Aggregation** — mean ± SD, 95 % CI (`1.96·SD/√n`), median, min, max — over the **test subjects** (n = 30 for T2, n = 20 for T2+FLAIR), not over slices.
- **Significance** — paired **Wilcoxon signed-rank** (`scipy.stats.wilcoxon`) on the per-subject metric, using only the subjects present in both members of a pair: (i) every method vs. the single best-SSIM method; (ii) within each backbone family, the 2.5D / 2D+3D-refine / 2D+3D-post / 3D variant vs. that backbone's plain-2D variant (architecture-effect test). Results in `evaluacion-final/rescore_wilcoxon.csv` / `table_wilcoxon_vs_best.csv` / `table_wilcoxon_architecture_effect.csv`.
- **Pre-op vs post-op stratification** — the same per-subject metrics, partitioned by the subject suffix (`-pre` / `-post`); summaries in `evaluacion-final/rescore_preop_summary.csv`, `rescore_postop_summary.csv`, compact tables `table_results_{t2,flair}_{preop,postop}.csv`, and a side-by-side delta table `table_preop_vs_postop_delta.csv`. Post-op studies are uniformly slightly harder (lower SSIM/PSNR, higher MAE) due to resection cavities and tissue shift.

**Deprecated alternative:** an earlier "slice-protocol" computed SSIM over the whole volume including background air → inflated SSIM (~0.81 vs the masked ~0.73 for the same model). It is not used; the foreground-masked protocol above is the only one reported.

---

## 6. Headline findings (under the unified protocol)

- **GANs (especially 2D+3D-post) win SSIM / PSNR / MAE** (best SSIM ≈ 0.814 T2 / 0.789 FLAIR; pix2pix-2D+3D-post is the top method on all three pixel metrics for both channels). All four GAN families occupy ranks 1–32 (T2) / 1–16 (FLAIR).
- **ResViT and SynDiff win LPIPS** (≈ 0.166–0.205 vs ≈ 0.20–0.28 for the GANs) but sit at SSIM ≈ 0.70–0.73 — a clear fidelity-vs-perceptual trade-off (Pareto). ResViT-3D / ResViT-2.5d give the lowest LPIPS; ResViT-2D+3D-refine the best SSIM among the transformer/diffusion methods.
- **Architecture effects** (paired Wilcoxon, per-subject SSIM, T2): 2D+3D-post ≫ 2D for every GAN family (p ≈ 2×10⁻⁹); 2.5D > 2D for the GANs (p ≈ 0.02–10⁻⁸) but **not significant** for ResViT (p ≈ 0.10) or SynDiff (p ≈ 0.81); full-3D < 2D for ResViT and SynDiff (3D hurts SSIM on this small cohort); ResViT-2D+3D-refine ≫ ResViT-2D (p ≈ 2×10⁻⁹). Cascading SynDiff onto a ResViT coarse prediction (cascade / joint fine-tune) recovers most of the gap to the ResViT-2.5d baseline (PSNR ≈ 15.2, LPIPS ≈ 0.167) and is far better than the untuned full SynDiff (PSNR ≈ 13.1, LPIPS ≈ 0.211).
- **Compute:** GAN-2D/2.5D ≈ 0.6–1.2 h to train and ≈ 16–30 s/volume to run; ResViT ≈ 3.9–8 h and < 1.3 s/volume; SynDiff-2D/2.5D ≈ 11–13 h and ≈ 2–4 s/volume (4 sampling steps — not a 1000-step DDPM). Refiners (3D-post, 3D-refine, ResNet3D refiner) and joint fine-tune are all cheap (≈ 15 min – 5 h).

---

## 7. File map

- `paper_assets/METHODOLOGY.md` — this document.
- `paper_assets/benchmarking_report.md` — long-form compendium (architectures, training config, results, §7 protocol recommendation).
- `paper_assets/compute_benchmark.md`, `_bench_results.json`, `_bench_run.py` — measured ResViT/SynDiff params/FLOPs/latency/VRAM/per-volume inference.
- `paper_assets/gan_compute_tf.csv`, `gan_compute_tf.md`, `gan_compute_tf.py` — TF-measured GAN generator/discriminator params + FLOPs (incl. the CUT-3D generator count).
- `evaluacion-final/rescore_all.py` → `rescore_methods_persubject.csv`, `rescore_methods_summary.csv`, `rescore_methods_rankings_{t2,flair}.csv`, `rescore_wilcoxon.csv`, `RESCORE_README.md` — unified re-score of all methods (ensemble-free after `build_paper_tables.py`).
- `evaluacion-final/build_paper_tables.py` → `master_benchmark_table.csv` / `_flair.csv` (metrics + compute + train time), `table_results_{t2,flair}.csv`, `table_compute.csv`, `table_wilcoxon_{vs_best,architecture_effect}.csv`, `rescore_{preop,postop}_summary.csv`, `table_results_{t2,flair}_{preop,postop}.csv`, `table_preop_vs_postop_delta.csv`.
- `evaluacion-final/DIFF_vs_initial.md` — how these differ from the original `synthesis_*.xlsx` (which were already protocol-consistent and ablation-free; the new files add pre/post stratification, the compute join, Wilcoxon stats, and the `ssim_gauss` column, and **exclude ensembles**).
- Original (still on disk, superseded): `evaluacion-final/synthesis_perSubject_FULL.xlsx`, `synthesis_rankings_FULL.xlsx`, `synthesis_rankings_PIVOT.xlsx`.
- Tools: xlsx reader without openpyxl → `a small local xlsx reader`. Eval/benchmark python env → `$IOUS2MR_PY_TORCH (see envs/environment-pytorch.yml)` (torch+cuda, nibabel, skimage, lpips, pandas). TF env for GAN compute → pip venv `$IOUS2MR_ROOT\paper_assets\tf_bench_venv\Scripts\python.exe` (TensorFlow 2.21.0 + tf-keras 2.21.0, `TF_USE_LEGACY_KERAS=1`; conda was blocked, so a plain venv was used).