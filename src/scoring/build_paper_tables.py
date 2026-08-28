# -*- coding: utf-8 -*-
"""Fill compute gaps + emit paper-ready CSV tables. Run with the mmhvae python."""
import os, csv, math
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
summ = pd.read_csv(os.path.join(HERE, "rescore_methods_summary.csv"))

# ----------------------------------------------------------------- drop ensembles
def _is_ens(df):
    return df["method"].astype(str).str.contains("Ensemble") | (df["backbone"].astype(str) == "ResViT+SynDiff")
summ = summ[~_is_ens(summ)].reset_index(drop=True)
summ.to_csv(os.path.join(HERE, "rescore_methods_summary.csv"), index=False)
# persubject (also overwrite, ensemble-free)
PS = pd.read_csv(os.path.join(HERE, "rescore_methods_persubject.csv"))
PS = PS[~_is_ens(PS)].reset_index(drop=True)
PS.to_csv(os.path.join(HERE, "rescore_methods_persubject.csv"), index=False)
# rankings (ensemble-free, re-ranked)
for ch, fn in [("t2","rescore_methods_rankings_t2.csv"), ("flair","rescore_methods_rankings_flair.csv")]:
    rk = pd.read_csv(os.path.join(HERE, fn))
    rk = rk[~_is_ens(rk)].sort_values("ssim_mean", ascending=False).reset_index(drop=True)
    rk["ssim_rank"] = range(1, len(rk)+1)
    rk.to_csv(os.path.join(HERE, fn), index=False)

# ---------------------------------------------------------------- compute lookup
# Keyed by (backbone, architecture, targetkind) where targetkind in {"single","dual"}.
# Params: GAN gen counts from training logs; GAN disc counts analytic (PatchGAN, see
# paper_assets/compute_benchmark.md); ResViT/SynDiff measured (paper_assets/_bench_results.json).
# gflops_fwd: GFLOPs per generator forward (FlopCounterMode). For diffusion add "(x4 sample)".
# infer_s_vol: per-volume inference seconds (median ~50 fg slices for 2D/2.5D; patch count for 3D).
# train_h: training wall-clock hours (see paper_assets/benchmarking_report.md sec 4).
# GAN discriminator trainable params, MEASURED in TensorFlow (count_params, incl. LayerNorm + tiny
# non-trainable spectral-norm u buffers); see paper_assets/gan_compute_tf.csv / .md.
GAN_DISC = {  # cond 2-scale 2D / uncond 2-scale 2D / cond 1-scale 3D / uncond 1-scale 3D
    "cond2d_s":  5_531_012, "cond2d_d":  5_533_060,
    "unc2d_s":   1_323_396, "unc2d_d":   1_325_444,
    "cond3d_s": 11_053_762, "cond3d_d": 11_057_858,
    "unc3d_s":   2_643_138, "unc3d_d":   2_647_234,
}
# CycleGAN has D_A + D_B; at 2.5D D_A takes a 3-ch US input (+4096 params -> 1,327,492); at 3D the same.
CG_DD = {  # (arch, kind) -> D_A + D_B total
    ("2D","single"):1_323_396+1_323_396, ("2D","dual"):1_323_396+1_325_444,
    ("2.5D","single"):1_327_492+1_323_396, ("2.5D","dual"):1_327_492+1_325_444,
    ("2D+3D-post","single"):1_323_396+1_323_396, ("2D+3D-post","dual"):1_323_396+1_325_444,
    ("3D","single"):2_643_138+2_643_138, ("3D","dual"):2_643_138+2_647_234,
}
REF2D_S, REF2D_D = 160_289, 161_154  # 2D+3D-post refiner (Keras), GFLOPs 3.36/3.49

C = {}  # (backbone, arch, kind) -> dict
def put(bb, arch, kind, gen, disc, gflops, infer, train, note=""):
    C[(bb, arch, kind)] = dict(gen=gen, disc=disc, gflops=gflops, infer=infer, train=train, note=note)

# ---- GAN: pix2pix  (gen GFLOPs/fwd MEASURED in TF; disc GFLOPs ~4.3 [2-scale] / ~6.0 [3D])
put("pix2pix","2D","single",37_022_817,GAN_DISC["cond2d_s"],"22.1 (gen) + 4.3 (disc)","~25 s (triplanar)","~0.8 h")
put("pix2pix","2D","dual",   37_023_394,GAN_DISC["cond2d_d"],"22.2 (gen) + 4.3 (disc)","~25 s (triplanar)","~0.8 h")
put("pix2pix","2.5D","single",37_024_865,GAN_DISC["cond2d_s"],"22.2 (gen) + 4.3 (disc)","~25 s (triplanar)","~0.8 h")
put("pix2pix","2.5D","dual",  37_025_442,GAN_DISC["cond2d_d"],"22.2 (gen) + 4.3 (disc)","~25 s (triplanar)","~0.8 h")
put("pix2pix","2D+3D-post","single",37_022_817+REF2D_S,GAN_DISC["cond2d_s"],"22.1 (gen) + 3.4 (refiner)","~25 s + 3D-refiner sw","~0.3-0.7 h (+2D base)")
put("pix2pix","2D+3D-post","dual",  37_023_394+REF2D_D,GAN_DISC["cond2d_d"],"22.2 (gen) + 3.5 (refiner)","~25 s + 3D-refiner sw","~0.7 h (+2D base)")
put("pix2pix","3D","single",13_967_377,GAN_DISC["cond3d_s"],"13.0 (gen, 64^3) + 6.0 (disc)","sw 64x64x32 patches","~4.7-5.0 h")
put("pix2pix","3D","dual",   13_968_242,GAN_DISC["cond3d_d"],"13.2 (gen, 64^3) + 6.1 (disc)","sw 64x64x32 patches","~5.0 h")
# ---- GAN: CUT
put("CUT","2D","single",37_090_593,GAN_DISC["unc2d_s"],"23.6 (gen) + 1.5 (disc)","~25 s (triplanar)","~0.6 h")
put("CUT","2D","dual",   37_091_170,GAN_DISC["unc2d_d"],"23.6 (gen) + 1.6 (disc)","~25 s (triplanar)","~0.6 h")
put("CUT","2.5D","single",37_096_865,GAN_DISC["unc2d_s"],"24.0 (gen) + 1.5 (disc)","~25 s (triplanar)","~0.6 h")
put("CUT","2.5D","dual",  37_097_442,GAN_DISC["unc2d_d"],"24.1 (gen) + 1.6 (disc)","~25 s (triplanar)","~0.6 h")
put("CUT","2D+3D-post","single",37_090_593+REF2D_S,GAN_DISC["unc2d_s"],"23.6 (gen) + 3.4 (refiner)","~25 s + 3D-refiner sw","~1.0-1.3 h (+2D base)")
put("CUT","2D+3D-post","dual",  37_091_170+REF2D_D,GAN_DISC["unc2d_d"],"23.6 (gen) + 3.5 (refiner)","~25 s + 3D-refiner sw","~1.0 h (+2D base)")
put("CUT","3D","single",23_629_057,GAN_DISC["unc3d_s"],"46.2 (gen, 64^3) + 3.4 (disc)","sw 64x64x32 patches","~3.6-4.0 h","CUT-3D generator = same 3D ResNet backbone as CycleGAN-3D (PatchNCE hooks add 0 weights)")
put("CUT","3D","dual",   23_640_034,GAN_DISC["unc3d_d"],"49.1 (gen, 64^3) + 3.5 (disc)","sw 64x64x32 patches","~4.0 h","idem")
# ---- GAN: CycleGAN (two G + two D)
put("CycleGAN","2D","single",11_370_881*2,CG_DD[("2D","single")],"63.1 x2 (G_AB+G_BA) + 1.5 x2 (D_A+D_B)","~16 s (triplanar)","~4.5-5.0 h")
put("CycleGAN","2D","dual",  11_374_018+11_374_017,CG_DD[("2D","dual")],"63.4 x2 (gen) + 1.5 x2 (disc)","~16 s (triplanar)","~5.0 h")
put("CycleGAN","2.5D","single",11_377_153+11_377_155,CG_DD[("2.5D","single")],"63.6 x2 (gen) + 1.5/1.6 (D_A/D_B)","~16 s (triplanar)","~4.5 h")
put("CycleGAN","2.5D","dual",  11_380_290+11_380_291,CG_DD[("2.5D","dual")],"63.8 x2 (gen) + ~1.6 x2 (disc)","~16 s (triplanar)","~4.5 h")
put("CycleGAN","2D+3D-post","single",11_370_881*2+REF2D_S,CG_DD[("2D+3D-post","single")],"63.1 x2 (gen) + 3.4 (refiner)","~16 s + 3D-refiner sw","~0.7-0.9 h (+2D base)")
put("CycleGAN","2D+3D-post","dual",  11_374_018+11_374_017+REF2D_D,CG_DD[("2D+3D-post","dual")],"63.4 x2 (gen) + 3.5 (refiner)","~16 s + 3D-refiner sw","~0.7 h (+2D base)")
put("CycleGAN","3D","single",23_629_057*2,CG_DD[("3D","single")],"46.2 x2 (gen, 64^3) + 3.4 x2 (disc)","sw 64x64x32 patches","~9.6-12.4 h")
put("CycleGAN","3D","dual",  23_640_034+23_640_033,CG_DD[("3D","dual")],"49.1 x2 (gen, 64^3) + ~3.4 x2 (disc)","sw 64x64x32 patches","~9.6 h")
# ---- GAN: SwinPix2Pix (Swin gen for 2D/2.5D; reuses pix2pix-3D generator for 3D)
put("SwinPix2Pix","2D","single",1_444_120,GAN_DISC["cond2d_s"],"8.4 (gen) + 4.3 (disc)","~30 s (triplanar)","~1.2 h")
put("SwinPix2Pix","2D","dual",   1_444_157,GAN_DISC["cond2d_d"],"8.4 (gen) + 4.3 (disc)","~30 s (triplanar)","~1.2 h")
put("SwinPix2Pix","2.5D","single",1_445_056,GAN_DISC["cond2d_s"],"8.4 (gen) + 4.3 (disc)","~30 s (triplanar)","~1.2 h")
put("SwinPix2Pix","2.5D","dual",  1_445_093,GAN_DISC["cond2d_d"],"8.4 (gen) + 4.3 (disc)","~30 s (triplanar)","~1.2 h")
put("SwinPix2Pix","2D+3D-post","single",1_444_120+REF2D_S,GAN_DISC["cond2d_s"],"8.4 (gen) + 3.4 (refiner)","~30 s + 3D-refiner sw","~1.1-1.5 h (+2D base)")
put("SwinPix2Pix","2D+3D-post","dual",  1_444_157+REF2D_D,GAN_DISC["cond2d_d"],"8.4 (gen) + 3.5 (refiner)","~30 s + 3D-refiner sw","~1.1 h (+2D base)")
put("SwinPix2Pix","3D","single",13_967_377,GAN_DISC["cond3d_s"],"13.0 (gen, 64^3) + 6.0 (disc)","sw 64x64x32 patches","~4.9-5.0 h","no 3D Swin: reuses pix2pix-3D generator")
put("SwinPix2Pix","3D","dual",   13_968_242,GAN_DISC["cond3d_d"],"13.2 (gen, 64^3) + 6.1 (disc)","sw 64x64x32 patches","~5.0 h","idem")

# ---- ResViT (measured)
put("ResViT","2D","single",18_247_489,2_763_713,"157.1","~0.70 / 1.26 s (~50/90 sl)","~5.0 h")
put("ResViT","2D","dual",  18_250_626,2_764_737,"157.5","~0.68 / 1.23 s","~4.3 h")
put("ResViT","2.5D","single",18_253_761,2_765_761,"157.9","~0.71 / 1.27 s","~3.85 h")
put("ResViT","2.5D","dual",  18_256_898,2_766_785,"158.3","~0.74 / 1.33 s","~3.9 h")
put("ResViT","2D+3D-refine","single",18_247_489+57_121,2_763_713,"157.1 + 239.2 (refine 128^3)","~0.72 / 1.28 s (+1 Refine3D)","~5.5 h")
put("ResViT","2D+3D-refine","dual",  18_250_626+58_850,2_764_737,"157.5 + 246.4","~0.70 / 1.25 s","~7-8 h")
put("ResViT","3D","single",23_778_113,6_224_593,"800.1 (96^3 patch, ngf=48)","~0.39 s (8x 96^3 patches)","~6.0 h")
put("ResViT","3D","dual",  23_794_578,6_227_665,"829.2 (96^3 patch)","~0.39 s (8x 96^3 patches)","~6.5 h")

# ---- SynDiff (measured; diffusion = 4 sampling steps)
put("SynDiff","2D","single",12_089_442,7_216_705,"286.0/fwd (x4 = 1144 / sample)","~2.15 / 3.69 s (~50/90 sl, b8)","~13.1 h")
put("SynDiff","2D","dual",  12_090_307,7_216_833,"286.7/fwd (x4 = 1147)","~2.18 / 3.73 s","~11.2 h")
put("SynDiff","2.5D","single",12_091_172,7_216_705,"287.4/fwd (x4 = 1150)","~2.16 / 3.70 s","~13.2 h")
put("SynDiff","2.5D","dual",  12_092_037,7_216_833,"288.1/fwd (x4 = 1152)","~2.15 / 3.68 s","~6.9 h")
put("SynDiff","3D","single",5_421_921,9_119_089,"254.2/fwd (x4 = 1017, 64^3 patch)","~4.89 s (27x 64^3 patches)","~5.4 h")
put("SynDiff","3D","dual",  5_423_218,9_119_185,"254.9/fwd (x4 = 1020)","~4.89 s (27x 64^3 patches)","~4.7 h")
# 3D+3D-refine in rescore CSV == SynDiff-2D-diffusion (Stage1, frozen) + 3D ResNet refiner
put("SynDiff","3D+3D-refine","single",12_089_442+212_257,7_216_705+1_564_345,"286.0/fwd (x4) + 886.3 (refiner 128^3)","~2.31 / 3.85 s (2D coarse + 1 Refiner3D)","~4.0 h (+2D Stage1)","label '3D+3D-refine' in rescore CSV; actually 2D diffusion Stage1 + 3D ResNet refiner")
put("SynDiff","3D+3D-refine","dual",  12_090_307+228_722,7_216_833+1_565_881,"286.7/fwd (x4) + 955.4 (refiner 128^3)","~2.34 / 3.89 s","~4.0 h (+2D Stage1)","idem")
# cascade: ResViT-2.5D (frozen) -> SynDiff diffusion refiner (num_channels=3)
put("SynDiff","ResViT->SynDiff-cascade","single",18_253_761+12_089_442,2_765_761+7_216_705,"157.9 + 286.0/fwd (x4)","~0.71 + ~2.15 s ~ 2.86 s","~6.5 h (+ ResViT 3.85 h prereq)","trainable part = the 12.1M diffusion refiner; ResViT frozen")
# joint: ResViT-2.5D + SynDiff diffusion refiner, end-to-end fine-tune
put("SynDiff","ResViT+SynDiff-joint","single",18_253_761+12_089_442,2_765_761+7_216_705,"157.9 + 286.0/fwd (x4)","~0.71 + ~2.15 s ~ 2.86 s","~4.7 h (+ ResViT 3.85 h + cascade prereq)","")
# untuned full SynDiff (ngf=64, CycleGAN-style 2 diffusive gens) - not benchmarked
put("SynDiff","2D (full, untuned)","single","~156M (full SynDiff, not benchmarked)","~? (not benchmarked)","not measured","not measured (~10-15 s/vol est.)","~? (80 ep, ngf 64, bs 1; not measured)","reference only; n=4 subjects have saved volumes")
# ensembles: weighted average of two already-trained models -> no extra training
put("Ensemble","2.5D-single","single",18_253_761+12_091_172,2_765_761+7_216_705,"157.9 + 287.4/fwd (x4)","~0.71 + ~2.16 s ~ 2.87 s","0 (combines pretrained ResViT-2.5D + SynDiff-2.5D)","")
put("Ensemble","2.5D-dual","single",18_256_898+12_092_037,2_766_785+7_216_833,"158.3 + 288.1/fwd (x4)","~0.74 + ~2.15 s ~ 2.89 s","0 (combines pretrained ResViT-2.5D-dual + SynDiff-2.5D-dual)","")
put("Ensemble","2.5D-dual","dual",  18_256_898+12_092_037,2_766_785+7_216_833,"158.3 + 288.1/fwd (x4)","~0.74 + ~2.15 s ~ 2.89 s","0 (combines pretrained dual models)","")

# --------------------------------------------------------- map a summary row -> compute key
def key_for(row):
    bb, fam, arch, tgt, meth = (str(row["backbone"]), str(row.get("family","")),
                                str(row["architecture"]), str(row["target"]), str(row["method"]))
    kind = "dual" if tgt.lower() in ("t2_flair","t2+flair","dual") else "single"
    if bb == "GAN":
        return (fam, arch, kind)
    if bb in ("ResViT", "SynDiff"):
        amap = {
            "ResViT->SynDiff cascade":"ResViT->SynDiff-cascade",
            "ResViT+SynDiff joint":"ResViT+SynDiff-joint",
            "2D (full SynDiff, untuned)":"2D (full, untuned)",
        }
        return (bb, amap.get(arch, arch), kind)
    # ensembles
    if "Ensemble" in meth or bb == "ResViT+SynDiff":
        ml = meth.lower()
        sub = "2.5D-dual" if "dual" in ml else "2.5D-single"
        return ("Ensemble", sub, kind)
    return (bb, arch, kind)

def gint(v):
    try: return f"{int(v):,}"
    except Exception: return str(v)

# ----------------------------------------------------------------- build master tables
def total_params(g, d):
    try: return f"{int(g)+int(d):,}"
    except Exception: return ""

rows_master = []
for _, r in summ.iterrows():
    k = key_for(r)
    c = C.get(k, {})
    g, d = c.get("gen",""), c.get("disc","")
    rows_master.append(dict(
        method=r["method"], family=r.get("family",""), backbone=r["backbone"],
        architecture=r["architecture"], target=r["target"], channel=r["channel"], n=int(r["n"]),
        gen_params=gint(g), disc_params=gint(d), total_params=total_params(g,d),
        gflops_fwd=c.get("gflops",""), train_time_h=c.get("train",""),
        infer_s_per_vol=c.get("infer",""),
        ssim_mean=round(r["ssim_mean"],4), ssim_sd=round(r["ssim_sd"],4), ssim_ci95=round(r["ssim_ci95"],4),
        psnr_mean=round(r["psnr_mean"],3), psnr_sd=round(r["psnr_sd"],3),
        mae_mean=round(r["mae_mean"],4), mae_sd=round(r["mae_sd"],4),
        lpips_mean=round(r["lpips_mean"],4), lpips_sd=round(r["lpips_sd"],4),
        compute_note=c.get("note",""),
    ))
M = pd.DataFrame(rows_master)

# rank within channel
def add_rank(df, ch):
    sub = df[df.channel==ch].copy()
    sub = sub.sort_values("ssim_mean", ascending=False)
    sub["ssim_rank"] = range(1, len(sub)+1)
    return sub

M_t2 = add_rank(M, "t2")
M_fl = add_rank(M, "flair")
M_t2.to_csv(os.path.join(HERE,"master_benchmark_table.csv"), index=False)
M_fl.to_csv(os.path.join(HERE,"master_benchmark_table_flair.csv"), index=False)

# ---------------------------------------------------- paper results tables (compact)
def fmt(m, s, nd=4): return f"{m:.{nd}f} ± {s:.{nd}f}"
def results_table(df, ch):
    sub = df[df.channel==ch].sort_values("ssim_mean", ascending=False).reset_index(drop=True)
    out = []
    for i, r in sub.iterrows():
        out.append(dict(rank=i+1, method=r["method"], backbone=r["backbone"],
            architecture=r["architecture"], target=r["target"], n=r["n"],
            SSIM=fmt(r["ssim_mean"], r["ssim_sd"]),
            PSNR_dB=fmt(r["psnr_mean"], r["psnr_sd"], 2),
            MAE=fmt(r["mae_mean"], r["mae_sd"]),
            LPIPS=fmt(r["lpips_mean"], r["lpips_sd"])))
    return pd.DataFrame(out)
results_table(M, "t2").to_csv(os.path.join(HERE,"table_results_t2.csv"), index=False)
results_table(M, "flair").to_csv(os.path.join(HERE,"table_results_flair.csv"), index=False)

# ---------------------------------------------------- compute table (one row per backbone x arch x target)
seen=set(); crows=[]
for _, r in M.iterrows():
    fam = r["family"] if r["backbone"]=="GAN" else r["backbone"]
    key=(fam, r["architecture"], r["target"])
    if key in seen: continue
    seen.add(key)
    crows.append(dict(model=fam, backbone=r["backbone"], architecture=r["architecture"], target=r["target"],
        gen_params=r["gen_params"], disc_params=r["disc_params"], total_params=r["total_params"],
        gflops_fwd=r["gflops_fwd"], train_time=r["train_time_h"], infer_s_per_vol=r["infer_s_per_vol"],
        note=r["compute_note"]))
pd.DataFrame(crows).to_csv(os.path.join(HERE,"table_compute.csv"), index=False)

# ---------------------------------------------------- architecture-effect Wilcoxon table (ensemble-free)
wil = pd.read_csv(os.path.join(HERE,"rescore_wilcoxon.csv"))
wil = wil[~wil["comparison"].astype(str).str.contains("Ensemble", regex=False)].copy()
wil.to_csv(os.path.join(HERE,"rescore_wilcoxon.csv"), index=False)
ae = wil[wil["comparison"].str.contains("  vs  ", regex=False) & ~wil["comparison"].str.contains("BEST(", regex=False)].copy()
ae.to_csv(os.path.join(HERE,"table_wilcoxon_architecture_effect.csv"), index=False)
vb = wil[wil["comparison"].str.contains("BEST(", regex=False)].copy()
vb.to_csv(os.path.join(HERE,"table_wilcoxon_vs_best.csv"), index=False)

# ---------------------------------------------------- PRE-OP / POST-OP stratification
# subject suffix '-pre' / '-post' determines the surgical phase.
PSf = PS.copy()
PSf["phase"] = PSf["subject"].astype(str).apply(lambda s: "preop" if s.endswith("-pre") else ("postop" if s.endswith("-post") else "other"))
metr = ["ssim","psnr","mae","lpips"]
def summarize(df):
    out = []
    for (meth, fam, bb, arch, tgt, ch), g in df.groupby(["method","family","backbone","architecture","target","channel"], sort=False):
        n = len(g)
        row = dict(method=meth, family=fam, backbone=bb, architecture=arch, target=tgt, channel=ch, n=n)
        for m in metr:
            v = g[m].astype(float)
            row[f"{m}_mean"] = round(v.mean(),5); row[f"{m}_sd"] = round(v.std(ddof=1),5) if n>1 else 0.0
            row[f"{m}_ci95"] = round(1.96*v.std(ddof=1)/math.sqrt(n),5) if n>1 else 0.0
            row[f"{m}_median"] = round(v.median(),5); row[f"{m}_min"] = round(v.min(),5); row[f"{m}_max"] = round(v.max(),5)
        out.append(row)
    return pd.DataFrame(out)
for ph in ("preop","postop"):
    sp = summarize(PSf[PSf.phase==ph])
    sp.to_csv(os.path.join(HERE, f"rescore_{ph}_summary.csv"), index=False)
    # compact results table per phase (T2 channel + FLAIR channel)
    for ch in ("t2","flair"):
        sub = sp[sp.channel==ch].sort_values("ssim_mean", ascending=False).reset_index(drop=True)
        if len(sub)==0: continue
        comp = []
        for i, r in sub.iterrows():
            comp.append(dict(rank=i+1, method=r["method"], backbone=r["backbone"], architecture=r["architecture"],
                target=r["target"], n=r["n"],
                SSIM=fmt(r["ssim_mean"], r["ssim_sd"]), PSNR_dB=fmt(r["psnr_mean"], r["psnr_sd"], 2),
                MAE=fmt(r["mae_mean"], r["mae_sd"]), LPIPS=fmt(r["lpips_mean"], r["lpips_sd"])))
        pd.DataFrame(comp).to_csv(os.path.join(HERE, f"table_results_{ch}_{ph}.csv"), index=False)
# also a side-by-side preop-vs-postop delta table (T2 SSIM/LPIPS), per method
prs = summarize(PSf[PSf.phase=="preop"]); pos = summarize(PSf[PSf.phase=="postop"])
key = ["method","channel"]
d = prs.merge(pos, on=key, suffixes=("_preop","_postop"))
d = d[["method","channel","target_preop","n_preop","n_postop",
       "ssim_mean_preop","ssim_mean_postop","psnr_mean_preop","psnr_mean_postop",
       "mae_mean_preop","mae_mean_postop","lpips_mean_preop","lpips_mean_postop"]].copy()
d["d_ssim_post_minus_pre"] = (d["ssim_mean_postop"]-d["ssim_mean_preop"]).round(4)
d["d_psnr_post_minus_pre"] = (d["psnr_mean_postop"]-d["psnr_mean_preop"]).round(3)
d["d_lpips_post_minus_pre"] = (d["lpips_mean_postop"]-d["lpips_mean_preop"]).round(4)
d.rename(columns={"target_preop":"target"}, inplace=True)
d.sort_values(["channel","ssim_mean_preop"], ascending=[True,False]).to_csv(os.path.join(HERE,"table_preop_vs_postop_delta.csv"), index=False)

print("WROTE:")
for f in ["master_benchmark_table.csv","master_benchmark_table_flair.csv","table_results_t2.csv",
          "table_results_flair.csv","table_compute.csv","table_wilcoxon_architecture_effect.csv",
          "table_wilcoxon_vs_best.csv"]:
    print("  ", f, os.path.getsize(os.path.join(HERE,f)), "bytes")
print("\n--- table_results_t2.csv (top 15) ---")
print(results_table(M,"t2").head(15).to_string(index=False))
print("\n--- table_compute.csv (sample rows) ---")
ct=pd.DataFrame(crows)
print(ct[ct.target=="t2"].to_string(index=False))
# sanity: any gaps remaining?
gaps = M[(M.gen_params=="") | (M.disc_params=="")]
print("\nremaining param gaps:", len(gaps))
if len(gaps): print(gaps[["method"]].to_string(index=False))
