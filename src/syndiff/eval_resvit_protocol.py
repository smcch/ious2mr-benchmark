"""
Re-evaluate SynDiff checkpoints on the test split using EXACTLY the same
metric protocol as resvit/eval_resvit_metrics.py and COMPARATIVA-3 — so the
numbers go straight into the unified comparison CSV.

Protocol (lifted from eval_resvit_metrics.py):
  - Both pred and target stored in [-1, 1].
  - (x + 1) / 2 -> [0, 1] before metrics.
  - SSIM_3D = mean over axial slices that have >=1% pixels with target>0.025.
    Per-slice: skimage.structural_similarity(data_range=1.0).
  - PSNR_3D = MSE over fg (target>0.025), psnr = 10 log10(1/mse).
  - MAE_3D = mean |target - pred| over fg (target>0.025).

Outputs in $IOUS2MR_ROOT/synthdiff/results/<exp>_resvit_protocol/
  per_subject.csv     — columns matching ResViT's per-subject schema
  summary.csv         — columns matching ResViT's global_summary schema
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from common.paths import PROJECT_ROOT, DATA_ROOT, CKPT_ROOT, EXTERNAL_ROOT  # noqa: E402
import os  # noqa: E402,F811
import os, sys, json, time, glob, argparse, csv, math
ROOT = os.path.join(str(PROJECT_ROOT), "synthdiff")
sys.path.insert(0, os.path.join(ROOT, "syndiff_src"))
sys.path.insert(0, ROOT)

import numpy as np, torch, nibabel as nib
from scipy.ndimage import zoom
from skimage.metrics import structural_similarity

from eval_volume import (TestArgs, load_checkpoint, Posterior_Coefficients,
                          sample_from_model, fg_mask, norm_pct, norm_z, rsz,
                          find_mr_t2, load_nii)
from backbones.ncsnpp_generator_adagn import NCSNpp


# ---- ResViT metric protocol (mirrors resvit/eval_resvit_metrics.py) --------
def ssim_3d_resvit(target, pred):
    """target, pred in [-1, 1]; per-slice mean SSIM filtered by foreground."""
    t01 = (target + 1.0) / 2.0
    p01 = (pred + 1.0) / 2.0
    vals = []
    for z in range(t01.shape[2]):
        t_sl, p_sl = t01[:, :, z], p01[:, :, z]
        if np.mean(t_sl > 0.025) < 0.01:
            continue
        vals.append(structural_similarity(t_sl, p_sl, data_range=1.0))
    return float(np.mean(vals)) if vals else 0.0

def psnr_3d_resvit(target, pred):
    t01 = (target + 1.0) / 2.0
    p01 = (pred + 1.0) / 2.0
    fg = t01 > 0.025
    if fg.sum() < 100:
        return 0.0
    mse = float(np.mean((t01[fg] - p01[fg]) ** 2))
    if mse < 1e-10:
        return 50.0
    return float(10.0 * np.log10(1.0 / mse))

def mae_3d_resvit(target, pred):
    t01 = (target + 1.0) / 2.0
    p01 = (pred + 1.0) / 2.0
    fg = t01 > 0.025
    if fg.sum() < 100:
        return 1.0
    return float(np.mean(np.abs(t01[fg] - p01[fg])))


def predict_volume(gen2, pos_coeff, targs, us_n, fg, slice_batch, min_fg, device):
    """Run gen_diffusive_2 on every fg slice and return predicted volume in [-1,1].

    IMPORTANT: SynDiff's dataset.LoadDataSet applies np.transpose(arr, (0,2,1))
    on the (N, H, W) tensor before returning to the training loop, so the
    network was trained on transposed slices. We mirror that here: transpose
    the slice into the model, transpose the output back, so both ends match
    the training-time orientation.
    """
    H, W, D = us_n.shape
    pred_vol = np.full_like(us_n, -1., dtype=np.float32)
    slice_indices = [k for k in range(D) if float(np.mean(fg[:, :, k])) >= min_fg]
    if not slice_indices:
        return pred_vol, []
    us256 = np.stack([rsz(us_n[:, :, k], 256, 256) for k in slice_indices], 0)
    # Match LoadDataSet's transpose(0, 2, 1) so the model sees the same
    # orientation it was trained on.
    us256_t = np.transpose(us256, (0, 2, 1)).copy()
    preds_lowres_t = np.zeros_like(us256_t, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(slice_indices), slice_batch):
            chunk = us256_t[start:start + slice_batch]
            src = torch.from_numpy(chunk[:, None]).to(device)
            noise = torch.randn_like(src)
            x_init = torch.cat((noise, src), axis=1)
            out = sample_from_model(pos_coeff, gen2, targs.num_timesteps, x_init, targs)
            preds_lowres_t[start:start + slice_batch] = out[:, 0].cpu().numpy()
    # Undo the training-time transpose for the prediction.
    preds_lowres = np.transpose(preds_lowres_t, (0, 2, 1)).copy()
    for j, k in enumerate(slice_indices):
        pred_vol[:, :, k] = rsz(preds_lowres[j], H, W)
    pred_vol = np.clip(pred_vol, -1., 1.)
    return pred_vol, slice_indices


def evaluate_on_test(epoch, exp, test_names, us_dir, mr_dir, device,
                    slice_batch, min_fg, save_volumes=False, ngf=64):
    targs = TestArgs()
    # Override generator width when evaluating models trained with a smaller
    # backbone (e.g. paired runs use ngf=32).
    targs.num_channels_dae = ngf
    targs.ngf = ngf
    targs.nf = ngf
    gen2 = NCSNpp(targs).to(device)
    ckpt = os.path.join(ROOT, "output", exp, f"gen_diffusive_2_{epoch}.pth")
    load_checkpoint(ckpt, gen2, device)
    pos_coeff = Posterior_Coefficients(targs, device)
    print(f"[ckpt] {ckpt}")

    rows = []
    out_dir = os.path.join(ROOT, "results",
                           f"{exp}_resvit_protocol_ep{epoch}")
    os.makedirs(out_dir, exist_ok=True)
    if save_volumes:
        os.makedirs(os.path.join(out_dir, "volumes"), exist_ok=True)

    for name in test_names:
        us_path = os.path.join(us_dir, f"{name}-us.nii.gz")
        mr_path = find_mr_t2(mr_dir, name)
        if (not os.path.exists(us_path)) or (mr_path is None):
            continue
        us_raw, aff = load_nii(us_path)
        mr_raw, _ = load_nii(mr_path)
        if us_raw.shape != mr_raw.shape:
            continue
        fg_us = fg_mask(us_raw, 0.01)
        fg_mr = fg_mask(mr_raw, 0.01)
        fg = fg_us | fg_mr
        us_n = norm_pct(us_raw, fg_us, 2, 98)         # [-1, 1]
        mr_n = norm_z(mr_raw, fg_mr, 3.0)              # [-1, 1]
        pred_vol, _ = predict_volume(gen2, pos_coeff, targs, us_n, fg,
                                     slice_batch, min_fg, device)
        # ResViT protocol expects the prediction to share the same intensity
        # convention as the target; we feed [-1,1] for both.
        ssim = ssim_3d_resvit(mr_n, pred_vol)
        psnr = psnr_3d_resvit(mr_n, pred_vol)
        mae  = mae_3d_resvit(mr_n, pred_vol)
        rows.append(dict(experiment=f"SynDiff_ep{epoch}", target="t2",
                         subject=name, ssim_t2=ssim, psnr_t2=psnr,
                         mae_t2=mae, lpips_t2=float("nan")))
        print(f"  {name:25s} SSIM={ssim:.4f} PSNR={psnr:.2f} MAE={mae:.4f}")
        if save_volumes:
            nib.save(nib.Nifti1Image(((pred_vol + 1) / 2).astype(np.float32), aff),
                     os.path.join(out_dir, "volumes", f"{name}_predT2.nii.gz"))
            nib.save(nib.Nifti1Image(((mr_n + 1) / 2).astype(np.float32), aff),
                     os.path.join(out_dir, "volumes", f"{name}_gtT2.nii.gz"))

    # Per-subject CSV (matches ResViT global_per_subject schema)
    per_subj_csv = os.path.join(out_dir, "per_subject.csv")
    with open(per_subj_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "experiment", "target", "subject",
                                          "ssim_t2", "psnr_t2", "mae_t2", "lpips_t2",
                                          "ssim_flair", "psnr_flair", "mae_flair", "lpips_flair"])
        w.writeheader()
        for r in rows:
            w.writerow({"group": "syndiff", **r,
                        "ssim_flair": "", "psnr_flair": "", "mae_flair": "", "lpips_flair": ""})

    # Summary
    if not rows:
        print("[warn] no rows"); return None
    ssim = np.array([r["ssim_t2"] for r in rows], dtype=np.float64)
    psnr = np.array([r["psnr_t2"] for r in rows], dtype=np.float64)
    mae  = np.array([r["mae_t2"]  for r in rows], dtype=np.float64)
    n = len(rows)
    summary = dict(
        group="syndiff",
        experiment=f"SynDiff_ep{epoch}",
        n_t2=n,
        ssim_t2_mean=float(np.mean(ssim)),
        ssim_t2_sd  =float(np.std(ssim, ddof=1)) if n > 1 else 0.0,
        ssim_t2_ci95=float(1.96 * np.std(ssim, ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
        psnr_t2_mean=float(np.mean(psnr)),
        psnr_t2_sd  =float(np.std(psnr, ddof=1)) if n > 1 else 0.0,
        psnr_t2_ci95=float(1.96 * np.std(psnr, ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
        mae_t2_mean =float(np.mean(mae)),
        mae_t2_sd   =float(np.std(mae, ddof=1)) if n > 1 else 0.0,
        mae_t2_ci95 =float(1.96 * np.std(mae, ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
        lpips_t2_mean=float("nan"), lpips_t2_sd=float("nan"), lpips_t2_ci95=float("nan"),
        n_flair=0,
        ssim_fl_mean="", ssim_fl_sd="", ssim_fl_ci95="",
        psnr_fl_mean="", psnr_fl_sd="", psnr_fl_ci95="",
        mae_fl_mean="", mae_fl_sd="", mae_fl_ci95="",
        lpips_fl_mean="", lpips_fl_sd="", lpips_fl_ci95="",
    )
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader(); w.writerow(summary)
    print(f"[summary epoch {epoch}] N={n} "
          f"SSIM={summary['ssim_t2_mean']:.4f}±{summary['ssim_t2_sd']:.3f}  "
          f"PSNR={summary['psnr_t2_mean']:.2f}±{summary['psnr_t2_sd']:.2f}  "
          f"MAE={summary['mae_t2_mean']:.4f}±{summary['mae_t2_sd']:.3f}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp",          default="syndiff_us_t2")
    ap.add_argument("--epochs",       default="0,15,35,55,80",
                    help="comma list; default uses checkpoints of interest")
    ap.add_argument("--max_subjects", type=int, default=0)
    ap.add_argument("--slice_batch",  type=int, default=4)
    ap.add_argument("--save_volumes", action="store_true",
                    help="save predicted/GT NIfTI volumes (only do for best ckpt)")
    ap.add_argument("--ngf", type=int, default=64,
                    help="generator width; use 32 for paired runs and 64 for "
                         "the original bidirectional checkpoints")
    args = ap.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda:0")

    with open(os.path.join(ROOT, "data", "split_used.json")) as f:
        split = json.load(f)
    test_names = split["test"]
    if args.max_subjects:
        test_names = test_names[:args.max_subjects]
    us_dir = os.path.join(str(PROJECT_ROOT), "resvit", "dataset-registration-corrected-cropped", "US")
    mr_dir = os.path.join(str(PROJECT_ROOT), "resvit", "dataset-registration-corrected-cropped", "MR-T2")

    epoch_list = [int(x) for x in args.epochs.split(",")]
    print(f"[run] epochs={epoch_list}, n_test={len(test_names)}")

    summaries = []
    for ep in epoch_list:
        s = evaluate_on_test(ep, args.exp, test_names, us_dir, mr_dir,
                             device, args.slice_batch, 0.02,
                             save_volumes=args.save_volumes, ngf=args.ngf)
        if s: summaries.append(s)

    # combined summary across epochs
    if summaries:
        comb = os.path.join(ROOT, "results",
                            f"{args.exp}_resvit_protocol_summary.csv")
        with open(comb, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader(); w.writerows(summaries)
        print(f"[done] {comb}")


if __name__ == "__main__":
    main()
