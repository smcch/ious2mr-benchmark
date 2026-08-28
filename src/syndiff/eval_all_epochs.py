"""
Evaluate every saved gen_diffusive_2_*.pth on the full test split (30 subjects).
Stores only per-subject metrics, no NIfTI/PNG (cheap on disk).

After we know the best epoch we can re-run eval_volume.py on it for full
qualitative outputs.

Output:
  results/all_epochs_per_subject.csv     — long format: epoch, subject, ssim, psnr, mae
  results/all_epochs_summary.csv         — wide format: epoch, n, ssim_mean, psnr_mean, mae_mean, ...
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from common.paths import PROJECT_ROOT, DATA_ROOT, CKPT_ROOT, EXTERNAL_ROOT  # noqa: E402
import os  # noqa: E402,F811
import os, sys, json, time, glob, argparse
ROOT = os.path.join(str(PROJECT_ROOT), "synthdiff")
sys.path.insert(0, os.path.join(ROOT, "syndiff_src"))
sys.path.insert(0, ROOT)

import numpy as np, torch, nibabel as nib, csv
from scipy.ndimage import zoom
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

from eval_volume import (TestArgs, load_checkpoint, Posterior_Coefficients,
                          sample_from_model, fg_mask, norm_pct, norm_z, rsz,
                          find_mr_t2, load_nii, volume_metrics)
from backbones.ncsnpp_generator_adagn import NCSNpp


def evaluate_checkpoint(ckpt_path, test_names, us_dir, mr_dir,
                        device, slice_batch=1, min_fg=0.02):
    targs = TestArgs()
    gen2 = NCSNpp(targs).to(device)
    load_checkpoint(ckpt_path, gen2, device)
    pos_coeff = Posterior_Coefficients(targs, device)

    rows = []
    for name in test_names:
        us_path = os.path.join(us_dir, f"{name}-us.nii.gz")
        mr_path = find_mr_t2(mr_dir, name)
        if (not os.path.exists(us_path)) or (mr_path is None):
            continue
        us_raw, _ = load_nii(us_path)
        mr_raw, _ = load_nii(mr_path)
        if us_raw.shape != mr_raw.shape:
            continue
        H, W, D = us_raw.shape
        fg = (fg_mask(us_raw, 0.01) | fg_mask(mr_raw, 0.01))

        us_n = norm_pct(us_raw, fg_mask(us_raw, 0.01), 2, 98)
        mr_n = norm_z(mr_raw, fg_mask(mr_raw, 0.01), 3.0)

        slice_indices = [k for k in range(D) if float(np.mean(fg[:, :, k])) >= min_fg]
        if not slice_indices:
            continue
        us256 = np.stack([rsz(us_n[:, :, k], 256, 256) for k in slice_indices], 0)

        preds_lowres = np.zeros_like(us256, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(slice_indices), slice_batch):
                chunk = us256[start:start + slice_batch]
                src = torch.from_numpy(chunk[:, None]).to(device)
                noise = torch.randn_like(src)
                x_init = torch.cat((noise, src), axis=1)
                out = sample_from_model(pos_coeff, gen2, targs.num_timesteps, x_init, targs)
                preds_lowres[start:start + slice_batch] = out[:, 0].cpu().numpy()

        pred_vol = np.full_like(us_n, -1., dtype=np.float32)
        for j, k in enumerate(slice_indices):
            pred_vol[:, :, k] = rsz(preds_lowres[j], H, W)
        pred_vol[~fg] = -1.

        gt01 = (np.clip(mr_n, -1, 1) + 1.0) / 2.0
        pred01 = (np.clip(pred_vol, -1, 1) + 1.0) / 2.0
        m = volume_metrics(pred01, gt01, fg)
        rows.append(dict(name=name, n_slices=len(slice_indices), **m))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp",          default="syndiff_us_t2")
    ap.add_argument("--epochs",       default="all", help="'all' or comma list")
    ap.add_argument("--max_subjects", type=int, default=0, help="0 = full test split")
    ap.add_argument("--slice_batch",  type=int, default=1)
    ap.add_argument("--out_dir",      default=os.path.join(ROOT, "results"))
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")

    exp_path = os.path.join(ROOT, "output", args.exp)
    if args.epochs == "all":
        ckpts = sorted(glob.glob(os.path.join(exp_path, "gen_diffusive_2_*.pth")))
        epoch_list = [int(os.path.basename(p).split("_")[-1].split(".")[0]) for p in ckpts]
        epoch_list = sorted(epoch_list)
    else:
        epoch_list = [int(x) for x in args.epochs.split(",")]
    print(f"[run] {len(epoch_list)} checkpoints: {epoch_list}")

    with open(os.path.join(ROOT, "data", "split_used.json")) as f:
        split = json.load(f)
    test_names = split["test"]
    if args.max_subjects:
        test_names = test_names[:args.max_subjects]
    print(f"[run] {len(test_names)} test volumes")

    us_dir = os.path.join(str(PROJECT_ROOT), "resvit", "dataset-registration-corrected-cropped", "US")
    mr_dir = os.path.join(str(PROJECT_ROOT), "resvit", "dataset-registration-corrected-cropped", "MR-T2")

    os.makedirs(args.out_dir, exist_ok=True)
    long_csv  = os.path.join(args.out_dir, "all_epochs_per_subject.csv")
    summary_csv = os.path.join(args.out_dir, "all_epochs_summary.csv")

    long_rows = []
    summary_rows = []
    for epoch in epoch_list:
        ckpt = os.path.join(exp_path, f"gen_diffusive_2_{epoch}.pth")
        if not os.path.exists(ckpt):
            print(f"  [skip] epoch {epoch}: ckpt missing")
            continue
        t0 = time.time()
        rows = evaluate_checkpoint(ckpt, test_names, us_dir, mr_dir,
                                   device, args.slice_batch)
        dt = time.time() - t0
        ssim = np.array([r["ssim"] for r in rows], dtype=np.float64)
        psnr = np.array([r["psnr"] for r in rows], dtype=np.float64)
        mae  = np.array([r["mae"]  for r in rows], dtype=np.float64)
        s = dict(epoch=epoch, n=len(rows),
                 ssim_mean=float(np.nanmean(ssim)),
                 ssim_std =float(np.nanstd(ssim)),
                 psnr_mean=float(np.nanmean(psnr)),
                 psnr_std =float(np.nanstd(psnr)),
                 mae_mean =float(np.nanmean(mae)),
                 mae_std  =float(np.nanstd(mae)),
                 seconds  =round(dt, 1))
        summary_rows.append(s)
        for r in rows:
            long_rows.append(dict(epoch=epoch, **r))
        print(f"  [epoch {epoch:3d}] N={s['n']} "
              f"SSIM={s['ssim_mean']:.4f}±{s['ssim_std']:.3f}  "
              f"PSNR={s['psnr_mean']:.2f}±{s['psnr_std']:.2f}  "
              f"MAE={s['mae_mean']:.4f}±{s['mae_std']:.3f}  "
              f"({dt:.0f}s)")

    # write outputs
    if long_rows:
        keys = list(long_rows[0].keys())
        with open(long_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(long_rows)
        print(f"[done] {long_csv}")
    if summary_rows:
        keys = list(summary_rows[0].keys())
        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(summary_rows)
        print(f"[done] {summary_csv}")

    if summary_rows:
        best = max(summary_rows, key=lambda x: x["ssim_mean"])
        print(f"\n[BEST] epoch {best['epoch']} SSIM={best['ssim_mean']:.4f} "
              f"PSNR={best['psnr_mean']:.2f} MAE={best['mae_mean']:.4f}")


if __name__ == "__main__":
    main()
