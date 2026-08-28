"""
recover_missing_subject.py

Re-runs inference for test subjects that were dropped during the original
training run (results.json reports n_test < 31 for these 4 experiments
because load_volume silently skipped them). The trained checkpoints are
intact, so we just rebuild the model, load p2_best.pth, and save the
missing predictions into predictions/<subject_id>/.

Targets the exact same save format that resvit_final.main() uses at its
final [SAVE] step (lines 1343-1356 in resvit_final.py).

Run from the nnunet conda env.
"""
import json
import os
import sys
import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import resvit_final as rf

MISSING_SUBJECTS = ["ReMIND-023-post"]

EXPERIMENTS = [
    ("full_3d", "t2"),
    ("full_3d", "t2_flair"),
]


def recover(variant, target, subject):
    cfg = rf.Config()
    cfg.arch_variant = variant
    cfg.target_mode = target
    dev = torch.device(cfg.device)

    print(f"\n{'='*70}")
    print(f"[recover] {cfg.run_name} | subject={subject}")
    print(f"{'='*70}")

    # Applies GPU profile first (sets patch_size_3d, image_size, etc.).
    cfg.apply_gpu_profile()

    # Override ngf from the saved results.json so we match the checkpoint's
    # channel widths exactly — training may have used a different GPU profile.
    results_path = os.path.join(cfg.pred_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            meta = json.load(f)
        if "ngf" in meta:
            if cfg.ngf != meta["ngf"]:
                print(f"  [override] ngf {cfg.ngf} -> {meta['ngf']} "
                      f"(from results.json)")
            cfg.ngf = meta["ngf"]

    # Build the same model used in training.
    G, D, R = rf.build_all(cfg, dev)

    ckpt_path = os.path.join(cfg.ckpt_dir, "p2_best.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(cfg.ckpt_dir, "p2_final.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No p2 checkpoint in {cfg.ckpt_dir}")

    print(f"  loading {os.path.basename(ckpt_path)}")
    state = torch.load(ckpt_path, map_location=dev, weights_only=False)
    # state can be {"G":..., "D":...} or a bare state_dict
    G_state = state["G"] if isinstance(state, dict) and "G" in state else state
    G.load_state_dict(G_state)
    G.eval()

    # Load the missing volume exactly like the training pipeline does.
    s = rf.load_volume(subject, cfg)
    if s is None:
        print(f"  [SKIP] load_volume returned None for {subject} "
              f"(target_mode={target} requires "
              f"{'T2' if target=='t2' else 'T2 or FLAIR'})")
        return False
    has = []
    if s["has_t2"]:
        has.append("T2")
    if s["has_fl"]:
        has.append("FL")
    print(f"  loaded {subject} shape={s['us'].shape} "
          f"avail={'+'.join(has) or 'NONE'}")

    # Inference (bypasses torch.compile via _unwrap inside run_inference).
    with torch.inference_mode():
        pred = rf.run_inference(G, s, cfg, dev)
    # pred: (out_ch, H, W, D) float32

    out_dir = os.path.join(cfg.pred_dir, subject)
    os.makedirs(out_dir, exist_ok=True)

    rf.save_nii(s["us"], os.path.join(out_dir, "us.nii.gz"), s["aff"])
    if s["has_t2"]:
        rf.save_nii(s["t2"], os.path.join(out_dir, "tgt_t2.nii.gz"), s["aff"])
        rf.save_nii(pred[0], os.path.join(out_dir, "pred_t2.nii.gz"),
                    s["aff"])
    if s["has_fl"] and cfg.out_ch > 1:
        rf.save_nii(s["flair"], os.path.join(out_dir, "tgt_fl.nii.gz"),
                    s["aff"])
        rf.save_nii(pred[1], os.path.join(out_dir, "pred_fl.nii.gz"),
                    s["aff"])

    # Verify what was written
    written = sorted(os.listdir(out_dir))
    print(f"  saved {out_dir}")
    for f in written:
        print(f"    {f}")

    # Clean up to free VRAM between experiments.
    del G, D, R, state
    torch.cuda.empty_cache()
    return True


def main():
    print(f"device={rf.cfg.device}  torch={torch.__version__}")
    report = []
    for variant, target in EXPERIMENTS:
        for subject in MISSING_SUBJECTS:
            ok = recover(variant, target, subject)
            report.append((variant, target, subject, "OK" if ok else "SKIP"))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for v, t, s, st in report:
        print(f"  {v:>14}  {t:<9}  {s:<20}  {st}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
