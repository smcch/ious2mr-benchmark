#!/usr/bin/env python3
"""Assemble the public repository from the authors' working tree.

This is a ONE-OFF provenance script: it records exactly which files of the
private working tree at E:\\SINTESIS were published, and where they landed. It is
kept in the repository so the mapping is auditable, not because users need it.

Nothing under a private-cohort path is ever copied; the ALLOW list below is
explicit and the script refuses to copy anything matching DENY.
"""
from __future__ import annotations

import os

import shutil
import sys
from pathlib import Path

SRC = Path(os.environ.get("IOUS2MR_SOURCE_TREE", "."))
DST = Path(__file__).resolve().parents[1]

# Any path containing one of these fragments is never copied, whatever the manifest says.
DENY = ("bratislava", "Bratislava", "realtime_video", "low_grade_outputs",
        "ANON_", "subjects_corrected", "RHUH", "venv", "site-packages")

# (source, destination) — destination is relative to the repository root.
MANIFEST: list[tuple[str, str]] = [
    # ---------------- GAN baselines (TensorFlow) ----------------
    ("COMPARATIVA-3/common.py",                      "src/gan/common.py"),
    ("COMPARATIVA-3/architectures_2d.py",            "src/gan/architectures.py"),
    ("COMPARATIVA-3/run_all_experiments.py",         "src/gan/train_all_experiments.py"),
    ("COMPARATIVA-3/run_single_axis_inference.py",   "src/gan/infer_single_axis.py"),
    ("COMPARATIVA-3/run_eval_only.py",               "src/gan/eval_only.py"),
    ("COMPARATIVA-3/run_lpips_eval.py",              "src/gan/eval_lpips.py"),

    # ---------------- ResViT (PyTorch) ----------------
    ("resvit/resvit_final.py",                       "src/resvit/resvit_final.py"),
    ("resvit/resvit_fullres.py",                     "src/resvit/resvit_fullres.py"),
    ("resvit/resvit_mrspace.py",                     "src/resvit/resvit_mrspace.py"),
    ("resvit/resvit_ablation_v2_win.py",             "src/resvit/resvit_ablation.py"),
    ("resvit/eval_resvit_metrics.py",                "src/resvit/eval_metrics.py"),
    ("resvit/build_unified_tables.py",               "src/resvit/build_unified_tables.py"),
    ("resvit/analyze_pareto.py",                     "src/resvit/analyze_pareto.py"),
    ("resvit/recover_missing_subject.py",            "src/resvit/recover_missing_subject.py"),
    ("resvit/run_experiments.sh",                    "src/resvit/launchers/run_experiments.sh"),
    ("resvit/run_ablations_v2.sh",                   "src/resvit/launchers/run_ablations_v2.sh"),

    # ---------------- SynDiff (PyTorch, adversarial diffusion) ----------------
    # NOTE: upstream SynDiff/DDGAN code is NOT vendored (non-commercial licence);
    # scripts/setup_syndiff_upstream.py fetches it at a pinned commit.
    ("synthdiff/backbones3d.py",                     "src/syndiff/backbones3d.py"),
    ("synthdiff/build_unified_final.py",             "src/syndiff/build_unified_final.py"),
    ("synthdiff/eval_resvit_protocol.py",            "src/syndiff/eval_resvit_protocol.py"),
    ("synthdiff/eval_all_epochs.py",                 "src/syndiff/eval_all_epochs.py"),

    # ---------------- Downstream segmentation (nnU-Net v2) ----------------
    ("downstream_seg/build_datasets.py",                    "src/downstream/build_datasets.py"),
    ("downstream_seg/stage_test_inputs.py",                 "src/downstream/stage_test_inputs.py"),
    ("downstream_seg/compute_seg_metrics_paper_protocol.py","src/downstream/compute_seg_metrics.py"),
    ("downstream_seg/make_lesion_primary_tables.py",        "src/downstream/make_lesion_primary_tables.py"),
    ("downstream_seg/correlation_robustness.py",            "src/downstream/correlation_robustness.py"),
    ("downstream_seg/interobserver_MR.py",                  "src/downstream/interobserver_MR.py"),
    ("downstream_seg/interobserver_US.py",                  "src/downstream/interobserver_US.py"),
    ("downstream_seg/make_paper_figures.py",                "src/downstream/make_paper_figures.py"),
    ("downstream_seg/train_one_fold.sh",                    "src/downstream/train_one_fold.sh"),
    ("downstream_seg/train_remaining_folds.sh",             "src/downstream/train_remaining_folds.sh"),
    ("downstream_seg/run_inference.sh",                     "src/downstream/run_inference.sh"),
    ("downstream_seg/run_inference_incremental.sh",         "src/downstream/run_inference_incremental.sh"),
    ("downstream_seg/orchestrate_T2_then_FLAIR.sh",         "src/downstream/orchestrate_T2_then_FLAIR.sh"),
    ("downstream_seg/METHODOLOGY_downstream.md",            "docs/methodology_downstream.md"),
    ("downstream_seg/test_cohort.json",                     "configs/downstream_test_cohort.json"),

    # nnU-Net configuration needed to rebuild the datasets / reuse the released models
    ("downstream_seg/nnUNet_raw/Dataset501_T2/dataset.json",
     "configs/nnunet/Dataset501_T2/dataset.json"),
    ("downstream_seg/nnUNet_raw/Dataset502_FLAIR/dataset.json",
     "configs/nnunet/Dataset502_FLAIR/dataset.json"),
    ("downstream_seg/nnUNet_preprocessed/Dataset501_T2/splits_final.json",
     "configs/nnunet/Dataset501_T2/splits_final.json"),
    ("downstream_seg/nnUNet_preprocessed/Dataset502_FLAIR/splits_final.json",
     "configs/nnunet/Dataset502_FLAIR/splits_final.json"),
    ("downstream_seg/nnUNet_preprocessed/Dataset501_T2/nnUNetPlans.json",
     "configs/nnunet/Dataset501_T2/nnUNetPlans.json"),
    ("downstream_seg/nnUNet_preprocessed/Dataset502_FLAIR/nnUNetPlans.json",
     "configs/nnunet/Dataset502_FLAIR/nnUNetPlans.json"),

    # ---------------- Unified scoring harness ----------------
    ("evaluacion-final/rescore_all.py",              "src/scoring/rescore_all.py"),
    ("evaluacion-final/rescore_roi.py",              "src/scoring/rescore_roi.py"),
    ("evaluacion-final/build_paper_tables.py",       "src/scoring/build_paper_tables.py"),
    ("evaluacion-final/ROI_METHODOLOGY.md",          "docs/roi_methodology.md"),
    ("paper_assets/METHODOLOGY.md",                  "docs/methodology_training.md"),

    # ---------------- Figures ----------------
    ("paper_assets/make_fig_multitask_v2.py",        "src/figures/make_fig_multitask.py"),
    ("paper_assets/make_fig_downstream_v3.py",       "src/figures/make_fig_downstream.py"),
    ("paper_assets/make_figs_may_v2.py",             "src/figures/make_figs_quality.py"),
    ("paper_assets/make_fig_external.py",            "src/figures/make_fig_external.py"),
    ("paper_assets/fix_syndiff_dual_fidelity.py",    "src/figures/fix_syndiff_dual_fidelity.py"),

    # ---------------- Derived results (small, publishable: ReMIND IDs only) ----------------
    ("paper_assets/all_experiments_metrics.csv",     "results/fidelity/all_experiments_metrics.csv"),
    ("paper_assets/per_experiment_ci.csv",           "results/fidelity/per_experiment_ci.csv"),
    ("evaluacion-final/rescore_methods_persubject.csv", "results/fidelity/per_subject_metrics.csv"),
    ("evaluacion-final/rescore_methods_summary.csv",    "results/fidelity/method_summary.csv"),
    ("evaluacion-final/rescore_preop_summary.csv",      "results/fidelity/preop_summary.csv"),
    ("evaluacion-final/rescore_postop_summary.csv",     "results/fidelity/postop_summary.csv"),
    ("evaluacion-final/rescore_wilcoxon.csv",           "results/fidelity/wilcoxon.csv"),
    ("evaluacion-final/roi_methods_summary.csv",        "results/roi/roi_methods_summary.csv"),
    ("evaluacion-final/roi_methods_persubject.csv",     "results/roi/roi_methods_persubject.csv"),
    ("evaluacion-final/roi_methods_wilcoxon.csv",       "results/roi/roi_methods_wilcoxon.csv"),
    ("evaluacion-final/table_compute.csv",              "results/compute/table_compute.csv"),

    ("downstream_seg/results_paper_protocol/seg_results_T2.csv",
     "results/downstream/seg_results_T2.csv"),
    ("downstream_seg/results_paper_protocol/seg_results_FLAIR.csv",
     "results/downstream/seg_results_FLAIR.csv"),
    ("downstream_seg/results_paper_protocol/seg_metrics_T2_per_study.csv",
     "results/downstream/seg_metrics_T2_per_study.csv"),
    ("downstream_seg/results_paper_protocol/seg_metrics_FLAIR_per_study.csv",
     "results/downstream/seg_metrics_FLAIR_per_study.csv"),
    ("downstream_seg/results_paper_protocol/seg_wilcoxon_T2.csv",
     "results/downstream/seg_wilcoxon_T2.csv"),
    ("downstream_seg/results_paper_protocol/seg_wilcoxon_FLAIR.csv",
     "results/downstream/seg_wilcoxon_FLAIR.csv"),
    ("downstream_seg/results_paper_protocol/headline_lesion_primary.csv",
     "results/downstream/headline_lesion_primary.csv"),
    ("downstream_seg/results_paper_protocol/headline_tumor_cavity_secondary.csv",
     "results/downstream/headline_tumor_cavity_secondary.csv"),
    ("downstream_seg/results_paper_protocol/phase_breakdown.csv",
     "results/downstream/phase_breakdown.csv"),
    ("downstream_seg/results_paper_protocol/correlation_robustness.csv",
     "results/downstream/correlation_robustness.csv"),
    ("downstream_seg/results_paper_protocol/human_ceiling_vs_models.csv",
     "results/downstream/human_ceiling_vs_models.csv"),
    ("downstream_seg/results_paper_protocol/interobs_MR_summary.csv",
     "results/downstream/interobs_MR_summary.csv"),
    ("downstream_seg/results_paper_protocol/interobs_MR_per_study.csv",
     "results/downstream/interobs_MR_per_study.csv"),
]

# Whole directories to mirror (filtered by DENY and by suffix).
DIR_MANIFEST: list[tuple[str, str, tuple[str, ...]]] = [
    ("synthdiff/syndiff_src", "src/syndiff/vendor", (".py", ".cpp", ".cu", ".md", "LICENSE", "LICENSE_MIT")),
    ("downstream_seg/results_paper_protocol/lesion_primary", "results/downstream/lesion_primary", (".csv",)),
]


def denied(path: str) -> bool:
    return any(frag in path for frag in DENY)


def main() -> int:
    copied, missing, blocked = 0, [], []
    for rel_src, rel_dst in MANIFEST:
        if denied(rel_src):
            blocked.append(rel_src)
            continue
        s, d = SRC / rel_src, DST / rel_dst
        if not s.exists():
            missing.append(rel_src)
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        copied += 1

    for rel_src, rel_dst, suffixes in DIR_MANIFEST:
        base = SRC / rel_src
        if not base.exists():
            missing.append(rel_src + "/")
            continue
        for f in base.rglob("*"):
            if not f.is_file():
                continue
            rel = f.relative_to(base).as_posix()
            if denied(rel) or "__pycache__" in rel:
                continue
            if not (f.name in suffixes or f.suffix in suffixes):
                continue
            out = DST / rel_dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)
            copied += 1

    print(f"copied : {copied}")
    if blocked:
        print(f"BLOCKED by DENY ({len(blocked)}): {blocked}")
    if missing:
        print(f"missing ({len(missing)}):")
        for m in missing:
            print("   ", m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
