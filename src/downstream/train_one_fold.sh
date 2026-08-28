#!/usr/bin/env bash
# Usage: train_one_fold.sh <DATASET_ID> <FOLD>
set -e
export nnUNet_raw="${IOUS2MR_ROOT}/downstream_seg/nnUNet_raw"
export nnUNet_preprocessed="${IOUS2MR_ROOT}/downstream_seg/nnUNet_preprocessed"
export nnUNet_results="${IOUS2MR_ROOT}/downstream_seg/nnUNet_results"
DSID=$1
FOLD=$2
${IOUS2MR_PY_TORCH:-python} -m nnunetv2.run.run_training $DSID 3d_fullres $FOLD -tr nnUNetTrainer_500epochs --npz
