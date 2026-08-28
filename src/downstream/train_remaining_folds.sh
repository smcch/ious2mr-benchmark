#!/usr/bin/env bash
# Chained training: waits for Dataset501 fold 0 (already running) to finish,
# then trains Dataset501 folds 1-4 and Dataset502 folds 0-4 sequentially.
# Each fold uses nnUNetTrainer_500epochs (500 ep, ~6 h on RTX 3090).
set -e
export nnUNet_raw="${IOUS2MR_ROOT}/downstream_seg/nnUNet_raw"
export nnUNet_preprocessed="${IOUS2MR_ROOT}/downstream_seg/nnUNet_preprocessed"
export nnUNet_results="${IOUS2MR_ROOT}/downstream_seg/nnUNet_results"
PY=${IOUS2MR_PY_TORCH:-python}

LOGDIR=${IOUS2MR_ROOT}/downstream_seg/logs
mkdir -p "$LOGDIR"

DS501_FINAL="${IOUS2MR_ROOT}/downstream_seg/nnUNet_results/Dataset501_T2/nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"

echo "[wait] $(date) waiting for Dataset501 fold 0 to finish (checkpoint_final.pth)"
until [ -f "$DS501_FINAL" ]; do sleep 60; done
echo "[wait] $(date) fold 0 done."

train() {
    DSID=$1; FOLD=$2; LOG=$3
    echo "[train] $(date) Dataset${DSID} fold ${FOLD} -> $LOG"
    $PY -m nnunetv2.run.run_training $DSID 3d_fullres $FOLD -tr nnUNetTrainer_500epochs --npz > "$LOG" 2>&1
    echo "[train] $(date) Dataset${DSID} fold ${FOLD} DONE"
}

for f in 1 2 3 4; do
    train 501 $f "$LOGDIR/train_501_fold${f}.log"
done
for f in 0 1 2 3 4; do
    train 502 $f "$LOGDIR/train_502_fold${f}.log"
done

echo "[done] $(date) all folds trained"
