#!/usr/bin/env bash
# Orchestration:
#   1) Wait for Dataset501 fold 4 to finish (the existing chain trains 1-4).
#   2) Hard-stop the existing chain BEFORE it can launch Dataset502 fold 0.
#   3) Run T2 inference for all 51 sets (Dataset501, 5-fold ensemble).
#   4) Compute T2 metrics + Wilcoxon.
#   5) Train Dataset502 folds 0-4 (FLAIR, 500 ep each).
#   6) Run FLAIR inference for all 25 sets.
#   7) Compute FLAIR metrics + cross-modal dual T2↔FLAIR table.
set -u
export nnUNet_raw="${IOUS2MR_ROOT}/downstream_seg/nnUNet_raw"
export nnUNet_preprocessed="${IOUS2MR_ROOT}/downstream_seg/nnUNet_preprocessed"
export nnUNet_results="${IOUS2MR_ROOT}/downstream_seg/nnUNet_results"
PY=${IOUS2MR_PY_TORCH:-python}
DS=${IOUS2MR_ROOT}/downstream_seg
LOGDIR=$DS/logs
mkdir -p "$LOGDIR"

DS501=$DS/nnUNet_results/Dataset501_T2/nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres
FOLD4_DONE=$DS501/fold_4/checkpoint_final.pth

echo "[orch] $(date) check 501 fold 4 final checkpoint"
if [ ! -f "$FOLD4_DONE" ]; then
    echo "[orch] fold 4 not yet present, waiting"
    while [ ! -f "$FOLD4_DONE" ]; do sleep 30; done
fi
echo "[orch] $(date) 501 ready; ensuring no stale GPU processes"
# best-effort cleanup of any lingering nnUNet python (relaunch-friendly)
$PY -c "
import psutil, time
for p in psutil.process_iter(['pid','cmdline']):
    try:
        cl = ' '.join(p.info['cmdline'] or [])
        if 'nnunetv2' in cl and ('run_training' in cl or 'predict_from_raw_data' in cl):
            p.kill()
    except: pass
time.sleep(2)
" 2>/dev/null || true
sleep 3

# ----------------------------------------------- 3) T2 INFERENCE
echo "[orch] $(date) ===== T2 INFERENCE (51 sets) ====="
for d in "$DS/test_inputs_T2"/*/; do
    name=$(basename "$d")
    out=$DS/predictions_T2/$name
    if [ -d "$out" ] && [ "$(ls "$out"/*.nii.gz 2>/dev/null | wc -l)" -ge 1 ]; then
        echo "[orch] skip $name (already predicted)"
        continue
    fi
    mkdir -p "$out"
    LOG=$LOGDIR/predict_T2_${name}.log
    echo "[orch] $(date) predict T2 $name"
    $PY -c "from nnunetv2.inference.predict_from_raw_data import predict_entry_point; predict_entry_point()" \
        -i "$d" -o "$out" -d 501 -c 3d_fullres \
        -tr nnUNetTrainer_500epochs -f 0 1 2 3 4 \
        --disable_progress_bar > "$LOG" 2>&1 \
        || { echo "[orch] FAILED predict $name -- check $LOG"; }
done
echo "[orch] $(date) T2 inference done"

# ----------------------------------------------- 4) T2 METRICS
echo "[orch] $(date) ===== T2 METRICS ====="
$PY $DS/compute_seg_metrics.py > $LOGDIR/compute_metrics_T2only.log 2>&1
echo "[orch] $(date) T2 metrics done. CSVs:"
ls -la $DS/seg_*T2*.csv 2>/dev/null || true

# ----------------------------------------------- 5) FLAIR TRAINING
echo "[orch] $(date) ===== FLAIR TRAINING (502 folds 0-4) ====="
for f in 0 1 2 3 4; do
    LOG=$LOGDIR/train_502_fold${f}.log
    echo "[orch] $(date) train 502 fold $f -> $LOG"
    $PY -m nnunetv2.run.run_training 502 3d_fullres $f \
        -tr nnUNetTrainer_500epochs --npz > "$LOG" 2>&1 \
        || { echo "[orch] FAILED train 502 fold $f -- check $LOG"; exit 1; }
done
echo "[orch] $(date) FLAIR training done"

# ----------------------------------------------- 6) FLAIR INFERENCE
echo "[orch] $(date) ===== FLAIR INFERENCE (25 sets) ====="
for d in "$DS/test_inputs_FLAIR"/*/; do
    name=$(basename "$d")
    out=$DS/predictions_FLAIR/$name
    if [ -d "$out" ] && [ "$(ls "$out"/*.nii.gz 2>/dev/null | wc -l)" -ge 1 ]; then
        echo "[orch] skip $name (already predicted)"
        continue
    fi
    mkdir -p "$out"
    LOG=$LOGDIR/predict_FLAIR_${name}.log
    echo "[orch] $(date) predict FLAIR $name"
    $PY -c "from nnunetv2.inference.predict_from_raw_data import predict_entry_point; predict_entry_point()" \
        -i "$d" -o "$out" -d 502 -c 3d_fullres \
        -tr nnUNetTrainer_500epochs -f 0 1 2 3 4 \
        --disable_progress_bar > "$LOG" 2>&1 \
        || { echo "[orch] FAILED predict $name -- check $LOG"; }
done
echo "[orch] $(date) FLAIR inference done"

# ----------------------------------------------- 7) FINAL METRICS (T2+FLAIR+cross-modal)
echo "[orch] $(date) ===== FINAL METRICS (T2+FLAIR+cross) ====="
$PY $DS/compute_seg_metrics.py > $LOGDIR/compute_metrics_final.log 2>&1
echo "[orch] $(date) all metrics done. CSVs:"
ls -la $DS/seg_*.csv 2>/dev/null || true

echo "[orch] $(date) ALL DONE"
