#!/usr/bin/env bash
# Re-run inference on all test sets with --continue_prediction so only
# the new (un-predicted) studies are processed. Existing predictions
# from the prior orchestrator run are preserved.
set -e
export nnUNet_raw="${IOUS2MR_ROOT}/downstream_seg/nnUNet_raw"
export nnUNet_preprocessed="${IOUS2MR_ROOT}/downstream_seg/nnUNet_preprocessed"
export nnUNet_results="${IOUS2MR_ROOT}/downstream_seg/nnUNet_results"
PY=${IOUS2MR_PY_TORCH:-python}
DS=${IOUS2MR_ROOT}/downstream_seg
LOGDIR=$DS/logs

echo "[inc] $(date) ===== T2 INCREMENTAL ====="
for d in "$DS/test_inputs_T2"/*/; do
    name=$(basename "$d")
    out=$DS/predictions_T2/$name
    mkdir -p "$out"
    in_count=$(ls "$d"*.nii.gz 2>/dev/null | wc -l)
    out_count=$(ls "$out"/*.nii.gz 2>/dev/null | wc -l)
    if [ "$in_count" -eq "$out_count" ]; then
        continue  # already up to date
    fi
    echo "[inc] $(date) T2 $name  in=$in_count  out=$out_count"
    $PY -c "from nnunetv2.inference.predict_from_raw_data import predict_entry_point; predict_entry_point()" \
        -i "$d" -o "$out" -d 501 -c 3d_fullres \
        -tr nnUNetTrainer_500epochs -f 0 1 2 3 4 \
        --continue_prediction --disable_progress_bar \
        > $LOGDIR/predict_T2_inc_${name}.log 2>&1 || echo "[inc] FAILED $name"
done

echo "[inc] $(date) ===== FLAIR INCREMENTAL ====="
for d in "$DS/test_inputs_FLAIR"/*/; do
    name=$(basename "$d")
    out=$DS/predictions_FLAIR/$name
    mkdir -p "$out"
    in_count=$(ls "$d"*.nii.gz 2>/dev/null | wc -l)
    out_count=$(ls "$out"/*.nii.gz 2>/dev/null | wc -l)
    if [ "$in_count" -eq "$out_count" ]; then
        continue
    fi
    echo "[inc] $(date) FLAIR $name  in=$in_count  out=$out_count"
    $PY -c "from nnunetv2.inference.predict_from_raw_data import predict_entry_point; predict_entry_point()" \
        -i "$d" -o "$out" -d 502 -c 3d_fullres \
        -tr nnUNetTrainer_500epochs -f 0 1 2 3 4 \
        --continue_prediction --disable_progress_bar \
        > $LOGDIR/predict_FLAIR_inc_${name}.log 2>&1 || echo "[inc] FAILED $name"
done

echo "[inc] $(date) ===== RECOMPUTE METRICS ====="
$PY $DS/compute_seg_metrics.py > $LOGDIR/compute_metrics_expanded.log 2>&1
echo "[inc] $(date) DONE"
