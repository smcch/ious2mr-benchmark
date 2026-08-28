#!/usr/bin/env bash
# Run 5-fold ensemble inference for every test set.
# Skips sets whose output dir already exists with the expected file count.
set -e
export nnUNet_raw="${IOUS2MR_ROOT}/downstream_seg/nnUNet_raw"
export nnUNet_preprocessed="${IOUS2MR_ROOT}/downstream_seg/nnUNet_preprocessed"
export nnUNet_results="${IOUS2MR_ROOT}/downstream_seg/nnUNet_results"
PY=${IOUS2MR_PY_TORCH:-python}

DS=${IOUS2MR_ROOT}/downstream_seg

predict_one() {
    DSID=$1; INDIR=$2; OUTDIR=$3
    if [ -d "$OUTDIR" ] && [ "$(ls -1 "$OUTDIR"/*.nii.gz 2>/dev/null | wc -l)" -ge 1 ]; then
        echo "[skip] $OUTDIR already populated"
        return 0
    fi
    mkdir -p "$OUTDIR"
    echo "[pred] $(date) -> $OUTDIR"
    $PY -m nnunetv2.inference.predict_from_raw_data \
        -i "$INDIR" -o "$OUTDIR" -d $DSID -c 3d_fullres \
        -tr nnUNetTrainer_500epochs -f 0 1 2 3 4 \
        --save_probabilities --disable_progress_bar 2>&1 | tail -5
}

echo "=== T2 inference (Dataset501, 5-fold ensemble) ==="
for d in "$DS/test_inputs_T2"/*/; do
    name=$(basename "$d")
    predict_one 501 "$d" "$DS/predictions_T2/$name"
done

echo "=== FLAIR inference (Dataset502, 5-fold ensemble) ==="
for d in "$DS/test_inputs_FLAIR"/*/; do
    name=$(basename "$d")
    predict_one 502 "$d" "$DS/predictions_FLAIR/$name"
done

echo "[done] $(date)"
