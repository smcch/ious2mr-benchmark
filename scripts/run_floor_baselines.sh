#!/usr/bin/env bash
# Floor baselines for the downstream endpoint: segment the raw ioUS (and a histogram-matched
# variant) with the same frozen Seg-T2 nnU-Net used for every synthesis.
#
#   python src/downstream/stage_floor_baselines.py   # writes test_inputs_T2/FLOOR_US{,_HISTM}
#   bash   scripts/run_floor_baselines.sh            # this script
#   python src/downstream/floor_baseline.py          # scores them
set -e
DS=${IOUS2MR_SOURCE_TREE:-E:/SINTESIS}/downstream_seg
export nnUNet_raw="$DS/nnUNet_raw"
export nnUNet_preprocessed="$DS/nnUNet_preprocessed"
export nnUNet_results="$DS/nnUNet_results"
PY=${IOUS2MR_PYTHON:-python}

mkdir -p "$DS/logs"
for name in FLOOR_US FLOOR_US_HISTM; do
    in="$DS/test_inputs_T2/$name"
    out="$DS/predictions_T2/$name"
    mkdir -p "$out"
    echo "[floor] $(date) $name"
    "$PY" -c "from nnunetv2.inference.predict_from_raw_data import predict_entry_point; predict_entry_point()" \
        -i "$in" -o "$out" -d 501 -c 3d_fullres \
        -tr nnUNetTrainer_500epochs -f 0 1 2 3 4 \
        --continue_prediction --disable_progress_bar \
        > "$DS/logs/predict_T2_${name}.log" 2>&1
    echo "[floor] $(date) $name done: $(ls "$out"/*.nii.gz 2>/dev/null | wc -l) volumes"
done
echo "[floor] ALL DONE"
