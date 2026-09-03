#!/usr/bin/env bash
# Seg-US control for the synthesis benchmark: train the direct-ioUS segmenters and predict the
# 29 test ioUS volumes.  Sequential on one GPU, resumable: a fold with checkpoint_final.pth is
# skipped, a fold with checkpoint_latest.pth is continued (--c).
#
#   Dataset503_US     ioUS -> MR-drawn labels      (primary control)   folds 0-4, 500 epochs
#   Dataset504_USlab  ioUS -> US-drawn labels      (sensitivity)       folds 0-4, 500 epochs
#
# Launch detached (PowerShell):
#   bash scripts/run_seg_us_chain.sh          (detached on Windows: Start-Process bash.exe ...)
# Progress: $IOUS2MR_SOURCE_TREE/downstream_seg/logs/seg_us_chain.log (+ train_503_fold*.log)
set -u
DS=${IOUS2MR_SOURCE_TREE:-E:/SINTESIS}/downstream_seg
export nnUNet_raw="$DS/nnUNet_raw"
export nnUNet_preprocessed="$DS/nnUNet_preprocessed"
export nnUNet_results="$DS/nnUNet_results"
export PYTHONIOENCODING=utf-8
PY=${IOUS2MR_PYTHON:-python}
LOGDIR=$DS/logs
MAIN=$LOGDIR/seg_us_chain.log
mkdir -p "$LOGDIR"
exec >> "$MAIN" 2>&1

say() { echo "[chain] $(date '+%F %T') $*"; }

train_ds() {   # train_ds <id> <name>
    local id=$1 name=$2 f
    local res=$nnUNet_results/Dataset${id}_${name}/nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres
    for f in 0 1 2 3 4; do
        if [ -f "$res/fold_$f/checkpoint_final.pth" ]; then say "Dataset$id fold $f already final -> skip"; continue; fi
        local cont=""
        [ -f "$res/fold_$f/checkpoint_latest.pth" ] && cont="--c" && say "Dataset$id fold $f: continuing from checkpoint_latest"
        say "Dataset$id fold $f: training start"
        $PY -m nnunetv2.run.run_training $id 3d_fullres $f -tr nnUNetTrainer_500epochs --npz $cont \
            > "$LOGDIR/train_${id}_fold${f}.log" 2>&1
        if [ -f "$res/fold_$f/checkpoint_final.pth" ]; then say "Dataset$id fold $f: DONE"
        else say "Dataset$id fold $f: FAILED (no checkpoint_final) -- see train_${id}_fold${f}.log"; return 1; fi
    done
}

predict_ds() {   # predict_ds <id> <name> <outname>
    local id=$1 name=$2 out=$DS/predictions_US/$3
    if [ -d "$out" ] && [ "$(ls "$out"/*.nii.gz 2>/dev/null | wc -l)" -ge 29 ]; then say "predictions $3 already present -> skip"; return 0; fi
    mkdir -p "$out"
    say "predict Dataset$id on test_inputs_US/REAL_US -> $3"
    $PY -c "from nnunetv2.inference.predict_from_raw_data import predict_entry_point; predict_entry_point()" \
        -i "$DS/test_inputs_US/REAL_US" -o "$out" -d $id -c 3d_fullres \
        -tr nnUNetTrainer_500epochs -f 0 1 2 3 4 --disable_progress_bar \
        > "$LOGDIR/predict_US_$3.log" 2>&1 && say "predict $3 DONE ($(ls "$out"/*.nii.gz | wc -l) files)" \
        || say "predict $3 FAILED -- see predict_US_$3.log"
}

say "===== Seg-US chain start (GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)) ====="
train_ds 503 US     && predict_ds 503 US    SEG_US_mrlab
# Dataset504_USlab (ioUS -> US-drawn labels) is built by build_dataset_us.py as a sensitivity
# variant but is not part of the reported analysis; add train_ds/predict_ds calls to run it.
say "===== chain finished ====="
