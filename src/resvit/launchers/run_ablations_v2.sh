#!/bin/bash
# Sequential ablation runner for resvit_ablation_v2.py
# Variant: 2.5d (best visually). Targets: t2 and t2_flair.
# Ablations: B (perceptual), C (perceptual+freq), D (perceptual+freq+hier).
# Each run does phase1 (100 ep CNN) + phase2 (100 ep ART).
set -u
cd "${IOUS2MR_ROOT}/resvit"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate nnunet

export PYTHONUNBUFFERED=1

LOGDIR="output/ablation/_logs"
mkdir -p "$LOGDIR"

run_one() {
    local ABL="$1" TGT="$2"
    local LOG="$LOGDIR/run_25d_${TGT}_abl${ABL}.log"
    echo "==========================================================" | tee -a "$LOG"
    echo "[$(date)] START v2 ablation ${ABL}  target=${TGT}"           | tee -a "$LOG"
    echo "==========================================================" | tee -a "$LOG"
    python resvit_ablation_v2.py --variant 2.5d --target "$TGT" --ablation "$ABL" \
        2>&1 | tee -a "$LOG"
    local EXIT=${PIPESTATUS[0]}
    echo "[$(date)] END ablation ${ABL}/${TGT} (exit=$EXIT)" | tee -a "$LOG"
    if [ "$EXIT" -ne 0 ]; then
        echo "[$(date)] Run ${ABL}/${TGT} failed — continuing" | tee -a "$LOG"
    fi
}

# Order: B first (both targets), then C, then D
for ABL in B C D; do
    for TGT in t2 t2_flair; do
        run_one "$ABL" "$TGT"
    done
done

echo "[$(date)] ALL v2 ABLATIONS DONE." | tee -a "$LOGDIR/summary.log"
