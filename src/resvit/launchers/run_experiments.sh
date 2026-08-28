#!/bin/bash
cd "${IOUS2MR_ROOT}/resvit"

echo "========================================"
echo "[$(date)] Experiment 1: 2.5d t2"
echo "========================================"
python resvit_final.py --variant 2.5d --target t2 2>&1 | tee output/run_25d_t2.log
E1=$?
echo "[$(date)] Experiment 1 finished (exit=$E1)"

echo ""
echo "========================================"
echo "[$(date)] Experiment 2: 2.5d t2_flair"
echo "========================================"
python resvit_final.py --variant 2.5d --target t2_flair 2>&1 | tee output/run_25d_t2_flair.log
E2=$?
echo "[$(date)] Experiment 2 finished (exit=$E2)"

echo ""
echo "[$(date)] ALL DONE. Exit codes: exp1=$E1 exp2=$E2"
