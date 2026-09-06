#!/bin/bash
# ============================================================
# ICML Ablation Study Sweep
# Runs the ablation-only evaluation across all tasks.
# Much faster than the full sweep — no FPR, no TL-ACDC.
#
# Usage:
#   bash scripts/run_ablation_sweep.sh
#   MODEL=gpt2 N_SEEDS=10 bash scripts/run_ablation_sweep.sh
# ============================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL_ID="${MODEL:-gpt2}"
N_SEEDS="${N_SEEDS:-50}"
N_BEHAVIORS="${N_BEHAVIORS:-3}"
DEVICE="${DEVICE:-cuda}"
OUT_DIR="${OUT_DIR:-output/icml_ablations}"

echo "======================================================"
echo " Ablation Sweep: $MODEL_ID | n=$N_SEEDS"
echo "======================================================"
mkdir -p "$OUT_DIR"

for TASK in induction greater_than agentic; do
    echo ""
    echo "  Task: $TASK"
    python run_icml_eval.py \
        --model "$MODEL_ID" \
        --task "$TASK" \
        --device "$DEVICE" \
        --n-seeds "$N_SEEDS" \
        --n-behaviors "$N_BEHAVIORS" \
        --methods "reference,ablation_no_skeptic,ablation_no_seap,ablation_random_layers,ablation_acdc_only" \
        --out-dir "$OUT_DIR" \
        --ablation-only \
        --skip-tl \
        2>&1 | tee -a "$OUT_DIR/ablations_${TASK}.log"
done

echo ""
echo "Ablation study complete. Results in $OUT_DIR/"
