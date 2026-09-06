#!/bin/bash
# ============================================================
# ICML Large-Model Evaluation Sweep
# Runs the full ICML eval pipeline across model scale ladder.
#
# Models: GPT-2 → Pythia-1.4B → Pythia-6.9B → Llama-3-8B
# Tasks:  induction, greater_than, agentic (all three)
# Seeds:  50 per task (adjustable via N_SEEDS)
#
# Usage:
#   bash scripts/run_large_model_sweep.sh
#   N_SEEDS=10 bash scripts/run_large_model_sweep.sh   # fast
#   MODEL=gpt2 bash scripts/run_large_model_sweep.sh  # single model
# ============================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

N_SEEDS="${N_SEEDS:-50}"
N_BEHAVIORS="${N_BEHAVIORS:-3}"
DEVICE="${DEVICE:-cuda}"
OUT_DIR="${OUT_DIR:-output/icml_large_model}"
SP_STEPS="${SP_STEPS:-200}"
ACD_BUDGET="${ACD_BUDGET:-40}"
RL_EPISODES="${RL_EPISODES:-60}"

# Ordered model ladder (small → large)
if [ -n "${MODEL:-}" ]; then
    MODELS=("$MODEL")
else
    MODELS=(
        "gpt2"
        "EleutherAI/pythia-1.4b"
        "EleutherAI/pythia-6.9b"
        "meta-llama/Llama-3.1-8B"
    )
fi

TASKS=("induction" "greater_than" "agentic")

echo "======================================================"
echo " ICML Large-Model Sweep"
echo "  Models:    ${MODELS[*]}"
echo "  Tasks:     ${TASKS[*]}"
echo "  Seeds:     $N_SEEDS"
echo "  Out dir:   $OUT_DIR"
echo "======================================================"

mkdir -p "$OUT_DIR"

for MODEL_ID in "${MODELS[@]}"; do
    echo ""
    echo "======================================================"
    echo " Model: $MODEL_ID"
    echo "======================================================"
    for TASK in "${TASKS[@]}"; do
        echo ""
        echo "  Task: $TASK"
        python run_icml_eval.py \
            --model "$MODEL_ID" \
            --task "$TASK" \
            --device "$DEVICE" \
            --n-seeds "$N_SEEDS" \
            --n-behaviors "$N_BEHAVIORS" \
            --out-dir "$OUT_DIR" \
            --sp-n-steps "$SP_STEPS" \
            --acd-budget "$ACD_BUDGET" \
            --rl-episodes "$RL_EPISODES" \
            --skip-tl \
            2>&1 | tee -a "$OUT_DIR/${MODEL_ID//\//_}_${TASK}.log"
        echo "  [Done] $MODEL_ID / $TASK"
    done
done

echo ""
echo "======================================================"
echo " Sweep complete. Results in $OUT_DIR/"
echo "======================================================"
