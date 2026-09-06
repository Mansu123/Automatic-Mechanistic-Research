#!/bin/bash
# ============================================================
# ICML Smoke Test — runs everything in <5 minutes
# n=3 seeds, 1 behavior per task, GPT-2, all methods
# Use this to verify the full pipeline before a long run.
#
# Usage:
#   bash scripts/smoke_test_icml.sh
# ============================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "======================================================"
echo " ICML Pipeline Smoke Test (n=3, 1 behavior, GPT-2)"
echo "======================================================"

python run_icml_eval.py \
    --model gpt2 \
    --task all \
    --n-seeds 3 \
    --n-behaviors 1 \
    --sp-n-steps 30 \
    --acd-budget 10 \
    --rl-episodes 10 \
    --n-controls 3 \
    --out-dir output/smoke_test \
    --skip-tl \
    2>&1 | tee output/smoke_test.log

echo ""
echo "Smoke test complete. Check output/smoke_test/"
