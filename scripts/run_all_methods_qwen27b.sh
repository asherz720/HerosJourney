#!/usr/bin/env bash
# run_all_methods_qwen27b.sh — Phase 2 episode + rule scoring for all 4 methods, Qwen 27B.
#
# Steps:
#   1. Episode Phase 2 for react, hr, idea, ace (already-completed tasks skipped).
#   2. QA Phase 2 for react (standard pipeline with LLM judge).
#   3. Rule hypothesis scoring for hr, idea, ace:
#        hr / idea : judge the final hypothesis from phase_2 episode results directly.
#        ace       : generate rule articulation with frozen playbook, then judge.
#   4. Recompute metrics CSV.
#
# Requires the Qwen 27B vLLM server running on port 8001, e.g.:
#   CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
#       --model /home/az22555/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/b7ca741b86de18df552fd2cc952861e04621a4bd \
#       --tensor-parallel-size 2 --port 8001
#
# Usage:
#   ./scripts/run_all_methods_qwen27b.sh               # all 4 methods, all 8 tasks
#   ./scripts/run_all_methods_qwen27b.sh --only conditional

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

RESULTS_DIR="results/qwen27b"
MODEL_PATH="/home/az22555/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/b7ca741b86de18df552fd2cc952861e04621a4bd"
JUDGE_MODEL="/home/az22555/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/b7ca741b86de18df552fd2cc952861e04621a4bd"
CONVERTER_MODEL="small"
DISTRACTOR_RULES="tree_management/distractors/canonical_property_distractor.json"
NUM_VARIANTS=20
NUM_DISTRACTOR_SAMPLES=2

EXTRA_ARGS=("$@")

# ---------------------------------------------------------------------------
# Step 1 — Episode Phase 2 for all methods
# ---------------------------------------------------------------------------
echo "========================================"
echo "Step 1: Episode Phase 2 — all methods"
echo "========================================"
for METHOD in react hr idea ace; do
    echo ""
    echo "--- episode / $METHOD ---"
    "$SCRIPT_DIR/run_experiments.sh" "benchmark_episode_${METHOD}_qwen27b" "${EXTRA_ARGS[@]}"
done

# ---------------------------------------------------------------------------
# Step 2 — QA Phase 2 for react
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Step 2: QA Phase 2 — react"
echo "========================================"
"$SCRIPT_DIR/run_experiments.sh" benchmark_qa_react_qwen27b "${EXTRA_ARGS[@]}"

# ---------------------------------------------------------------------------
# Step 3 — Rule hypothesis scoring for hr, idea, ace
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Step 3: Rule hypothesis scoring — hr, idea (judge-only)"
echo "========================================"
python analysis/score_rule_hypotheses.py \
    --methods    hr,idea \
    --base       qwen27b \
    --results_dir "$RESULTS_DIR" \
    --judge_model "$JUDGE_MODEL" \
    --converter_model "$CONVERTER_MODEL" \
    --num_variants "$NUM_VARIANTS" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "========================================"
echo "Step 3: Rule hypothesis scoring — ace (model + judge)"
echo "========================================"
python analysis/score_rule_hypotheses.py \
    --methods    ace \
    --base       qwen27b \
    --results_dir "$RESULTS_DIR" \
    --model_path "$MODEL_PATH" \
    --judge_model "$JUDGE_MODEL" \
    --converter_model "$CONVERTER_MODEL" \
    --distractor_rules "$DISTRACTOR_RULES" \
    --num_distractor_samples "$NUM_DISTRACTOR_SAMPLES" \
    --num_variants "$NUM_VARIANTS" \
    "${EXTRA_ARGS[@]}"

# ---------------------------------------------------------------------------
# Step 4 — Recompute metrics
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Step 4: Recomputing metrics"
echo "========================================"
python analysis/compute_phase2_metrics.py \
    --base_models qwen27b \
    --methods     react,hr,idea,ace \
    --include_phase1 \
    --output      analysis/figures/metrics_phase2_qwen27b.csv

echo ""
echo "All done. Results in $RESULTS_DIR, metrics in analysis/figures/metrics_phase2_qwen27b.csv"
