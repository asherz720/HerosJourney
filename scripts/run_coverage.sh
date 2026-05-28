#!/usr/bin/env bash
# run_coverage.sh — identifiability coverage sweep (k=0 through k=2×src)
#
# Varies the number of source demos shown (k) across a fixed step grid per task,
# running both the episode pipeline (Phase 1 only) and QA (structure_exp) at each step.
# k=0 (no demos, no distractors) is always run first as the baseline.
#
# Step grids (src_size → [k1, k2, k3, k_double]):
#   additive      (src=6):  2-4-6-12
#   compositional (src=6):  2-4-6-12
#   override      (src=7):  2-4-7-14
#   conditional   (src=8):  3-5-8-16
#   proc_add      (src=3):  1-2-3-6
#   proc_comp     (src=3):  1-2-3-6
#   proc_cond     (src=7):  2-4-7-14
#   proc_over     (src=9):  3-6-9-18
#
# Usage:
#   ./scripts/run_coverage.sh <model_path> [--judge_model <path>] [--episode_only] [--parallel] [--only <task>] [extra flags...]
#
# Examples:
#   ./scripts/run_coverage.sh gemini --num_workers 4
#   ./scripts/run_coverage.sh /path/to/qwen27b --num_workers 4 --only additive
#   ./scripts/run_coverage.sh gemini --only proc_cond --num_variants 5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
RULES_DIR="${REPO_DIR}/tree_management/rules"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
MODEL_PATH=""
ONLY_TASK=""
JUDGE_MODEL=""
EPISODE_ONLY=false
PARALLEL=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --only)
            ONLY_TASK="$2"; shift 2 ;;
        --judge_model)
            JUDGE_MODEL="$2"; shift 2 ;;
        --episode_only)
            EPISODE_ONLY=true; shift ;;
        --parallel)
            PARALLEL=true; shift ;;
        *)
            if [ -z "$MODEL_PATH" ] && [[ "$1" != --* ]]; then
                MODEL_PATH="$1"; shift
            else
                EXTRA_ARGS+=("$1"); shift
            fi
            ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <model_path> [--judge_model <path>] [--episode_only] [--only <task_type>] [extra flags...]"
    echo ""
    echo "  model_path      : 'gemini', 'gpt-5.4-mini', or /path/to/local/model[@PORT]"
    echo "  --judge_model   : model for LLM-as-judge in structure_exp (default: same as model_path)"
    echo "  --episode_only  : skip QA, run only episode pipeline"
    echo "  --parallel      : run all tasks simultaneously in background processes"
    echo "  --only TASK     : run only this task type"
    exit 1
fi

# Derive a short stem for save names
MODEL_STEM="$(basename "$MODEL_PATH" | tr '/' '_' | cut -c1-20)"

cd "$REPO_DIR"

# ---------------------------------------------------------------------------
# Task step grids
# ---------------------------------------------------------------------------
# Format: "task_type:k1,k2,k3,k_double"
declare -A GRIDS=(
    [additive]="2,4,6,12"
    [compositional]="2,4,6,12"
    [override]="2,4,7,14"
    [conditional]="3,5,8,16"
    [proc_add]="1,2,3,6"
    [proc_comp]="1,2,3,6"
    [proc_cond]="2,4,7,14"
    [proc_over]="3,6,9,18"
)

TASK_ORDER="additive compositional override conditional proc_add proc_comp proc_cond proc_over"

# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------
_run_episode_k0() {
    local TASK="$1" SAVE="$2"
    echo "  [episodes] task=${TASK} k=0 → ${SAVE}"
    python3 -m pipeline.run_adventure_story \
        --task_type              "$TASK" \
        --elements               "${RULES_DIR}/${TASK}.json" \
        --model                  "$MODEL_PATH" \
        --save_name              "$SAVE" \
        --n_source_demos         0 \
        --n_gen_tasks            1 \
        --num_variants           10 \
        --num_tries              3 \
        --num_distractor_samples 0 \
        --skip_phase2 \
        "${EXTRA_ARGS[@]}"
}

_run_episode_step() {
    local TASK="$1" K="$2" SAVE="$3"
    echo "  [episodes] task=${TASK} k=${K} → ${SAVE}"
    python3 -m pipeline.run_adventure_story \
        --task_type      "$TASK" \
        --elements       "${RULES_DIR}/${TASK}.json" \
        --model          "$MODEL_PATH" \
        --save_name      "$SAVE" \
        --n_source_demos "$K" \
        --n_gen_tasks    1 \
        --num_variants   10 \
        --num_tries      max \
        --skip_phase2 \
        "${EXTRA_ARGS[@]}"
}

_run_qa_step() {
    local TASK="$1" K="$2" SAVE="$3"
    echo "  [qa] task=${TASK} k=${K} → ${SAVE}"
    local JUDGE_ARGS=()
    [ -n "$JUDGE_MODEL" ] && JUDGE_ARGS=(--judge_model "$JUDGE_MODEL")
    python3 -m pipeline.run_adventure_story \
        --task_type      "$TASK" \
        --elements       "${RULES_DIR}/${TASK}.json" \
        --model          "$MODEL_PATH" \
        --save_name      "$SAVE" \
        --n_source_demos "$K" \
        --n_gen_tasks    1 \
        --num_variants   10 \
        --qa_mode        structure_exp \
        "${JUDGE_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
}

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
echo "========================================"
echo "Coverage sweep"
echo "Model        : $MODEL_PATH"
echo "Judge        : ${JUDGE_MODEL:-"(same as model)"}"
echo "Episode only : $EPISODE_ONLY"
echo "Parallel     : $PARALLEL"
echo "Num variants : 10 (fixed)"
echo "========================================"
echo ""

_run_task() {
    local TASK="$1"
    local GRID="${GRIDS[$TASK]}"
    echo "========================================"
    echo "Task: ${TASK}  grid: 0,${GRID}"
    echo "========================================"

    # k=0 baseline: no demos, no distractors, capped at 3 tries
    echo "[${TASK}] Step k=0"
    _run_episode_k0 "$TASK" "coverage_${TASK}_k0_${MODEL_STEM}"
    if [ "$EPISODE_ONLY" = false ]; then
        local JUDGE_ARGS=()
        [ -n "$JUDGE_MODEL" ] && JUDGE_ARGS=(--judge_model "$JUDGE_MODEL")
        python3 -m pipeline.run_adventure_story \
            --task_type              "$TASK" \
            --elements               "${RULES_DIR}/${TASK}.json" \
            --model                  "$MODEL_PATH" \
            --save_name              "coverage_${TASK}_k0_${MODEL_STEM}_qa" \
            --n_source_demos         0 \
            --n_gen_tasks            1 \
            --num_variants           10 \
            --num_distractor_samples 0 \
            --qa_mode                structure_exp \
            "${JUDGE_ARGS[@]}" \
            "${EXTRA_ARGS[@]}"
    fi

    IFS=',' read -ra STEPS <<< "$GRID"
    for K in "${STEPS[@]}"; do
        SAVE="coverage_${TASK}_k${K}_${MODEL_STEM}"
        echo "[${TASK}] Step k=${K}"
        _run_episode_step "$TASK" "$K" "$SAVE"
        if [ "$EPISODE_ONLY" = false ]; then
            _run_qa_step "$TASK" "$K" "${SAVE}_qa"
        fi
    done
    echo "[${TASK}] done."
}

for TASK in $TASK_ORDER; do
    if [ -n "$ONLY_TASK" ] && [ "$TASK" != "$ONLY_TASK" ]; then
        continue
    fi

    ELEM="${RULES_DIR}/${TASK}.json"
    if [ ! -f "$ELEM" ]; then
        echo "Warning: rule file not found, skipping ${TASK}: ${ELEM}"
        continue
    fi

    if [ "$PARALLEL" = true ]; then
        _run_task "$TASK" >> "${REPO_DIR}/results/log_coverage_${TASK}_${MODEL_STEM}.txt" 2>&1 &
        echo "Launched ${TASK} in background (log: results/log_coverage_${TASK}_${MODEL_STEM}.txt)"
    else
        _run_task "$TASK"
    fi
done

if [ "$PARALLEL" = true ]; then
    echo ""
    echo "All tasks launched. Waiting for completion..."
    wait
fi

echo "========================================"
echo "Coverage sweep complete."
echo "Results saved to: ${REPO_DIR}/results/coverage_*.json"
echo "========================================"
