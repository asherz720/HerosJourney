#!/usr/bin/env bash
# End-to-end smoke test for Qwen 27B Phase 2 pipeline.
#
# Runs a minimal version (1 variant, 2 gen tasks) of episode + QA for one
# method (default: react) on one task (default: additive), then verifies the
# output files exist and are well-formed.
#
# Usage (from repo root):
#   ./scripts/smoke_test_run_qwen27b.sh                          # react / additive (local)
#   ./scripts/smoke_test_run_qwen27b.sh --method hr              # hr / additive (local)
#   ./scripts/smoke_test_run_qwen27b.sh --method idea --task conditional
#   ./scripts/smoke_test_run_qwen27b.sh --model Qwen/Qwen3.5-27B # TACC
#
# The test writes to results/smoke_qwen27b/ so it never touches real results.
# Clean up afterwards: rm -rf results/smoke_qwen27b/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

METHOD="react"
TASK="additive"
MODEL_PATH="/home/az22555/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/b7ca741b86de18df552fd2cc952861e04621a4bd"

while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD="$2"; shift 2 ;;
        --task)   TASK="$2";   shift 2 ;;
        --model)  MODEL_PATH="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done
ELEMENTS="tree_management/rules/${TASK}.json"
SMOKE_DIR="results/smoke_qwen27b"
SAVE_NAME="${TASK}_smoke_qwen27b_${METHOD}"
P1_FILE="${SMOKE_DIR}/phase_1_${TASK}_smoke_qwen27b.json"

echo "========================================"
echo "Smoke test: Qwen 27B Phase 2"
echo "  method  : $METHOD"
echo "  task    : $TASK"
echo "  output  : $SMOKE_DIR"
echo "========================================"

cd "$REPO_DIR"

# --------------------------------------------------------------------------
# Step 1: Phase 1 (1 variant, 2 gen tasks, 1 worker)
# --------------------------------------------------------------------------
echo ""
echo "--- Step 1: Phase 1 (baseline, 1 variant) ---"
python3 -m pipeline.run_adventure_story \
    --task_type "$TASK" \
    --elements "$ELEMENTS" \
    --model "$MODEL_PATH" \
    --num_variants 1 \
    --n_gen_tasks 2 \
    --num_workers 1 \
    --num_tries max \
    --converter_model small \
    --distractor_rules tree_management/distractors/canonical_property_distractor.json \
    --num_distractor_samples 2 \
    --demo_repeats 1 \
    --skip_phase2 \
    --save_name "${TASK}_smoke_qwen27b" \
    --results_dir "$SMOKE_DIR"

if [ ! -f "$P1_FILE" ]; then
    echo "FAIL: Phase 1 file not created: $P1_FILE"
    exit 1
fi
echo "OK: $P1_FILE"

# --------------------------------------------------------------------------
# Step 2: Episode Phase 2 (uses Phase 1 results)
# --------------------------------------------------------------------------
echo ""
echo "--- Step 2: Episode Phase 2 (method=$METHOD) ---"
python3 -m pipeline.run_adventure_story \
    --task_type "$TASK" \
    --elements "$ELEMENTS" \
    --model "$MODEL_PATH" \
    --num_variants 1 \
    --n_gen_tasks 2 \
    --num_workers 1 \
    --num_tries max \
    --converter_model small \
    --distractor_rules tree_management/distractors/canonical_property_distractor.json \
    --num_distractor_samples 2 \
    --demo_repeats 1 \
    --skip_phase1 \
    --phase1_results "$P1_FILE" \
    --teaching_strategy "$METHOD" \
    --save_name "$SAVE_NAME" \
    --results_dir "$SMOKE_DIR"

EP2_FILE="${SMOKE_DIR}/phase_2_${SAVE_NAME}.json"
P11_FILE="${SMOKE_DIR}/phase_1_1_${SAVE_NAME}.json"

if [ ! -f "$EP2_FILE" ]; then
    echo "FAIL: Episode Phase 2 file not created: $EP2_FILE"
    exit 1
fi
echo "OK: $EP2_FILE"

if [ "$METHOD" != "react" ] && [ ! -f "$P11_FILE" ]; then
    echo "FAIL: Phase 1.1 file not created: $P11_FILE"
    exit 1
fi
[ -f "$P11_FILE" ] && echo "OK: $P11_FILE"

# --------------------------------------------------------------------------
# Step 3: QA Phase 2 (conditioned on episode Phase 2 teaching message)
# --------------------------------------------------------------------------
echo ""
echo "--- Step 3: QA Phase 2 (method=$METHOD) ---"

P11_FLAG=""
if [ "$METHOD" != "react" ] && [ -f "$P11_FILE" ]; then
    P11_FLAG="--phase1_1_results $P11_FILE"
fi

python3 -m pipeline.run_adventure_story \
    --task_type "$TASK" \
    --elements "$ELEMENTS" \
    --model "$MODEL_PATH" \
    --num_variants 1 \
    --n_gen_tasks 2 \
    --num_workers 1 \
    --converter_model small \
    --distractor_rules tree_management/distractors/canonical_property_distractor.json \
    --num_distractor_samples 2 \
    --demo_repeats 1 \
    --qa_mode all \
    --qa_teaching_strategy "$METHOD" \
    $P11_FLAG \
    --save_name "$SAVE_NAME" \
    --results_dir "$SMOKE_DIR"

QA2_FILE="${SMOKE_DIR}/qa_phase2_${SAVE_NAME}.json"
QA1_FILE="${SMOKE_DIR}/qa_phase1_${SAVE_NAME}.json"

if [ ! -f "$QA1_FILE" ]; then
    echo "FAIL: QA Phase 1 file not created: $QA1_FILE"
    exit 1
fi
echo "OK: $QA1_FILE"

if [ ! -f "$QA2_FILE" ]; then
    echo "FAIL: QA Phase 2 file not created: $QA2_FILE"
    exit 1
fi
echo "OK: $QA2_FILE"

# --------------------------------------------------------------------------
# Step 4: Verify content
# --------------------------------------------------------------------------
echo ""
echo "--- Step 4: Content check ---"
python3 - <<PYEOF
import json, sys

def check(path, label):
    try:
        d = json.load(open(path))
    except Exception as e:
        print(f"  FAIL {label}: {e}")
        return False
    variants = d.get("variants", [])
    print(f"  OK   {label}: {len(variants)} variant(s)")

    # Check teaching_message present for Phase 2 episode
    if "phase_2" in path:
        for v in variants:
            for ep in list(v.get("episodes", {}).values())[:1]:
                tm = ep.get("teaching_message", "")
                if not tm:
                    print(f"       WARNING: teaching_message is empty in episode")
                else:
                    print(f"       teaching_message: {tm[:80]}...")
    # Check QA Phase 2 has instance results
    if "qa_phase2" in path:
        for v in variants[:1]:
            inst = v.get("instance", {})
            sexp = v.get("structure_exp", {})
            print(f"       instance: {inst.get('correct',0)}/{inst.get('total',0)}")
            print(f"       rule_score: {sexp.get('rule_score','?')}")
    return True

import sys
ok = True
ok &= check("${EP2_FILE}", "episode phase_2")
ok &= check("${QA1_FILE}", "qa_phase1")
ok &= check("${QA2_FILE}", "qa_phase2")
sys.exit(0 if ok else 1)
PYEOF

echo ""
echo "========================================"
echo "Smoke test PASSED: method=$METHOD task=$TASK"
echo "Output in: $SMOKE_DIR"
echo "Clean up:  rm -rf $SMOKE_DIR"
echo "========================================"
