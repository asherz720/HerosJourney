#!/usr/bin/env bash
# run_experiments.sh — generalization benchmark experiment runner
#
# Supported experiment_type values in experiments.json:
#   phase1_only     — Phase 1 (demo only) only; skips Phase 2
#   phase2_only     — Phase 2 (demo + teaching) only; loads existing Phase 1 results
#                     Requires phase1_results in config or --phase1_results override
#   both_phases     — Run both Phase 1 and Phase 2
#   sweep           — Loop over property tasks (additive/compositional/conditional/override)
#                     Use elements_dir in config (default: tree_management/rules)
#   qa              — Q&A pipeline only (single task_type); episode pipeline skipped
#                     Requires qa_mode in config (instance|structure_mc|structure_exp|all)
#   qa_sweep        — Q&A pipeline over property tasks
#   proc_sweep      — Episode pipeline over proc tasks (proc_add/proc_comp/proc_cond/proc_over)
#   proc_qa_sweep   — Q&A pipeline over proc tasks
#   full_sweep      — Episode pipeline over all 8 tasks (property + proc)
#   full_qa_sweep   — Q&A pipeline over all 8 tasks (property + proc)
#
# Non-reserved config keys are forwarded as CLI flags to run_adventure_story.py.
#
# Usage:
#   ./run_experiments.sh <exp_name> [--config <file>] [extra flags...]
#
# Examples:
#   ./run_experiments.sh proc_add_episode_verify
#   ./run_experiments.sh proc_sweep_qwen27b
#   ./run_experiments.sh proc_sweep_qwen27b --only proc_add
#   ./run_experiments.sh proc_qa_sweep_qwen27b --num_variants 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${SCRIPT_DIR}/experiments.json"

EXPERIMENT_NAME=""
OVERRIDE_ARGS=()
ONLY_TASK=""      # if set, sweep runs only this task type

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --only)
            ONLY_TASK="$2"
            shift 2
            ;;
        --*)
            OVERRIDE_ARGS+=("$1")
            if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then
                OVERRIDE_ARGS+=("$2")
                shift 2
            else
                shift 1
            fi
            ;;
        *)
            EXPERIMENT_NAME="$1"
            shift
            ;;
    esac
done

if [ -z "$EXPERIMENT_NAME" ]; then
    echo "Usage: $0 <experiment_name> [--config <file>] [--only <task_type>] [run_adventure_story.py overrides...]"
    echo ""
    echo "  --only TASK    (sweep experiments) run only this task type"
    echo "                 property: additive|compositional|conditional|override"
    echo "                 proc:     proc_add|proc_comp|proc_cond|proc_over"
    echo ""
    echo "Available experiments:"
    export _CFG_PATH="$CONFIG_FILE"
    python3 - <<'PY'
import json, sys, os
cfg_path = os.environ["_CFG_PATH"]
try:
    with open(cfg_path) as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"  (config file not found: {cfg_path})", file=sys.stderr)
    sys.exit(1)
for k, v in data.items():
    if k.startswith("_"):
        continue
    etype = v.get("experiment_type", "phase1_only")
    desc  = v.get("description", "")
    nonce = " [nonce]" if v.get("nonce") else ""
    print(f"  [{etype:16s}] {k}{nonce}: {desc}")
PY
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

cd "$REPO_DIR"

# ---------------------------------------------------------------------------
# Parse config — emit unit-separator-delimited fields
# ---------------------------------------------------------------------------
export _CFG_PATH="$CONFIG_FILE"
export _EXP_NAME="$EXPERIMENT_NAME"

IFS=$'\x1f' read -r MODEL_TYPE MODEL_PATH BASE_NAME EXP_TYPE TEACHING_STRATEGY RESULTS_DIR_CFG \
               ELEMENTS ELEMENTS_DIR PHASE1_RESULTS PHASE1_1_RESULTS PHASE2_RESULTS \
               CLI_ARGS \
< <(python3 - <<'PY'
import json, shlex, sys, os

cfg_path = os.environ["_CFG_PATH"]
name     = os.environ["_EXP_NAME"]

with open(cfg_path) as f:
    data = json.load(f)

if name not in data:
    raise SystemExit(f"Error: experiment '{name}' not found in {cfg_path}")

exp = data[name]

model_type          = exp.get("model_type", "vllm")
model_path          = exp.get("model_path", "")
base_name           = exp.get("base_name", name)
exp_type            = exp.get("experiment_type", "phase1_only")
teaching_strategy   = exp.get("teaching_strategy", "none")
results_dir         = exp.get("results_dir", "")
elements            = exp.get("elements", "")
elements_dir        = exp.get("elements_dir", "tree_management/rules")
phase1_results      = exp.get("phase1_results", "")
phase1_1_results    = exp.get("phase1_1_results", "")
phase2_results      = exp.get("phase2_results", "")

# Reserved keys — handled explicitly, not forwarded verbatim to CLI
reserved = {
    "exp_number", "description", "experiment_type", "model_type",
    "base_name", "elements_dir", "elements",
    "model_path", "results_dir",
    "cuda_visible_devices", "port", "max_len", "tensor_parallel_size",
    "google_cloud_project", "google_cloud_location",
    "phase1_results", "phase1_1_results", "phase2_results",
}

cli_parts = []
for k, v in exp.items():
    if k in reserved:
        continue
    flag = f"--{k}"
    if isinstance(v, bool):
        if v:
            cli_parts.append(flag)
    else:
        cli_parts.extend([flag, str(v)])

print(model_type, model_path, base_name, exp_type, teaching_strategy, results_dir,
      elements, elements_dir,
      phase1_results, phase1_1_results, phase2_results,
      shlex.join(cli_parts),
      sep="\x1f")
PY
)

if [ -z "$MODEL_PATH" ]; then
    echo "Error: model_path is required in config for $EXPERIMENT_NAME"
    exit 1
fi

# ---------------------------------------------------------------------------
# Set up environment
# ---------------------------------------------------------------------------

# Google Cloud env for Gemini
if [ "$MODEL_TYPE" = "api" ] && [ "$MODEL_PATH" = "gemini" ]; then
    export GOOGLE_CLOUD_PROJECT="ling-discolab-np"
    export GOOGLE_CLOUD_LOCATION="global"
    export GOOGLE_GENAI_USE_VERTEXAI=True
fi

# Resolve results paths relative to REPO_DIR
if [ -n "$PHASE1_RESULTS" ] && [[ "$PHASE1_RESULTS" != /* ]]; then
    PHASE1_RESULTS="${REPO_DIR}/${PHASE1_RESULTS}"
fi
if [ -n "$PHASE1_1_RESULTS" ] && [[ "$PHASE1_1_RESULTS" != /* ]]; then
    PHASE1_1_RESULTS="${REPO_DIR}/${PHASE1_1_RESULTS}"
fi
if [ -n "$PHASE2_RESULTS" ] && [[ "$PHASE2_RESULTS" != /* ]]; then
    PHASE2_RESULTS="${REPO_DIR}/${PHASE2_RESULTS}"
fi

echo "========================================"
echo "Experiment : $EXPERIMENT_NAME"
echo "Type       : $EXP_TYPE"
echo "Model      : $MODEL_PATH"
echo "Base name  : $BASE_NAME"
echo "========================================"
echo ""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_run_one_task() {
    # $1 = task_type
    # $2 = elements path
    # $3 = save_name
    # Remaining args are extra flags (--skip_phase1, --phase1_results, etc.)
    local TASK_TYPE="$1"
    local ELEM_PATH="$2"
    local SAVE="$3"
    shift 3

    local RESULTS_DIR_FLAG=""
    if [ -n "$RESULTS_DIR_CFG" ]; then
        RESULTS_DIR_FLAG="--results_dir ${RESULTS_DIR_CFG}"
    fi

    echo "--- Task: $TASK_TYPE | save: $SAVE ---"
    python3 -m pipeline.run_adventure_story \
        --task_type   "$TASK_TYPE" \
        --elements    "$ELEM_PATH" \
        --model       "$MODEL_PATH" \
        --save_name   "$SAVE" \
        $RESULTS_DIR_FLAG \
        $CLI_ARGS \
        "$@" \
        "${OVERRIDE_ARGS[@]}"
    echo ""
}

# Sweep helper: iterate over a list of tasks, filter by $ONLY_TASK if set,
# skip missing rule files with a warning.
# Args: extra_flags...  (e.g. --skip_phase2)
# Uses globals: TASK_LIST, ELEMENTS_DIR, BASE_NAME
_run_sweep() {
    local EXTRA_FLAGS=("$@")
    for TASK in $TASK_LIST; do
        if [ -n "$ONLY_TASK" ] && [ "$TASK" != "$ONLY_TASK" ]; then
            continue
        fi
        ELEM="${ELEMENTS_DIR}/${TASK}.json"
        if [ ! -f "$ELEM" ]; then
            echo "Warning: rule file not found, skipping $TASK: $ELEM"
            continue
        fi
        _run_one_task "$TASK" "$ELEM" "${TASK}_${BASE_NAME}" "${EXTRA_FLAGS[@]}"
    done
}

# ---------------------------------------------------------------------------
# Dispatch on experiment type
# ---------------------------------------------------------------------------
case "$EXP_TYPE" in

    phase1_only)
        _run_one_task \
            "$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['$EXPERIMENT_NAME']['task_type'])")" \
            "$ELEMENTS" \
            "${BASE_NAME}" \
            --skip_phase2
        ;;

    phase2_only)
        if [ -z "$PHASE1_RESULTS" ]; then
            echo "Error: 'phase2_only' requires 'phase1_results' in config or --phase1_results override"
            exit 1
        fi
        if [ ! -f "$PHASE1_RESULTS" ]; then
            echo "Error: Phase 1 results file not found: $PHASE1_RESULTS"
            exit 1
        fi
        EXTRA_11=""
        if [ -n "$PHASE1_1_RESULTS" ] && [ -f "$PHASE1_1_RESULTS" ]; then
            EXTRA_11="--phase1_1_results $PHASE1_1_RESULTS"
        fi
        _run_one_task \
            "$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['$EXPERIMENT_NAME']['task_type'])")" \
            "$ELEMENTS" \
            "${BASE_NAME}" \
            --skip_phase1 \
            --phase1_results "$PHASE1_RESULTS" \
            $EXTRA_11
        ;;

    both_phases)
        _run_one_task \
            "$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['$EXPERIMENT_NAME']['task_type'])")" \
            "$ELEMENTS" \
            "${BASE_NAME}"
        ;;

    sweep)
        TASK_LIST="additive compositional conditional override"
        _run_sweep --skip_phase2
        ;;

    qa)
        _run_one_task \
            "$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['$EXPERIMENT_NAME']['task_type'])")" \
            "$ELEMENTS" \
            "${BASE_NAME}"
        ;;

    qa_sweep)
        TASK_LIST="additive compositional conditional override"
        _run_sweep
        ;;

    sweep_phase2)
        # Phase 2 sweep over property tasks using a teaching strategy.
        # Phase 1 results are located in results_dir (if set) or results/.
        # Phase 2 results are saved as {task}_{base_name}_{teaching_strategy}.
        METHOD="${TEACHING_STRATEGY:-none}"
        P1_DIR="${RESULTS_DIR_CFG:-${REPO_DIR}/results}"
        # Resolve relative paths
        if [[ "$P1_DIR" != /* ]]; then
            P1_DIR="${REPO_DIR}/${P1_DIR}"
        fi
        TASK_LIST="additive compositional conditional override"
        for TASK in $TASK_LIST; do
            if [ -n "$ONLY_TASK" ] && [ "$TASK" != "$ONLY_TASK" ]; then
                continue
            fi
            ELEM="${ELEMENTS_DIR}/${TASK}.json"
            if [ ! -f "$ELEM" ]; then
                echo "Warning: rule file not found, skipping $TASK: $ELEM"
                continue
            fi
            P1_FILE="${P1_DIR}/phase_1_${TASK}_${BASE_NAME}.json"
            if [ ! -f "$P1_FILE" ]; then
                echo "Warning: Phase 1 results not found for $TASK, skipping."
                echo "  Expected: $P1_FILE"
                echo "  Run the corresponding phase1 experiment first."
                continue
            fi
            _run_one_task "$TASK" "$ELEM" "${TASK}_${BASE_NAME}_${METHOD}" \
                --skip_phase1 \
                --phase1_results "$P1_FILE"
        done
        ;;

    full_sweep_phase2)
        # Phase 2 sweep over all 8 tasks (property + proc) using a teaching strategy.
        # Already-completed Phase 2 files are loaded and skipped automatically.
        METHOD="${TEACHING_STRATEGY:-none}"
        P1_DIR="${RESULTS_DIR_CFG:-${REPO_DIR}/results}"
        if [[ "$P1_DIR" != /* ]]; then
            P1_DIR="${REPO_DIR}/${P1_DIR}"
        fi
        TASK_LIST="additive compositional conditional override proc_add proc_comp proc_cond proc_over"
        for TASK in $TASK_LIST; do
            if [ -n "$ONLY_TASK" ] && [ "$TASK" != "$ONLY_TASK" ]; then
                continue
            fi
            ELEM="${ELEMENTS_DIR}/${TASK}.json"
            if [ ! -f "$ELEM" ]; then
                echo "Warning: rule file not found, skipping $TASK: $ELEM"
                continue
            fi
            P1_FILE="${P1_DIR}/phase_1_${TASK}_${BASE_NAME}.json"
            if [ ! -f "$P1_FILE" ]; then
                echo "Warning: Phase 1 results not found for $TASK, skipping."
                echo "  Expected: $P1_FILE"
                echo "  Run the corresponding phase1 experiment first."
                continue
            fi
            _run_one_task "$TASK" "$ELEM" "${TASK}_${BASE_NAME}_${METHOD}" \
                --skip_phase1 \
                --phase1_results "$P1_FILE"
        done
        ;;

    qa_full_sweep_phase2)
        # QA Phase 2 sweep over all 8 tasks using a teaching strategy.
        # Loads per-variant teaching messages from Phase 1.1 files (episode pipeline).
        # react builds its message from the demo tree; hr/idea/ace load from phase_1_1 files.
        METHOD="${TEACHING_STRATEGY:-none}"
        P1_DIR="${RESULTS_DIR_CFG:-${REPO_DIR}/results}"
        if [[ "$P1_DIR" != /* ]]; then
            P1_DIR="${REPO_DIR}/${P1_DIR}"
        fi
        TASK_LIST="additive compositional conditional override proc_add proc_comp proc_cond proc_over"
        for TASK in $TASK_LIST; do
            if [ -n "$ONLY_TASK" ] && [ "$TASK" != "$ONLY_TASK" ]; then
                continue
            fi
            ELEM="${ELEMENTS_DIR}/${TASK}.json"
            if [ ! -f "$ELEM" ]; then
                echo "Warning: rule file not found, skipping $TASK: $ELEM"
                continue
            fi
            P11_FLAG=""
            if [ "$METHOD" != "react" ]; then
                P11_FILE="${P1_DIR}/phase_1_1_${TASK}_${BASE_NAME}_${METHOD}.json"
                if [ ! -f "$P11_FILE" ]; then
                    echo "Warning: Phase 1.1 file not found for $TASK/$METHOD, skipping."
                    echo "  Expected: $P11_FILE"
                    echo "  Run the episode Phase 2 experiment first."
                    continue
                fi
                P11_FLAG="--phase1_1_results $P11_FILE"
            fi
            _run_one_task "$TASK" "$ELEM" "${TASK}_${BASE_NAME}_${METHOD}" \
                --qa_teaching_strategy "$METHOD" \
                $P11_FLAG
        done
        ;;

    proc_sweep)
        TASK_LIST="proc_add proc_comp proc_cond proc_over"
        _run_sweep --skip_phase2
        ;;

    proc_qa_sweep)
        TASK_LIST="proc_add proc_comp proc_cond proc_over"
        _run_sweep
        ;;

    full_sweep)
        # All 8 task types: property first, then proc
        TASK_LIST="additive compositional conditional override proc_add proc_comp proc_cond proc_over"
        _run_sweep --skip_phase2
        ;;

    full_qa_sweep)
        TASK_LIST="additive compositional conditional override proc_add proc_comp proc_cond proc_over"
        _run_sweep
        ;;

    *)
        echo "Error: Unknown experiment_type '$EXP_TYPE'"
        echo "Supported: phase1_only | phase2_only | both_phases"
        echo "           sweep | sweep_phase2 | full_sweep_phase2 | qa | qa_sweep | qa_full_sweep_phase2"
        echo "           proc_sweep | proc_qa_sweep"
        echo "           full_sweep | full_qa_sweep"
        exit 1
        ;;
esac

echo "========================================"
echo "Done: $EXPERIMENT_NAME"
echo "========================================"
