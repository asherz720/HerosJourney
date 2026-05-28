#!/usr/bin/env python3
"""
smoke_test_methods.py

Mirrors the full pipeline (episodes + QA) with small numbers to verify every
steering method works end-to-end before running full experiments.

Defaults: 1 variant, 2 gen tasks, 1 try.

Usage (from repo root):
    # Run everything fresh:
    python scripts/smoke_test_methods.py --model gpt-5.4-mini

    # Reuse existing Phase 1 results from a folder (skip re-running Phase 1):
    python scripts/smoke_test_methods.py --model gpt-5.4-mini \\
        --phase1_dir results/gpt5

    # Single task or method:
    python scripts/smoke_test_methods.py --model gpt-5.4-mini --task_type additive
    python scripts/smoke_test_methods.py --model gpt-5.4-mini --method hr

Results are saved to results/smoke_test/.

Phase 1 auto-detection:
    Once Phase 1 has run for a model+task (either fresh or via --phase1_dir),
    subsequent runs automatically reuse the saved results — no flags needed.
"""

import sys
import os
import glob
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pipeline.adventure_pipeline as _ap
from pipeline.adventure_pipeline import run_two_phase_pipeline, run_qa_pipeline

TASK_TYPES  = ["additive", "compositional", "conditional", "override"]
METHODS     = ["react", "hr", "idea", "ace"]
RESULTS_DIR = "results/smoke_test"
QA_MODES    = ["instance", "structure_exp"]


# ---------------------------------------------------------------------------
# Phase 1 file discovery
# ---------------------------------------------------------------------------

def _find_file(directory: str, prefix: str, task_type: str) -> str | None:
    """
    Look for  {prefix}_{task_type}_*.json  in directory.
    Returns the first match, or None.
    """
    pattern = os.path.join(directory, f"{prefix}_{task_type}_*.json")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def _find_ep_phase1(directory: str, task_type: str) -> str | None:
    return _find_file(directory, "phase_1", task_type)


def _find_qa_phase1(directory: str, task_type: str) -> str | None:
    return _find_file(directory, "qa_phase1", task_type)


# ---------------------------------------------------------------------------
# ACE variant search
# ---------------------------------------------------------------------------

def _make_ace_phase1_file(phase1_file: str, out_dir: str) -> str:
    """
    Repack a phase1 file so that variant 0 is the failure-richest variant.
    ACE curation gets the most useful traces regardless of which variant seed
    originally contained failures.
    """
    with open(phase1_file) as f:
        data = json.load(f)

    variants   = data.get("variants", [data])
    best_seed  = max(
        range(len(variants)),
        key=lambda i: sum(
            1 for ep in variants[i].get("episodes", {}).values()
            if not ep.get("success", True)
        ),
    )
    best_var = variants[best_seed]
    n_failed = sum(1 for ep in best_var.get("episodes", {}).values()
                   if not ep.get("success", True))

    if n_failed == 0:
        print(f"  [ACE] Warning: no failures found in {os.path.basename(phase1_file)}. "
              f"Playbook will be empty.")
    else:
        print(f"  [ACE] Picked variant {best_seed} ({n_failed} failure(s)) "
              f"from {os.path.basename(phase1_file)}")

    os.makedirs(out_dir, exist_ok=True)
    stem     = os.path.splitext(os.path.basename(phase1_file))[0]
    out_path = os.path.join(out_dir, f"_ace_tmp_{stem}.json")
    with open(out_path, "w") as f:
        json.dump({**data, "variants": [best_var], "num_variants": 1}, f)
    return out_path


# ---------------------------------------------------------------------------
# Single method runners (episode and QA)
# ---------------------------------------------------------------------------

def _run_ep(task_type, method, model, teacher_model,
            num_variants, n_gen_tasks, num_tries,
            p1_file, ace_p1_file, verbose):
    """Run one episode pipeline phase for task_type + method."""
    elements_path = f"tree_management/rules/{task_type}.json"
    model_stem    = os.path.basename(model.rstrip("/")).replace(" ", "_")[:15]
    orig_dir      = _ap.RESULTS_DIR
    _ap.RESULTS_DIR = RESULTS_DIR
    os.makedirs(RESULTS_DIR, exist_ok=True)
    try:
        if method == "phase1":
            p1, _ = run_two_phase_pipeline(
                task_type=task_type, elements_path=elements_path,
                model_path=model,
                teaching_strategy="none",
                num_variants=num_variants, n_gen_tasks=n_gen_tasks,
                num_tries=num_tries,
                save_name=f"smoke_{task_type}_{model_stem}",
                phase1_results_file=p1_file,   # None = run fresh; path = auto-load
                skip_phase2=True,
                verbose=verbose,
            )
            return p1 or {}
        else:
            p1_to_use = ace_p1_file if method == "ace" else p1_file
            _, p2 = run_two_phase_pipeline(
                task_type=task_type, elements_path=elements_path,
                model_path=model, teacher_model_path=teacher_model,
                teaching_strategy=method,
                num_variants=num_variants, n_gen_tasks=n_gen_tasks,
                num_tries=num_tries,
                save_name=f"smoke_{task_type}_{model_stem}_{method}",
                skip_phase1=True, phase1_results_file=p1_to_use,
                verbose=verbose,
            )
            return p2 or {}
    finally:
        _ap.RESULTS_DIR = orig_dir


def _extract_ep2_teaching_message(ep_p2_file: str) -> str:
    """
    Return the teaching_message from the first episode of Episode Phase 2 results.

    All methods now store the right thing in teaching_message:
      react — REACT_STUDENT_PROMPT
      hr    — pre-episode hypothesis text
      idea  — final (possibly revised) hypothesis
      ace   — ACE playbook
    """
    if not ep_p2_file or not os.path.exists(ep_p2_file):
        return ""
    with open(ep_p2_file) as f:
        data = json.load(f)
    variants = data.get("variants", [data])
    if not variants:
        return ""
    first_ep = next(iter(variants[0].get("episodes", {}).values()), None)
    return first_ep.get("teaching_message", "") if first_ep else ""


def _run_qa(task_type, method, model, teacher_model,
            num_variants, n_gen_tasks,
            qa_p1_file, ep_p11_ace_file, ep_p2_file, verbose):
    """Run one QA pipeline phase for task_type + method.

    QA Phase 2 teaching message is aligned with Episode Phase 2:
      phase1  — no teaching (Phase 1 only)
      react   — static REACT_STUDENT_PROMPT
      ace     — ACE playbook from Episode Phase 1.1
      hr      — pre-episode hypothesis from Episode Phase 2 results
      idea    — final hypothesis extracted from Episode Phase 2 trace
    """
    elements_path = f"tree_management/rules/{task_type}.json"
    model_stem    = os.path.basename(model.rstrip("/")).replace(" ", "_")[:15]
    orig_dir      = _ap.RESULTS_DIR
    _ap.RESULTS_DIR = RESULTS_DIR
    os.makedirs(RESULTS_DIR, exist_ok=True)
    try:
        if method == "phase1":
            p1, _ = run_qa_pipeline(
                task_type=task_type, elements_path=elements_path,
                model_path=model,
                qa_modes=QA_MODES,
                teaching_strategy="none",
                num_variants=num_variants, n_gen_tasks=n_gen_tasks,
                judge_model_path=model,
                save_name=f"smoke_{task_type}_{model_stem}",
                skip_qa_phase2=True,
                verbose=verbose,
            )
            return p1 or {}

        elif method == "react":
            # Static prompt; no episode results needed
            _, p2 = run_qa_pipeline(
                task_type=task_type, elements_path=elements_path,
                model_path=model, teacher_model_path=teacher_model,
                qa_modes=QA_MODES,
                teaching_strategy=method,
                num_variants=num_variants, n_gen_tasks=n_gen_tasks,
                judge_model_path=model,
                save_name=f"smoke_{task_type}_{model_stem}_{method}",
                skip_qa_phase1=True,
                verbose=verbose,
            )
            return p2 or {}

        elif method == "ace":
            # ACE playbook from Episode Phase 1.1
            _, p2 = run_qa_pipeline(
                task_type=task_type, elements_path=elements_path,
                model_path=model, teacher_model_path=teacher_model,
                qa_modes=QA_MODES,
                teaching_strategy=method,
                num_variants=num_variants, n_gen_tasks=n_gen_tasks,
                judge_model_path=model,
                save_name=f"smoke_{task_type}_{model_stem}_{method}",
                skip_qa_phase1=True,
                phase1_1_results_file=ep_p11_ace_file,
                verbose=verbose,
            )
            return p2 or {}

        else:
            # hr/idea: extract the hypothesis reached in Episode Phase 2
            msg = _extract_ep2_teaching_message(ep_p2_file)
            if msg:
                print(f"  [QA {method.upper()}] Using Episode Phase 2 hypothesis "
                      f"({len(msg)} chars)")
            else:
                print(f"  [QA {method.upper()}] No Episode Phase 2 hypothesis found "
                      f"(episode pipeline may not have run yet)")
            _, p2 = run_qa_pipeline(
                task_type=task_type, elements_path=elements_path,
                model_path=model, teacher_model_path=teacher_model,
                qa_modes=QA_MODES,
                teaching_strategy=method,
                num_variants=num_variants, n_gen_tasks=n_gen_tasks,
                judge_model_path=model,
                save_name=f"smoke_{task_type}_{model_stem}_{method}",
                skip_qa_phase1=True,
                fixed_teaching_message=msg,
                verbose=verbose,
            )
            return p2 or {}

    finally:
        _ap.RESULTS_DIR = orig_dir


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _sr(result: dict, key: str = "success_rate") -> str:
    if not result:
        return "—"
    n  = result.get("num_episodes", 0) or result.get("num_variants", 0)
    ok = result.get("successes", 0)
    sr = result.get(key, result.get("success_rate", result.get("instance_accuracy", 0)))
    return f"{ok}/{n}  ({sr:.0%})" if isinstance(ok, int) else f"({sr:.0%})"


def _print_summary(all_ep: dict, all_qa: dict) -> None:
    tasks = sorted(set(list(all_ep) + list(all_qa)))
    for task in tasks:
        print(f"\n  {task}")
        ep_rec  = all_ep.get(task, {})
        qa_rec  = all_qa.get(task, {})
        methods = sorted(set(list(ep_rec) + list(qa_rec)))
        print(f"    {'method':<12}  {'episodes':>14}  {'QA (instance)':>14}")
        print(f"    {'-'*12}  {'-'*14}  {'-'*14}")
        for m in methods:
            ep = _sr(ep_rec.get(m, {}))
            qa_r = qa_rec.get(m, {})
            qa = _sr({"success_rate": qa_r.get("instance_accuracy", 0),
                      "num_episodes": qa_r.get("num_episodes", 0),
                      "successes":    round(qa_r.get("instance_accuracy", 0)
                                            * qa_r.get("num_episodes", 0))},
                     "success_rate") if qa_r else "—"
            print(f"    {m:<12}  {ep:>14}  {qa:>14}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Smoke-test all steering methods (episodes + QA) with small counts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="gemini",
                        help="Student model path.")
    parser.add_argument("--teacher_model", default=None,
                        help="Teacher model for ACE (default: same as --model).")
    parser.add_argument("--task_type", default="all",
                        choices=["all"] + TASK_TYPES,
                        help="Task type(s) to test.")
    parser.add_argument("--method", default="all",
                        choices=["all", "phase1"] + METHODS,
                        help="Method(s) to test.")
    parser.add_argument("--phase1_dir", default=None,
                        help="Folder containing existing phase 1 results "
                             "(e.g. results/gpt5). The script looks for "
                             "phase_1_{task}_*.json and qa_phase1_{task}_*.json "
                             "inside it. If a file is found it is used instead of "
                             "re-running Phase 1. Auto-detection of previously "
                             "saved smoke-test results also works without this flag.")
    parser.add_argument("--num_variants", type=int, default=1,
                        help="Number of surface variants per task.")
    parser.add_argument("--n_gen_tasks", type=int, default=2,
                        help="Number of gen tasks per variant.")
    parser.add_argument("--num_tries", type=int, default=1,
                        help="Number of attempts per episode.")
    parser.add_argument("--episodes_only", action="store_true",
                        help="Skip QA pipeline.")
    parser.add_argument("--qa_only", action="store_true",
                        help="Skip episode pipeline.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print step-by-step episode output.")
    args = parser.parse_args()

    teacher_model = args.teacher_model or args.model
    task_types    = TASK_TYPES if args.task_type == "all" else [args.task_type]
    methods_p2    = METHODS    if args.method   == "all" else (
                    []         if args.method   == "phase1" else [args.method])
    run_episodes  = not args.qa_only
    run_qa        = not args.episodes_only

    print(f"\nSmoke-test  |  model={args.model}  variants={args.num_variants}"
          f"  gen_tasks={args.n_gen_tasks}  tries={args.num_tries}")
    print(f"  task_type(s) : {', '.join(task_types)}")
    print(f"  method(s)    : {args.method}")
    print(f"  pipelines    : {'episodes + QA' if run_episodes and run_qa else 'episodes' if run_episodes else 'QA'}")
    if args.phase1_dir:
        print(f"  phase1_dir   : {args.phase1_dir}  (searched for phase_1_{{task}}_*.json)")

    model_stem  = os.path.basename(args.model.rstrip("/")).replace(" ", "_")[:15]
    all_ep: dict  = {}
    all_qa: dict  = {}

    for task_type in task_types:
        print(f"\n{'='*58}")
        print(f"  TASK: {task_type.upper()}")
        print(f"{'='*58}")

        ep_records: dict = {}
        qa_records: dict = {}

        # ----------------------------------------------------------------
        # Resolve Phase 1 file paths
        # ----------------------------------------------------------------

        # Episode Phase 1
        smoke_ep_p1 = os.path.join(RESULTS_DIR, f"phase_1_smoke_{task_type}_{model_stem}.json")
        if args.phase1_dir:
            ext_ep_p1 = _find_ep_phase1(args.phase1_dir, task_type)
            ep_p1_file = ext_ep_p1 or smoke_ep_p1
            if ext_ep_p1:
                print(f"\n  [EP  Phase 1] Using {ext_ep_p1}")
            else:
                print(f"\n  [EP  Phase 1] Not found in {args.phase1_dir} — will run fresh")
        else:
            ep_p1_file = smoke_ep_p1

        # QA Phase 1
        smoke_qa_p1 = os.path.join(RESULTS_DIR, f"qa_phase1_smoke_{task_type}_{model_stem}.json")
        if args.phase1_dir:
            ext_qa_p1 = _find_qa_phase1(args.phase1_dir, task_type)
            qa_p1_file = ext_qa_p1 or smoke_qa_p1
            if ext_qa_p1:
                print(f"  [QA  Phase 1] Using {ext_qa_p1}")
            else:
                print(f"  [QA  Phase 1] Not found in {args.phase1_dir} — will run fresh")
        else:
            qa_p1_file = smoke_qa_p1

        # ----------------------------------------------------------------
        # Episode pipeline
        # ----------------------------------------------------------------
        if run_episodes:
            # Phase 1 (auto-skipped if file already exists)
            if os.path.exists(ep_p1_file):
                print(f"\n  [EP  Phase 1] Auto-detected: {ep_p1_file}")
            else:
                print(f"\n  [EP  Phase 1] Running baseline...")
            ep_records["phase1"] = _run_ep(
                task_type, "phase1", args.model, teacher_model,
                args.num_variants, args.n_gen_tasks, args.num_tries,
                ep_p1_file, None, args.verbose,
            )

            # ACE needs a file remapped to the failure-richest variant
            ace_ep_p1 = (
                _make_ace_phase1_file(ep_p1_file, RESULTS_DIR)
                if "ace" in methods_p2 and os.path.exists(ep_p1_file)
                else ep_p1_file
            )

            for method in methods_p2:
                print(f"\n  [EP  {method.upper()}] Running Phase 2...")
                ep_records[method] = _run_ep(
                    task_type, method, args.model, teacher_model,
                    args.num_variants, args.n_gen_tasks, args.num_tries,
                    ep_p1_file, ace_ep_p1, args.verbose,
                )

        # Episode Phase 1.1 file for ACE (saved by episode pipeline after ACE Phase 2)
        ep_p11_ace_file = os.path.join(
            RESULTS_DIR, f"phase_1_1_smoke_{task_type}_{model_stem}_ace.json"
        )

        # ----------------------------------------------------------------
        # QA pipeline
        # ----------------------------------------------------------------
        if run_qa:
            if os.path.exists(qa_p1_file):
                print(f"\n  [QA  Phase 1] Auto-detected: {qa_p1_file}")
            else:
                print(f"\n  [QA  Phase 1] Running baseline...")
            qa_records["phase1"] = _run_qa(
                task_type, "phase1", args.model, teacher_model,
                args.num_variants, args.n_gen_tasks,
                qa_p1_file, ep_p11_ace_file, None, args.verbose,
            )

            for method in methods_p2:
                print(f"\n  [QA  {method.upper()}] Running Phase 2...")
                ep_p2_file = os.path.join(
                    RESULTS_DIR,
                    f"phase_2_smoke_{task_type}_{model_stem}_{method}.json"
                )
                qa_records[method] = _run_qa(
                    task_type, method, args.model, teacher_model,
                    args.num_variants, args.n_gen_tasks,
                    qa_p1_file, ep_p11_ace_file, ep_p2_file, args.verbose,
                )

        all_ep[task_type] = ep_records
        all_qa[task_type] = qa_records

    # ----------------------------------------------------------------
    # Save and print summary
    # ----------------------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"smoke_summary_{model_stem}_{ts}.json")

    def _compact(r):
        return {
            "success_rate":      r.get("success_rate", 0),
            "instance_accuracy": r.get("instance_accuracy"),
            "successes":         r.get("successes", 0),
            "num_episodes":      r.get("num_episodes", 0),
        }

    with open(path, "w") as f:
        json.dump({
            "model": args.model, "num_variants": args.num_variants,
            "n_gen_tasks": args.n_gen_tasks, "num_tries": args.num_tries,
            "episodes": {t: {m: _compact(r) for m, r in rec.items()}
                         for t, rec in all_ep.items()},
            "qa":       {t: {m: _compact(r) for m, r in rec.items()}
                         for t, rec in all_qa.items()},
        }, f, indent=2)

    print(f"\n{'='*58}")
    print("  SUMMARY")
    print(f"{'='*58}")
    _print_summary(all_ep, all_qa)
    print(f"\n  Full results : {RESULTS_DIR}/")
    print(f"  Summary file : {path}\n")


if __name__ == "__main__":
    main()
