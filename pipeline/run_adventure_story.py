"""
CLI entry point for the two-phase adventure-story generalization benchmark.

"""

import os
import sys
import argparse

# Allow running from repo root as  python -m pipeline.run_adventure_story  or directly
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from pipeline.adventure_pipeline import run_two_phase_pipeline, run_qa_pipeline


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Two-phase adventure-story generalization benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Task ---
    p.add_argument(
        "--task_type", required=True,
        choices=[
            # Property tasks (item-selection)
            "additive", "compositional", "conditional", "override",
            # Procedural tasks (process-selection)
            "proc_add", "proc_comp", "proc_cond", "proc_over",
        ],
        help="Task type to evaluate.",
    )
    p.add_argument(
        "--elements", required=True,
        help="Path to the concept elements JSON file.",
    )
    p.add_argument(
        "--distractor_rules", default=None,
        help="Path to distractor elements JSON (optional).  When provided, "
             "distractor episodes are interleaved with source demos.",
    )
    p.add_argument(
        "--num_distractor_samples", type=int, default=4,
        help="Number of distractor entities to sample for the demo context.",
    )
    p.add_argument(
        "--nonce", action="store_true",
        help="Use nonce (non-semantic) names for entities and items.",
    )

    # --- Model ---
    p.add_argument(
        "--model", default="gemini",
        help="Model identifier.  Built-in: 'gemini', 'claude4.5', 'gpt5-nano'. "
             "For local vLLM: /path/to/model or /path/to/model@PORT.",
    )
    p.add_argument(
        "--converter_model", default="small", choices=["small", "gemini"],
        help="Model used to repair malformed JSON action outputs.",
    )

    # --- Episode settings ---
    def _num_tries_type(v: str):
        if v == "max":
            return "max"
        try:
            return int(v)
        except ValueError:
            raise argparse.ArgumentTypeError(f"--num_tries must be a positive integer or 'max', got: {v!r}")

    p.add_argument(
        "--num_tries", type=_num_tries_type, default=2,
        help="Number of attempts per episode: max_runs = ref_length × num_tries. "
             "All task types share the same standard, making success rates comparable. "
             "num_tries=1: must solve on first attempt. num_tries=2 (default): one wrong path allowed. "
             "num_tries=max: use the registered per-task max (= number of distinct items).",
    )
    p.add_argument(
        "--max_runs", type=int, default=None,
        help="Override max actions per episode directly (ignores --num_tries).",
    )
    p.add_argument(
        "--truncate_window", type=int, default=None,
        help="Keep only the last N action-observation pairs in context.",
    )
    p.add_argument(
        "--initial_currency", type=int, default=500,
        help="Starting currency for each episode.",
    )
    p.add_argument(
        "--num_workers", type=int, default=1,
        help="Parallel episode workers.",
    )
    p.add_argument(
        "--num_variants", type=int, default=1,
        help="Number of independent surface variants to run. "
             "Each variant uses different entity/attribute/item names "
             "while preserving the same concept structure and source/gen split. "
             "Variants are saved to variants/<task_type>/v{seed}_filled.json.",
    )
    p.add_argument(
        "--demo_repeats", type=int, default=1,
        help="Times to repeat the source pairs in demos with different entity names. "
             "demo_repeats=2 shows 12 demo episodes (6 original + 6 renamed) for C1/C2. "
             "Demonstrations are shuffled with the variant seed for reproducibility.",
    )
    p.add_argument(
        "--variants_dir", default=None,
        help="Directory to save generated variant elements files "
             "(default: variants/<task_type>).",
    )
    p.add_argument(
        "--n_source_demos", type=int, default=None,
        help="Coverage sweep: use only the first k source demos (sampled from the "
             "shuffled source list). Default: all source demos.",
    )
    p.add_argument(
        "--n_gen_tasks", type=int, default=None,
        help="Coverage sweep: evaluate only the first k gen tasks. Default: all gen tasks.",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print step-by-step episode output.",
    )
    p.add_argument(
        "--no_reasoning", action="store_true",
        help="Omit 'reasoning' field from the action JSON prompt and schema. "
             "Speeds up inference for models where chain-of-thought in the "
             "action output is unnecessary (e.g. local vLLM models).",
    )

    # --- Phase 2 teaching ---
    p.add_argument(
        "--teaching_strategy", default="none",
        choices=["none", "react", "hr", "idea", "ace"],
        help="Teaching strategy for Phase 2.  "
             "Static: react — explicit hypothesis prompt appended each step.  "
             "Dynamic: hr (Hypothesis Refinement, pre-episode call), "
             "idea (I→D→A online loop), "
             "ace (offline playbook from Phase 1 failure traces).  "
             "'none': Phase 2 runs with no teaching message (repeat-run baseline).",
    )
    p.add_argument(
        "--teacher_model_path", default=None,
        help="Model path for the teacher agent used by dynamic strategies "
             "(separate from the student --model).  "
             "Falls back to TEACHER_MODEL_PATH in teacher.py if unset.",
    )

    # --- Persistence / skip ---
    p.add_argument(
        "--save_name", default=None,
        help="Filename stem for saved results JSON files "
             "(default: <task_type>_<model_stem>).",
    )
    p.add_argument(
        "--results_dir", default=None,
        help="Directory for saving/loading results JSON files "
             "(default: ./results). Overrides the pipeline's RESULTS_DIR.",
    )
    p.add_argument(
        "--skip_phase1", action="store_true",
        help="Skip Phase 1 and load existing results (requires saved file).",
    )
    p.add_argument(
        "--skip_phase2", action="store_true",
        help="Skip Phase 2.",
    )
    p.add_argument(
        "--phase1_results", default=None,
        help="Path to existing Phase 1 results JSON (implies --skip_phase1).",
    )
    p.add_argument(
        "--phase1_1_results", default=None,
        help="Path to existing Phase 1.1 curation results JSON.  "
             "When provided, the pipeline loads the saved teaching message "
             "and skips re-running teacher curation.",
    )
    p.add_argument(
        "--phase2_results", default=None,
        help="Path to existing Phase 2 results JSON (implies --skip_phase2).",
    )

    # --- Q&A evaluation ---
    p.add_argument(
        "--qa_mode", nargs="+", default=[],
        choices=["instance", "structure_exp", "all"],
        help=(
            "Q&A evaluation mode(s) to run alongside or instead of the episode pipeline.\n"
            "  instance      — per gen task: predict the item for a specific entity (exact match)\n"
            "  structure_exp — per variant: describe the rule in free form (LLM-as-judge)\n"
            "  all           — run all modes\n"
            "When --qa_mode is given, the episode pipeline is skipped entirely.\n"
            "Results saved to results/qa_phase1_<save_name>.json (and qa_phase2_<save_name>.json "
            "when --qa_teaching_strategy is set)."
        ),
    )
    p.add_argument(
        "--qa_teaching_strategy", default="none",
        choices=["none", "react", "hr", "idea", "ace"],
        help=(
            "Teaching strategy for Q&A Phase 2.  Same choices as --teaching_strategy. "
            "'none': only Phase 1 Q&A is run (no teaching).  "
            "Static strategies build a teaching message from gen task trees.  "
            "Dynamic strategies require Phase 1 results for curation."
        ),
    )
    p.add_argument(
        "--skip_qa_phase1", action="store_true",
        help="Skip Q&A Phase 1 (demo-only).",
    )
    p.add_argument(
        "--skip_qa_phase2", action="store_true",
        help="Skip Q&A Phase 2 (+ teaching).",
    )
    p.add_argument(
        "--judge_model", default=None,
        help="Model path for the LLM-as-judge (structure_exp scoring). "
             "Defaults to --model when not set. Use this to fix the judge "
             "(e.g. a Qwen 27B vLLM server) while sweeping different test models.",
    )

    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    # Override results directory if specified
    if args.results_dir:
        import pipeline.adventure_pipeline as _ap
        _ap.RESULTS_DIR = args.results_dir

    # Resolve qa_modes: "all" expands to all three modes
    qa_modes = args.qa_mode or []
    if "all" in qa_modes:
        qa_modes = ["instance", "structure_exp"]

    # Episode pipeline — only when no Q&A modes are requested
    if not qa_modes:
        skip_phase1 = args.skip_phase1 or bool(args.phase1_results)
        skip_phase2 = args.skip_phase2 or bool(args.phase2_results)

        run_two_phase_pipeline(
            task_type=args.task_type,
            elements_path=args.elements,
            model_path=args.model,
            distractor_rules_path=args.distractor_rules,
            num_distractor_samples=args.num_distractor_samples,
            use_nonce=args.nonce,
            max_runs=args.max_runs,
            num_tries=args.num_tries,
            verbose=args.verbose,
            truncate_window=args.truncate_window,
            initial_currency=args.initial_currency,
            num_workers=args.num_workers,
            converter_model=args.converter_model,
            teaching_strategy=args.teaching_strategy,
            teacher_model_path=args.teacher_model_path,
            num_variants=args.num_variants,
            demo_repeats=args.demo_repeats,
            variants_dir=args.variants_dir,
            n_source_demos=args.n_source_demos,
            n_gen_tasks=args.n_gen_tasks,
            save_name=args.save_name,
            skip_phase1=skip_phase1,
            skip_phase2=skip_phase2,
            phase1_results_file=args.phase1_results,
            phase1_1_results_file=args.phase1_1_results,
            phase2_results_file=args.phase2_results,
            no_reasoning=args.no_reasoning,
        )

    # Q&A pipeline
    if qa_modes:
        run_qa_pipeline(
            task_type=args.task_type,
            elements_path=args.elements,
            model_path=args.model,
            qa_modes=qa_modes,
            distractor_rules_path=args.distractor_rules,
            num_distractor_samples=args.num_distractor_samples,
            use_nonce=args.nonce,
            num_variants=args.num_variants,
            demo_repeats=args.demo_repeats,
            variants_dir=args.variants_dir,
            n_source_demos=args.n_source_demos,
            n_gen_tasks=args.n_gen_tasks,
            converter_model=args.converter_model,
            num_workers=args.num_workers,
            verbose=args.verbose,
            save_name=args.save_name,
            teaching_strategy=args.qa_teaching_strategy,
            teacher_model_path=args.teacher_model_path,
            phase1_1_results_file=args.phase1_1_results,
            skip_qa_phase1=args.skip_qa_phase1,
            skip_qa_phase2=args.skip_qa_phase2,
            judge_model_path=args.judge_model,
        )
