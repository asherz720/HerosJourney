"""
Example 05 — Run the full benchmark: a model on all eight tasks, scored by ECSR and RV.

This mirrors how the paper evaluates a model. For each task it:
  - builds `num_variants` variants (different surface names + source/gen splits),
  - ECSR: runs each gen-split entity as an episode and computes the
    efficiency-calibrated success rate,
  - RV: asks the model to verbalize the hidden rule and scores that explanation
    with an LLM judge (0-2 on four dimensions, normalized to [0,1]),
  - averages both metrics across variants.

The model is served behind any OpenAI-compatible endpoint. Point the generic
adapter at it via env vars, then pass the served model name with --model:

    # e.g. a local vLLM server:
    #   python -m vllm.entrypoints.openai.api_server \
    #       --model Qwen/Qwen2.5-27B-Instruct --port 8000
    export OPENAI_BASE_URL=http://localhost:8000/v1
    export OPENAI_API_KEY=EMPTY

    python examples/05_run_benchmark.py --model Qwen/Qwen2.5-27B-Instruct --num-variants 5

Use --judge-model to score RV with a different model than the one under test
(the paper uses a separate judge). Add --nonce to test on nonsense surface names.
"""
import argparse
import json
import statistics

from herosjourney import get_task, compute_ecsr
from herosjourney.core.elements import fill_elements, load_lexicons
from herosjourney.core.demo_generator import generate_mixed_demos
from herosjourney.runner.adventure_episode import run_single_episode, construct_demo_context
from herosjourney.runner.qa_episode import run_qa_structure_exp
from herosjourney.eval.judge import QA_BASE_PROMPT

ALL_TASKS = [
    "additive", "compositional", "conditional", "override",   # attribute induction
    "proc_add", "proc_comp", "proc_cond", "proc_over",         # procedural induction
]


def run_one_task(task_type, model, judge_model, num_variants, num_tries,
                 converter_model, use_nonce, verbose):
    """Return (mean_ECSR, mean_RV) for one task over num_variants variants."""
    spec = get_task(task_type)
    sem_lex, nonce_lex = load_lexicons()
    with open(spec.rules) as f:
        rule = json.load(f)

    ecsr_per_variant, rv_per_variant = [], []
    for seed in range(num_variants):
        # One variant: surface-realize the rule, then split into source/gen.
        elements     = fill_elements(rule, sem_lex, nonce_lex, seed=seed, split_spec=spec.split)
        source_tasks = spec.gen_fn(elements, split="source", use_nonce=use_nonce)
        gen_tasks    = spec.gen_fn(elements, split="gen",    use_nonce=use_nonce)
        demo_context = construct_demo_context(generate_mixed_demos(source_tasks, distractor_tasks=[]))

        # --- ECSR: run every gen entity as an episode ---
        results = [
            run_single_episode(
                episode_idx=i, task=task, demo_context=demo_context,
                max_runs=None, verbose=verbose, truncate_window=None,
                model_path=model, source_tasks=source_tasks,
                num_tries=num_tries, converter_model=converter_model,
            )
            for i, task in enumerate(gen_tasks)
        ]
        ecsr_per_variant.append(compute_ecsr(results, n_tries=spec.max_tries))

        # --- RV: one rule-verbalization per variant, judged ---
        qa = run_qa_structure_exp(
            source_tasks, gen_tasks, demo_context, task_type,
            model=model, base_prompt=QA_BASE_PROMPT,
            converter_model=converter_model, judge_model=judge_model,
            verbose=verbose,
        )
        rv_per_variant.append(qa["overall"])

    mean = statistics.mean
    return mean(ecsr_per_variant), mean(rv_per_variant)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="Model name served by your OpenAI-compatible endpoint.")
    p.add_argument("--judge-model", default=None,
                   help="Model for RV judging (default: same as --model).")
    p.add_argument("--tasks", nargs="+", default=ALL_TASKS,
                   help=f"Subset of tasks (default: all 8). Choices: {ALL_TASKS}")
    p.add_argument("--num-variants", type=int, default=5,
                   help="Variants per task to average over (paper uses 20).")
    p.add_argument("--num-tries", type=int, default=2,
                   help="Episode budget multiplier (max_runs = reference_length * num_tries).")
    p.add_argument("--converter-model", default="small",
                   help="JSON-repair helper for malformed actions (uses the same endpoint).")
    p.add_argument("--nonce", action="store_true",
                   help="Use nonce (nonsense) surface names instead of semantic ones.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    print(f"model={args.model}  judge={args.judge_model or args.model}  "
          f"variants={args.num_variants}  nonce={args.nonce}\n")
    print(f"{'task':<14}{'ECSR':>8}{'RV':>8}")
    print("-" * 30)

    ecsr_all, rv_all = [], []
    for task_type in args.tasks:
        ecsr, rv = run_one_task(
            task_type, args.model, args.judge_model, args.num_variants,
            args.num_tries, args.converter_model, args.nonce, args.verbose,
        )
        ecsr_all.append(ecsr); rv_all.append(rv)
        print(f"{task_type:<14}{ecsr:>8.3f}{rv:>8.3f}")

    print("-" * 30)
    print(f"{'mean':<14}{statistics.mean(ecsr_all):>8.3f}{statistics.mean(rv_all):>8.3f}")


if __name__ == "__main__":
    main()
