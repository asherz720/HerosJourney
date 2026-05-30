"""
Example 04 — Apply an induction method (steering strategy) on top of an agent.

A "method" wraps extra reasoning around the base agent loop. Select it with
episode_mode:
    "standard"  base loop (default)
    "react"     ReAct-style hypothesis/evidence reasoning each step
    "hr"        Hypothesis Refinement: generate+verify candidate rules upfront
    "idea"      Induction-Deduction-Abduction: hypothesize, then revise on failure

The method only changes how the agent is prompted/looped — the same model_fn is
reused. Run:  python examples/04_methods.py
"""
import json

from herosjourney import get_task, compute_ecsr
from herosjourney.core.elements import fill_elements, load_lexicons
from herosjourney.core.demo_generator import generate_mixed_demos
from herosjourney.runner.adventure_episode import (
    run_single_episode, construct_demo_context,
)

METHODS = ["standard", "react", "hr", "idea"]


def main():
    spec = get_task("additive")
    sem_lex, nonce_lex = load_lexicons()
    with open(spec.rules) as f:
        rule = json.load(f)
    elements = fill_elements(rule, sem_lex, nonce_lex, seed=0, split_spec=spec.split)
    source_tasks = spec.gen_fn(elements, split="source", use_nonce=False)
    gen_tasks    = spec.gen_fn(elements, split="gen",    use_nonce=False)
    demo_context = construct_demo_context(generate_mixed_demos(source_tasks, distractor_tasks=[]))

    # NOTE: hr/idea make extra plain-text reasoning calls, so a real model_fn is
    # needed to see their effect. With model_path + the [runner] extra:
    #
    #   for method in METHODS:
    #       results = [
    #           run_single_episode(
    #               episode_idx=i, task=t, demo_context=demo_context,
    #               max_runs=None, verbose=False, truncate_window=None,
    #               model_path="my-model", source_tasks=source_tasks,
    #               episode_mode=method,
    #           )
    #           for i, t in enumerate(gen_tasks)
    #       ]
    #       print(method, compute_ecsr(results, n_tries=spec.max_tries))

    print("Available induction methods:", METHODS)
    print("Pass episode_mode=<method> to run_single_episode (see commented loop).")


if __name__ == "__main__":
    main()
