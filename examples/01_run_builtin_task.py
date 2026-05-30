"""
Example 01 — Run a built-in task end to end with a mock agent.

Demonstrates the full flow: RULE -> TASK -> demos -> episode -> ECSR.
Uses an "oracle" model_fn (follows the reference solution) so it runs with no
API calls. Replace oracle_model_fn with a real model to actually test induction.

Run:  python examples/01_run_builtin_task.py
"""
import json

from herosjourney import get_task, compute_ecsr
from herosjourney.core.elements import fill_elements, load_lexicons
from herosjourney.core.demo_generator import generate_mixed_demos
from herosjourney.runner.adventure_episode import (
    run_single_episode, construct_demo_context,
)


def main():
    # 1. RULE + TASK
    spec = get_task("additive")

    # 2. Surface-realize the rule into a concrete variant
    sem_lex, nonce_lex = load_lexicons()
    with open(spec.rules) as f:
        rule = json.load(f)
    elements = fill_elements(rule, sem_lex, nonce_lex, seed=0, split_spec=spec.split)

    source_tasks = spec.gen_fn(elements, split="source", use_nonce=False)
    gen_tasks    = spec.gen_fn(elements, split="gen",    use_nonce=False)
    print(f"{len(source_tasks)} source (demo) entities, {len(gen_tasks)} gen (test) entities")

    # 3. Demonstrations shown in-context
    demos        = generate_mixed_demos(source_tasks, distractor_tasks=[])
    demo_context = construct_demo_context(demos)

    # 4. AGENT — a mock oracle that follows the reference solution.
    results = []
    for i, task in enumerate(gen_tasks):
        solution = task.tree.get_solution()
        step = {"i": 0}

        def oracle_model_fn(prompt, max_tokens=512, _sol=solution, _step=step):
            if _step["i"] < len(_sol):
                act = _sol[_step["i"]]; _step["i"] += 1
                verb, _, arg = act.partition(" ")
                return json.dumps({"action": verb, "argument": arg, "reasoning": "oracle"}), None, None
            return json.dumps({"action": "check_inventory", "argument": "", "reasoning": "done"}), None, None

        res = run_single_episode(
            episode_idx=i, task=task, demo_context=demo_context,
            max_runs=None, verbose=False, truncate_window=None,
            model_fn=oracle_model_fn, source_tasks=source_tasks,
        )
        results.append(res)
        print(f"  gen[{i}] success={res.success} efficiency={res.efficiency}")

    # 5. Score
    ecsr = compute_ecsr(results, n_tries=spec.max_tries)
    print(f"\nECSR over {len(results)} gen episodes: {ecsr:.3f}")


if __name__ == "__main__":
    main()
