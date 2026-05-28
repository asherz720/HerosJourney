"""
Score rule hypotheses for hr, idea, and ace methods.

hr / idea  — extracts the final hypothesis from phase_2 episode results
             (last episode's teaching_message per variant) and judges it
             directly without any new model generation.

ace        — picks a frozen playbook from phase_1_1 results (first non-empty
             teaching_message across variants), then for each variant generates
             a rule articulation via [base_prompt + demo_context + playbook]
             → model → judge.

Output: results/gpt5/qa_phase2_{task}_{base}_{method}.json
        (same format as the existing qa_phase2_react files)

Usage (from repo root):
    python analysis/score_rule_hypotheses.py --methods hr,idea,ace
    python analysis/score_rule_hypotheses.py --methods ace --only conditional
"""
from __future__ import annotations

import argparse
import json
import random as _random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repo root is on sys.path when run directly (e.g. python analysis/score_rule_hypotheses.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROPERTY_TASKS = ["additive", "compositional", "conditional", "override"]
PROC_TASKS     = ["proc_add", "proc_comp", "proc_cond", "proc_over"]
ALL_TASKS      = PROPERTY_TASKS + PROC_TASKS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_episode_teaching_message(variant: Dict) -> str:
    """Return the teaching_message from the last episode in a phase_2 variant."""
    eps = variant.get("episodes", {})
    if not eps:
        return ""
    def _ep_idx(k: str) -> int:
        try:
            return int(k.split("_")[-1])
        except ValueError:
            return 0
    last_key = max(eps.keys(), key=_ep_idx)
    return eps[last_key].get("teaching_message", "")


def _get_frozen_playbook(p11_path: Path) -> str:
    """Return the first non-empty teaching_message from a phase_1_1 file."""
    data = json.loads(p11_path.read_text())
    for v in data.get("variants", []):
        msg = v.get("teaching_message", "")
        if msg:
            return msg
    return ""


def _build_variant_output(variant_seed: int, judge_result: Dict) -> Dict:
    return {
        "variant_seed": variant_seed,
        "structure_exp": {
            "mode":                 "structure_exp",
            "overall":              judge_result.get("overall", 0.0),
            "input_score":          judge_result.get("input_score",          0),
            "output_score":         judge_result.get("output_score",         0),
            "rule_score":           judge_result.get("rule_score",           0),
            "generalization_score": judge_result.get("generalization_score", 0),
            "judge_reasoning":      judge_result.get("reasoning", ""),
            "judge_failed":         judge_result.get("judge_failed", False),
            "raw_judge":            judge_result.get("raw_judge_response"),
        },
    }


def _save_results(variants: List[Dict], task: str, base: str, method: str,
                  results_dir: Path) -> None:
    out = {
        "task_type":         task,
        "teaching_strategy": method,
        "num_variants":      len(variants),
        "variants":          variants,
    }
    path = results_dir / f"qa_phase2_{task}_{base}_{method}.json"
    path.write_text(json.dumps(out, indent=4))
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# hr / idea: judge last-episode hypothesis directly
# ---------------------------------------------------------------------------

def score_iterative_method(
    task: str,
    base: str,
    method: str,
    results_dir: Path,
    judge_model: str,
    converter_model: str = "gemini",
    num_variants: int = 20,
    verbose: bool = False,
) -> None:
    from pipeline.qa_episode import judge_structure_explanation

    p2_path = results_dir / f"phase_2_{task}_{base}_{method}.json"
    if not p2_path.exists():
        print(f"  SKIP: {p2_path} not found")
        return

    data = json.loads(p2_path.read_text())
    variant_outputs: List[Dict] = []

    for v in data.get("variants", [])[:num_variants]:
        seed = v.get("variant_seed", len(variant_outputs))
        hyp  = _last_episode_teaching_message(v)
        print(f"  Variant {seed}: hypothesis len={len(hyp)}", end=" ... ")

        if not hyp:
            print("empty, score=0")
            judge_result: Dict[str, Any] = {
                "input_score": 0, "output_score": 0,
                "rule_score": 0, "generalization_score": 0,
                "overall": 0.0, "reasoning": "No hypothesis.",
                "judge_failed": True,
            }
        else:
            judge_result = judge_structure_explanation(
                hyp, task, judge_model, converter_model, verbose=verbose,
            )
            print(f"rule_score={judge_result['rule_score']}, overall={judge_result['overall']:.3f}")

        variant_outputs.append(_build_variant_output(seed, judge_result))

    _save_results(variant_outputs, task, base, method, results_dir)


# ---------------------------------------------------------------------------
# ace: generate rule articulation with frozen playbook then judge
# ---------------------------------------------------------------------------

def score_ace(
    task: str,
    base: str,
    results_dir: Path,
    model_path: str,
    judge_model: str,
    converter_model: str = "gemini",
    distractor_rules_path: Optional[str] = None,
    num_distractor_samples: int = 2,
    num_variants: int = 20,
    verbose: bool = False,
) -> None:
    from tree_management.registry import get_task
    from tree_management.elements import fill_elements, load_lexicons
    from tree_management.generator import generate_tasks, get_template
    from tree_management.demo_generator import (
        generate_demos, generate_mixed_demos, build_world_listing,
    )
    from pipeline.adventure_episode import construct_demo_context
    from pipeline.qa_episode import run_qa_structure_exp
    from pipeline.prompts import QA_BASE_PROMPT

    p11_path = results_dir / f"phase_1_1_{task}_{base}_ace.json"
    if not p11_path.exists():
        print(f"  SKIP: {p11_path} not found")
        return

    playbook = _get_frozen_playbook(p11_path)
    if not playbook:
        print(f"  SKIP: no non-empty playbook found in {p11_path.name}")
        return
    print(f"  Frozen playbook: {len(playbook)} chars (from {p11_path.name})")

    spec   = get_task(task)
    gen_fn = spec.gen_fn

    with open(f"tree_management/rules/{task}.json") as f:
        rule_spec = json.load(f)
    sem_lex, nonce_lex = load_lexicons(rule_spec.get("lexicon", "default"))

    dist_spec = dist_sem_lex = dist_nonce_lex = None
    if distractor_rules_path:
        with open(distractor_rules_path) as f:
            dist_spec = json.load(f)
        dist_sem_lex, dist_nonce_lex = load_lexicons(dist_spec.get("lexicon", "default"))

    variant_outputs: List[Dict] = []

    for seed in range(num_variants):
        print(f"  Variant {seed}/{num_variants - 1}", end=" ... ")

        elements      = fill_elements(rule_spec, sem_lex, nonce_lex, seed=seed,
                                      split_spec=spec.split)
        source_tasks  = gen_fn(elements, split="source", use_nonce=False)
        gen_tasks     = gen_fn(elements, split="gen",    use_nonce=False)

        if dist_spec is not None:
            dist_elems  = fill_elements(dist_spec, dist_sem_lex, dist_nonce_lex, seed=seed)
            _dist_tmpl  = get_template(spec.distractor_template or spec.template_name)
            all_dist    = generate_tasks(dist_elems, process=None, split=None,
                                         use_nonce=False, task_type="distractor",
                                         template_name=_dist_tmpl.name, seed=seed)
            dist_tasks  = (all_dist if num_distractor_samples >= len(all_dist)
                           else _random.Random(seed).sample(all_dist, num_distractor_samples))
            demos = generate_mixed_demos(source_tasks, dist_tasks, gen_tasks=gen_tasks,
                                         initial_currency=500, seed=seed)
        else:
            demos = generate_demos(source_tasks, initial_currency=500, seed=seed)
            world_listing = build_world_listing(source_tasks + gen_tasks)
            for d in demos:
                d.metadata["world_listing"] = world_listing

        demo_context = construct_demo_context(demos, include_world=True)

        result = run_qa_structure_exp(
            source_tasks=source_tasks,
            gen_tasks=gen_tasks,
            demo_context=demo_context,
            task_type=task,
            model=model_path,
            base_prompt=QA_BASE_PROMPT,
            converter_model=converter_model,
            teaching_message=playbook,
            verbose=verbose,
            judge_model=judge_model,
        )
        print(f"rule_score={result['rule_score']}, overall={result['overall']:.3f}")

        variant_outputs.append({
            "variant_seed":  seed,
            "structure_exp": result,
        })

    _save_results(variant_outputs, task, base, "ace", results_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Score rule hypotheses for hr/idea/ace")
    parser.add_argument("--base",        default="gpt5")
    parser.add_argument("--methods",     default="hr,idea,ace")
    parser.add_argument("--tasks",       default=",".join(ALL_TASKS))
    parser.add_argument("--only",        default=None,
                        help="Comma-separated task subset (e.g. additive,conditional)")
    parser.add_argument("--results_dir", default="results/gpt5")
    parser.add_argument("--model_path",  default="gpt-5.4-mini",
                        help="Model for ace generation (not used for hr/idea)")
    parser.add_argument("--judge_model", default=None,
                        help="Model for judging (defaults to model_path)")
    parser.add_argument("--converter_model", default="gemini")
    parser.add_argument("--distractor_rules",
                        default="tree_management/distractors/canonical_property_distractor.json")
    parser.add_argument("--num_distractor_samples", type=int, default=2)
    parser.add_argument("--num_variants", type=int, default=20)
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    methods     = [m.strip() for m in args.methods.split(",") if m.strip()]
    tasks       = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if args.only:
        only_set = {t.strip() for t in args.only.split(",") if t.strip()}
        tasks = [t for t in tasks if t in only_set]

    judge_model = args.judge_model or args.model_path

    for method in methods:
        for task in tasks:
            print(f"\n=== {method} / {task} ===")
            out_path = results_dir / f"qa_phase2_{task}_{args.base}_{method}.json"
            if out_path.exists():
                print(f"  Already exists, skipping. (Delete to re-run.)")
                continue

            if method in ("hr", "idea"):
                score_iterative_method(
                    task=task,
                    base=args.base,
                    method=method,
                    results_dir=results_dir,
                    judge_model=judge_model,
                    converter_model=args.converter_model,
                    num_variants=args.num_variants,
                    verbose=args.verbose,
                )
            elif method == "ace":
                score_ace(
                    task=task,
                    base=args.base,
                    results_dir=results_dir,
                    model_path=args.model_path,
                    judge_model=judge_model,
                    converter_model=args.converter_model,
                    distractor_rules_path=args.distractor_rules,
                    num_distractor_samples=args.num_distractor_samples,
                    num_variants=args.num_variants,
                    verbose=args.verbose,
                )
            else:
                print(f"  Unknown method '{method}', skipping.")


if __name__ == "__main__":
    main()
