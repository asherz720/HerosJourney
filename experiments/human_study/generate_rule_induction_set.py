"""
Generate one demo context per task type for the human rule induction experiment.

Saves human_experiment/rule_induction_set.json — load this into rule_induction_tool.html.

Usage (from repo root):
    python analysis/generate_rule_induction_set.py --seed 42
    python analysis/generate_rule_induction_set.py --seed 42 --nonce
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from herosjourney.core.demo_generator import generate_demos, generate_mixed_demos
from herosjourney.core.elements import fill_elements, load_lexicons
from herosjourney.core.generator import generate_tasks, get_template
from herosjourney.core.registry import get_task

ALL_TASK_TYPES = [
    "additive", "compositional", "conditional", "override",
    "proc_add", "proc_comp", "proc_cond", "proc_over",
]
RULES_DIR = Path(__file__).parent.parent / "tree_management" / "rules"


def construct_demo_context(demos, include_world=True):
    if not demos:
        return ""
    lines = []
    if include_world:
        world = demos[0].metadata.get("world_listing", "")
        if world:
            lines.append(world)
            lines.append("")
    lines.append("[Start of Demonstration Episodes]")
    for demo in demos:
        lines.append(demo.format(show_world=False))
        lines.append("")
    lines.append("[End of Demonstration Episodes]")
    return "\n".join(lines)


def generate_one(task_type, use_nonce, seed, initial_currency,
                 distractor_rules_path, num_distractor_samples):
    rules_path = RULES_DIR / f"{task_type}.json"
    with open(rules_path) as f:
        rule_spec = json.load(f)

    spec = get_task(task_type)
    sem_lex, nonce_lex = load_lexicons(rule_spec.get("lexicon", "default"))
    elements = fill_elements(rule_spec, sem_lex, nonce_lex, seed=seed, split_spec=spec.split)

    source_tasks = spec.gen_fn(elements, split="source", use_nonce=use_nonce)
    gen_tasks    = spec.gen_fn(elements, split="gen",    use_nonce=use_nonce)

    demo_source = list(source_tasks)
    random.Random(seed).shuffle(demo_source)

    dist_path = distractor_rules_path or spec.distractor_rules
    if dist_path:
        with open(dist_path) as f:
            dist_spec = json.load(f)
        dist_sem_lex, dist_nonce_lex = load_lexicons(dist_spec.get("lexicon", "default"))
        dist_elems = fill_elements(dist_spec, dist_sem_lex, dist_nonce_lex, seed=seed)
        dist_tmpl  = get_template(spec.distractor_template or spec.template_name)
        all_dist   = generate_tasks(
            dist_elems, process=None, split=None,
            use_nonce=use_nonce, task_type="distractor",
            template_name=dist_tmpl.name, seed=seed,
        )
        distractor_tasks = (
            all_dist if num_distractor_samples >= len(all_dist)
            else random.Random(seed).sample(all_dist, num_distractor_samples)
        )
        demos = generate_mixed_demos(
            property_tasks=demo_source,
            distractor_tasks=distractor_tasks,
            gen_tasks=gen_tasks,
            initial_currency=initial_currency,
            seed=seed,
        )
    else:
        demos = generate_demos(demo_source, initial_currency=initial_currency, seed=seed)

    demo_context = construct_demo_context(demos)
    correct_rule = spec.correct_rule or ""
    print(f"  [{task_type}] demo context: {len(demo_context)} chars")
    return demo_context, correct_rule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",                    type=int, default=42)
    parser.add_argument("--nonce",                   action="store_true")
    parser.add_argument("--initial_currency",        type=int, default=500)
    parser.add_argument("--distractor_rules",        default="herosjourney/core/distractors/canonical_property_distractor.json")
    parser.add_argument("--num_distractor_samples",  type=int, default=2)
    parser.add_argument("--output",                  default="human_experiment/rule_induction_set.json")
    args = parser.parse_args()

    print(f"Generating rule induction set (seed={args.seed}, nonce={args.nonce}) ...")
    items = []
    for task_type in ALL_TASK_TYPES:
        demo_context, correct_rule = generate_one(
            task_type,
            use_nonce=args.nonce,
            seed=args.seed,
            initial_currency=args.initial_currency,
            distractor_rules_path=args.distractor_rules,
            num_distractor_samples=args.num_distractor_samples,
        )
        items.append({
            "task_type":    task_type,
            "seed":         args.seed,
            "nonce":        args.nonce,
            "demo_context": demo_context,
            "correct_rule": correct_rule,
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, indent=2))
    print(f"Saved {len(items)} tasks → {out}")


if __name__ == "__main__":
    main()
