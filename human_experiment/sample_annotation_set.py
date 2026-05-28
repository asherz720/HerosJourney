"""
Sample a stratified annotation set for LLM judge validation.

Stratifies by rule_score (0 / 1 / 2) to ensure balanced coverage,
then shuffles so score level isn't revealed by item order.

Usage (from repo root):
    python analysis/sample_annotation_set.py \
        --results_dir results \
        --models qwen27b,gemini,gptoss \
        --n_per_level 33 \
        --seed 42 \
        --output analysis/annotation_set.json
"""

import argparse
import json
import random
from pathlib import Path

ALL_TASKS = [
    "additive", "compositional", "conditional", "override",
    "proc_add", "proc_comp", "proc_cond", "proc_over",
]


def collect_items(results_dir: Path, models: list[str]) -> list[dict]:
    items = []
    for model in models:
        for task in ALL_TASKS:
            fpath = results_dir / model / f"qa_phase1_{task}_{model}.json"
            if not fpath.exists():
                continue
            data = json.loads(fpath.read_text())
            for var in data["variants"]:
                sexp = var.get("structure_exp", {})
                if sexp.get("judge_failed", False):
                    continue
                score = sexp.get("rule_score")
                if score is None:
                    continue
                items.append(dict(
                    model=model,
                    task=task,
                    variant_seed=var.get("variant_seed"),
                    correct_rule=sexp.get("correct_rule", ""),
                    explanation=sexp.get("explanation", ""),
                    llm_rule_score=int(score),
                    llm_reasoning=sexp.get("judge_reasoning", ""),
                ))
    return items


def sample_stratified(items: list[dict], n_per_level: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_score = {0: [], 1: [], 2: []}
    for item in items:
        s = item["llm_rule_score"]
        if s in by_score:
            by_score[s].append(item)

    sampled = []
    for score, pool in by_score.items():
        n = min(n_per_level, len(pool))
        chosen = rng.sample(pool, n)
        if n < n_per_level:
            print(f"  Warning: only {n} items with score={score} (requested {n_per_level})")
        sampled.extend(chosen)

    rng.shuffle(sampled)
    for i, item in enumerate(sampled):
        item["item_id"] = i
    return sampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--models",      default="qwen27b,gemini,gptoss")
    parser.add_argument("--n_per_level", type=int, default=33,
                        help="Items to sample per score level (0/1/2)")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output",      default="analysis/annotation_set.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    models      = [m.strip() for m in args.models.split(",")]

    print(f"Collecting items from {results_dir} for models: {models}")
    items = collect_items(results_dir, models)
    print(f"  Total items: {len(items)}")

    from collections import Counter
    dist = Counter(it["llm_rule_score"] for it in items)
    print(f"  Score distribution: {dict(sorted(dist.items()))}")

    sampled = sample_stratified(items, args.n_per_level, args.seed)
    print(f"  Sampled: {len(sampled)} items  "
          f"({Counter(it['llm_rule_score'] for it in sampled)})")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sampled, indent=2))
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
