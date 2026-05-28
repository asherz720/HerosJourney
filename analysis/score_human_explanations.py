"""
Score human rule explanations using the Qwen3.5 judge.

Reads an exported explanations JSON (from rule_induction_tool.html) and calls
judge_structure_explanation() for each item — the same judge used for LLM scoring.

Usage (from repo root):
    python analysis/score_human_explanations.py \
        --input human_experiment/explanations_alice.json \
        --judge_model /path/to/qwen3.5@8001 \
        --output human_experiment/scored_alice.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.qa_episode import judge_structure_explanation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",           required=True,
                        help="Exported explanations JSON from rule_induction_tool.html")
    parser.add_argument("--judge_model",     required=True,
                        help="Judge model path (e.g. /path/to/qwen3.5@8001)")
    parser.add_argument("--converter_model", default="small")
    parser.add_argument("--output",          default=None,
                        help="Output path (default: scored_<input>)")
    parser.add_argument("--verbose",         action="store_true")
    args = parser.parse_args()

    inp = Path(args.input)
    data = json.loads(inp.read_text())
    annotator   = data.get("annotator", "unknown")
    explanations = data.get("explanations", [])

    print(f"Scoring {len(explanations)} explanations from '{annotator}' ...")

    results = []
    for i, item in enumerate(explanations):
        task_type   = item["task_type"]
        explanation = (item.get("explanation") or "").strip()
        print(f"  [{i+1}/{len(explanations)}] {task_type} — {len(explanation)} chars")

        if not explanation:
            scores = {
                "input_score": 0, "output_score": 0,
                "rule_score": 0, "generalization_score": 0,
                "overall": 0.0, "reasoning": "No explanation provided.",
                "judge_failed": True,
            }
        else:
            scores = judge_structure_explanation(
                explanation=explanation,
                task_type=task_type,
                model=args.judge_model,
                converter_model=args.converter_model,
                verbose=args.verbose,
            )

        results.append({
            "task_type":            task_type,
            "seed":                 item.get("seed"),
            "explanation":          explanation,
            "input_score":          scores.get("input_score", 0),
            "output_score":         scores.get("output_score", 0),
            "rule_score":           scores.get("rule_score", 0),
            "generalization_score": scores.get("generalization_score", 0),
            "overall":              scores.get("overall", 0.0),
            "judge_reasoning":      scores.get("reasoning", ""),
            "judge_failed":         scores.get("judge_failed", False),
        })

    out_path = Path(args.output) if args.output else inp.parent / f"scored_{inp.name}"
    output = {
        "annotator":    annotator,
        "judge_model":  args.judge_model,
        "n_scored":     len(results),
        "results":      results,
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved → {out_path}")

    # Summary
    import numpy as np
    for task_type in sorted(set(r["task_type"] for r in results)):
        rs = [r["rule_score"] for r in results if r["task_type"] == task_type and not r["judge_failed"]]
        if rs:
            print(f"  {task_type:15s}  rule_score={np.mean(rs):.2f}/2")


if __name__ == "__main__":
    main()
