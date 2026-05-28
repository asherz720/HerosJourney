"""
Smoke test for Qwen 27B Phase 2 results.

Checks that all expected episode and QA Phase 2 files exist and are well-formed,
and prints a summary table of key metrics.

Usage (from repo root):
    python scripts/smoke_test_qwen27b.py
    python scripts/smoke_test_qwen27b.py --results_dir results/qwen27b
    python scripts/smoke_test_qwen27b.py --tasks additive compositional  # subset
"""

import argparse
import json
from pathlib import Path

PROPERTY_TASKS = ["additive", "compositional", "conditional", "override"]
PROC_TASKS     = ["proc_add", "proc_comp", "proc_cond", "proc_over"]
ALL_TASKS      = PROPERTY_TASKS + PROC_TASKS
METHODS        = ["react", "hr", "idea", "ace"]
DYNAMIC        = {"hr", "idea", "ace"}
BASE           = "qwen27b"


def _load(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        return f"ERROR: {e}"


def _ep_stats(data) -> str:
    if data is None:
        return "MISSING"
    if isinstance(data, str):
        return data
    variants = data.get("variants", [])
    n_var = len(variants)
    n_ep  = sum(len(v.get("episodes", {})) for v in variants)
    sr    = data.get("success_rate")
    sr_str = f"{sr:.1%}" if sr is not None else "?"
    # Check teaching_message is non-empty in at least first episode
    tm_ok = False
    for v in variants[:1]:
        for ep in list(v.get("episodes", {}).values())[:1]:
            if ep.get("teaching_message", ""):
                tm_ok = True
    tm_flag = "" if tm_ok else " [!no teaching_msg]"
    return f"ok  {n_var}v/{n_ep}ep  SR={sr_str}{tm_flag}"


def _qa_stats(data) -> str:
    if data is None:
        return "MISSING"
    if isinstance(data, str):
        return data
    variants = data.get("variants", [])
    n_var = len(variants)
    # Instance accuracy
    inst_correct = inst_total = 0
    rule_scores = []
    for v in variants:
        inst  = v.get("instance", {})
        inst_correct += inst.get("correct", 0)
        inst_total   += inst.get("total", 0)
        sexp = v.get("structure_exp", {})
        if sexp.get("rule_score") is not None:
            rule_scores.append(sexp["rule_score"] / 2.0)
    inst_acc  = f"{inst_correct/inst_total:.1%}" if inst_total else "?"
    rule_mean = f"{sum(rule_scores)/len(rule_scores):.2f}" if rule_scores else "?"
    return f"ok  {n_var}v  inst={inst_acc}  rule={rule_mean}"


def _p11_stats(data) -> str:
    if data is None:
        return "MISSING"
    if isinstance(data, str):
        return data
    variants = data.get("variants", [])
    filled = sum(1 for v in variants if v.get("teaching_message", ""))
    return f"ok  {filled}/{len(variants)} variants have teaching_msg"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/qwen27b")
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS)
    args = parser.parse_args()

    rdir = Path(args.results_dir)
    tasks = args.tasks

    print(f"\nSmoke test: Qwen 27B Phase 2  [{rdir}]")
    print(f"Tasks: {', '.join(tasks)}")
    print()

    # ---- Episode Phase 2 ----
    print("=" * 70)
    print("EPISODE Phase 2")
    print("=" * 70)
    ep_missing = []
    for method in METHODS:
        print(f"\n  [{method}]")
        for task in tasks:
            path = rdir / f"phase_2_{task}_{BASE}_{method}.json"
            data = _load(path)
            status = _ep_stats(data)
            tag = "✓" if status.startswith("ok") else "✗"
            print(f"    {tag} {task:<20} {status}")
            if not status.startswith("ok"):
                ep_missing.append(f"phase_2_{task}_{BASE}_{method}.json")

    # ---- Phase 1.1 (dynamic methods) ----
    print()
    print("=" * 70)
    print("PHASE 1.1 curation files (hr / idea / ace)")
    print("=" * 70)
    p11_missing = []
    for method in DYNAMIC:
        print(f"\n  [{method}]")
        for task in tasks:
            path = rdir / f"phase_1_1_{task}_{BASE}_{method}.json"
            data = _load(path)
            status = _p11_stats(data)
            tag = "✓" if status.startswith("ok") else "✗"
            print(f"    {tag} {task:<20} {status}")
            if not status.startswith("ok"):
                p11_missing.append(f"phase_1_1_{task}_{BASE}_{method}.json")

    # ---- QA Phase 2 ----
    print()
    print("=" * 70)
    print("QA Phase 2")
    print("=" * 70)
    qa_missing = []
    for method in METHODS:
        print(f"\n  [{method}]")
        for task in tasks:
            path = rdir / f"qa_phase2_{task}_{BASE}_{method}.json"
            data = _load(path)
            status = _qa_stats(data)
            tag = "✓" if status.startswith("ok") else "✗"
            print(f"    {tag} {task:<20} {status}")
            if not status.startswith("ok"):
                qa_missing.append(f"qa_phase2_{task}_{BASE}_{method}.json")

    # ---- Summary ----
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_ep  = len(METHODS) * len(tasks)
    total_p11 = len(DYNAMIC) * len(tasks)
    total_qa  = len(METHODS) * len(tasks)
    print(f"  Episode Phase 2 : {total_ep - len(ep_missing)}/{total_ep} present")
    print(f"  Phase 1.1       : {total_p11 - len(p11_missing)}/{total_p11} present")
    print(f"  QA Phase 2      : {total_qa - len(qa_missing)}/{total_qa} present")

    all_missing = ep_missing + p11_missing + qa_missing
    if all_missing:
        print(f"\n  Missing files ({len(all_missing)}):")
        for f in all_missing:
            print(f"    - {f}")
    else:
        print("\n  All files present and well-formed.")
    print()


if __name__ == "__main__":
    main()
