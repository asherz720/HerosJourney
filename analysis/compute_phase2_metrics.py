"""
Compute metrics.csv for Phase 2 steering methods.

Loads phase_2_{task}_{base_stem}_{method}.json files and outputs a CSV in the
same format as analysis/figures/metrics.csv (success_rate, efficiency_on_success,
expected_efficiency, ecsr, ecsr_se, ecsr_ci95).

Usage (from repo root):
    python analysis/compute_phase2_metrics.py \
        --base_models gpt5 \
        --methods react,hr,idea,ace \
        --output analysis/figures/metrics_phase2.csv

    # Include both models:
    python analysis/compute_phase2_metrics.py \
        --base_models gpt5,qwen27b \
        --methods react,hr,idea,ace \
        --output analysis/figures/metrics_phase2.csv

    # Also include Phase 1 baseline rows:
    python analysis/compute_phase2_metrics.py \
        --base_models gpt5 \
        --methods react,hr,idea,ace \
        --include_phase1 \
        --output analysis/figures/metrics_phase2.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Task config (mirrors viz_results.py)
# ---------------------------------------------------------------------------
PROPERTY_TASKS = ["additive", "compositional", "conditional", "override"]
PROC_TASKS     = ["proc_add", "proc_comp", "proc_cond", "proc_over"]
ALL_TASKS      = PROPERTY_TASKS + PROC_TASKS

TASK_MAX_TRIES = {
    "additive":      5,
    "compositional": 9,
    "conditional":   6,
    "override":      4,
    "proc_add":      3,
    "proc_comp":     4,
    "proc_cond":     3,
    "proc_over":     4,
}

# Where each model's results live (relative to results/)
MODEL_RESULTS_SUBDIR = {
    "gpt5":    "gpt5",
    "qwen27b": "qwen27b",
}


def _new_efficiency(ep: dict) -> float | None:
    if not ep.get("success", False):
        return None
    num_runs = ep.get("num_runs")
    ref_len  = ep.get("reference_length")
    if num_runs and ref_len:
        return ref_len / num_runs
    stored = ep.get("efficiency")
    if stored and stored > 0:
        return 1.0 / stored
    return None


def _load_qa(results_dir: Path, model_dir: str, filename: str) -> list[dict]:
    """Load per-variant RV scores from a qa_phase results file."""
    fpath = results_dir / model_dir / filename
    if not fpath.exists():
        return []
    data = json.loads(fpath.read_text())
    rows = []
    for var in data.get("variants", []):
        seed = var.get("variant_seed")
        sexp = var.get("structure_exp", {})
        rule_s = (sexp.get("rule_score", 0) or 0) / 2.0
        rows.append(dict(variant_seed=seed, rule_score=rule_s))
    return rows


def _load_episodes(results_dir: Path, model_dir: str, filename: str) -> list[dict]:
    """Load all episodes from a phase results file. Returns flat list of episode dicts."""
    fpath = results_dir / model_dir / filename
    if not fpath.exists():
        return []
    data = json.loads(fpath.read_text())
    rows = []
    for var in data.get("variants", []):
        seed = var.get("variant_seed")
        for ep in var.get("episodes", {}).values():
            rows.append(dict(
                variant_seed=seed,
                success=bool(ep.get("success", False)),
                efficiency=_new_efficiency(ep),
                num_runs=ep.get("num_runs"),
                reference_length=ep.get("reference_length"),
            ))
    return rows


def _variant_ecsr(results_dir: Path, base_model: str, task: str,
                  method: str = "") -> dict[int, float]:
    """Return {variant_seed: ecsr} from a phase_2 (or phase_1 baseline) JSON file."""
    subdir = MODEL_RESULTS_SUBDIR.get(base_model, base_model)
    fname  = (f"phase_2_{task}_{base_model}_{method}.json" if method
              else f"phase_1_{task}_{base_model}.json")
    rows = _load_episodes(results_dir, subdir, fname)
    if not rows:
        return {}
    floor = 1.0 / TASK_MAX_TRIES.get(task, 2)
    out = {}
    from itertools import groupby
    rows_sorted = sorted(rows, key=lambda r: r["variant_seed"])
    for seed, group in groupby(rows_sorted, key=lambda r: r["variant_seed"]):
        eps = list(group)
        v_sr  = sum(e["success"] for e in eps) / len(eps)
        effs  = [e["efficiency"] for e in eps if e["efficiency"] is not None]
        v_eff = sum(effs) / len(effs) if effs else 0.0
        v_norm = max((v_eff - floor) / (1.0 - floor), 0.0) if v_eff > 0 else 0.0
        out[seed] = v_sr * v_norm
    return out


def _bootstrap_pval(deltas: np.ndarray, n_boot: int = 10000,
                    rng: np.random.Generator = None) -> float:
    """Two-tailed paired bootstrap p-value for H0: mean(delta) = 0."""
    d_obs    = deltas.mean()
    centered = deltas - d_obs
    boot_means = np.array([
        rng.choice(centered, size=len(centered), replace=True).mean()
        for _ in range(n_boot)
    ])
    return float((np.abs(boot_means) >= np.abs(d_obs)).mean())


def add_delta_significance(
    csv_path: Path,
    results_dir: Path,
    n_boot: int = 10000,
    seed: int = 42,
) -> None:
    """Add delta_ecsr, delta_ecsr_p_raw, delta_ecsr_p_adj, delta_ecsr_sig columns
    to an existing phase2 metrics CSV.  Bonferroni correction is applied across all
    teaching-method rows in the file (4 methods × 8 tasks per base model)."""

    df = pd.read_csv(csv_path)

    def _parse(model_str):
        for base in ("gpt5", "qwen27b"):
            if model_str == base:
                return base, ""
            if model_str.startswith(base + "_"):
                return base, model_str[len(base) + 1:]
        return model_str, ""

    df[["_base", "_method"]] = df["model"].apply(lambda x: pd.Series(_parse(x)))

    # Baseline ECSR from the CSV (model = base, no method suffix)
    baseline_ecsr = (
        df[df["_method"] == ""][["_base", "task", "ecsr"]]
        .rename(columns={"ecsr": "_ecsr_base"})
    )
    df = df.merge(baseline_ecsr, on=["_base", "task"], how="left")
    df["delta_ecsr"] = (df["ecsr"] - df["_ecsr_base"]).round(4)

    # Per-task bootstrap p-values for teaching rows
    rng = np.random.default_rng(seed)
    teaching_mask = df["_method"] != ""
    raw_pvals: dict[tuple, float] = {}

    for _, row in df[teaching_mask].iterrows():
        base_v  = _variant_ecsr(results_dir, row["_base"], row["task"])
        teach_v = _variant_ecsr(results_dir, row["_base"], row["task"], row["_method"])
        deltas  = [teach_v[s] - base_v[s] for s in set(base_v) & set(teach_v)]
        if len(deltas) >= 2:
            p = _bootstrap_pval(np.array(deltas), n_boot=n_boot, rng=rng)
            raw_pvals[(row["_base"], row["_method"], row["task"])] = p

    # Bonferroni across all teaching rows in this file
    n_tests = len(raw_pvals)
    adj_pvals = {k: min(v * n_tests, 1.0) for k, v in raw_pvals.items()}

    def _stars(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return ""

    df["delta_ecsr_p_raw"] = df.apply(
        lambda r: round(raw_pvals.get((r["_base"], r["_method"], r["task"]), float("nan")), 4),
        axis=1,
    )
    df["delta_ecsr_p_adj"] = df.apply(
        lambda r: round(adj_pvals.get((r["_base"], r["_method"], r["task"]), float("nan")), 4),
        axis=1,
    )
    df["delta_ecsr_sig"] = df["delta_ecsr_p_adj"].apply(
        lambda p: _stars(p) if not (isinstance(p, float) and np.isnan(p)) else ""
    )

    df = df.drop(columns=["_base", "_method", "_ecsr_base"])
    df.to_csv(csv_path, index=False)
    print(f"Updated: {csv_path}  ({n_tests} tests, Bonferroni-corrected)")


def _load_model_phase(
    results_dir: Path,
    base_model: str,
    tasks: list[str],
    phase: str,      # "1" or "2"
    method: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (episode_df, qa_df)."""
    subdir = MODEL_RESULTS_SUBDIR.get(base_model, base_model)
    ep_rows, qa_rows = [], []
    model_key = f"{base_model}_{method}" if method else base_model
    for task in tasks:
        if phase == "2":
            ep_fname = f"phase_2_{task}_{base_model}_{method}.json"
            qa_fname = f"qa_phase2_{task}_{base_model}_{method}.json"
        else:
            ep_fname = f"phase_1_{task}_{base_model}.json"
            qa_fname = f"qa_phase1_{task}_{base_model}.json"
        for ep in _load_episodes(results_dir, subdir, ep_fname):
            ep_rows.append(dict(model=model_key, task=task, **ep))
        for row in _load_qa(results_dir, subdir, qa_fname):
            qa_rows.append(dict(model=model_key, task=task, **row))
    return pd.DataFrame(ep_rows), pd.DataFrame(qa_rows)


def _ci95(arr: list[float]):
    n = len(arr)
    if n < 2:
        return float("nan"), float("nan")
    a = np.array(arr)
    se = a.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, df=n - 1)
    return se, t * se


def _compute_metrics(edf: pd.DataFrame, qdf: pd.DataFrame, tasks: list[str]) -> pd.DataFrame:
    rows = []
    models = edf["model"].unique() if not edf.empty else []
    for model in models:
        for task in tasks:
            sub = edf[(edf["model"] == model) & (edf["task"] == task)]
            if sub.empty:
                continue

            sr          = sub["success"].mean()
            eff_vals    = sub["efficiency"].dropna().values
            eff_on_succ = eff_vals.mean() if len(eff_vals) > 0 else 0.0
            floor       = 1.0 / TASK_MAX_TRIES.get(task, 2)
            norm_eff    = max((eff_on_succ - floor) / (1.0 - floor), 0.0) if eff_on_succ > 0 else 0.0

            ecsr_per_var = []
            for _, vg in sub.groupby("variant_seed"):
                v_sr  = vg["success"].mean()
                v_eff = vg["efficiency"].dropna().values
                v_eff_on_succ = v_eff.mean() if len(v_eff) > 0 else 0.0
                v_norm = max((v_eff_on_succ - floor) / (1.0 - floor), 0.0) if v_eff_on_succ > 0 else 0.0
                ecsr_per_var.append(v_sr * v_norm)

            ecsr_se, ecsr_ci95 = _ci95(ecsr_per_var)

            # RV from QA data
            rv_val = float("nan")
            rv_se  = float("nan")
            rv_ci95_val = float("nan")
            if not qdf.empty:
                q_sub = qdf[(qdf["model"] == model) & (qdf["task"] == task)]
                if not q_sub.empty:
                    rv_per_var = [vg["rule_score"].mean()
                                  for _, vg in q_sub.groupby("variant_seed")]
                    rv_val = float(np.mean(rv_per_var))
                    rv_se, rv_ci95_val = _ci95(rv_per_var)

            rows.append(dict(
                model=model,
                task=task,
                n_episodes=len(sub),
                n_variants=len(ecsr_per_var),
                success_rate=round(sr, 4),
                efficiency_on_success=round(eff_on_succ, 4),
                expected_efficiency=round(sr * eff_on_succ, 4),
                ecsr=round(sr * norm_eff, 4),
                ecsr_se=round(ecsr_se, 4),
                ecsr_ci95=round(ecsr_ci95, 4),
                rule_score=round(rv_val, 4) if not np.isnan(rv_val) else "",
                rule_score_se=round(rv_se, 4) if not np.isnan(rv_se) else "",
                rule_score_ci95=round(rv_ci95_val, 4) if not np.isnan(rv_ci95_val) else "",
            ))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Phase 2 metrics CSV")
    parser.add_argument("--results_dir",   default="results",
                        help="Root results directory (default: results)")
    parser.add_argument("--base_models",   default="gpt5",
                        help="Comma-separated base model names (default: gpt5)")
    parser.add_argument("--methods",       default="react,hr,idea,ace",
                        help="Comma-separated Phase 2 methods (default: react,hr,idea,ace)")
    parser.add_argument("--tasks",         default=",".join(ALL_TASKS),
                        help="Comma-separated task names")
    parser.add_argument("--include_phase1", action="store_true",
                        help="Also include Phase 1 baseline rows")
    parser.add_argument("--output",        default="analysis/figures/metrics_phase2.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    results_dir  = Path(args.results_dir)
    base_models  = [m.strip() for m in args.base_models.split(",") if m.strip()]
    methods      = [m.strip() for m in args.methods.split(",") if m.strip()]
    tasks        = [t.strip() for t in args.tasks.split(",") if t.strip()]
    output_path  = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_ep, all_qa = [], []

    for base_model in base_models:
        if args.include_phase1:
            ep1, qa1 = _load_model_phase(results_dir, base_model, tasks, phase="1")
            if not ep1.empty:
                all_ep.append(ep1)
                print(f"  Loaded Phase 1: {base_model} ({len(ep1)} episodes)")
            if not qa1.empty:
                all_qa.append(qa1)

        for method in methods:
            ep2, qa2 = _load_model_phase(results_dir, base_model, tasks, phase="2", method=method)
            if ep2.empty:
                print(f"  Warning: no Phase 2 episode data for {base_model} / {method}")
                continue
            all_ep.append(ep2)
            print(f"  Loaded Phase 2 {method}: {base_model} ({len(ep2)} episodes)")
            if not qa2.empty:
                all_qa.append(qa2)
                print(f"  Loaded QA Phase 2 {method}: {base_model} ({len(qa2)} variants)")

    if not all_ep:
        print("No data loaded. Check --results_dir and file names.")
        return

    edf     = pd.concat(all_ep, ignore_index=True)
    qdf     = pd.concat(all_qa, ignore_index=True) if all_qa else pd.DataFrame()
    metrics = _compute_metrics(edf, qdf, tasks)
    metrics = metrics.sort_values(["model", "task"]).reset_index(drop=True)
    metrics.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
