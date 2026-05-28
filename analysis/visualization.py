"""
visualization.py — streamlined figure generation.

Generates the key figures and metrics.csv:
  fig4_combined_ecsr       — rule score vs ECSR scatter (all models + human panel)
  fig14_combined_scatter   — single-panel: models + humans co-plotted, two regression lines
  fig15_radar              — radar charts: ECSR (left) and RV (right) across all task types
  fig8_gap_ecsr            — instance accuracy vs ECSR gap (Qwen + GPT only)
  fig9_nonce_comparison    — semantic vs nonce ECSR/RV (nonce pairs only)
  fig13_gap_nonce_combined — 1×4 combined: fig8 panels A/B + fig9 panels C/D (one row)
  fig10b_coverage_sweep    — ECSR vs coverage k (GPT + GPT-OSS only)
  fig11_phase2_methods     — phase 2 ECSR after vs before, by teaching method (scatter)
  fig12_phase2_delta       — mean Δ ECSR per method × task family (diverging bar chart)
  metrics.csv              — ECSR and RV for all models

Usage:
  python analysis/visualization.py --models qwen27b,gpt5,gptoss,gemini3.1,llama,olmo \\
      --include_humans --output_dir analysis/figures_main
"""

import argparse
import json
import sys
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # noqa: F401 (imported for potential future use)
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.viz_results import (
    load_all_data,
    make_figure1b_normeff,
    make_figure4_combined_ecsr,
    make_figure8,
    make_figure9,
    make_figure13,
    make_figure14,
    make_figure13a,
    make_figure13b,
    make_figure15,
    make_figure16,
    make_figure10b,
    save_metrics_csv,
    save_figure13a_csv,
    save_figure13b_csv,
    print_family_correlations,
    COVERAGE_MODEL_MAP,
    ALL_TASKS,
    PROPERTY_TASKS,
    PROC_TASKS,
    _apply_nature_style,
    _despine,
)


# ---------------------------------------------------------------------------
# Figure 11 & 12 — phase 2 teaching methods
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Significance testing helpers for fig12
# ---------------------------------------------------------------------------

_TASK_MAX_TRIES = {
    "additive": 5, "compositional": 9, "conditional": 6, "override": 4,
    "proc_add": 3, "proc_comp": 4, "proc_cond": 3, "proc_over": 4,
}
_MODEL_SUBDIRS = {"gpt5": "gpt5", "qwen27b": "qwen27b"}


def _load_variant_ecsr(results_dir: Path, model: str, task: str,
                       method: str = "") -> dict[int, float]:
    """Return {variant_seed: ecsr} from a phase_2 (or phase_1 baseline) JSON file."""
    subdir = _MODEL_SUBDIRS.get(model, model)
    if method:
        fname = f"phase_2_{task}_{model}_{method}.json"
    else:
        fname = f"phase_1_{task}_{model}.json"
    fpath = results_dir / subdir / fname
    if not fpath.exists():
        return {}
    data = json.loads(fpath.read_text())
    floor = 1.0 / _TASK_MAX_TRIES.get(task, 2)
    out = {}
    for var in data.get("variants", []):
        seed = var.get("variant_seed")
        eps  = var.get("episodes", {})
        successes = [bool(ep.get("success", False)) for ep in eps.values()]
        effs = []
        for ep in eps.values():
            if ep.get("success"):
                nr = ep.get("num_runs")
                rl = ep.get("reference_length")
                if nr and rl:
                    effs.append(rl / nr)
                elif ep.get("efficiency", 0) > 0:
                    effs.append(1.0 / ep["efficiency"])
        v_sr   = sum(successes) / len(successes) if successes else 0.0
        v_eff  = sum(effs) / len(effs) if effs else 0.0
        v_norm = max((v_eff - floor) / (1.0 - floor), 0.0) if v_eff > 0 else 0.0
        out[seed] = v_sr * v_norm
    return out


def _bootstrap_pval(deltas: np.ndarray, n_boot: int = 10000,
                    rng: np.random.Generator = None) -> float:
    """Two-tailed paired bootstrap p-value for H0: mean(delta) = 0.
    Centers the bootstrap distribution under H0 by subtracting the observed mean."""
    d_obs = deltas.mean()
    centered = deltas - d_obs          # shift so null mean = 0
    boot_means = np.array([
        rng.choice(centered, size=len(centered), replace=True).mean()
        for _ in range(n_boot)
    ])
    return float((np.abs(boot_means) >= np.abs(d_obs)).mean())


def _compute_delta_pvals(
    results_dir: Path,
    methods: list[str],
    models: list[str],
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[tuple, float]:
    """Return {(method, model, family): Bonferroni-adjusted p_value}.
    Correction ×4 within each method (2 models × 2 families).
    Each method is a separate claim; models and families corrected jointly.
    Family is 'Attribute' or 'Procedural'."""
    rng = np.random.default_rng(seed)
    method_keys:  dict[str, list] = {m: [] for m in methods}
    method_pvals: dict[str, list] = {m: [] for m in methods}
    for method in methods:
        for model in models:
            for family, task_list in (
                ("Attribute", PROPERTY_TASKS),
                ("Procedural", PROC_TASKS),
            ):
                deltas = []
                for task in task_list:
                    base  = _load_variant_ecsr(results_dir, model, task)
                    teach = _load_variant_ecsr(results_dir, model, task, method)
                    for seed_ in set(base) & set(teach):
                        deltas.append(teach[seed_] - base[seed_])
                if len(deltas) >= 2:
                    p = _bootstrap_pval(np.array(deltas), n_boot=n_boot, rng=rng)
                    method_keys[method].append((method, model, family))
                    method_pvals[method].append(p)
    result = {}
    for method in methods:
        n = len(method_pvals[method])
        for k, p in zip(method_keys[method], method_pvals[method]):
            result[k] = min(p * n, 1.0)
    return result


def compute_aggregated_method_tests(
    results_dir: Path,
    methods: list[str],
    models: list[str],
    output_path: Path,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict:
    """One aggregated test per method: pool Δ ECSR across all models × families.
    H0: mean Δ = 0. Four tests total (one per method), no correction needed.
    Results saved to output_path as JSON.
    """
    rng = np.random.default_rng(seed)
    results = {}

    for method in methods:
        all_deltas = []
        for model in models:
            for task_list in (PROPERTY_TASKS, PROC_TASKS):
                for task in task_list:
                    base  = _load_variant_ecsr(results_dir, model, task)
                    teach = _load_variant_ecsr(results_dir, model, task, method)
                    for seed_ in set(base) & set(teach):
                        all_deltas.append(teach[seed_] - base[seed_])

        if len(all_deltas) < 2:
            results[method] = None
            continue

        arr     = np.array(all_deltas)
        mean_d  = float(arr.mean())
        p       = _bootstrap_pval(arr, n_boot=n_boot, rng=rng)
        # 95% CI via bootstrap percentile
        boot_means = np.array([
            rng.choice(arr, size=len(arr), replace=True).mean()
            for _ in range(n_boot)
        ])
        ci_lo, ci_hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

        results[method] = dict(
            n       = len(all_deltas),
            mean    = round(mean_d, 4),
            ci_95   = [round(ci_lo, 4), round(ci_hi, 4)],
            p       = round(p, 4),
            stars   = _sig_stars(p),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved aggregated method tests: {output_path}")
    return results


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Shape = method, color = model (consistent with prior figures)
_METHOD_MARKERS = {"ace": "s", "hr": "^", "idea": "o", "react": "D"}
_METHOD_LABELS  = {"ace": "ACE", "hr": "HR", "idea": "IDEA", "react": "ReAct"}
_MODEL_COLORS   = {"gpt5": "#4878CF", "qwen27b": "#E87722"}
_MODEL_LABELS   = {"gpt5": "GPT", "qwen27b": "Qwen"}
_FAMILY_COLORS  = {"Attribute": "#5B8DB8", "Procedural": "#D4845A"}


def _load_phase2_merged(
    phase2_gpt_path: str | Path,
    phase2_qwen_path: str | Path,
) -> pd.DataFrame:
    """Load and merge phase2 CSVs; add base_model, method, delta columns."""
    gpt_df  = pd.read_csv(phase2_gpt_path)
    qwen_df = pd.read_csv(phase2_qwen_path)
    df = pd.concat([gpt_df, qwen_df], ignore_index=True)

    def _parse(model_str):
        for base in ("gpt5", "qwen27b"):
            if model_str == base:
                return base, "baseline"
            if model_str.startswith(base + "_"):
                return base, model_str[len(base) + 1:]
        return model_str, "baseline"

    df[["base_model", "method"]] = df["model"].apply(lambda x: pd.Series(_parse(x)))
    baseline = (
        df[df["method"] == "baseline"][["base_model", "task", "ecsr"]]
        .rename(columns={"ecsr": "ecsr_baseline"})
    )
    teaching = df[df["method"].isin(_METHOD_MARKERS)]
    merged   = teaching.merge(baseline, on=["base_model", "task"], how="inner")
    merged["delta"] = merged["ecsr"] - merged["ecsr_baseline"]
    return merged


def make_figure11(
    phase2_gpt_path: str | Path,
    phase2_qwen_path: str | Path,
    output_path: Path,
) -> None:
    """Two-panel scatter: ECSR after (y) vs ECSR before (x).
    Shape = method, color = model."""

    merged = _load_phase2_merged(phase2_gpt_path, phase2_qwen_path)

    _apply_nature_style()
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=False)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.16, wspace=0.18)

    rng = np.random.default_rng(42)
    jitter = 0.015

    for ax, task_list, title in (
        (ax_l, PROPERTY_TASKS, "Attribute induction"),
        (ax_r, PROC_TASKS,     "Procedural induction"),
    ):
        sub = merged[merged["task"].isin(task_list)]

        is_proc = task_list is PROC_TASKS
        if is_proc:
            data_max = max(sub["ecsr_baseline"].max(), sub["ecsr"].max())
            ax_max = min(round(data_max + 0.15, 1), 1.0)
            margin = 0.03
        else:
            ax_max = 1.0
            margin = 0.05

        ax.plot([0, ax_max], [0, ax_max], color="#aaaaaa", linewidth=0.8,
                linestyle="--", zorder=1)

        for method, marker in _METHOD_MARKERS.items():
            for model, color in _MODEL_COLORS.items():
                pts = sub[(sub["method"] == method) & (sub["base_model"] == model)]
                if pts.empty:
                    continue
                x = pts["ecsr_baseline"].values + rng.uniform(-jitter, jitter, len(pts))
                y = pts["ecsr"].values          + rng.uniform(-jitter, jitter, len(pts))
                ax.scatter(
                    x, y, marker=marker, color=color, s=32,
                    edgecolors="white", linewidths=0.5,
                    zorder=4, alpha=0.75,
                )

        ax.set_xlim(-margin, ax_max + margin)
        ax.set_ylim(-margin, ax_max + margin)
        ax.set_aspect("equal")
        ax.set_xlabel("ECSR (before)", labelpad=3)
        if ax is ax_l:
            ax.set_ylabel("ECSR (after)", labelpad=3)
        ax.set_title(title, pad=4)
        tick_step = 0.25 if ax_max >= 0.75 else 0.10
        ax.xaxis.set_major_locator(mticker.MultipleLocator(tick_step))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(tick_step))
        _despine(ax)

    method_handles = [
        mpl.lines.Line2D([], [], marker=_METHOD_MARKERS[k], color="#555555",
                         linestyle="None", markersize=4, label=_METHOD_LABELS[k])
        for k in _METHOD_MARKERS
    ]
    model_handles = [
        mpatches.Patch(color=c, label=_MODEL_LABELS[k])
        for k, c in _MODEL_COLORS.items()
    ]
    diag_handle = mpl.lines.Line2D(
        [], [], color="#aaaaaa", linewidth=0.8, linestyle="--", label="No change"
    )
    ax_r.legend(
        handles=method_handles + model_handles + [diag_handle],
        loc="lower right",
        frameon=False,
        handlelength=0.9, handletextpad=0.4, borderpad=0.3,
        labelspacing=0.3, fontsize=5.5,
    )

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_figure12(
    phase2_gpt_path: str | Path,
    phase2_qwen_path: str | Path,
    output_path: Path,
    results_dir: str | Path = "results",
) -> None:
    """Diverging horizontal bar chart: Δ ECSR per method × model × task family.
    y = method (best attr Δ at top). Four grouped bars per method:
      attr/GPT, attr/Qwen (upper pair) and proc/GPT, proc/Qwen (lower pair).
    Color = model, hatch = family (solid = attribute, /// = procedural).
    Significance stars from paired variant-level t-test."""

    merged = _load_phase2_merged(phase2_gpt_path, phase2_qwen_path)
    results_dir = Path(results_dir)
    pvals = _compute_delta_pvals(
        results_dir,
        methods=list(_METHOD_MARKERS.keys()),
        models=list(_MODEL_COLORS.keys()),
    )

    attr_delta = (
        merged[merged["task"].isin(PROPERTY_TASKS)]
        .groupby("method")["delta"].mean()
    )
    # ascending=True → worst at y=0 (bottom), best at y=n-1 (top)
    methods_sorted = attr_delta.sort_values(ascending=True).index.tolist()
    for m in _METHOD_MARKERS:
        if m not in methods_sorted:
            methods_sorted.insert(0, m)

    bar_h      = 0.11
    gap_inner  = 0.02
    gap_outer  = 0.07

    half_inner = bar_h / 2 + gap_inner / 2
    half_outer = bar_h / 2 + gap_outer / 2

    _BAR_SPECS = [
        ("gpt5",   PROPERTY_TASKS, _MODEL_COLORS["gpt5"],   "",    half_outer + gap_inner + bar_h),
        ("qwen27b",PROPERTY_TASKS, _MODEL_COLORS["qwen27b"],"",    half_outer),
        ("gpt5",   PROC_TASKS,     _MODEL_COLORS["gpt5"],   "///", -half_outer),
        ("qwen27b",PROC_TASKS,     _MODEL_COLORS["qwen27b"],"///",-half_outer - gap_inner - bar_h),
    ]

    y_centers = {m: i for i, m in enumerate(methods_sorted)}
    n_methods = len(methods_sorted)

    _apply_nature_style()
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    fig.subplots_adjust(left=0.24, right=0.97, top=0.95, bottom=0.17)

    family_of = {id(PROPERTY_TASKS): "Attribute", id(PROC_TASKS): "Procedural"}

    for model, task_list, color, hatch, y_offset in _BAR_SPECS:
        family = family_of[id(task_list)]
        sub = merged[
            (merged["task"].isin(task_list)) & (merged["base_model"] == model)
        ]
        for method in methods_sorted:
            m_sub = sub[sub["method"] == method]
            if m_sub.empty:
                continue
            mean_d = m_sub["delta"].mean()
            ypos   = y_centers[method] + y_offset
            ax.barh(
                ypos, mean_d,
                height=bar_h,
                color=color, alpha=0.85,
                hatch=hatch, edgecolor="white", linewidth=0.4,
                zorder=3,
            )
            stars = _sig_stars(pvals.get((method, model, family), 1.0))
            if stars:
                x_txt = mean_d + (0.004 if mean_d >= 0 else -0.004)
                ha    = "left" if mean_d >= 0 else "right"
                ax.text(x_txt, ypos, stars, va="center", ha=ha,
                        fontsize=5.5, color="#333333", zorder=5)

    ax.axvline(0, color="#555555", linewidth=0.8, zorder=4)
    ax.set_yticks(list(y_centers.values()))
    ax.set_yticklabels([_METHOD_LABELS[m] for m in methods_sorted])
    ax.set_xlabel("Δ ECSR", labelpad=3)
    ax.set_ylim(-0.5, n_methods - 0.5)
    _despine(ax)

    gpt_c  = _MODEL_COLORS["gpt5"]
    qwen_c = _MODEL_COLORS["qwen27b"]
    legend_handles = [
        mpatches.Patch(facecolor=gpt_c,  edgecolor="none", alpha=0.85, label="GPT"),
        mpatches.Patch(facecolor=qwen_c, edgecolor="none", alpha=0.85, label="Qwen"),
        mpatches.Patch(facecolor="#888888", edgecolor="none", alpha=0.85, label="Attribute"),
        mpatches.Patch(facecolor="#888888", edgecolor="white", hatch="///", alpha=0.85,
                       label="Procedural"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=False,
        handlelength=1.0, handletextpad=0.4,
        labelspacing=0.25, fontsize=5.0,
    )

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def save_figure12_csv(
    phase2_gpt_path: str | Path,
    phase2_qwen_path: str | Path,
    output_path: Path,
    results_dir: str | Path = "results",
) -> None:
    """CSV for fig12: per-method × model × family Δ ECSR with ×4 Bonferroni stars."""
    merged = _load_phase2_merged(phase2_gpt_path, phase2_qwen_path)
    results_dir = Path(results_dir)
    pvals = _compute_delta_pvals(
        results_dir,
        methods=list(_METHOD_MARKERS.keys()),
        models=list(_MODEL_COLORS.keys()),
    )
    rows = []
    for method in _METHOD_MARKERS:
        for model in _MODEL_COLORS:
            for family, task_list in [("Attribute", PROPERTY_TASKS), ("Procedural", PROC_TASKS)]:
                sub = merged[merged["task"].isin(task_list) & (merged["base_model"] == model)]
                m_sub = sub[sub["method"] == method]
                mean_d = float(m_sub["delta"].mean()) if not m_sub.empty else float("nan")
                p_adj  = pvals.get((method, model, family), float("nan"))
                rows.append({
                    "method":  _METHOD_LABELS.get(method, method),
                    "model":   "GPT" if model == "gpt5" else "Qwen",
                    "family":  family,
                    "mean_delta": round(mean_d, 4),
                    "p_adj":   round(p_adj, 4) if not (isinstance(p_adj, float) and p_adj != p_adj) else float("nan"),
                    "stars":   _sig_stars(p_adj),
                })
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    results_dir: str,
    models: list[str],
    include_humans: bool,
    output_dir: str,
    phase2_gpt_csv: str | None = None,
    phase2_qwen_csv: str | None = None,
) -> None:
    results_dir = Path(results_dir)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _nonce_only = {"gpt5_nonce", "qwen_nonce"}
    plot_models = [m for m in models if m not in _nonce_only]

    # Always include nonce variants for fig9 if their result dirs exist
    all_models = list(models)
    for nonce_key in ("gpt5_nonce", "qwen_nonce"):
        if nonce_key not in all_models and (results_dir / nonce_key).exists():
            all_models.append(nonce_key)

    print(f"Loading data — models: {all_models}  humans: {include_humans}")
    data = load_all_data(results_dir, all_models, include_humans=include_humans)
    print(f"  Episodes: {len(data['episodes'])} rows  |  QA: {len(data['qa'])} rows")

    print("Generating figures …")

    # fig1b_normeff — success rate + normalized efficiency strips
    make_figure1b_normeff(
        data, plot_models,
        output_dir / "fig1b_normeff.pdf",
        include_humans=include_humans,
    )

    # fig4_combined_ecsr — all models
    make_figure4_combined_ecsr(
        data, plot_models, ALL_TASKS,
        output_dir / "fig4_combined_ecsr.pdf",
    )

    # fig14 — single-panel: models + humans co-plotted, separate regression lines
    make_figure14(
        data, plot_models, ALL_TASKS,
        output_dir / "fig14_combined_scatter.pdf",
    )

    # fig15 — radar charts: ECSR (left) and RV (right) across all task types
    make_figure15(
        data, plot_models, ALL_TASKS,
        output_dir / "fig15_radar.pdf",
    )

    # fig16 — horizontal bar chart: Attribute vs Procedural avg per model
    make_figure16(
        data, plot_models,
        output_dir / "fig16_hbar.pdf",
    )

    # fig8_gap_ecsr — Qwen + GPT only
    fig8_models = [m for m in plot_models if m in {"qwen27b", "gpt5"}]
    if fig8_models:
        make_figure8(data, fig8_models, output_dir / "fig8_gap_ecsr.pdf")
    else:
        print("  Skipped fig8_gap_ecsr: neither qwen27b nor gpt5 in model list")

    # fig9_nonce_comparison — nonce pairs only
    nonce_pairs = []
    if "gpt5" in all_models and "gpt5_nonce" in all_models:
        nonce_pairs.append(("gpt5", "gpt5_nonce"))
    if "qwen27b" in all_models and "qwen_nonce" in all_models:
        nonce_pairs.append(("qwen27b", "qwen_nonce"))
    if nonce_pairs:
        make_figure9(data, output_dir / "fig9_nonce_comparison.pdf", model_pairs=nonce_pairs)
    else:
        print("  Skipped fig9_nonce_comparison: no nonce pairs found in model list")

    # fig13 — combined 1×4: gap panels (A/B) + nonce panels (C/D) on one row
    if fig8_models and nonce_pairs:
        make_figure13(
            data, fig8_models,
            output_dir / "fig13_gap_nonce_combined.pdf",
            model_pairs_9=nonce_pairs,
        )
    else:
        print("  Skipped fig13: requires qwen27b/gpt5 models and nonce pairs")

    # fig13a — aggregated gap panel; fig13b — aggregated nonce panel (both single-panel, fig14 size)
    if fig8_models and nonce_pairs:
        make_figure13a(data, fig8_models, output_dir / "fig13a_gap.pdf")
        make_figure13b(
            data, fig8_models,
            output_dir / "fig13b_nonce.pdf",
            model_pairs_9=nonce_pairs,
        )
    else:
        print("  Skipped fig13a/13b: requires qwen27b/gpt5 models and nonce pairs")

    # fig10b_coverage_sweep — GPT + GPT-OSS only
    cov_entries = []
    for m in ("gpt5", "gptoss"):
        if m not in COVERAGE_MODEL_MAP:
            continue
        lbl, subdir, stem, main_fstem = COVERAGE_MODEL_MAP[m]
        cov_dir = results_dir / subdir
        if cov_dir.exists():
            cov_entries.append((lbl, cov_dir, stem, main_fstem))
    if cov_entries:
        make_figure10b(
            output_dir / "fig10b_coverage_sweep.pdf",
            coverage_entries=cov_entries,
            main_results_dir=None,
        )
    else:
        print("  Skipped fig10b_coverage_sweep: no coverage dirs found")

    # fig11 + fig12 — phase 2 teaching methods
    gpt_csv  = Path(phase2_gpt_csv)  if phase2_gpt_csv  else None
    qwen_csv = Path(phase2_qwen_csv) if phase2_qwen_csv else None
    if gpt_csv and qwen_csv and gpt_csv.exists() and qwen_csv.exists():
        make_figure11(gpt_csv, qwen_csv, output_dir / "fig11_phase2_methods.pdf")
        make_figure12(gpt_csv, qwen_csv, output_dir / "fig12_phase2_delta.pdf",
                      results_dir=results_dir)
    else:
        print("  Skipped fig11/fig12: provide --phase2_gpt_csv and --phase2_qwen_csv")

    # metrics.csv — all plot_models
    save_metrics_csv(data, plot_models, ALL_TASKS, output_dir / "metrics.csv")

    print_family_correlations(data, plot_models)
    print(f"\nDone. Figures saved to {output_dir}/")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate key figures and metrics CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results_dir",    default="results")
    p.add_argument("--models",         default="qwen27b,gpt5,gptoss,gemini3.1,llama,olmo",
                   help="Comma-separated model keys (nonce variants auto-added for fig9)")
    p.add_argument("--include_humans", action="store_true")
    p.add_argument("--output_dir",     default="analysis/figures_main")
    p.add_argument("--phase2_gpt_csv",  default=None,
                   help="Path to phase 2 GPT metrics CSV (for fig11/fig12)")
    p.add_argument("--phase2_qwen_csv", default=None,
                   help="Path to phase 2 Qwen metrics CSV (for fig11/fig12)")
    args = p.parse_args()

    run(
        results_dir    = args.results_dir,
        models         = args.models.split(","),
        include_humans = args.include_humans,
        output_dir     = args.output_dir,
        phase2_gpt_csv = args.phase2_gpt_csv,
        phase2_qwen_csv= args.phase2_qwen_csv,
    )


if __name__ == "__main__":
    main()
