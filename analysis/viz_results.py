"""
Publication-quality (Nature-style) visualizations for adventure-story generalization results.

Six figures:
  fig1_success_efficiency.pdf     — success-rate bars + per-episode efficiency scatter (same axis)
  fig2_mean_efficiency.pdf        — mean efficiency dot plot per model × task
  fig3_rule_score.pdf             — LLM-judge rule induction score bars
  fig4_efficiency_vs_rule.pdf     — scatter: efficiency vs rule score  (one point per model × task)
  fig5_instance_vs_rule.pdf       — scatter: QA instance acc vs rule score
  fig6_efficiency_vs_instance.pdf — scatter: efficiency vs QA instance acc

Efficiency convention (higher = better):
    success episode:  eff = reference_length / num_runs   ∈ (0, 1]
    failure episode:  eff = 0

Usage (from repo root):
    python analysis/viz_results.py \
        --results_dir results \
        --models qwen27b,gemini,gptoss \
        --include_humans \
        --output_dir analysis/figures
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from scipy import stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Nature-style rcParams
# ---------------------------------------------------------------------------
NATURE_FULL_W = 7.08
NATURE_HALF_W = 3.46

_NATURE_RC = {
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":            7,
    "axes.titlesize":       8,
    "axes.labelsize":       8,
    "xtick.labelsize":      7,
    "ytick.labelsize":      7,
    "legend.fontsize":      7,
    "legend.title_fontsize": 7,
    "axes.linewidth":       0.6,
    "xtick.major.width":    0.6,
    "ytick.major.width":    0.6,
    "xtick.major.size":     2.5,
    "ytick.major.size":     2.5,
    "lines.linewidth":      1.0,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            True,
    "grid.color":           "#E5E5E5",
    "grid.linewidth":       0.4,
    "grid.alpha":           0.8,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.02,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
}

# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------
PROPERTY_TASKS = ["additive", "compositional", "conditional", "override"]
PROC_TASKS     = ["proc_add", "proc_comp", "proc_cond", "proc_over"]
ALL_TASKS      = PROPERTY_TASKS + PROC_TASKS

TASK_DISPLAY = {
    "additive":      "A-Add",
    "compositional": "A-Comp",
    "conditional":   "A-Cond",
    "override":      "A-Over",
    "proc_add":      "P-Add",
    "proc_comp":     "P-Comp",
    "proc_cond":     "P-Cond",
    "proc_over":     "P-Over",
}

# Short form used in fig4_combined where A-/P- encoding is in the legend separately
_TASK_DISPLAY_SHORT = {
    "additive":      "Add",
    "compositional": "Comp",
    "conditional":   "Cond",
    "override":      "Over",
}

# max_tries per task (used for efficiency floor reference line = 1/max_tries)
TASK_MAX_TRIES = {
    # Property tasks: shortcut-adjusted floor = 4/(M+3) where M=num_items,
    # so effective max_tries = (M+3)/4. Model can chain buys without re-navigating.
    "additive":      5,   # M=5: (5+3)/4=2
    "compositional": 9,   # M=9: (9+3)/4=3
    "conditional":   6,   # M=6: (6+3)/4≈2
    "override":      4,   # M=4: (4+3)/4≈2
    # Proc tasks: no shortcut; full re-execution required each attempt.
    "proc_add":      3,
    "proc_comp":     4,
    "proc_cond":     3,
    "proc_over":     4,
}

TASK_BASE_TYPE = {
    "additive":      "additive",
    "compositional": "compositional",
    "conditional":   "conditional",
    "override":      "override",
    "proc_add":      "additive",
    "proc_comp":     "compositional",
    "proc_cond":     "conditional",
    "proc_over":     "override",
}

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------
MODEL_PALETTE = [
    "#E87722",   # burnt orange
    "#4878CF",   # steel blue
    "#6BAF92",   # sage green
    "#C44D58",   # crimson
    "#8E6DC3",   # purple
    "#3FA0B0",   # teal
    "#888888",   # grey — human
]

TASK_TYPE_COLOR = {
    "additive":      "#4878CF",
    "compositional": "#6BAF92",
    "conditional":   "#E87722",
    "override":      "#C44D58",
}

MODEL_DISPLAY = {
    "qwen27b":    "Qwen",
    "qwen_nonce": "Qwen (nonce)",
    "gemini":     "Gemini 2.5",
    "gemini3.1":  "Gemini",
    "olmo":       "Olmo",
    "gptoss":     "GPT-OSS",
    "gpt5":       "GPT",
    "gpt5_nonce": "GPT (nonce)",
    "llama":      "LLaMA",
    "human":      "Human",
}

# Maps model key → filename stem when the two differ
MODEL_FILE_STEM = {
    "llama":      "llama4",
    "gpt5_nonce": "nonce_gpt5",
    "qwen_nonce": "nonce_qwen27b",
}

# model key → (display_label, coverage_subdir, file_stem)
COVERAGE_MODEL_MAP = {
    # model_key → (display_label, coverage_subdir, coverage_file_stem, main_file_stem)
    "gpt5":    ("GPT",     "gpt5_coverage",   "gpt-5.4-mini",         "gpt5"),
    "gptoss":  ("GPT-OSS", "gptoss_coverage",  "gpt-oss-120b",         "gptoss"),
    "qwen27b": ("Qwen",    "qwen_coverage",    "b7ca741b86de18df552f", "qwen27b"),
}

# ---------------------------------------------------------------------------
# Efficiency conversion
# ---------------------------------------------------------------------------

def _new_efficiency(ep: dict) -> float | None:
    """ref_length / num_runs for successful episodes; None for failures.
    Always conditioned on success — failures are captured by success_rate."""
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_model_episodes(results_dir: Path, model: str) -> pd.DataFrame:
    rows = []
    model_dir = results_dir / model
    fstem = MODEL_FILE_STEM.get(model, model)
    for task in ALL_TASKS:
        fpath = model_dir / f"phase_1_{task}_{fstem}.json"
        if not fpath.exists():
            continue
        data = json.loads(fpath.read_text())
        for var in data["variants"]:
            for ep in var["episodes"].values():
                rows.append(dict(
                    model=model,
                    task=task,
                    variant_seed=var.get("variant_seed"),
                    success=bool(ep.get("success", False)),
                    efficiency=_new_efficiency(ep),   # None for failures
                    num_runs=ep.get("num_runs"),
                    reference_length=ep.get("reference_length"),
                ))
    return pd.DataFrame(rows)


def _load_human_episodes(results_dir: Path) -> pd.DataFrame:
    rows = []
    human_dir = results_dir / "humans"
    if not human_dir.exists():
        return pd.DataFrame()
    for fpath in human_dir.glob("*.json"):
        data = json.loads(fpath.read_text())
        participant = data.get("participant_id", fpath.stem)
        for ep in data.get("episode_results", []):
            task = ep.get("task_type")
            if task not in ALL_TASKS:
                continue
            success  = bool(ep.get("success", False))
            num_runs = ep.get("num_runs")
            ref_len  = ep.get("reference_length")
            max_tries = TASK_MAX_TRIES.get(task, 2)
            over_budget = (num_runs is not None and ref_len is not None
                           and num_runs > max_tries * ref_len)
            if success and not over_budget and num_runs and ref_len:
                eff = ref_len / num_runs
            else:
                eff = None
            success = success and not over_budget
            rows.append(dict(
                model="human",
                task=task,
                variant_seed=None,
                participant=participant,
                success=success,
                efficiency=eff,
                num_runs=num_runs,
                reference_length=ref_len,
            ))
    return pd.DataFrame(rows)


def _load_model_qa(results_dir: Path, model: str) -> pd.DataFrame:
    rows = []
    model_dir = results_dir / model
    fstem = MODEL_FILE_STEM.get(model, model)
    for task in ALL_TASKS:
        fpath = model_dir / f"qa_phase1_{task}_{fstem}.json"
        if not fpath.exists():
            continue
        data = json.loads(fpath.read_text())
        for var in data["variants"]:
            inst  = var.get("instance", {})
            total = inst.get("total", 0)
            inst_acc = inst.get("correct", 0) / total if total else float("nan")

            sexp = var.get("structure_exp", {})
            rule_s = (sexp.get("rule_score", 0) or 0) / 2.0
            in_s   = (sexp.get("input_score", 0) or 0) / 2.0
            out_s  = (sexp.get("output_score", 0) or 0) / 2.0
            gen_s  = (sexp.get("generalization_score", 0) or 0) / 2.0
            overall = sexp.get("overall")

            rows.append(dict(
                model=model,
                task=task,
                variant_seed=var.get("variant_seed"),
                instance_accuracy=inst_acc,
                rule_score=rule_s,
                input_score=in_s,
                output_score=out_s,
                gen_score=gen_s,
                exp_overall=overall,
            ))
    return pd.DataFrame(rows)


def load_all_data(
    results_dir: str | Path,
    models: list[str],
    include_humans: bool = True,
) -> dict[str, pd.DataFrame]:
    results_dir = Path(results_dir)
    ep_dfs, qa_dfs = [], []

    for m in models:
        ep_dfs.append(_load_model_episodes(results_dir, m))
        qa_dfs.append(_load_model_qa(results_dir, m))

    if include_humans:
        ep_dfs.append(_load_human_episodes(results_dir))

    human_rule_path = results_dir / "human_explanations" / "human_rule_scores.json"
    human_rule = {}
    if human_rule_path.exists():
        human_rule = json.loads(human_rule_path.read_text()).get("scores", {})

    return {
        "episodes":   pd.concat(ep_dfs, ignore_index=True) if ep_dfs else pd.DataFrame(),
        "qa":         pd.concat(qa_dfs, ignore_index=True) if qa_dfs else pd.DataFrame(),
        "human_rule": human_rule,
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _apply_nature_style() -> None:
    mpl.rcParams.update(_NATURE_RC)


def _despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_label(ax: plt.Axes, label: str, x: float = -0.18, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")


def _task_ticks(ax: plt.Axes, tasks: list[str], rotation: int = 0) -> None:
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASK_DISPLAY[t] for t in tasks], rotation=rotation, ha="center")


def _model_color(model: str, all_models: list[str]) -> str:
    if model == "human":
        return MODEL_PALETTE[-1]
    idx = all_models.index(model) if model in all_models else 0
    return MODEL_PALETTE[idx % (len(MODEL_PALETTE) - 1)]


def _model_label(model: str) -> str:
    return MODEL_DISPLAY.get(model, model)


# ---------------------------------------------------------------------------
# Shared: bar + scatter panel (one task group)
# ---------------------------------------------------------------------------

def _draw_bar_scatter_panel(
    ax: plt.Axes,
    edf: pd.DataFrame,
    tasks: list[str],
    all_models: list[str],
    llm_models: list[str],
    title: str,
) -> None:
    """Success-rate bar + efficiency-on-success scatter strip, same y-axis."""
    n_models = len(all_models)
    unit_w   = 0.80 / n_models
    half_grp = n_models * unit_w / 2        # half-width of the bar group
    rng      = np.random.default_rng(42)

    # Collect mean-efficiency positions per task to draw connecting lines later
    mean_eff_xs = {t: [] for t in tasks}
    mean_eff_ys = {t: [] for t in tasks}

    for t_idx, task in enumerate(tasks):
        t_sub = edf[edf["task"] == task]

        for m_idx, model in enumerate(all_models):
            m_sub = t_sub[t_sub["model"] == model]
            if m_sub.empty:
                continue

            color  = _model_color(model, llm_models)
            offset = (m_idx - (n_models - 1) / 2) * unit_w

            # success rate bar (centered)
            bar_x = t_idx + offset
            sr    = m_sub["success"].mean()
            bar_w = unit_w * 0.72
            ax.bar(bar_x, sr, width=bar_w,
                   color=color, alpha=0.85, zorder=3)

            # efficiency scatter overlaid on bar
            effs = m_sub["efficiency"].dropna().values
            if len(effs) == 0:
                continue
            jitter = rng.uniform(-bar_w * 0.28, bar_w * 0.28, size=len(effs))
            ax.scatter(bar_x + jitter, effs,
                       color="white", s=4, alpha=0.60, linewidths=0, zorder=4)
            ax.scatter([bar_x], [effs.mean()],
                       color=color, s=12, marker="D",
                       linewidths=0.5, edgecolors="white", zorder=5)
            mean_eff_xs[task].append(bar_x)
            mean_eff_ys[task].append(effs.mean())

        # per-task floor line — spans only this task's bar group, label inside
        floor    = 1.0 / TASK_MAX_TRIES.get(task, 2)
        x_left   = t_idx - half_grp + 0.02
        x_right  = t_idx + half_grp - 0.02
        ax.plot([x_left, x_right], [floor, floor],
                color="#666666", linewidth=0.8, linestyle=":", zorder=6)
        ax.text(x_right, floor + 0.013, f"1/{TASK_MAX_TRIES.get(task,'?')}",
                va="bottom", ha="right", fontsize=5.5, color="#666666", zorder=6)

    # Connect mean-efficiency diamonds across models within each task
    for task in tasks:
        xs, ys = mean_eff_xs[task], mean_eff_ys[task]
        if len(xs) < 2:
            continue
        ax.plot(xs, ys, color="#888888", linewidth=0.7, alpha=0.5,
                linestyle="-", zorder=4, marker=None)

    ax.axhline(1.0, color="#888888", linewidth=0.7, linestyle="--", zorder=2)
    ax.set_xlim(-0.5, len(tasks) - 0.5)
    ax.set_ylim(-0.02, 1.12)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.set_ylabel("Success rate  /  Efficiency (on success)")
    ax.set_title(title, pad=4)
    _task_ticks(ax, tasks)
    _despine(ax)


# ---------------------------------------------------------------------------
# Figure 1 — Success rate bars + per-episode efficiency scatter (same axis)
# ---------------------------------------------------------------------------

def make_figure1(
    data: dict[str, pd.DataFrame],
    models: list[str],
    output_path: Path,
    include_humans: bool = True,
) -> None:
    """Two panels (property | proc). Bar = success rate; scatter = efficiency on success."""
    _apply_nature_style()

    all_models = list(models) + (["human"] if include_humans else [])
    edf = data["episodes"].copy()

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(NATURE_FULL_W, NATURE_FULL_W * 0.46))
    fig.subplots_adjust(left=0.09, right=0.76, top=0.88, bottom=0.14, wspace=0.32)

    _draw_bar_scatter_panel(ax_l, edf, PROPERTY_TASKS, all_models, models, "Attribute induction")
    _draw_bar_scatter_panel(ax_r, edf, PROC_TASKS,     all_models, models, "Procedural induction")

    for ax in (ax_l, ax_r):
        ax.set_ylim(-0.02, 1.12)

    model_handles = [
        mpatches.Patch(color=_model_color(m, models), label=_model_label(m))
        for m in all_models
    ]
    encoding_handles = [
        mpatches.Patch(color="#555555", alpha=0.85, label="Bar = success rate"),
        mpl.lines.Line2D([], [], marker="o", color="w", markerfacecolor="white",
                         markeredgecolor="#555555", markersize=5, linewidth=0,
                         label="Dot = efficiency (on success)"),
        mpl.lines.Line2D([], [], marker="D", color="w", markerfacecolor="#555555",
                         markeredgecolor="white", markersize=5, linewidth=0,
                         label="Mean efficiency"),
    ]
    fig.legend(handles=model_handles + encoding_handles,
               loc="center left", bbox_to_anchor=(0.78, 0.50),
               frameon=False, handlelength=0.9, handleheight=0.8,
               borderpad=0, labelspacing=0.5)

    _panel_label(ax_l, "A")
    _panel_label(ax_r, "B")

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 1b — Success rate (top row) + efficiency strip (bottom row)
# ---------------------------------------------------------------------------

def _draw_sr_only_panel(
    ax: plt.Axes,
    edf: pd.DataFrame,
    tasks: list[str],
    all_models: list[str],
    llm_models: list[str],
    title: str,
) -> None:
    """Success rate bars only — no efficiency overlay."""
    n_models = len(all_models)
    unit_w   = 0.80 / n_models

    for t_idx, task in enumerate(tasks):
        t_sub = edf[edf["task"] == task]
        for m_idx, model in enumerate(all_models):
            m_sub = t_sub[t_sub["model"] == model]
            if m_sub.empty:
                continue
            color  = _model_color(model, llm_models)
            offset = (m_idx - (n_models - 1) / 2) * unit_w
            bar_x  = t_idx + offset
            ax.bar(bar_x, m_sub["success"].mean(), width=unit_w * 0.72,
                   color=color, alpha=0.85, zorder=3)

    ax.axhline(1.0, color="#888888", linewidth=0.7, linestyle="--", zorder=2)
    ax.set_xlim(-0.5, len(tasks) - 0.5)
    ax.set_ylim(-0.02, 1.12)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.set_ylabel("Success rate")
    ax.set_title(title, pad=4)
    _task_ticks(ax, tasks)
    _despine(ax)


def _draw_eff_strip_panel(
    ax: plt.Axes,
    edf: pd.DataFrame,
    tasks: list[str],
    all_models: list[str],
    llm_models: list[str],
    use_normalized: bool = False,
) -> None:
    """Efficiency strip: individual dots + mean diamond + within-task connecting line.
    use_normalized: map raw eff through (eff - floor)/(1 - floor) → [0,1]."""
    n_models = len(all_models)
    unit_w   = 0.80 / n_models
    half_grp = n_models * unit_w / 2
    rng      = np.random.default_rng(42)

    mean_xs = {t: [] for t in tasks}
    mean_ys = {t: [] for t in tasks}

    for t_idx, task in enumerate(tasks):
        t_sub = edf[edf["task"] == task]
        floor = 1.0 / TASK_MAX_TRIES.get(task, 2)
        for m_idx, model in enumerate(all_models):
            m_sub = t_sub[t_sub["model"] == model]
            if m_sub.empty:
                continue
            color  = _model_color(model, llm_models)
            offset = (m_idx - (n_models - 1) / 2) * unit_w
            bar_x  = t_idx + offset
            bar_w  = unit_w * 0.72

            effs = m_sub["efficiency"].dropna().values
            if len(effs) == 0:
                continue
            if use_normalized:
                effs = np.clip((effs - floor) / (1.0 - floor), 0.0, 1.0)
            jitter = rng.uniform(-bar_w * 0.28, bar_w * 0.28, size=len(effs))
            ax.scatter(bar_x + jitter, effs,
                       color=color, s=4, alpha=0.35, linewidths=0, zorder=3)
            ax.scatter([bar_x], [effs.mean()],
                       color=color, s=12, marker="D",
                       linewidths=0.5, edgecolors="white", zorder=5)
            mean_xs[task].append(bar_x)
            mean_ys[task].append(effs.mean())

        if not use_normalized:
            # floor line within task group
            x_left  = t_idx - half_grp + 0.02
            x_right = t_idx + half_grp - 0.02
            ax.plot([x_left, x_right], [floor, floor],
                    color="#666666", linewidth=0.8, linestyle=":", zorder=6)
            ax.text(x_right, floor + 0.013, f"1/{TASK_MAX_TRIES.get(task,'?')}",
                    va="bottom", ha="right", fontsize=5.5, color="#666666", zorder=6)

    # Connect mean diamonds across models within each task
    for task in tasks:
        if len(mean_xs[task]) < 2:
            continue
        ax.plot(mean_xs[task], mean_ys[task],
                color="#888888", linewidth=0.7, alpha=0.5, linestyle="-", zorder=4)

    ax.axhline(1.0, color="#888888", linewidth=0.7, linestyle="--", zorder=2)
    ax.set_xlim(-0.5, len(tasks) - 0.5)
    ax.set_ylim(-0.02, 1.12)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ylabel = "Normalized efficiency (on success)" if use_normalized else "Efficiency (on success)"
    ax.set_ylabel(ylabel)
    _task_ticks(ax, tasks)
    _despine(ax)


def make_figure1b(
    data: dict[str, pd.DataFrame],
    models: list[str],
    output_path: Path,
    include_humans: bool = True,
) -> None:
    """2×2 layout: top = success rate bars, bottom = efficiency strip."""
    _apply_nature_style()

    all_models = list(models) + (["human"] if include_humans else [])
    edf = data["episodes"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(NATURE_FULL_W, NATURE_FULL_W * 0.80),
                             sharey="row")
    fig.subplots_adjust(left=0.09, right=0.76, top=0.93, bottom=0.10,
                        wspace=0.18, hspace=0.32)

    # Top row — success rate
    _draw_sr_only_panel(axes[0, 0], edf, PROPERTY_TASKS, all_models, models,
                        "Attribute induction")
    _draw_sr_only_panel(axes[0, 1], edf, PROC_TASKS, all_models, models,
                        "Procedural induction")

    # Bottom row — efficiency strip
    _draw_eff_strip_panel(axes[1, 0], edf, PROPERTY_TASKS, all_models, models)
    _draw_eff_strip_panel(axes[1, 1], edf, PROC_TASKS,     all_models, models)

    # Remove redundant y-labels on right column
    for row in range(2):
        axes[row, 1].set_ylabel("")

    model_handles = [
        mpatches.Patch(color=_model_color(m, models), label=_model_label(m))
        for m in all_models
    ]
    encoding_handles = [
        mpatches.Patch(color="#555555", alpha=0.85, label="Bar = success rate"),
        mpl.lines.Line2D([], [], marker="o", color="w", markerfacecolor="#555555",
                         markersize=5, linewidth=0, label="Dot = efficiency"),
        mpl.lines.Line2D([], [], marker="D", color="w", markerfacecolor="#555555",
                         markeredgecolor="white", markersize=5, linewidth=0,
                         label="Mean efficiency"),
    ]
    fig.legend(handles=model_handles + encoding_handles,
               loc="center left", bbox_to_anchor=(0.78, 0.50),
               frameon=False, handlelength=0.9, handleheight=0.8,
               borderpad=0, labelspacing=0.5)

    _panel_label(axes[0, 0], "A")
    _panel_label(axes[0, 1], "B")
    _panel_label(axes[1, 0], "C")
    _panel_label(axes[1, 1], "D")

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_figure1b_normeff(
    data: dict[str, pd.DataFrame],
    models: list[str],
    output_path: Path,
    include_humans: bool = True,
) -> None:
    """Same as fig1b but bottom row uses normalized efficiency → floor disappears."""
    _apply_nature_style()

    all_models = list(models) + (["human"] if include_humans else [])
    edf = data["episodes"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(NATURE_FULL_W, NATURE_FULL_W * 0.80),
                             sharey="row")
    fig.subplots_adjust(left=0.09, right=0.76, top=0.93, bottom=0.10,
                        wspace=0.18, hspace=0.32)

    _draw_sr_only_panel(axes[0, 0], edf, PROPERTY_TASKS, all_models, models,
                        "Attribute induction")
    _draw_sr_only_panel(axes[0, 1], edf, PROC_TASKS, all_models, models,
                        "Procedural induction")
    _draw_eff_strip_panel(axes[1, 0], edf, PROPERTY_TASKS, all_models, models,
                          use_normalized=True)
    _draw_eff_strip_panel(axes[1, 1], edf, PROC_TASKS,     all_models, models,
                          use_normalized=True)

    for row in range(2):
        axes[row, 1].set_ylabel("")

    model_handles = [
        mpatches.Patch(color=_model_color(m, models), label=_model_label(m))
        for m in all_models
    ]
    encoding_handles = [
        mpl.lines.Line2D([], [], marker="o", color="w", markerfacecolor="#555555",
                         markersize=5, linewidth=0, label="Dot = norm. efficiency"),
        mpl.lines.Line2D([], [], marker="D", color="w", markerfacecolor="#555555",
                         markeredgecolor="white", markersize=5, linewidth=0,
                         label="Mean norm. efficiency"),
    ]
    fig.legend(handles=model_handles + encoding_handles,
               loc="center left", bbox_to_anchor=(0.78, 0.50),
               frameon=False, handlelength=0.9, handleheight=0.8,
               borderpad=0, labelspacing=0.5)

    _panel_label(axes[0, 0], "A")
    _panel_label(axes[0, 1], "B")
    _panel_label(axes[1, 0], "C")
    _panel_label(axes[1, 1], "D")

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 2 — Mean efficiency dot plot (one dot per model × task)
# ---------------------------------------------------------------------------

def _draw_mean_eff_panel(
    ax: plt.Axes,
    edf: pd.DataFrame,
    tasks: list[str],
    all_models: list[str],
    llm_models: list[str],
    title: str,
    use_ecsr: bool = False,
) -> None:
    """Expected efficiency (or ECSR) per model × task."""
    n_models = len(all_models)
    offsets  = np.linspace(-0.22, 0.22, n_models) if n_models > 1 else [0.0]

    for m_idx, model in enumerate(all_models):
        color = _model_color(model, llm_models)
        m_sub = edf[edf["model"] == model]
        xs, ys = [], []
        for t_idx, task in enumerate(tasks):
            task_sub = m_sub[m_sub["task"] == task]
            if task_sub.empty:
                continue
            sr          = task_sub["success"].mean()
            eff_vals    = task_sub["efficiency"].dropna().values
            eff_on_succ = eff_vals.mean() if len(eff_vals) > 0 else 0.0
            if use_ecsr:
                floor   = 1.0 / TASK_MAX_TRIES.get(task, 2)
                norm_eff = max((eff_on_succ - floor) / (1.0 - floor), 0.0) if eff_on_succ > 0 else 0.0
                value   = sr * norm_eff
            else:
                value   = sr * eff_on_succ
            xs.append(t_idx + offsets[m_idx])
            ys.append(value)
        if xs:
            ax.plot(xs, ys, color=color, linewidth=0.6, alpha=0.4, zorder=2)
            ax.scatter(xs, ys, color=color, s=20, zorder=4,
                       label=_model_label(model),
                       linewidths=0.4, edgecolors="white")

    ax.axhline(1.0, color="#888888", linewidth=0.8, linestyle="--", zorder=3,
               label="Optimal (SR=1, 1 attempt)")

    ax.set_xlim(-0.5, len(tasks) - 0.5)
    ax.set_ylim(-0.02, 1.12)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ylabel = ("ECSR (SR × normalized efficiency)"
              if use_ecsr else "Expected efficiency (SR × eff on success)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4)
    _task_ticks(ax, tasks)
    _despine(ax)


def make_figure2(
    data: dict[str, pd.DataFrame],
    models: list[str],
    output_path: Path,
    include_humans: bool = True,
) -> None:
    """Two panels (property | proc). Mean efficiency conditioned on success."""
    _apply_nature_style()

    all_models = list(models) + (["human"] if include_humans else [])
    edf = data["episodes"]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(NATURE_FULL_W, NATURE_FULL_W * 0.44))
    fig.subplots_adjust(left=0.10, right=0.76, top=0.88, bottom=0.16, wspace=0.32)

    _draw_mean_eff_panel(ax_l, edf, PROPERTY_TASKS, all_models, models, "Attribute induction")
    _draw_mean_eff_panel(ax_r, edf, PROC_TASKS,     all_models, models, "Procedural induction")

    for ax in (ax_l, ax_r):
        ax.set_ylim(-0.02, 1.12)

    # Deduplicate: take handles/labels from left panel only
    handles, labels = ax_l.get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels,
               frameon=False, loc="center left", bbox_to_anchor=(0.78, 0.50),
               handlelength=1.0, handleheight=0.8, borderpad=0, labelspacing=0.5)

    _panel_label(ax_l, "A")
    _panel_label(ax_r, "B")

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_figure2_ecsr(
    data: dict[str, pd.DataFrame],
    models: list[str],
    output_path: Path,
    include_humans: bool = True,
) -> None:
    """Same layout as fig2 but y-axis = ECSR instead of EE."""
    _apply_nature_style()

    all_models = list(models) + (["human"] if include_humans else [])
    edf = data["episodes"]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(NATURE_FULL_W, NATURE_FULL_W * 0.44))
    fig.subplots_adjust(left=0.10, right=0.76, top=0.88, bottom=0.16, wspace=0.32)

    _draw_mean_eff_panel(ax_l, edf, PROPERTY_TASKS, all_models, models,
                        "Attribute induction", use_ecsr=True)
    _draw_mean_eff_panel(ax_r, edf, PROC_TASKS,     all_models, models,
                        "Procedural induction", use_ecsr=True)

    for ax in (ax_l, ax_r):
        ax.set_ylim(-0.02, 1.12)

    handles, labels = ax_l.get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels,
               frameon=False, loc="center left", bbox_to_anchor=(0.78, 0.50),
               handlelength=1.0, handleheight=0.8, borderpad=0, labelspacing=0.5)

    _panel_label(ax_l, "A")
    _panel_label(ax_r, "B")

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 3 — Rule verbalization score bars
# ---------------------------------------------------------------------------

def make_figure3(
    data: dict[str, pd.DataFrame],
    models: list[str],
    tasks: list[str],
    output_path: Path,
) -> None:
    """Grouped bars: LLM-judge rule_score per task × model."""
    _apply_nature_style()

    qdf = data["qa"][data["qa"]["task"].isin(tasks)]
    n_models = len(models)
    group_w  = 0.72
    bar_w    = group_w / n_models

    fig, ax = plt.subplots(figsize=(NATURE_FULL_W, NATURE_HALF_W * 1.1))
    fig.subplots_adjust(left=0.10, right=0.76, top=0.90, bottom=0.14)

    for m_idx, model in enumerate(models):
        m_sub  = qdf[qdf["model"] == model]
        color  = _model_color(model, models)
        means, xs = [], []
        for t_idx, task in enumerate(tasks):
            vals = m_sub[m_sub["task"] == task]["rule_score"].dropna().values
            if len(vals) == 0:
                continue
            means.append(vals.mean())
            offset = (m_idx - (n_models - 1) / 2) * bar_w
            xs.append(t_idx + offset)

        ax.bar(xs, means, width=bar_w * 0.88,
               color=color, alpha=0.85,
               label=_model_label(model),
               zorder=3)

    ax.axhline(1.0, color="#AAAAAA", linewidth=0.6, linestyle="--", zorder=2)
    ax.set_xlim(-0.5, len(tasks) - 0.5)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.set_ylabel("Rule verbalization score")
    _task_ticks(ax, tasks, rotation=20)
    _despine(ax)
    fig.legend(frameon=False, loc="center left", bbox_to_anchor=(0.78, 0.50),
               handlelength=0.9, handleheight=0.8, borderpad=0, labelspacing=0.5)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 3b — Rule induction sub-scores (input / output / rule)
# ---------------------------------------------------------------------------

SUBSCORE_COLS   = ["input_score",  "output_score",  "rule_score"]
SUBSCORE_LABELS = ["Input score",  "Output score",  "Rule score"]
SUBSCORE_COLORS = ["#4878CF",      "#8E6DC3",        "#6BAF92"]


def _draw_subscore_panel(
    ax: plt.Axes,
    qdf: pd.DataFrame,
    tasks: list[str],
    models: list[str],
    col: str,
    title: str,
    color: str,
) -> None:
    n_models = len(models)
    bar_w    = 0.72 / n_models

    for m_idx, model in enumerate(models):
        m_sub  = qdf[qdf["model"] == model]
        alpha  = 0.90 - m_idx * 0.15   # slight fade for each subsequent model
        means, xs = [], []
        for t_idx, task in enumerate(tasks):
            vals = m_sub[m_sub["task"] == task][col].dropna().values
            if len(vals) == 0:
                continue
            means.append(vals.mean())
            offset = (m_idx - (n_models - 1) / 2) * bar_w
            xs.append(t_idx + offset)

        ax.bar(xs, means, width=bar_w * 0.88,
               color=color, alpha=max(alpha, 0.45),
               label=_model_label(model) if col == SUBSCORE_COLS[0] else None,
               zorder=3)

    ax.axhline(1.0, color="#AAAAAA", linewidth=0.6, linestyle="--", zorder=2)
    ax.set_xlim(-0.5, len(tasks) - 0.5)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.set_title(title, pad=4)
    _task_ticks(ax, tasks, rotation=20)
    _despine(ax)


def make_figure3b(
    data: dict[str, pd.DataFrame],
    models: list[str],
    tasks: list[str],
    output_path: Path,
) -> None:
    """Three panels: input score | output score | rule score, per task × model."""
    _apply_nature_style()

    qdf = data["qa"][data["qa"]["task"].isin(tasks)]

    fig, axes = plt.subplots(1, 3, figsize=(NATURE_FULL_W, NATURE_HALF_W * 1.1),
                             sharey=True)
    fig.subplots_adjust(left=0.08, right=0.76, top=0.90, bottom=0.18, wspace=0.12)

    for ax, col, label, color in zip(axes, SUBSCORE_COLS, SUBSCORE_LABELS, SUBSCORE_COLORS):
        _draw_subscore_panel(ax, qdf, tasks, models, col, label, color)

    axes[0].set_ylabel("Sub-score (0–1)")

    # Model legend: use alpha gradient to show model distinction
    model_handles = []
    for m_idx, model in enumerate(models):
        alpha = max(0.90 - m_idx * 0.15, 0.45)
        # Use a neutral grey patch with matching alpha, labeled by model
        h = mpatches.Patch(facecolor="#777777", alpha=alpha, label=_model_label(model))
        model_handles.append(h)

    # Sub-score legend
    subscore_handles = [
        mpatches.Patch(facecolor=c, alpha=0.85, label=l)
        for c, l in zip(SUBSCORE_COLORS, SUBSCORE_LABELS)
    ]

    fig.legend(handles=subscore_handles + model_handles,
               loc="center left", bbox_to_anchor=(0.78, 0.50),
               frameon=False, handlelength=0.9, handleheight=0.8,
               borderpad=0, labelspacing=0.5)

    _panel_label(axes[0], "A")
    _panel_label(axes[1], "B", x=-0.08)
    _panel_label(axes[2], "C", x=-0.08)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figures 4–6 — Correlation scatter (one point per model × task)
# ---------------------------------------------------------------------------

def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def _make_scatter(
    ax: plt.Axes,
    merged: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    models: list[str],
) -> None:
    """Scatter: color = task type, marker = model, fill = property (solid) vs proc (hollow).
    Adds OLS trend line with 95% CI and ρ / p annotation."""
    markers = ["o", "s", "^", "D", "v", "P"]
    lim = (-0.05, 1.10)

    for m_idx, model in enumerate(models):
        m_sub  = merged[merged["model"] == model]
        marker = markers[m_idx % len(markers)]
        for _, row in m_sub.iterrows():
            color     = TASK_TYPE_COLOR.get(TASK_BASE_TYPE.get(row["task"], ""), "#777777")
            is_proc   = row["task"] in PROC_TASKS
            facecolor = "none" if is_proc else color
            ax.scatter(
                row[x_col], row[y_col],
                facecolors=facecolor, edgecolors=color,
                marker=marker, s=28, linewidths=0.8, zorder=4,
            )

    # Overall OLS trend line + 95% CI band
    xy = merged[[x_col, y_col]].dropna()
    x, y = xy[x_col].values, xy[y_col].values
    n = len(x)

    if n >= 4:
        x_line  = np.linspace(0, 1, 200)
        slope, intercept, r_val, p_pearson, _ = stats.linregress(x, y)
        y_line  = slope * x_line + intercept
        y_hat   = slope * x + intercept
        se_res  = np.sqrt(np.sum((y - y_hat) ** 2) / (n - 2))
        x_mean  = x.mean()
        se_line = se_res * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))
        t_crit  = stats.t.ppf(0.975, df=n - 2)

        ax.plot(x_line, y_line, color="#333333", linewidth=1.2, zorder=4)
        ax.fill_between(x_line,
                        y_line - t_crit * se_line,
                        y_line + t_crit * se_line,
                        color="#333333", alpha=0.10, zorder=2)

        rho_val, p_spearman = stats.spearmanr(x, y)
        annot  = (f"$\\rho$ = {rho_val:.2f},  {_fmt_p(p_spearman)}"
                  f"  (n = {n})")
        ax.text(0.04, 0.96, annot,
                transform=ax.transAxes,
                fontsize=6, va="top", ha="left", color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="none", alpha=0.7))

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    _despine(ax)


def _scatter_legend(fig: plt.Figure, models: list[str], x: float, y: float) -> None:
    """Legend: task-type colors, model markers, fill = property/proc."""
    markers = ["o", "s", "^", "D", "v", "P"]
    task_handles = [
        mpl.lines.Line2D([], [], marker="o", color="w",
                         markerfacecolor=TASK_TYPE_COLOR[bt],
                         markersize=6, linewidth=0, label=TASK_DISPLAY[bt])
        for bt in ["additive", "compositional", "conditional", "override"]
    ]
    model_handles = [
        mpl.lines.Line2D([], [], marker=markers[i % len(markers)],
                         markerfacecolor="#777777", markeredgecolor="#777777",
                         markersize=6, linewidth=0, label=_model_label(m))
        for i, m in enumerate(models)
    ]
    trend_handles = [
        mpl.lines.Line2D([], [], color="#333333", linewidth=1.2, label="Overall trend"),
    ]
    fill_handles = [
        mpl.lines.Line2D([], [], marker="o", markerfacecolor="#777777",
                         markeredgecolor="#777777", markersize=6,
                         linewidth=0, label="Attribute (filled)"),
        mpl.lines.Line2D([], [], marker="o", markerfacecolor="none",
                         markeredgecolor="#777777", markeredgewidth=1.0,
                         markersize=6, linewidth=0, label="Procedural (hollow)"),
    ]
    fig.legend(
        handles=task_handles + fill_handles + model_handles + trend_handles,
        loc="center left",
        bbox_to_anchor=(x, y),
        ncol=1, frameon=False,
        handlelength=0.7, handletextpad=0.4,
        labelspacing=0.5, borderpad=0,
    )


def _build_merged(data: dict[str, pd.DataFrame], models: list[str], tasks: list[str]) -> pd.DataFrame:
    rows = []
    edf = data["episodes"][data["episodes"]["task"].isin(tasks)]
    for model in models:
        for task in tasks:
            sub = edf[(edf["model"] == model) & (edf["task"] == task)]
            if sub.empty:
                continue
            sr           = sub["success"].mean()
            eff_vals     = sub["efficiency"].dropna().values
            eff_on_succ  = eff_vals.mean() if len(eff_vals) > 0 else 0.0
            floor        = 1.0 / TASK_MAX_TRIES.get(task, 2)
            norm_eff     = max((eff_on_succ - floor) / (1.0 - floor), 0.0) if eff_on_succ > 0 else 0.0
            rows.append(dict(model=model, task=task,
                             success_rate=sr,
                             efficiency_on_success=eff_on_succ,
                             expected_efficiency=sr * eff_on_succ,
                             ecsr=sr * norm_eff,
                             n_episodes=len(sub)))
    ep_summary = pd.DataFrame(rows)

    qa_summary = (
        data["qa"][data["qa"]["task"].isin(tasks)]
        .groupby(["model", "task"])[["instance_accuracy", "rule_score", "input_score", "output_score"]]
        .mean()
        .reset_index()
    )
    return ep_summary.merge(qa_summary, on=["model", "task"], how="left")


def _compute_variant_ci(
    data: dict,
    models: list[str],
    tasks: list[str],
) -> pd.DataFrame:
    """95% CI for ECSR and rule_score across variants (per model × task)."""
    edf = data["episodes"]
    qdf = data["qa"]
    rows = []
    for model in models:
        for task in tasks:
            e_sub = edf[(edf["model"] == model) & (edf["task"] == task)]
            q_sub = qdf[(qdf["model"] == model) & (qdf["task"] == task)]
            if e_sub.empty:
                continue
            floor = 1.0 / TASK_MAX_TRIES.get(task, 2)

            ecsr_per_var = []
            for _, vg in e_sub.groupby("variant_seed"):
                sr = vg["success"].mean()
                effs = vg["efficiency"].dropna().values
                eff_on_succ = effs.mean() if len(effs) > 0 else 0.0
                norm_eff = max((eff_on_succ - floor) / (1.0 - floor), 0.0) if eff_on_succ > 0 else 0.0
                ecsr_per_var.append(sr * norm_eff)

            rv_per_var = (
                [vg["rule_score"].mean() for _, vg in q_sub.groupby("variant_seed")]
                if not q_sub.empty else []
            )

            def _ci95(arr):
                n = len(arr)
                if n < 2:
                    return float("nan"), float("nan")
                a = np.array(arr)
                se = a.std(ddof=1) / np.sqrt(n)
                t = stats.t.ppf(0.975, df=n - 1)
                return se, t * se

            ecsr_se, ecsr_ci95 = _ci95(ecsr_per_var)
            rv_se,   rv_ci95   = _ci95(rv_per_var)

            rows.append(dict(
                model=model, task=task,
                n_variants=len(ecsr_per_var),
                ecsr_se=ecsr_se,       ecsr_ci95=ecsr_ci95,
                rule_score_se=rv_se,   rule_score_ci95=rv_ci95,
            ))
    return pd.DataFrame(rows)


def save_metrics_csv(
    data: dict[str, pd.DataFrame],
    models: list[str],
    tasks: list[str],
    output_path: Path,
) -> None:
    merged = _build_merged(data, models, tasks)

    # Append human aggregate rows (mean across annotators per task)
    human_panel = _build_human_panel_df(data, tasks)
    if human_panel is not None and not human_panel.empty:
        human_ep = _build_merged(data, ["human"], tasks)
        rule_mean = human_panel.groupby("task")["rule_score"].mean()
        human_ep["rule_score"] = human_ep["task"].map(rule_mean)
        merged = pd.concat([merged, human_ep], ignore_index=True)

    # Merge in per-variant CI columns for LLM models
    ci_df = _compute_variant_ci(data, models, tasks)
    if not ci_df.empty:
        merged = merged.merge(ci_df, on=["model", "task"], how="left")

    cols = [
        "model", "task", "n_episodes", "n_variants",
        "success_rate", "efficiency_on_success", "expected_efficiency", "ecsr",
        "ecsr_se", "ecsr_ci95",
        "rule_score", "rule_score_se", "rule_score_ci95",
        "instance_accuracy", "input_score", "output_score",
    ]
    cols = [c for c in cols if c in merged.columns]
    merged = merged[cols].sort_values(["model", "task"]).reset_index(drop=True)
    merged = merged.round(4)
    merged.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")


def print_family_correlations(
    data: dict[str, pd.DataFrame],
    models: list[str],
) -> None:
    """Print Pearson r and Spearman rho between ECSR and RV, split by task family."""
    families = {
        "Attribute induction": PROPERTY_TASKS,
        "Procedural induction": PROC_TASKS,
        "All tasks": ALL_TASKS,
    }
    for family_name, task_list in families.items():
        merged = _build_merged(data, models, task_list)
        sub = merged[["rule_score", "ecsr"]].dropna()
        n = len(sub)
        if n < 3:
            print(f"  {family_name}: n={n} — too few points for correlation")
            continue
        r, p_r = stats.pearsonr(sub["rule_score"], sub["ecsr"])
        rho, p_rho = stats.spearmanr(sub["rule_score"], sub["ecsr"])
        print(
            f"  {family_name} (n={n}): "
            f"Pearson r={r:.3f} (p={p_r:.3f}), "
            f"Spearman rho={rho:.3f} (p={p_rho:.3f})"
        )


def make_figure4(
    data: dict[str, pd.DataFrame],
    models: list[str],
    tasks: list[str],
    output_path: Path,
) -> None:
    """Scatter: mean efficiency (y) vs rule induction score (x)."""
    _apply_nature_style()
    merged = _build_merged(data, models, tasks)

    fig, ax = plt.subplots(figsize=(NATURE_HALF_W * 1.3, NATURE_HALF_W * 1.3))
    fig.subplots_adjust(left=0.15, right=0.72, top=0.90, bottom=0.16)

    _make_scatter(ax, merged, "rule_score", "expected_efficiency",
                  "Rule verbalization score", "Expected efficiency", models)
    _scatter_legend(fig, models, x=0.74, y=0.55)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_figure5(
    data: dict[str, pd.DataFrame],
    models: list[str],
    tasks: list[str],
    output_path: Path,
) -> None:
    """Scatter: QA instance accuracy (y) vs rule induction score (x)."""
    _apply_nature_style()
    merged = _build_merged(data, models, tasks)

    fig, ax = plt.subplots(figsize=(NATURE_HALF_W * 1.3, NATURE_HALF_W * 1.3))
    fig.subplots_adjust(left=0.15, right=0.72, top=0.90, bottom=0.16)

    _make_scatter(ax, merged, "rule_score", "instance_accuracy",
                  "Rule verbalization score", "QA instance accuracy", models)
    _scatter_legend(fig, models, x=0.74, y=0.55)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_figure6(
    data: dict[str, pd.DataFrame],
    models: list[str],
    tasks: list[str],
    output_path: Path,
) -> None:
    """Scatter: mean efficiency (y) vs QA instance accuracy (x)."""
    _apply_nature_style()
    merged = _build_merged(data, models, tasks)

    fig, ax = plt.subplots(figsize=(NATURE_HALF_W * 1.3, NATURE_HALF_W * 1.3))
    fig.subplots_adjust(left=0.15, right=0.72, top=0.90, bottom=0.16)

    _make_scatter(ax, merged, "instance_accuracy", "expected_efficiency",
                  "QA instance accuracy", "Expected efficiency", models)
    _scatter_legend(fig, models, x=0.74, y=0.55)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figures 4b / 5b / 6b — Per-model scatter panels (3 panels, one per model)
# ---------------------------------------------------------------------------

def _draw_model_scatter_panel(
    ax: plt.Axes,
    m_sub: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    model: str,
    models: list[str],
    show_task_labels: bool = True,
) -> None:
    """One scatter panel for a single model's 8 task points."""
    lim = (-0.05, 1.10)

    for _, row in m_sub.iterrows():
        color    = TASK_TYPE_COLOR.get(TASK_BASE_TYPE.get(row["task"], ""), "#777777")
        is_proc  = row["task"] in PROC_TASKS
        facecolor = "none" if is_proc else color
        ax.scatter(row[x_col], row[y_col],
                   facecolors=facecolor, edgecolors=color,
                   marker="*", s=72, linewidths=0.8, zorder=4)
        if show_task_labels:
            ax.text(row[x_col] + 0.02, row[y_col] + 0.02,
                    TASK_DISPLAY[row["task"]],
                    fontsize=4.5, color="#555555", zorder=5)

    xy = m_sub[[x_col, y_col]].dropna()
    x, y = xy[x_col].values, xy[y_col].values
    n = len(x)

    if n >= 3:
        slope, intercept, r_val, p_pearson, _ = stats.linregress(x, y)
        x_line  = np.linspace(0, 1, 200)
        y_line  = slope * x_line + intercept
        y_hat   = slope * x + intercept
        se_res  = np.sqrt(np.sum((y - y_hat) ** 2) / max(n - 2, 1))
        x_mean  = x.mean()
        denom   = np.sum((x - x_mean) ** 2)
        se_line = se_res * np.sqrt(1/n + (x_line - x_mean)**2 / denom) if denom > 0 else np.zeros(200)
        t_crit  = stats.t.ppf(0.975, df=max(n - 2, 1))
        color_m = _model_color(model, models)

        ax.plot(x_line, y_line, color=color_m, linewidth=1.0, zorder=3)
        ax.fill_between(x_line,
                        y_line - t_crit * se_line,
                        y_line + t_crit * se_line,
                        color=color_m, alpha=0.12, zorder=2)

        rho_val, p_spearman = stats.spearmanr(x, y)
        annot  = (f"$\\rho$ = {rho_val:.2f},  {_fmt_p(p_spearman)}"
                  f"  (n = {n})")
        ax.text(0.04, 0.96, annot, transform=ax.transAxes,
                fontsize=5.5, va="top", ha="left", color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="none", alpha=0.7))

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.set_xlabel(x_label, fontsize=7)
    ax.set_ylabel(y_label, fontsize=7)
    ax.set_title(_model_label(model), pad=4)
    _despine(ax)


def _build_human_panel_df(data: dict, tasks: list[str]) -> pd.DataFrame | None:
    """One row per (participant, task): per-annotator ECSR + per-annotator rule score."""
    human_rule = data.get("human_rule", {})
    edf = data["episodes"]
    edf_h = edf[edf["model"] == "human"]
    if edf_h.empty or not human_rule:
        return None
    rows = []
    for participant, p_scores in human_rule.items():
        p_sub = edf_h[edf_h["participant"] == participant]
        for task in tasks:
            if task not in p_scores:
                continue
            t_sub = p_sub[p_sub["task"] == task]
            if t_sub.empty:
                continue
            sr          = t_sub["success"].mean()
            effs        = t_sub["efficiency"].dropna().values
            eff_on_succ = effs.mean() if len(effs) > 0 else 0.0
            floor       = 1.0 / TASK_MAX_TRIES.get(task, 2)
            norm_eff    = max((eff_on_succ - floor) / (1.0 - floor), 0.0) if eff_on_succ > 0 else 0.0
            rows.append(dict(
                model="human", task=task,
                participant=participant,
                expected_efficiency=sr * eff_on_succ,
                ecsr=sr * norm_eff,
                rule_score=p_scores[task],
                instance_accuracy=float("nan"),
            ))
    return pd.DataFrame(rows) if rows else None


def _make_per_model_figure(
    data: dict[str, pd.DataFrame],
    models: list[str],
    tasks: list[str],
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    include_human_panel: bool = False,
) -> None:
    """Per-model scatter panels + optional human panel."""
    _apply_nature_style()
    merged = _build_merged(data, models, tasks)

    panel_models = list(models)
    human_df = None
    if include_human_panel:
        human_df = _build_human_panel_df(data, tasks)
        if human_df is not None and not human_df[[x_col, y_col]].dropna().empty:
            panel_models = list(models) + ["human"]

    n_panels = len(panel_models)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(NATURE_FULL_W * (n_panels / 4), NATURE_FULL_W * 0.38),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.26, wspace=0.12)

    for ax, model in zip(axes, panel_models):
        if model == "human" and human_df is not None:
            m_sub = human_df
        else:
            m_sub = merged[merged["model"] == model]
        _draw_model_scatter_panel(ax, m_sub, x_col, y_col,
                                  x_label, y_label, model, panel_models)

    # Remove redundant y-labels on non-leftmost panels
    for ax in axes[1:]:
        ax.set_ylabel("")

    # Single shared x-axis label centered across all panels
    for ax in axes:
        ax.set_xlabel("")
    fig.text(0.53, 0.14, x_label, ha="center", va="bottom", fontsize=7)

    # Task-type legend (colors + fill encoding) below all panels
    task_handles = [
        mpl.lines.Line2D([], [], marker="o", color="w",
                         markerfacecolor=TASK_TYPE_COLOR[bt],
                         markersize=6, linewidth=0, label=TASK_DISPLAY[bt])
        for bt in ["additive", "compositional", "conditional", "override"]
    ]
    fill_handles = [
        mpl.lines.Line2D([], [], marker="o", markerfacecolor="#777777",
                         markeredgecolor="#777777", markersize=6,
                         linewidth=0, label="Attribute (filled)"),
        mpl.lines.Line2D([], [], marker="o", markerfacecolor="none",
                         markeredgecolor="#777777", markeredgewidth=1.0,
                         markersize=6, linewidth=0, label="Procedural (hollow)"),
    ]
    fig.legend(handles=task_handles + fill_handles,
               loc="lower center", bbox_to_anchor=(0.5, 0.00),
               ncol=len(task_handles) + len(fill_handles),
               frameon=False, handlelength=0.7, handletextpad=0.4,
               columnspacing=1.0, borderpad=0)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 7 — Format comparison: success rate vs QA instance accuracy
# ---------------------------------------------------------------------------

def _draw_format_comparison_panel(
    ax: plt.Axes,
    merged: pd.DataFrame,
    tasks: list[str],
    models: list[str],
    title: str,
) -> None:
    """Grouped bars: success_rate (solid) vs instance_accuracy (hatched) per model × task."""
    n_models = len(models)
    group_w  = 0.80
    pair_w   = group_w / n_models
    bar_w    = pair_w * 0.42

    for t_idx, task in enumerate(tasks):
        t_sub = merged[merged["task"] == task]
        for m_idx, model in enumerate(models):
            row = t_sub[t_sub["model"] == model]
            if row.empty:
                continue
            color  = _model_color(model, models)
            center = t_idx + (m_idx - (n_models - 1) / 2) * pair_w

            sr  = float(row["success_rate"].iloc[0])
            acc = row["instance_accuracy"].iloc[0]

            ax.bar(center - bar_w * 0.55, sr, width=bar_w,
                   color=color, alpha=0.85, zorder=3)
            if not np.isnan(acc):
                ax.bar(center + bar_w * 0.55, acc, width=bar_w,
                       color=color, alpha=0.40, hatch="///", edgecolor=color,
                       linewidth=0.4, zorder=3)

    ax.axhline(1.0, color="#888888", linewidth=0.7, linestyle="--", zorder=2)
    ax.set_xlim(-0.5, len(tasks) - 0.5)
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.set_ylabel("Proportion correct")
    ax.set_title(title, pad=4)
    _task_ticks(ax, tasks)
    _despine(ax)


def make_figure7(
    data: dict[str, pd.DataFrame],
    models: list[str],
    output_path: Path,
) -> None:
    """Two panels (property | proc). Solid bar = episodic success rate; hatched = QA accuracy."""
    _apply_nature_style()
    merged_prop = _build_merged(data, models, PROPERTY_TASKS)
    merged_proc = _build_merged(data, models, PROC_TASKS)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(NATURE_FULL_W, NATURE_FULL_W * 0.44))
    fig.subplots_adjust(left=0.09, right=0.76, top=0.88, bottom=0.14, wspace=0.32)

    _draw_format_comparison_panel(ax_l, merged_prop, PROPERTY_TASKS, models, "Attribute induction")
    _draw_format_comparison_panel(ax_r, merged_proc, PROC_TASKS,     models, "Procedural induction")

    model_handles = [
        mpatches.Patch(color=_model_color(m, models), label=_model_label(m))
        for m in models
    ]
    encoding_handles = [
        mpatches.Patch(facecolor="#777777", alpha=0.85, label="Solid = episodic SR"),
        mpatches.Patch(facecolor="#777777", alpha=0.40, hatch="///",
                       edgecolor="#777777", label="Hatched = QA accuracy"),
    ]
    fig.legend(handles=model_handles + encoding_handles,
               loc="center left", bbox_to_anchor=(0.78, 0.50),
               frameon=False, handlelength=0.9, handleheight=0.8,
               borderpad=0, labelspacing=0.5)

    _panel_label(ax_l, "A")
    _panel_label(ax_r, "B")

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 7b — Format gap: instance_accuracy − success_rate
# ---------------------------------------------------------------------------

def _draw_gap_panel(
    ax: plt.Axes,
    merged: pd.DataFrame,
    tasks: list[str],
    models: list[str],
    title: str,
    use_ecsr: bool = False,
    sig_data: dict | None = None,
    show_region_labels: bool = True,
) -> None:
    """Bar chart of (RV − ECSR) per model × task.
    If sig_data provided: error bars show bootstrap 95% CI across variants;
    stars mark gaps significantly different from zero (one-sample permutation test)."""
    n_models = len(models)
    bar_w    = 0.72 / n_models
    base_col = "ecsr" if use_ecsr else "success_rate"
    ylabel   = "QA accuracy − ECSR" if use_ecsr else "QA accuracy − episodic SR"

    for m_idx, model in enumerate(models):
        color  = _model_color(model, models)
        offset = (m_idx - (n_models - 1) / 2) * bar_w
        for t_idx, task in enumerate(tasks):
            row = merged[(merged["model"] == model) & (merged["task"] == task)]
            if row.empty:
                continue
            base = float(row[base_col].iloc[0])
            acc  = row["instance_accuracy"].iloc[0]
            if np.isnan(acc):
                continue
            gap  = acc - base
            x    = t_idx + offset
            ax.bar(x, gap, width=bar_w * 0.88,
                   color=color, alpha=0.85 if gap >= 0 else 0.55, zorder=3)

            # Significance stars only (no error bars)
            td = (sig_data or {}).get(model, {}).get(task)
            if td is not None:
                stars = td["stars"]
                if stars != "ns":
                    y_text = gap + (0.025 if gap >= 0 else -0.025)
                    va     = "bottom" if gap >= 0 else "top"
                    ax.text(x, y_text, stars,
                            ha="center", va=va, fontsize=5.5, zorder=8)

    ax.axhline(0, color="#444444", linewidth=0.8, zorder=4)
    ax.set_xlim(-0.5, len(tasks) - 0.5)
    y_abs = 0.75
    ax.set_ylim(-y_abs, y_abs)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4)
    _task_ticks(ax, tasks)
    _despine(ax)

    if show_region_labels:
        top_label = ("QA > ECSR\n(execution bottleneck)" if use_ecsr
                     else "QA > SR\n(execution bottleneck)")
        bot_label = ("ECSR > QA\n(QA bottleneck)" if use_ecsr
                     else "SR > QA\n(QA bottleneck)")
        ax.text(len(tasks) - 0.48, y_abs * 0.92, top_label,
                fontsize=5.5, color="#666666", ha="right", va="top")
        ax.text(len(tasks) - 0.48, -y_abs * 0.92, bot_label,
                fontsize=5.5, color="#666666", ha="right", va="bottom")


def make_figure7b(
    data: dict[str, pd.DataFrame],
    models: list[str],
    output_path: Path,
) -> None:
    """Two panels (property | proc). Bar = QA accuracy − episodic success rate."""
    _apply_nature_style()
    merged_prop = _build_merged(data, models, PROPERTY_TASKS)
    merged_proc = _build_merged(data, models, PROC_TASKS)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(NATURE_FULL_W, NATURE_FULL_W * 0.44))
    fig.subplots_adjust(left=0.10, right=0.76, top=0.88, bottom=0.14, wspace=0.32)

    _draw_gap_panel(ax_l, merged_prop, PROPERTY_TASKS, models, "Attribute induction")
    _draw_gap_panel(ax_r, merged_proc, PROC_TASKS,     models, "Procedural induction")

    model_handles = [
        mpatches.Patch(color=_model_color(m, models), label=_model_label(m))
        for m in models
    ]
    agg_handle = mpl.lines.Line2D(
        [], [], marker="D", color="black", markersize=5, linewidth=0,
        label="Mean across models",
    )
    fig.legend(handles=model_handles + [agg_handle],
               loc="center left", bbox_to_anchor=(0.78, 0.50),
               frameon=False, handlelength=0.9, handleheight=0.8,
               borderpad=0, labelspacing=0.5)

    _panel_label(ax_l, "A")
    _panel_label(ax_r, "B")

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_figure8_comparison(
    data: dict[str, pd.DataFrame],
    models: list[str],
    output_path: Path,
) -> None:
    """Two panels (property | proc). Solid bar = ECSR; hatched = QA instance accuracy.
    Companion to fig8 — shows the raw values whose difference fig8 plots."""
    _apply_nature_style()
    merged_prop = _build_merged(data, models, PROPERTY_TASKS)
    merged_proc = _build_merged(data, models, PROC_TASKS)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(NATURE_FULL_W, NATURE_FULL_W * 0.44))
    fig.subplots_adjust(left=0.09, right=0.76, top=0.88, bottom=0.14, wspace=0.32)

    for ax, merged, tasks, title in [
        (ax_l, merged_prop, PROPERTY_TASKS, "Attribute induction"),
        (ax_r, merged_proc, PROC_TASKS,     "Procedural induction"),
    ]:
        n_models = len(models)
        pair_w   = 0.80 / n_models
        bar_w    = pair_w * 0.42

        for t_idx, task in enumerate(tasks):
            t_sub = merged[merged["task"] == task]
            for m_idx, model in enumerate(models):
                row = t_sub[t_sub["model"] == model]
                if row.empty:
                    continue
                color  = _model_color(model, models)
                center = t_idx + (m_idx - (n_models - 1) / 2) * pair_w

                ecsr_val = float(row["ecsr"].iloc[0])
                acc      = row["instance_accuracy"].iloc[0]

                ax.bar(center - bar_w * 0.55, ecsr_val, width=bar_w,
                       color=color, alpha=0.85, zorder=3)
                if not np.isnan(acc):
                    ax.bar(center + bar_w * 0.55, acc, width=bar_w,
                           color=color, alpha=0.40, hatch="///", edgecolor=color,
                           linewidth=0.4, zorder=3)

        ax.axhline(1.0, color="#888888", linewidth=0.7, linestyle="--", zorder=2)
        ax.set_xlim(-0.5, len(tasks) - 0.5)
        ax.set_ylim(0, 1.18)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
        ax.set_ylabel("Proportion")
        ax.set_title(title, pad=4)
        _task_ticks(ax, tasks)
        _despine(ax)

    model_handles = [
        mpatches.Patch(color=_model_color(m, models), label=_model_label(m))
        for m in models
    ]
    encoding_handles = [
        mpatches.Patch(facecolor="#777777", alpha=0.85, label="Solid = ECSR"),
        mpatches.Patch(facecolor="#777777", alpha=0.40, hatch="///",
                       edgecolor="#777777", label="Hatched = QA accuracy"),
    ]
    fig.legend(handles=model_handles + encoding_handles,
               loc="center left", bbox_to_anchor=(0.78, 0.50),
               frameon=False, handlelength=0.9, handleheight=0.8,
               borderpad=0, labelspacing=0.5)

    _panel_label(ax_l, "A")
    _panel_label(ax_r, "B")

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_figure8(
    data: dict[str, pd.DataFrame],
    models: list[str],
    output_path: Path,
) -> None:
    """Two panels (property | proc). Bar = QA instance accuracy − ECSR per model × task.
    Stars: one-sample sign-permutation test (H0: mean gap = 0)."""
    _apply_nature_style()
    edf = data["episodes"]
    qdf = data["qa"]
    merged_prop = _build_merged(data, models, PROPERTY_TASKS)
    merged_proc = _build_merged(data, models, PROC_TASKS)
    sig_prop = _compute_gap_sig_data(edf, qdf, models, PROPERTY_TASKS)
    sig_proc = _compute_gap_sig_data(edf, qdf, models, PROC_TASKS)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(6.5, 2.3))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.16, wspace=0.28)

    _draw_gap_panel(ax_l, merged_prop, PROPERTY_TASKS, models,
                    "Attribute induction", use_ecsr=True, sig_data=sig_prop)
    _draw_gap_panel(ax_r, merged_proc, PROC_TASKS,     models,
                    "Procedural induction", use_ecsr=True, sig_data=sig_proc)
    ax_r.set_ylabel("")

    model_handles = [
        mpatches.Patch(color=_model_color(m, models), label=_model_label(m))
        for m in models
    ]
    ax_l.legend(handles=model_handles,
                loc="upper left",
                frameon=True, facecolor="white", edgecolor="none",
                handlelength=0.9, handleheight=0.8,
                borderpad=0.4, labelspacing=0.3)

    _panel_label(ax_l, "A")
    _panel_label(ax_r, "B")

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_figure4c(
    data: dict,
    tasks: list[str],
    output_path: Path,
) -> None:
    """Scatter: human expected efficiency (y) vs human rule score (x), one point per task."""
    human_rule = data.get("human_rule", {})
    if not human_rule:
        print("  Skipped fig4c: no human_rule_scores.json found")
        return

    # Build human expected efficiency per task from episode data
    edf = data["episodes"]
    edf_h = edf[edf["model"] == "human"]
    rows = []
    for task in tasks:
        sub = edf_h[edf_h["task"] == task]
        if sub.empty or task not in human_rule:
            continue
        sr          = sub["success"].mean()
        eff_vals    = sub["efficiency"].dropna().values
        eff_on_succ = eff_vals.mean() if len(eff_vals) > 0 else 0.0
        rows.append(dict(
            task             = task,
            expected_efficiency = sr * eff_on_succ,
            rule_score       = human_rule[task],
            is_proc          = task in PROC_TASKS,
        ))

    if not rows:
        print("  Skipped fig4c: no overlapping human episode + rule score data")
        return

    df = pd.DataFrame(rows)

    _apply_nature_style()
    fig, ax = plt.subplots(figsize=(NATURE_HALF_W * 1.3, NATURE_HALF_W * 1.3))
    fig.subplots_adjust(left=0.15, right=0.88, top=0.90, bottom=0.16)

    prop_df = df[~df["is_proc"]]
    proc_df = df[ df["is_proc"]]

    human_color = "#888888"
    ax.scatter(prop_df["rule_score"], prop_df["expected_efficiency"],
               s=32, color=human_color, marker="o", zorder=3, label="Attribute")
    ax.scatter(proc_df["rule_score"], proc_df["expected_efficiency"],
               s=32, color=human_color, marker="o", facecolors="none", linewidths=1.0,
               zorder=3, label="Procedural")

    # Label each point with task name
    for _, row in df.iterrows():
        ax.annotate(TASK_DISPLAY.get(row["task"], row["task"]),
                    (row["rule_score"], row["expected_efficiency"]),
                    fontsize=6, xytext=(4, 3), textcoords="offset points", color="#555")

    # OLS trend if enough points
    x, y = df["rule_score"].values, df["expected_efficiency"].values
    if len(x) >= 4:
        m, b, r, p, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line, m * x_line + b, color=human_color, lw=1.2, ls="--", alpha=0.7)
        p_show = p
        p_str  = f"p<0.001" if p_show < 0.001 else f"p={p_show:.3f}"
        ax.text(0.97, 0.05, f"r={r:.2f}, {p_str}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color="#555")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Rule score (human)", fontsize=8)
    ax.set_ylabel("Expected efficiency (human)", fontsize=8)
    ax.axhline(1.0, color="#ccc", lw=0.6, ls="--", zorder=0)
    ax.axvline(1.0, color="#ccc", lw=0.6, ls="--", zorder=0)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=human_color,
               markersize=6, label="Attribute"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor=human_color, markersize=6, label="Procedural"),
    ]
    ax.legend(handles=handles, fontsize=6, frameon=False, loc="upper left")

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_figure4_combined(
    data: dict,
    models: list[str],
    tasks: list[str],
    output_path: Path,
) -> None:
    """Two-panel figure: left = all-models scatter (fig4), right = human panel."""
    _apply_nature_style()
    merged  = _build_merged(data, models, tasks)
    human_df = _build_human_panel_df(data, tasks)

    fig, (ax_models, ax_human) = plt.subplots(
        1, 2,
        figsize=(6.5, 2.5),
        sharey=True,
    )
    fig.subplots_adjust(left=0.11, right=0.66, top=0.92, bottom=0.17, wspace=0.04)

    # Left panel: all-models scatter
    _make_scatter(ax_models, merged, "rule_score", "expected_efficiency",
                  "Rule verbalization score", "Expected efficiency", models)
    ax_models.set_title("Models", pad=4)
    ax_models.xaxis.label.set_size(7)
    ax_models.yaxis.label.set_size(7)
    ax_models.xaxis.labelpad = ax_models.yaxis.labelpad

    # Right panel: human scatter
    if human_df is not None and not human_df[["rule_score", "expected_efficiency"]].dropna().empty:
        _draw_model_scatter_panel(
            ax_human, human_df, "rule_score", "expected_efficiency",
            "Rule verbalization score", "", "human", ["human"],
            show_task_labels=False,
        )
        ax_human.set_ylabel("")
        ax_human.xaxis.label.set_size(7)
        ax_human.xaxis.labelpad = ax_models.xaxis.labelpad
    else:
        ax_human.text(0.5, 0.5, "No human data", transform=ax_human.transAxes,
                      ha="center", va="center", fontsize=8, color="#999999")
        ax_human.set_title("Human", pad=4)

    # Right-side legend: 6 rows × 2 columns
    _markers = ["o", "s", "^", "D", "v", "P"]
    task_handles = [
        mpl.lines.Line2D([], [], marker="o", color="w",
                         markerfacecolor=TASK_TYPE_COLOR[bt],
                         markersize=6, linewidth=0, label=_TASK_DISPLAY_SHORT[bt])
        for bt in ["additive", "compositional", "conditional", "override"]
    ]
    fill_handles = [
        mpl.lines.Line2D([], [], marker="o", markerfacecolor="#777777",
                         markeredgecolor="#777777", markersize=6,
                         linewidth=0, label="A- (filled)"),
        mpl.lines.Line2D([], [], marker="o", markerfacecolor="none",
                         markeredgecolor="#777777", markeredgewidth=1.0,
                         markersize=6, linewidth=0, label="P- (hollow)"),
    ]
    model_handles = [
        mpl.lines.Line2D([], [], marker=_markers[i % len(_markers)],
                         markerfacecolor="#777777", markeredgecolor="#777777",
                         markersize=6, linewidth=0, label=_model_label(m))
        for i, m in enumerate(models)
    ]
    human_handle = mpl.lines.Line2D(
        [], [], marker="*", markerfacecolor="#777777", markeredgecolor="#777777",
        markersize=7, linewidth=0, label="Human",
    )
    # Single legend on the right side: 12 rows × 1 column
    all_handles = task_handles + fill_handles + model_handles + [human_handle]
    fig.legend(handles=all_handles,
               loc="center left", bbox_to_anchor=(0.675, 0.50),
               ncol=1,
               frameon=False, handlelength=0.8, handletextpad=0.35,
               columnspacing=0.8, borderpad=0, fontsize=6)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_figure4b(data, models, tasks, output_path):
    _make_per_model_figure(data, models, tasks,
                           "rule_score", "expected_efficiency",
                           "Rule verbalization score", "Expected efficiency",
                           output_path, include_human_panel=False)


def make_figure5b(data, models, tasks, output_path):
    _make_per_model_figure(data, models, tasks,
                           "rule_score", "instance_accuracy",
                           "Rule verbalization score", "QA instance accuracy",
                           output_path)


def make_figure6b(data, models, tasks, output_path):
    _make_per_model_figure(data, models, tasks,
                           "instance_accuracy", "expected_efficiency",
                           "QA instance accuracy", "Expected efficiency",
                           output_path)


# ---------------------------------------------------------------------------
# ECSR variants of figs 4, 4_combined, 4b, 6, 6b
# ---------------------------------------------------------------------------

def make_figure4_ecsr(
    data: dict[str, pd.DataFrame],
    models: list[str],
    tasks: list[str],
    output_path: Path,
) -> None:
    """Scatter: ECSR (y) vs rule verbalization score (x)."""
    _apply_nature_style()
    merged = _build_merged(data, models, tasks)

    fig, ax = plt.subplots(figsize=(NATURE_HALF_W * 1.3, NATURE_HALF_W * 1.3))
    fig.subplots_adjust(left=0.15, right=0.72, top=0.90, bottom=0.16)

    _make_scatter(ax, merged, "rule_score", "ecsr",
                  "Rule verbalization score", "ECSR", models)
    _scatter_legend(fig, models, x=0.74, y=0.55)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_figure4_combined_ecsr(
    data: dict,
    models: list[str],
    tasks: list[str],
    output_path: Path,
) -> None:
    """Same as fig4_combined but y-axis = ECSR."""
    _apply_nature_style()
    merged   = _build_merged(data, models, tasks)
    human_df = _build_human_panel_df(data, tasks)

    fig, (ax_models, ax_human) = plt.subplots(
        1, 2,
        figsize=(6.5, 2.2),
        sharey=True,
    )
    fig.subplots_adjust(left=0.11, right=0.66, top=0.92, bottom=0.19, wspace=0.04)

    _make_scatter(ax_models, merged, "rule_score", "ecsr",
                  "Rule verbalization score", "ECSR", models)
    ax_models.set_title("Models", pad=4)
    ax_models.xaxis.label.set_size(7)
    ax_models.yaxis.label.set_size(7)
    ax_models.xaxis.labelpad = ax_models.yaxis.labelpad

    if human_df is not None and "ecsr" in human_df.columns and \
            not human_df[["rule_score", "ecsr"]].dropna().empty:
        _draw_model_scatter_panel(
            ax_human, human_df, "rule_score", "ecsr",
            "Rule verbalization score", "", "human", ["human"],
            show_task_labels=False,
        )
        ax_human.set_ylabel("")
        ax_human.xaxis.label.set_size(7)
        ax_human.xaxis.labelpad = ax_models.xaxis.labelpad
    else:
        ax_human.text(0.5, 0.5, "No human data", transform=ax_human.transAxes,
                      ha="center", va="center", fontsize=8, color="#999999")
        ax_human.set_title("Human", pad=4)

    _markers = ["o", "s", "^", "D", "v", "P"]
    task_handles = [
        mpl.lines.Line2D([], [], marker="o", color="w",
                         markerfacecolor=TASK_TYPE_COLOR[bt],
                         markersize=6, linewidth=0, label=_TASK_DISPLAY_SHORT[bt])
        for bt in ["additive", "compositional", "conditional", "override"]
    ]
    fill_handles = [
        mpl.lines.Line2D([], [], marker="o", markerfacecolor="#777777",
                         markeredgecolor="#777777", markersize=6,
                         linewidth=0, label="A- (filled)"),
        mpl.lines.Line2D([], [], marker="o", markerfacecolor="none",
                         markeredgecolor="#777777", markeredgewidth=1.0,
                         markersize=6, linewidth=0, label="P- (hollow)"),
    ]
    model_handles = [
        mpl.lines.Line2D([], [], marker=_markers[i % len(_markers)],
                         markerfacecolor="#777777", markeredgecolor="#777777",
                         markersize=6, linewidth=0, label=_model_label(m))
        for i, m in enumerate(models)
    ]
    human_handle = mpl.lines.Line2D(
        [], [], marker="*", markerfacecolor="#777777", markeredgecolor="#777777",
        markersize=7, linewidth=0, label="Human",
    )
    # Single legend on the right side
    all_handles = task_handles + fill_handles + model_handles + [human_handle]
    fig.legend(handles=all_handles,
               loc="center left", bbox_to_anchor=(0.675, 0.50),
               ncol=1,
               frameon=False, handlelength=0.8, handletextpad=0.35,
               columnspacing=0.8, borderpad=0, fontsize=6)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_figure14(
    data: dict,
    models: list[str],
    tasks: list[str],
    output_path: Path,
) -> None:
    """Single-panel scatter: rule verbalization score (x) vs ECSR (y).
    AI models and human participants co-plotted; separate regression lines
    for each group so their trends can be compared directly.

    Encoding (consistent with fig4):
      color  = task type  |  fill solid/hollow = Attribute/Procedural
      marker = model (AI) |  star = human
    Regression lines:
      models → dark solid (#2C2C2C)  |  human → dashed red (#CC3333)
    """
    _apply_nature_style()

    merged   = _build_merged(data, models, tasks)
    human_df = _build_human_panel_df(data, tasks)

    has_human = (
        human_df is not None
        and "ecsr" in human_df.columns
        and not human_df[["rule_score", "ecsr"]].dropna().empty
    )

    # ── Figure & axes ────────────────────────────────────────────────────────
    # Square-ish single panel; legend placed to the right of the axes.
    fig, ax = plt.subplots(figsize=(NATURE_FULL_W * 0.60, NATURE_FULL_W * 0.58))
    fig.subplots_adjust(left=0.13, right=0.62, top=0.93, bottom=0.14)

    lim = (-0.05, 1.10)
    markers = ["o", "s", "^", "D", "v", "P"]

    # ── AI model points ──────────────────────────────────────────────────────
    for m_idx, model in enumerate(models):
        marker = markers[m_idx % len(markers)]
        m_sub  = merged[merged["model"] == model]
        for _, row in m_sub.iterrows():
            color     = TASK_TYPE_COLOR.get(TASK_BASE_TYPE.get(row["task"], ""), "#777777")
            facecolor = "none" if row["task"] in PROC_TASKS else color
            ax.scatter(row["rule_score"], row["ecsr"],
                       facecolors=facecolor, edgecolors=color,
                       marker=marker, s=28, linewidths=0.8, zorder=4)

    # ── Human points ─────────────────────────────────────────────────────────
    if has_human:
        for _, row in human_df.iterrows():
            color     = TASK_TYPE_COLOR.get(TASK_BASE_TYPE.get(row["task"], ""), "#777777")
            facecolor = "none" if row["task"] in PROC_TASKS else color
            ax.scatter(row["rule_score"], row["ecsr"],
                       facecolors=facecolor, edgecolors=color,
                       marker="*", s=55, linewidths=0.7, zorder=4, alpha=0.80)

    # ── Helper: draw one regression line + CI band ───────────────────────────
    def _reg_line(x_vals, y_vals, line_color, linestyle, label):
        xy = pd.DataFrame({"x": x_vals, "y": y_vals}).dropna()
        if len(xy) < 4:
            return None, None
        x, y = xy["x"].values, xy["y"].values
        n    = len(x)
        slope, intercept, r_val, p_pearson, _ = stats.linregress(x, y)
        x_line  = np.linspace(0, 1, 200)
        y_line  = slope * x_line + intercept
        y_hat   = slope * x + intercept
        se_res  = np.sqrt(np.sum((y - y_hat) ** 2) / (n - 2))
        x_mean  = x.mean()
        se_line = se_res * np.sqrt(1/n + (x_line - x_mean) ** 2
                                   / np.sum((x - x_mean) ** 2))
        t_crit  = stats.t.ppf(0.975, df=n - 2)
        ax.plot(x_line, y_line, color=line_color, linewidth=1.1,
                linestyle=linestyle, zorder=5, label=label)
        ax.fill_between(x_line,
                        y_line - t_crit * se_line,
                        y_line + t_crit * se_line,
                        color=line_color, alpha=0.08, zorder=2)
        rho, p_sp = stats.spearmanr(x, y)
        return rho, p_sp

    # Models regression — dark solid
    xy_m   = merged[["rule_score", "ecsr"]].dropna()
    rho_m, p_m = _reg_line(xy_m["rule_score"], xy_m["ecsr"],
                            "#2C2C2C", "-", "Models OLS")

    # Human regression — red dashed
    rho_h = p_h = None
    if has_human:
        xy_h = human_df[["rule_score", "ecsr"]].dropna()
        rho_h, p_h = _reg_line(xy_h["rule_score"], xy_h["ecsr"],
                                "#CC3333", "--", "Human OLS")

    # ── ρ annotations (stacked, upper-left) ──────────────────────────────────
    annot_lines = []
    if rho_m is not None:
        n_m = len(xy_m)
        annot_lines.append(
            f"Models:  $\\rho$ = {rho_m:.2f},  {_fmt_p(p_m)}  (n={n_m})"
        )
    if rho_h is not None:
        n_h = len(human_df[["rule_score", "ecsr"]].dropna())
        annot_lines.append(
            f"Human:  $\\rho$ = {rho_h:.2f},  {_fmt_p(p_h)}  (n={n_h})"
        )
    if annot_lines:
        ax.text(0.04, 0.97, "\n".join(annot_lines),
                transform=ax.transAxes, fontsize=5.5, va="top", ha="left",
                color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="none", alpha=0.75))

    # ── Axes formatting ──────────────────────────────────────────────────────
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.set_xlabel("Rule verbalization score")
    ax.set_ylabel("ECSR")
    _despine(ax)

    # ── Legend (right of axes) ───────────────────────────────────────────────
    task_handles = [
        mpl.lines.Line2D([], [], marker="o", color="w",
                         markerfacecolor=TASK_TYPE_COLOR[bt],
                         markersize=6, linewidth=0, label=_TASK_DISPLAY_SHORT[bt])
        for bt in ["additive", "compositional", "conditional", "override"]
    ]
    fill_handles = [
        mpl.lines.Line2D([], [], marker="o", markerfacecolor="#777",
                         markeredgecolor="#777", markersize=5,
                         linewidth=0, label="Attr. (filled)"),
        mpl.lines.Line2D([], [], marker="o", markerfacecolor="none",
                         markeredgecolor="#777", markeredgewidth=1.0,
                         markersize=5, linewidth=0, label="Proc. (hollow)"),
    ]
    model_handles = [
        mpl.lines.Line2D([], [], marker=markers[i % len(markers)],
                         markerfacecolor="#777", markeredgecolor="#777",
                         markersize=5, linewidth=0, label=_model_label(m))
        for i, m in enumerate(models)
    ]
    human_handle = mpl.lines.Line2D(
        [], [], marker="*", markerfacecolor="#777", markeredgecolor="#777",
        markersize=6, linewidth=0, label="Human",
    )
    trend_handles = [
        mpl.lines.Line2D([], [], color="#2C2C2C", linewidth=1.1,
                         linestyle="-",  label="Models OLS"),
        mpl.lines.Line2D([], [], color="#CC3333", linewidth=1.0,
                         linestyle="--", label="Human OLS"),
    ]
    all_handles = task_handles + fill_handles + model_handles
    if has_human:
        all_handles += [human_handle]
    all_handles += trend_handles

    fig.legend(handles=all_handles,
               loc="center left", bbox_to_anchor=(0.635, 0.50),
               ncol=1, frameon=False,
               handlelength=0.8, handletextpad=0.35,
               labelspacing=0.45, borderpad=0, fontsize=6)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 15 — Radar charts: ECSR and RV across task types
# ---------------------------------------------------------------------------

def make_figure15(
    data: dict,
    models: list[str],
    tasks: list[str],
    output_path: Path,
) -> None:
    """Two side-by-side radar charts: left = ECSR, right = RV (rule verbalization score).
    One polygon per AI model plus one for the human average across all participants.
    Spokes = task types.
    """
    _apply_nature_style()

    merged   = _build_merged(data, models, tasks)
    human_df = _build_human_panel_df(data, tasks)

    # ── Collect (entity → task → metric) values ───────────────────────────────
    entity_data: dict[str, dict[str, dict[str, float]]] = {}

    for model in models:
        m_sub = merged[merged["model"] == model]
        entity_data[model] = {
            row["task"]: {
                "ecsr": row["ecsr"],
                "rv":   row.get("rule_score", float("nan")),
            }
            for _, row in m_sub.iterrows()
        }

    if human_df is not None and not human_df.empty:
        h_agg = human_df.groupby("task")[["ecsr", "rule_score"]].mean()
        entity_data["human"] = {
            task: {
                "ecsr": h_agg.loc[task, "ecsr"]       if task in h_agg.index else float("nan"),
                "rv":   h_agg.loc[task, "rule_score"] if task in h_agg.index else float("nan"),
            }
            for task in tasks
        }

    entities   = list(entity_data.keys())
    n_tasks    = len(tasks)
    angles     = [2 * np.pi * i / n_tasks for i in range(n_tasks)]
    angles_cls = angles + [angles[0]]  # closed polygon

    def _vals(entity: str, metric: str) -> list[float]:
        d = entity_data[entity]
        raw = [d.get(t, {}).get(metric, float("nan")) for t in tasks]
        return [v if not np.isnan(v) else 0.0 for v in raw]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(NATURE_FULL_W, NATURE_FULL_W * 0.52))

    panels = [("ecsr", "ECSR"), ("rv", "Rule verbalization score")]
    for col, (metric, title) in enumerate(panels):
        ax = fig.add_subplot(1, 2, col + 1, projection="polar")
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=5.5, color="#555555")
        ax.set_xticks(angles)
        ax.set_xticklabels([TASK_DISPLAY[t] for t in tasks], fontsize=6.5)
        ax.set_title(title, fontsize=8, pad=14)
        ax.spines["polar"].set_linewidth(0.5)
        ax.grid(color="#E0E0E0", linewidth=0.4, alpha=0.9)

        for entity in entities:
            is_human   = entity == "human"
            color      = MODEL_PALETTE[-1] if is_human else _model_color(entity, models)
            linestyle  = "--" if is_human else "-"
            lw         = 1.3 if is_human else 0.9
            vals       = _vals(entity, metric)
            vals_cls   = vals + [vals[0]]
            ax.plot(angles_cls, vals_cls,
                    color=color, linewidth=lw, linestyle=linestyle,
                    alpha=0.90, zorder=4)
            ax.fill(angles_cls, vals_cls, color=color, alpha=0.06, zorder=3)

    # ── Shared legend below both panels ───────────────────────────────────────
    handles = []
    for entity in entities:
        is_human = entity == "human"
        color    = MODEL_PALETTE[-1] if is_human else _model_color(entity, models)
        ls       = "--" if is_human else "-"
        label    = "Human (avg)" if is_human else _model_label(entity)
        handles.append(
            mpl.lines.Line2D([], [], color=color, linewidth=1.2,
                             linestyle=ls, label=label)
        )

    fig.legend(handles=handles,
               loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=len(entities), frameon=False,
               handlelength=1.2, handletextpad=0.4,
               columnspacing=1.2, fontsize=6.5)

    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.12, wspace=0.45)

    fig.savefig(output_path, transparent=True)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 16 — Horizontal bar chart: overall avg (all 8 tasks) per model
# ---------------------------------------------------------------------------

def make_figure16(
    data: dict,
    models: list[str],
    output_path: Path,
) -> None:
    """Single-panel horizontal bar chart averaged across all 8 tasks.
    Each model gets two bars: ECSR (solid) and RV (hatched).
    Human: average across all participants.
    """
    _apply_nature_style()

    merged   = _build_merged(data, models, ALL_TASKS)
    human_df = _build_human_panel_df(data, ALL_TASKS)

    has_human = human_df is not None and not human_df.empty
    entities  = list(models) + (["human"] if has_human else [])

    def _avgs(entity: str) -> tuple[float, float]:
        if entity == "human":
            h_agg = human_df.groupby("task")[["ecsr", "rule_score"]].mean()
            return h_agg["ecsr"].mean(), h_agg["rule_score"].mean()
        sub = merged[merged["model"] == entity]
        return (sub["ecsr"].mean()       if not sub.empty else float("nan"),
                sub["rule_score"].mean() if not sub.empty else float("nan"))

    # ── Layout ────────────────────────────────────────────────────────────────
    bar_h  = 0.28
    gap    = 0.10
    slot_h = 2 * bar_h + gap + 0.18

    n       = len(entities)
    y_slots = np.arange(n) * slot_h
    y_ecsr  = y_slots + bar_h + gap / 2
    y_rv    = y_slots + gap / 2

    entities_td = list(reversed(entities))   # top of chart = first model

    fig, ax = plt.subplots(figsize=(NATURE_HALF_W * 0.95, slot_h * n + 0.55))
    fig.subplots_adjust(left=0.04, right=0.70, top=0.93, bottom=0.13)

    for i, entity in enumerate(entities_td):
        ecsr_val, rv_val = _avgs(entity)
        color = MODEL_PALETTE[-1] if entity == "human" else _model_color(entity, models)
        label = "Human (avg)" if entity == "human" else _model_label(entity)

        if not np.isnan(ecsr_val):
            ax.barh(y_ecsr[i], ecsr_val, height=bar_h,
                    color=color, alpha=0.88, zorder=3, linewidth=0.4, edgecolor=color)
        if not np.isnan(rv_val):
            ax.barh(y_rv[i], rv_val, height=bar_h,
                    color="none", hatch="////", zorder=3, linewidth=0.4, edgecolor=color)

        # Model name to the right of the longer bar
        x_label = max(v for v in [ecsr_val, rv_val] if not np.isnan(v))
        y_mid   = (y_ecsr[i] + y_rv[i]) / 2
        ax.text(x_label + 0.03, y_mid, label,
                va="center", ha="left", fontsize=7, color="#333333", clip_on=False)

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Score", fontsize=7.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.tick_params(axis="x", labelsize=7)
    ax.set_yticks([])
    ax.set_ylim(-0.15, y_slots[-1] + slot_h)
    ax.spines["left"].set_visible(False)
    _despine(ax)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpl.patches.Patch(facecolor="#555555", edgecolor="#555555",
                          alpha=0.88, label="ECSR"),
        mpl.patches.Patch(facecolor="none", edgecolor="#555555",
                          hatch="////", label="RV"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              frameon=False, fontsize=7.5,
              handlelength=1.2, handletextpad=0.5, columnspacing=1.0)

    fig.savefig(output_path, transparent=True)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_figure4b_ecsr(data, models, tasks, output_path):
    """Per-model scatter: ECSR (y) vs rule verbalization score (x)."""
    _make_per_model_figure(data, models, tasks,
                           "rule_score", "ecsr",
                           "Rule verbalization score", "ECSR",
                           output_path, include_human_panel=False)


def make_figure6_ecsr(
    data: dict[str, pd.DataFrame],
    models: list[str],
    tasks: list[str],
    output_path: Path,
) -> None:
    """Scatter: ECSR (y) vs QA instance accuracy (x)."""
    _apply_nature_style()
    merged = _build_merged(data, models, tasks)

    fig, ax = plt.subplots(figsize=(NATURE_HALF_W * 1.3, NATURE_HALF_W * 1.3))
    fig.subplots_adjust(left=0.15, right=0.72, top=0.90, bottom=0.16)

    _make_scatter(ax, merged, "instance_accuracy", "ecsr",
                  "QA instance accuracy", "ECSR", models)
    _scatter_legend(fig, models, x=0.74, y=0.55)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_figure6b_ecsr(data, models, tasks, output_path):
    """Per-model scatter: ECSR (y) vs QA instance accuracy (x)."""
    _make_per_model_figure(data, models, tasks,
                           "instance_accuracy", "ecsr",
                           "QA instance accuracy", "ECSR",
                           output_path)


# ---------------------------------------------------------------------------
# Figure 9 — Semantic vs Nonce comparison (two models)
# ---------------------------------------------------------------------------

# Per-model color pairs: (semantic, nonce)
NONCE_MODEL_COLORS = {
    "qwen27b": {"semantic": "#E87722", "nonce": "#F5C49A"},   # orange / light orange
    "gpt5":    {"semantic": "#4878CF", "nonce": "#A8C4EF"},   # blue / light blue
}


def _draw_nonce_comparison_panel_two_models(
    ax: plt.Axes,
    model_data: list,   # [(sem_key, sem_merged, nonce_merged), ...]
    tasks: list[str],
    title: str,
    sig_data: list[dict[str, dict | None]] | None = None,  # one dict per model entry
) -> None:
    """4 bars per task: [model0-sem, model0-nonce | model1-sem, model1-nonce].
    Error bars show bootstrap 95% CI across variants.
    Bracket + stars above each sem/nonce pair shows permutation p-value."""
    unit_w    = 0.17
    gap_inner = 0.03   # between sem/nonce within same model
    gap_outer = 0.10   # between model groups

    total_w = 4 * unit_w + 2 * gap_inner + gap_outer
    starts  = [
        0,
        unit_w + gap_inner,
        2 * unit_w + gap_inner + gap_outer,
        3 * unit_w + 2 * gap_inner + gap_outer,
    ]
    centers = [s + unit_w / 2 - total_w / 2 for s in starts]

    conditions = []
    for sem_key, sem_merged, nonce_merged in model_data:
        colors = NONCE_MODEL_COLORS[sem_key]
        conditions.append((sem_merged, colors["semantic"]))
        conditions.append((nonce_merged, colors["nonce"]))

    # collect bar heights for dynamic ylim
    bar_tops = []

    for t_idx, task in enumerate(tasks):
        for c_idx, (merged, color) in enumerate(conditions):
            row = merged[merged["task"] == task]
            if row.empty:
                continue
            bar_x    = t_idx + centers[c_idx]
            ecsr_val = float(row["ecsr"].iloc[0])
            rv_val   = row["rule_score"].iloc[0]

            # bootstrap CI if sig_data provided
            ci_lo = ci_hi = None
            if sig_data is not None:
                m_idx  = c_idx // 2          # which model
                is_noc = bool(c_idx % 2)     # semantic (0) or nonce (1)
                td = sig_data[m_idx].get(task) if sig_data[m_idx] else None
                if td is not None:
                    ci_lo, ci_hi = td["noc_ci"] if is_noc else td["sem_ci"]

            ax.bar(bar_x, ecsr_val, width=unit_w,
                   color=color, alpha=0.88, zorder=3)
            top = ci_hi if ci_hi is not None else ecsr_val
            bar_tops.append(top)

            if not np.isnan(rv_val):
                ax.scatter([bar_x], [rv_val],
                           marker="D", color=color, s=22, zorder=5,
                           edgecolors="white", linewidths=0.6)

    # significance brackets above each sem/nonce pair per model
    # Two rows: lower = ECSR, upper = RV
    if sig_data is not None:
        tick_h = 0.018
        row_gap = 0.09   # vertical gap between ECSR and RV bracket rows

        for t_idx, task in enumerate(tasks):
            for m_idx in range(len(model_data)):
                td = sig_data[m_idx].get(task) if sig_data[m_idx] else None
                if td is None:
                    continue
                x_sem  = t_idx + centers[2 * m_idx]
                x_noc  = t_idx + centers[2 * m_idx + 1]
                x_mid  = (x_sem + x_noc) / 2

                # base height: just above the taller bar
                sem_top = td["sem_ci"][1] if td.get("sem_ci") else 0.0
                noc_top = td["noc_ci"][1] if td.get("noc_ci") else 0.0
                y0 = max(sem_top, noc_top) + 0.05

                row_drawn = 0
                for stars_key, label, color in [
                    ("ecsr_stars", "E", "black"),
                    ("rv_stars",   "R", "#555555"),
                ]:
                    stars = td.get(stars_key, "")
                    if not stars or stars == "ns":
                        continue
                    y = y0 + row_drawn * row_gap
                    ax.plot([x_sem, x_sem, x_noc, x_noc],
                            [y - tick_h, y, y, y - tick_h],
                            color=color, linewidth=0.8, zorder=7)
                    ax.text(x_mid, y + 0.005,
                            f"{label}:{stars}",
                            ha="center", va="bottom", fontsize=5.0,
                            color=color, zorder=8)
                    row_drawn += 1

    ax.axhline(1.0, color="#888888", linewidth=0.7, linestyle="--", zorder=2)
    ax.set_xlim(-0.5, len(tasks) - 0.5)
    y_max = max(bar_tops) + 0.22 if bar_tops else 1.25
    ax.set_ylim(0, max(y_max, 1.25))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.set_title(title, pad=4)
    _task_ticks(ax, tasks)
    _despine(ax)


def make_figure9(
    data: dict[str, pd.DataFrame],
    output_path: Path,
    model_pairs: list[tuple[str, str]] | None = None,
) -> None:
    """Two panels (property | proc). 4 paired bars per task: sem/nonce × two models.
    Error bars: bootstrap 95% CI across variants.
    Bracket + stars: permutation p-value for semantic > nonce within each model × task.

    model_pairs: list of (sem_key, nonce_key), default GPT-5 and Qwen 27B.
    """
    if model_pairs is None:
        model_pairs = [("gpt5", "gpt5_nonce"), ("qwen27b", "qwen_nonce")]

    _apply_nature_style()

    edf = data["episodes"]
    qdf = data["qa"]
    model_data = []
    sig_data_prop = []
    sig_data_proc = []
    for sem_key, nonce_key in model_pairs:
        sem_merged   = _build_merged(data, [sem_key],   ALL_TASKS)
        nonce_merged = _build_merged(data, [nonce_key], ALL_TASKS)
        model_data.append((sem_key, sem_merged, nonce_merged))
        sig_data_prop.append(_compute_nonce_sig(edf, qdf, sem_key, nonce_key, PROPERTY_TASKS))
        sig_data_proc.append(_compute_nonce_sig(edf, qdf, sem_key, nonce_key, PROC_TASKS))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(6.5, 2.3))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.16, wspace=0.28)

    _draw_nonce_comparison_panel_two_models(ax_l, model_data, PROPERTY_TASKS, "Attribute induction",
                                            sig_data=sig_data_prop)
    _draw_nonce_comparison_panel_two_models(ax_r, model_data, PROC_TASKS,     "Procedural induction",
                                            sig_data=sig_data_proc)

    ax_l.set_ylabel("ECSR  /  Rule verbalization score")

    legend_handles = []
    for sem_key, _, __ in model_data:
        colors = NONCE_MODEL_COLORS[sem_key]
        label  = MODEL_DISPLAY.get(sem_key, sem_key)
        legend_handles += [
            mpatches.Patch(color=colors["semantic"], alpha=0.88, label=f"{label} (semantic)"),
            mpatches.Patch(color=colors["nonce"],    alpha=0.88, label=f"{label} (nonce)"),
        ]
    legend_handles.append(
        mpl.lines.Line2D([], [], marker="D", color="w",
                         markerfacecolor="#555555", markeredgecolor="white",
                         markersize=5, linewidth=0, label="Avg. RV"),
    )
    ax_r.legend(handles=legend_handles,
                loc="upper right",
                frameon=True, facecolor="white", edgecolor="none",
                handlelength=0.9, handleheight=0.8,
                borderpad=0.4, labelspacing=0.3)

    _panel_label(ax_l, "A")
    _panel_label(ax_r, "B")

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 13b — aggregated two-panel: gap (A) + nonce (B) at family level
# ---------------------------------------------------------------------------

def _compute_gap_family_data(
    data: dict, models_8: list[str],
) -> dict:
    """Compute per-model × family gap stats with ×2 Bonferroni within each model."""
    edf = data["episodes"]
    qdf = data["qa"]
    families = [("Attribute", PROPERTY_TASKS), ("Procedural", PROC_TASKS)]
    gap_data: dict = {}
    for model in models_8:
        raw_ps, keys = [], []
        for fam_name, tasks in families:
            pooled = []
            for task in tasks:
                ecsr_d = _get_variant_ecsrs_dict(edf, model, task)
                acc_d  = _get_variant_instance_acc_dict(qdf, model, task)
                for s in sorted(set(ecsr_d) & set(acc_d)):
                    pooled.append(acc_d[s] - ecsr_d[s])
            if len(pooled) >= 2:
                p = _permutation_p_zero(pooled)
                ci_lo, ci_hi = _bootstrap_mean_ci(pooled)
                gap_data[(model, fam_name)] = dict(
                    mean=float(np.mean(pooled)), ci_lo=ci_lo, ci_hi=ci_hi, p_raw=p)
                raw_ps.append(p); keys.append((model, fam_name))
        for k, p in zip(keys, raw_ps):
            gap_data[k]["stars"] = _sig_stars(min(p * len(raw_ps), 1.0))
    return gap_data


def _compute_nonce_family_data(
    data: dict,
    model_pairs_9: list[tuple[str, str]],
) -> dict:
    """Compute per-pair × metric × family nonce stats with ×2 Bonferroni within each pair × metric."""
    edf = data["episodes"]
    qdf = data["qa"]
    families = [("Attribute", PROPERTY_TASKS), ("Procedural", PROC_TASKS)]
    nonce_data: dict = {}
    for sem_key, nonce_key in model_pairs_9:
        for metric in ["ecsr", "rv"]:
            raw_ps, keys = [], []
            for fam_name, tasks in families:
                pooled = []
                for task in tasks:
                    if metric == "ecsr":
                        sem_d = _get_variant_ecsrs_dict(edf, sem_key,   task)
                        noc_d = _get_variant_ecsrs_dict(edf, nonce_key, task)
                    else:
                        sem_d = _get_variant_rv_dict(qdf, sem_key,   task)
                        noc_d = _get_variant_rv_dict(qdf, nonce_key, task)
                    for s in sorted(set(sem_d) & set(noc_d)):
                        pooled.append(sem_d[s] - noc_d[s])
                if len(pooled) >= 2:
                    p = _permutation_p_zero(pooled)
                    ci_lo, ci_hi = _bootstrap_mean_ci(pooled)
                    nonce_data[(sem_key, metric, fam_name)] = dict(
                        mean=float(np.mean(pooled)), ci_lo=ci_lo, ci_hi=ci_hi, p_raw=p)
                    raw_ps.append(p); keys.append((sem_key, metric, fam_name))
            for k, p in zip(keys, raw_ps):
                nonce_data[k]["stars"] = _sig_stars(min(p * len(raw_ps), 1.0))
    return nonce_data


def save_figure13a_csv(
    data: dict, models_8: list[str], output_path: Path,
) -> None:
    """CSV for fig13a: per-model × family gap (QA accuracy − ECSR)."""
    gap_data = _compute_gap_family_data(data, models_8)
    rows = []
    for model in models_8:
        for fam in ["Attribute", "Procedural"]:
            d = gap_data.get((model, fam), {})
            rows.append({
                "model":   _model_label(model),
                "family":  fam,
                "n":       len(d),
                "mean":    round(d.get("mean", float("nan")), 4),
                "p_raw":   round(d.get("p_raw", float("nan")), 4),
                "p_adj":   round(min(d.get("p_raw", 1.0) * 2, 1.0), 4) if "p_raw" in d else float("nan"),
                "stars":   d.get("stars", ""),
            })
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")


def save_figure13b_csv(
    data: dict,
    models_8: list[str],
    output_path: Path,
    model_pairs_9: list[tuple[str, str]] | None = None,
) -> None:
    """CSV for fig13b: per-model × metric × family nonce effect (semantic − nonce)."""
    if model_pairs_9 is None:
        model_pairs_9 = [("gpt5", "gpt5_nonce"), ("qwen27b", "qwen_nonce")]
    nonce_data = _compute_nonce_family_data(data, model_pairs_9)
    rows = []
    for sem_key, _ in model_pairs_9:
        for metric in ["ecsr", "rv"]:
            for fam in ["Attribute", "Procedural"]:
                d = nonce_data.get((sem_key, metric, fam), {})
                rows.append({
                    "model":   _model_label(sem_key),
                    "metric":  metric.upper(),
                    "family":  fam,
                    "mean":    round(d.get("mean", float("nan")), 4),
                    "p_raw":   round(d.get("p_raw", float("nan")), 4),
                    "p_adj":   round(min(d.get("p_raw", 1.0) * 2, 1.0), 4) if "p_raw" in d else float("nan"),
                    "stars":   d.get("stars", ""),
                })
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")


# shared figure size matching fig14
_FIG14_W = NATURE_FULL_W * 0.60
_FIG14_H = NATURE_FULL_W * 0.58


def make_figure13a(
    data: dict,
    models_8: list[str],
    output_path: Path,
) -> None:
    """Single-panel: QA gap (QA accuracy − ECSR) per model × family."""
    _apply_nature_style()
    gap_data   = _compute_gap_family_data(data, models_8)
    fam_keys   = ["Attribute", "Procedural"]   # keys used in gap_data
    fam_labels = ["Attr.", "Proc."]            # display labels
    x     = np.array([0, 0.8])   # tighter group spacing
    bar_w = 0.22

    # auto-scale y from data
    vals  = [gap_data.get((m, f), {}).get("mean", 0) for m in models_8 for f in fam_keys]
    v_abs = max(abs(v) for v in vals) if vals else 0.1
    y_hi  = np.ceil(max(v_abs, 0.15) * 1.55 * 10) / 10
    y_lo  = -np.ceil(max(v_abs, 0.10) * 1.30 * 10) / 10
    loc_step = 0.1 if (y_hi - y_lo) <= 0.6 else 0.25
    x_max = x[-1] + 0.35   # right xlim

    fig, ax = plt.subplots(figsize=(2.5, 2.0))
    fig.subplots_adjust(left=0.21, right=0.94, top=0.94, bottom=0.18)

    for m_idx, model in enumerate(models_8):
        offset = (m_idx - (len(models_8) - 1) / 2) * bar_w
        color  = _model_color(model, models_8)
        for fi, fam in enumerate(fam_keys):
            d     = gap_data.get((model, fam), {})
            v     = d.get("mean", 0)
            stars = d.get("stars", "ns")
            ax.bar(x[fi] + offset, v, width=bar_w * 0.85,
                   color=color, alpha=0.85 if v >= 0 else 0.55, zorder=3)
            if stars != "ns":
                y_text = v + (0.018 if v >= 0 else -0.018)
                va     = "bottom" if v >= 0 else "top"
                ax.text(x[fi] + offset, y_text, stars,
                        ha="center", va=va, fontsize=5.5, zorder=8)

    ax.axhline(0, color="#444444", linewidth=0.8, zorder=4)
    ax.set_xlim(x[0] - 0.32, x_max)
    ax.set_ylim(y_lo, y_hi)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(loc_step))
    ax.set_xticks(x); ax.set_xticklabels(fam_labels)
    ax.set_ylabel("QA accuracy − ECSR")

    model_handles = [mpatches.Patch(color=_model_color(m, models_8), alpha=0.85,
                                    label=_model_label(m)) for m in models_8]
    ax.legend(handles=model_handles, frameon=False, fontsize=6.5,
              loc="upper left", handlelength=1.0)
    _despine(ax)

    ax.text(x_max - 0.02, y_hi * 0.92,
            "QA > ECSR\n(exec. bottleneck)",
            fontsize=5.0, color="#666666", ha="right", va="top")
    ax.text(x_max - 0.02, y_lo * 0.92,
            "ECSR > QA\n(QA bottleneck)",
            fontsize=5.0, color="#666666", ha="right", va="bottom")

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_figure13b(
    data: dict,
    models_8: list[str],
    output_path: Path,
    model_pairs_9: list[tuple[str, str]] | None = None,
) -> None:
    """Single-panel: nonce effect (semantic − nonce) per model pair × metric × family.
    Same size as fig14."""
    if model_pairs_9 is None:
        model_pairs_9 = [("gpt5", "gpt5_nonce"), ("qwen27b", "qwen_nonce")]

    _apply_nature_style()
    nonce_data = _compute_nonce_family_data(data, model_pairs_9)
    pair_keys  = [sem for sem, _ in model_pairs_9]
    fam_keys   = ["Attribute", "Procedural"]
    fam_labels = ["Attr.", "Proc."]
    x     = np.array([0, 0.8])
    bar_w = 0.22

    metric_hatch  = {"ecsr": "",      "rv": "////"}
    metric_edge   = {"ecsr": "none",  "rv": "#333333"}
    metric_lw     = {"ecsr": 0.0,     "rv": 0.6}
    metric_alpha  = {"ecsr": 0.85,    "rv": 0.55}
    metric_labels = {"ecsr": "ECSR",  "rv": "RV"}

    bar_specs = [(sem, m) for sem in pair_keys for m in ["ecsr", "rv"]]
    n_bars    = len(bar_specs)
    offsets   = (np.arange(n_bars) - (n_bars - 1) / 2) * bar_w * 0.60

    b_vals = [nonce_data.get((sk, mt, fam), {}).get("mean", 0)
              for sk, mt in bar_specs for fam in fam_keys]
    b_abs  = max(abs(v) for v in b_vals) if b_vals else 0.1
    y_b    = np.ceil(max(b_abs * 1.5, 0.15) * 10) / 10
    loc_step = 0.1 if y_b <= 0.4 else 0.25

    fig, ax = plt.subplots(figsize=(2.5, 2.0))
    fig.subplots_adjust(left=0.28, right=0.94, top=0.94, bottom=0.18)

    for bi, (sem_key, metric) in enumerate(bar_specs):
        color = _model_color(sem_key, models_8)
        for fi, fam in enumerate(fam_keys):
            d     = nonce_data.get((sem_key, metric, fam), {})
            v     = d.get("mean", 0)
            stars = d.get("stars", "ns")
            label = f"{_model_label(sem_key)} {metric_labels[metric]}" if fi == 0 else None
            ax.bar(x[fi] + offsets[bi], v, width=bar_w * 0.55,
                   color=color,
                   alpha=metric_alpha[metric] if v >= 0 else metric_alpha[metric] * 0.65,
                   hatch=metric_hatch[metric],
                   edgecolor=metric_edge[metric], linewidth=metric_lw[metric],
                   zorder=3, label=label)
            if stars != "ns":
                y_text = v + (0.018 if v >= 0 else -0.018)
                va     = "bottom" if v >= 0 else "top"
                ax.text(x[fi] + offsets[bi], y_text, stars,
                        ha="center", va=va, fontsize=5.5, zorder=8)

    ax.axhline(0, color="#444444", linewidth=0.8, zorder=4)
    ax.set_xlim(x[0] - 0.32, x[-1] + 0.35)
    ax.set_ylim(-y_b, y_b)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(loc_step))
    ax.set_xticks(x); ax.set_xticklabels(fam_labels)
    ax.set_ylabel("Δ ECSR / Δ RV\n(Semantic − Nonce)")
    ax.legend(frameon=False, fontsize=5.5, loc="upper right",
              handlelength=1.0, ncol=2, columnspacing=0.8)
    _despine(ax)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 13 — combined 1×4: gap panels (A/B) + nonce comparison (C/D)
# ---------------------------------------------------------------------------

def make_figure13(
    data: dict[str, pd.DataFrame],
    models_8: list[str],
    output_path: Path,
    model_pairs_9: list[tuple[str, str]] | None = None,
) -> None:
    """1×4 figure combining fig8 (panels A–B, left) and fig9 (panels C–D, right).
    All four panels on a single row; fits EMNLP two-column text width (~7 in).
    Panels A/B share the gap y-axis; panels C/D share the nonce y-axis."""
    if model_pairs_9 is None:
        model_pairs_9 = [("gpt5", "gpt5_nonce"), ("qwen27b", "qwen_nonce")]

    _apply_nature_style()

    edf = data["episodes"]
    qdf = data["qa"]

    # Gap panels data (A = attribute, B = procedural)
    merged_prop = _build_merged(data, models_8, PROPERTY_TASKS)
    merged_proc = _build_merged(data, models_8, PROC_TASKS)
    sig_prop = _compute_gap_sig_data(edf, qdf, models_8, PROPERTY_TASKS)
    sig_proc = _compute_gap_sig_data(edf, qdf, models_8, PROC_TASKS)

    # Nonce panels data (C = attribute, D = procedural)
    model_data, sig_data_prop, sig_data_proc = [], [], []
    for sem_key, nonce_key in model_pairs_9:
        sem_merged   = _build_merged(data, [sem_key],   ALL_TASKS)
        nonce_merged = _build_merged(data, [nonce_key], ALL_TASKS)
        model_data.append((sem_key, sem_merged, nonce_merged))
        sig_data_prop.append(_compute_nonce_sig(edf, qdf, sem_key, nonce_key, PROPERTY_TASKS))
        sig_data_proc.append(_compute_nonce_sig(edf, qdf, sem_key, nonce_key, PROC_TASKS))

    # ── Layout ──────────────────────────────────────────────────────────────
    # Five-column GridSpec: [gap-attr | gap-proc | spacer | nonce-attr | nonce-proc]
    # The narrow spacer column creates a visible break between the two groups.
    fig = plt.figure(figsize=(NATURE_FULL_W, 2.6))
    gs = gridspec.GridSpec(
        1, 5,
        figure=fig,
        width_ratios=[1, 1, 0.18, 1, 1],
        left=0.09, right=0.97, top=0.88, bottom=0.21,
        wspace=0.40,
    )
    ax_ga = fig.add_subplot(gs[0, 0])
    ax_gb = fig.add_subplot(gs[0, 1], sharey=ax_ga)
    ax_na = fig.add_subplot(gs[0, 3])
    ax_nb = fig.add_subplot(gs[0, 4], sharey=ax_na)

    # ── Gap panels ───────────────────────────────────────────────────────────
    # Region labels ("QA > ECSR" / "ECSR > QA") only on panel A; legend lives on panel B.
    _draw_gap_panel(ax_ga, merged_prop, PROPERTY_TASKS, models_8,
                    "Attribute induction", use_ecsr=True, sig_data=sig_prop,
                    show_region_labels=True)
    _draw_gap_panel(ax_gb, merged_proc, PROC_TASKS, models_8,
                    "Procedural induction", use_ecsr=True, sig_data=sig_proc,
                    show_region_labels=False)
    ax_gb.set_ylabel("")
    ax_gb.tick_params(labelleft=False)

    # ── Nonce panels ─────────────────────────────────────────────────────────
    _draw_nonce_comparison_panel_two_models(ax_na, model_data, PROPERTY_TASKS,
                                            "Attribute induction",
                                            sig_data=sig_data_prop)
    _draw_nonce_comparison_panel_two_models(ax_nb, model_data, PROC_TASKS,
                                            "Procedural induction",
                                            sig_data=sig_data_proc)
    ax_na.set_ylabel("ECSR  /  Rule verbalization score")
    ax_nb.tick_params(labelleft=False)

    # ── Rotate x-tick labels to prevent overlap in narrow panels ─────────────
    for ax in (ax_ga, ax_gb, ax_na, ax_nb):
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

    # ── Legends ──────────────────────────────────────────────────────────────
    # Gap group legend on panel B (upper left) — panel A is occupied by region labels.
    model_handles = [
        mpatches.Patch(color=_model_color(m, models_8), label=_model_label(m))
        for m in models_8
    ]
    ax_gb.legend(handles=model_handles, loc="upper right",
                 frameon=True, facecolor="white", edgecolor="none",
                 handlelength=0.9, handleheight=0.8,
                 borderpad=0.4, labelspacing=0.3)

    nonce_handles = []
    for sem_key, _, __ in model_data:
        colors = NONCE_MODEL_COLORS[sem_key]
        label  = MODEL_DISPLAY.get(sem_key, sem_key)
        nonce_handles += [
            mpatches.Patch(color=colors["semantic"], alpha=0.88, label=f"{label} (sem.)"),
            mpatches.Patch(color=colors["nonce"],    alpha=0.88, label=f"{label} (nonce)"),
        ]
    nonce_handles.append(
        mpl.lines.Line2D([], [], marker="D", color="w",
                         markerfacecolor="#555555", markeredgecolor="white",
                         markersize=5, linewidth=0, label="Avg. RV"),
    )
    ax_nb.legend(handles=nonce_handles, loc="upper right",
                 frameon=True, facecolor="white", edgecolor="none",
                 handlelength=0.9, handleheight=0.8,
                 borderpad=0.4, labelspacing=0.3)

    # ── Panel labels ─────────────────────────────────────────────────────────
    # ax_ga and ax_na carry y-axis labels → use slightly more negative x to clear them
    _panel_label(ax_ga, "A", x=-0.22)
    _panel_label(ax_gb, "B", x=-0.12)
    _panel_label(ax_na, "C", x=-0.22)
    _panel_label(ax_nb, "D", x=-0.12)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 10 — identifiability coverage sweep (ECSR vs k)
# ---------------------------------------------------------------------------

# Source split sizes per task (identifiability threshold = full src)
_COVERAGE_SRC_SIZE = {
    "additive":      6,
    "compositional": 6,
    "conditional":   8,
    "override":      7,
    "proc_add":      3,
    "proc_comp":     3,
    "proc_cond":     7,
    "proc_over":     9,
}

# Step grids per task (k values tested); 0 = no property demos (baseline)
_COVERAGE_GRIDS = {
    "additive":      [0, 2, 4, 6, 12],
    "compositional": [0, 2, 4, 6, 12],
    "conditional":   [0, 3, 5, 8, 16],
    "override":      [0, 2, 4, 7, 14],
    "proc_add":      [0, 1, 2, 3, 6],
    "proc_comp":     [0, 1, 2, 3, 6],
    "proc_cond":     [0, 2, 4, 7, 14],
    "proc_over":     [0, 3, 6, 9, 18],
}


def _demo_items_from_trace(trace: str) -> set:
    """Items bought inside the demonstration section of an episode trace."""
    m = re.search(
        r'\[Start of Demonstration Episodes\](.*?)\[End of Demonstration Episodes\]',
        trace, re.DOTALL,
    )
    if not m:
        return set()
    return set(re.findall(r'> buy (\S+)', m.group(1)))


def _property_demo_items_from_trace(trace: str) -> set:
    """Items bought only in property demo episodes (entities with class/role attributes).

    Distractor demos feature entities with birthplace/eye_color attributes and buy
    from a different item location — including them inflates the diversity denominator
    with irrelevant items.  Property-only diversity correctly reflects how many
    candidate items the model could have seen for the gen entity's rule.
    """
    m = re.search(
        r'\[Start of Demonstration Episodes\](.*?)\[End of Demonstration Episodes\]',
        trace, re.DOTALL,
    )
    if not m:
        return set()
    items = set()
    for block in re.split(r'=== Episode: defeat \S+ ===', m.group(1)):
        if not block.strip():
            continue
        if 'class:' in block and 'role:' in block:
            buy_m = re.search(r'> buy (\S+)', block)
            if buy_m:
                items.add(buy_m.group(1))
    return items


def _correct_item_from_solution(ref_solution: list) -> str | None:
    for action in ref_solution:
        if action.startswith("buy "):
            return action[4:]
    return None


_STANDARD_ACTIONS = {"go", "buy", "get", "defeat", "rescue"}


def _correct_proc_sequence_from_solution(ref_solution: list) -> tuple | None:
    """Return the hidden procedure as a tuple of (action, 'before'|'after') pairs.

    Position relative to 'buy' is part of the rule (e.g. proc_comp encodes timing via
    role), so two episodes with the same drink/perform action but different positions
    represent different procedures and must not be treated as a match.
    For proc_add repetitions also matter (3× frost_mark ≠ 1× frost_mark).
    """
    try:
        buy_idx = next(i for i, a in enumerate(ref_solution) if a.startswith("buy "))
    except StopIteration:
        buy_idx = len(ref_solution)
    seq = tuple(
        (a, "before" if i < buy_idx else "after")
        for i, a in enumerate(ref_solution)
        if a.split()[0] not in _STANDARD_ACTIONS
    )
    return seq if seq else None


def _proc_demo_sequences_from_trace(trace: str) -> set:
    """Set of position-aware hidden-procedure tuples, one per property demo episode.

    Each tuple element is (action, 'before'|'after') relative to the buy step,
    preserving both which action and when it occurs.
    """
    m = re.search(
        r'\[Start of Demonstration Episodes\](.*?)\[End of Demonstration Episodes\]',
        trace, re.DOTALL,
    )
    if not m:
        return set()
    seqs = set()
    for block in re.split(r'=== Episode: defeat \S+ ===', m.group(1)):
        if not block.strip() or "class:" not in block or "role:" not in block:
            continue
        lines = [l.strip()[2:] for l in block.splitlines()
                 if l.strip().startswith("> ")]
        try:
            buy_idx = next(i for i, l in enumerate(lines) if l.startswith("buy "))
        except StopIteration:
            buy_idx = len(lines)
        seq = tuple(
            (l, "before" if i < buy_idx else "after")
            for i, l in enumerate(lines)
            if l.split()[0] not in _STANDARD_ACTIONS
        )
        if seq:
            seqs.add(seq)
    return seqs


def _variant_ecsr(var: dict, floor: float) -> float:
    """ECSR for a single variant (pooled across its episodes)."""
    succs, effs = [], []
    for ep in var.get("episodes", {}).values():
        success = bool(ep.get("success", False))
        succs.append(success)
        if success:
            eff = _new_efficiency(ep)
            if eff is not None:
                effs.append(eff)
    if not succs:
        return 0.0
    sr       = float(np.mean(succs))
    eff_mean = float(np.mean(effs)) if effs else 0.0
    norm_eff = max((eff_mean - floor) / (1.0 - floor), 0.0) if eff_mean > 0 else 0.0
    return sr * norm_eff


def _bootstrap_ci(
    values: list[float],
    n_boot: int = 1000,
    ci: float = 95,
) -> tuple[float, float]:
    """Bootstrap percentile CI over a list of per-variant ECSR values."""
    if len(values) < 2:
        m = float(np.mean(values)) if values else 0.0
        return m, m
    rng = np.random.default_rng(42)
    boot = [float(np.mean(rng.choice(values, size=len(values), replace=True)))
            for _ in range(n_boot)]
    lo = float(np.percentile(boot, (100 - ci) / 2))
    hi = float(np.percentile(boot, 100 - (100 - ci) / 2))
    return lo, hi


def _get_variant_ecsrs_dict(edf: pd.DataFrame, model: str, task: str) -> dict:
    """Per-variant ECSR keyed by variant_seed, for one model × task."""
    sub = edf[(edf["model"] == model) & (edf["task"] == task)]
    if sub.empty:
        return {}
    floor = 1.0 / TASK_MAX_TRIES.get(task, 2)
    result = {}
    for seed, vg in sub.groupby("variant_seed"):
        sr = float(vg["success"].mean())
        effs = vg["efficiency"].dropna().values
        eff_mean = float(effs.mean()) if len(effs) > 0 else 0.0
        norm_eff = max((eff_mean - floor) / (1.0 - floor), 0.0) if eff_mean > 0 else 0.0
        result[seed] = sr * norm_eff
    return result


def _bootstrap_mean_ci(values: list[float], n_boot: int = 2000) -> tuple[float, float]:
    """Bootstrap 95% percentile CI for mean(values). Returns (lo, hi)."""
    if len(values) < 2:
        m = float(np.mean(values)) if values else 0.0
        return m, m
    arr = np.array(values)
    rng = np.random.default_rng(42)
    boot = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _permutation_p_paired(x: list[float], y: list[float], n_perm: int = 5000) -> float:
    """Sign-permutation test on paired differences; two-sided p-value."""
    d = np.array(x) - np.array(y)
    observed = abs(d.mean())
    rng = np.random.default_rng(42)
    count = sum(
        1 for _ in range(n_perm)
        if abs((rng.choice([-1, 1], size=len(d)) * d).mean()) >= observed
    )
    return (count + 1) / (n_perm + 1)


def _sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def _apply_bonferroni(
    gap_sig_dicts: list[dict],   # each is model → task → {p, stars, ...} | None
    nonce_sig_dicts: list[dict], # each is task → {p_ecsr, p_rv, ecsr_stars, rv_stars, ...} | None
) -> None:
    """Bonferroni correction in-place across all gap and nonce tests jointly.
    Collects every finite p-value, multiplies by family size, rewrites stars."""
    # Gather all (container, key, p_key, stars_key) references
    refs = []   # (dict_entry, p_field, stars_field)

    for sig in gap_sig_dicts:
        for model_dict in sig.values():
            for entry in model_dict.values():
                if entry is not None and not np.isnan(entry.get("p", float("nan"))):
                    refs.append((entry, "p", "stars"))

    for sig in nonce_sig_dicts:
        for entry in sig.values():
            if entry is None:
                continue
            if not np.isnan(entry.get("p_ecsr", float("nan"))):
                refs.append((entry, "p_ecsr", "ecsr_stars"))
            if not np.isnan(entry.get("p_rv", float("nan"))):
                refs.append((entry, "p_rv", "rv_stars"))

    n = len(refs)
    if n == 0:
        return

    for entry, p_key, stars_key in refs:
        corrected = min(1.0, entry[p_key] * n)
        entry[stars_key] = _sig_stars(corrected)


def _permutation_p_zero(values: list[float], n_perm: int = 5000) -> float:
    """One-sample sign-permutation test: H0: mean(values) = 0. Two-sided."""
    d = np.array(values)
    observed = abs(d.mean())
    rng = np.random.default_rng(42)
    count = sum(
        1 for _ in range(n_perm)
        if abs((rng.choice([-1, 1], size=len(d)) * d).mean()) >= observed
    )
    return (count + 1) / (n_perm + 1)


def _get_variant_instance_acc_dict(qdf: pd.DataFrame, model: str, task: str) -> dict:
    """Per-variant mean instance accuracy keyed by variant_seed."""
    sub = qdf[(qdf["model"] == model) & (qdf["task"] == task)]
    if sub.empty:
        return {}
    return {seed: float(vg["instance_accuracy"].mean())
            for seed, vg in sub.groupby("variant_seed")
            if not vg["instance_accuracy"].isna().all()}


def _compute_gap_sig_data(
    edf: pd.DataFrame,
    qdf: pd.DataFrame,
    models: list[str],
    tasks: list[str],
) -> dict[str, dict[str, dict | None]]:
    """Per-variant (QA instance accuracy − ECSR) gap: bootstrap CI + one-sample permutation p.
    Returns nested dict: model → task → {ci_lo, ci_hi, p, stars}."""
    result: dict = {}
    for model in models:
        result[model] = {}
        for task in tasks:
            ecsr_dict = _get_variant_ecsrs_dict(edf, model, task)
            acc_dict  = _get_variant_instance_acc_dict(qdf, model, task)
            if not ecsr_dict or not acc_dict:
                result[model][task] = None
                continue
            common = sorted(set(ecsr_dict) & set(acc_dict))
            if len(common) < 4:
                result[model][task] = None
                continue
            gaps = [acc_dict[s] - ecsr_dict[s] for s in common]
            ci_lo, ci_hi = _bootstrap_mean_ci(gaps)
            p = _permutation_p_zero(gaps)
            result[model][task] = dict(ci_lo=ci_lo, ci_hi=ci_hi, p=p, stars=_sig_stars(p))
    return result


def _get_variant_rv_dict(qdf: pd.DataFrame, model: str, task: str) -> dict:
    """Per-variant mean RV score keyed by variant_seed."""
    sub = qdf[(qdf["model"] == model) & (qdf["task"] == task)]
    if sub.empty:
        return {}
    return {seed: float(vg["rule_score"].mean())
            for seed, vg in sub.groupby("variant_seed")}


def _compute_nonce_sig(
    edf: pd.DataFrame,
    qdf: pd.DataFrame,
    sem_key: str,
    nonce_key: str,
    tasks: list[str],
) -> dict[str, dict | None]:
    """Per-task permutation p-values for the semantic vs nonce gap on ECSR and RV."""
    result = {}
    for task in tasks:
        # ECSR
        sem_e = _get_variant_ecsrs_dict(edf, sem_key, task)
        noc_e = _get_variant_ecsrs_dict(edf, nonce_key, task)
        # RV
        sem_r = _get_variant_rv_dict(qdf, sem_key, task)
        noc_r = _get_variant_rv_dict(qdf, nonce_key, task)

        if not sem_e or not noc_e:
            result[task] = None
            continue

        common_e = sorted(set(sem_e) & set(noc_e))
        p_ecsr = (_permutation_p_paired([sem_e[s] for s in common_e],
                                        [noc_e[s] for s in common_e])
                  if len(common_e) >= 4 else float("nan"))

        common_r = sorted(set(sem_r) & set(noc_r))
        p_rv = (_permutation_p_paired([sem_r[s] for s in common_r],
                                      [noc_r[s] for s in common_r])
                if len(common_r) >= 4 else float("nan"))

        sem_ci = _bootstrap_mean_ci(list(sem_e.values()))
        noc_ci = _bootstrap_mean_ci(list(noc_e.values()))
        result[task] = dict(
            sem_ci=sem_ci, noc_ci=noc_ci,
            p_ecsr=p_ecsr, p_rv=p_rv,
            ecsr_stars=_sig_stars(p_ecsr),
            rv_stars=_sig_stars(p_rv),
        )
    return result


def compute_aggregated_tests(
    data: dict,
    gap_models: list[str],
    nonce_pairs: list[tuple[str, str]],
    output_path: Path,
    families: dict[str, list[str]] | None = None,
) -> dict:
    """Aggregated one-sample and paired permutation tests for the two general claims.

    Claim 1 — QA > ECSR: pool per-variant gaps across all tasks in each family,
    run one-sample sign-permutation test per model × family.

    Claim 2 — Semantic > Nonce: pool per-variant (sem − nonce) ECSR and RV differences
    across all tasks in each family, run paired sign-permutation test per pair × family.

    Results saved to output_path as JSON.
    """
    if families is None:
        families = {
            "attribute": PROPERTY_TASKS,
            "procedural": PROC_TASKS,
            "all":        ALL_TASKS,
        }

    edf = data["episodes"]
    qdf = data["qa"]
    results = {"gap": {}, "nonce": {}}

    # ── Claim 1: QA gap ───────────────────────────────────────────────────────
    for model in gap_models:
        results["gap"][model] = {}
        for fam_name, tasks in families.items():
            pooled_gaps = []
            for task in tasks:
                ecsr_d = _get_variant_ecsrs_dict(edf, model, task)
                acc_d  = _get_variant_instance_acc_dict(qdf, model, task)
                common = sorted(set(ecsr_d) & set(acc_d))
                pooled_gaps.extend(acc_d[s] - ecsr_d[s] for s in common)

            if len(pooled_gaps) < 4:
                results["gap"][model][fam_name] = None
                continue

            mean_gap = float(np.mean(pooled_gaps))
            ci_lo, ci_hi = _bootstrap_mean_ci(pooled_gaps)
            p = _permutation_p_zero(pooled_gaps)
            results["gap"][model][fam_name] = dict(
                n=len(pooled_gaps),
                mean=round(mean_gap, 4),
                ci_95=[round(ci_lo, 4), round(ci_hi, 4)],
                p=round(p, 4),
                stars=_sig_stars(p),
            )

    # ── Claim 2: Nonce vs Semantic ────────────────────────────────────────────
    for sem_key, nonce_key in nonce_pairs:
        pair_label = f"{sem_key}_vs_{nonce_key}"
        results["nonce"][pair_label] = {}
        for fam_name, tasks in families.items():
            pooled_ecsr_diff, pooled_rv_diff = [], []
            for task in tasks:
                sem_e  = _get_variant_ecsrs_dict(edf, sem_key,   task)
                noc_e  = _get_variant_ecsrs_dict(edf, nonce_key, task)
                sem_r  = _get_variant_rv_dict(qdf, sem_key,   task)
                noc_r  = _get_variant_rv_dict(qdf, nonce_key, task)

                for s in sorted(set(sem_e) & set(noc_e)):
                    pooled_ecsr_diff.append(sem_e[s] - noc_e[s])
                for s in sorted(set(sem_r) & set(noc_r)):
                    pooled_rv_diff.append(sem_r[s] - noc_r[s])

            entry = {}
            for metric, diffs in [("ecsr", pooled_ecsr_diff), ("rv", pooled_rv_diff)]:
                if len(diffs) < 4:
                    entry[metric] = None
                    continue
                p = _permutation_p_zero(diffs)   # one-sample on differences
                ci_lo, ci_hi = _bootstrap_mean_ci(diffs)
                entry[metric] = dict(
                    n=len(diffs),
                    mean=round(float(np.mean(diffs)), 4),
                    ci_95=[round(ci_lo, 4), round(ci_hi, 4)],
                    p=round(p, 4),
                    stars=_sig_stars(p),
                )
            results["nonce"][pair_label][fam_name] = entry

    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved aggregated tests: {output_path}")
    return results


def _load_coverage_stats(
    coverage_dir: Path,
    task: str,
    k: int,
    model_stem: str = "gpt-5.4-mini",
) -> dict | None:
    """Load one coverage file and return a stats dict, or None if missing.

    Returns keys: sr, ecsr, bias, variant_ecsrs
      sr            — overall success rate (pooled)
      ecsr          — pooled ECSR
      bias          — avg(in_demo / property_diversity) per episode
      variant_ecsrs — per-variant ECSR list (used for bootstrapped CI)
    """
    fname = coverage_dir / f"phase_1_coverage_{task}_k{k}_{model_stem}.json"
    if not fname.exists():
        return None
    try:
        data = json.loads(fname.read_text())
    except Exception:
        return None

    max_tries = TASK_MAX_TRIES.get(task, 2)
    floor     = 1.0 / max_tries

    successes, effs, bias_scores = [], [], []
    variant_ecsrs = []
    for var in data.get("variants", []):
        variant_ecsrs.append(_variant_ecsr(var, floor))
        for ep in var.get("episodes", {}).values():
            success = bool(ep.get("success", False))
            successes.append(success)
            if success:
                eff = _new_efficiency(ep)
                if eff is not None:
                    effs.append(eff)
            is_proc = task in PROC_TASKS
            ref = ep.get("reference_solution", [])
            trace = ep.get("trace", "")
            if is_proc:
                ci = _correct_proc_sequence_from_solution(ref)
                di = _proc_demo_sequences_from_trace(trace)
            else:
                ci = _correct_item_from_solution(ref)
                di = _property_demo_items_from_trace(trace)
            in_demo = int(bool(ci and ci in di))
            bias_scores.append(in_demo / len(di) if (in_demo and di) else 0.0)

    if not successes:
        return None

    sr          = float(np.mean(successes))
    eff_on_succ = float(np.mean(effs)) if effs else 0.0
    norm_eff    = max((eff_on_succ - floor) / (1.0 - floor), 0.0) if eff_on_succ > 0 else 0.0
    return dict(
        sr            = sr,
        ecsr          = sr * norm_eff,
        bias          = float(np.mean(bias_scores)),
        variant_ecsrs = variant_ecsrs,
    )


def _load_main_benchmark_stats(results_dir: Path, task: str, main_fstem: str) -> dict | None:
    """Load main benchmark results to use as the k=k* anchor in the coverage sweep."""
    fpath = results_dir / main_fstem / f"phase_1_{task}_{main_fstem}.json"
    if not fpath.exists():
        return None
    try:
        data = json.loads(fpath.read_text())
    except Exception:
        return None

    max_tries = TASK_MAX_TRIES.get(task, 2)
    floor     = 1.0 / max_tries

    successes, effs, bias_scores = [], [], []
    variant_ecsrs = []
    for var in data.get("variants", []):
        variant_ecsrs.append(_variant_ecsr(var, floor))
        for ep in var.get("episodes", {}).values():
            success = bool(ep.get("success", False))
            successes.append(success)
            if success:
                eff = _new_efficiency(ep)
                if eff is not None:
                    effs.append(eff)
            is_proc = task in PROC_TASKS
            ref = ep.get("reference_solution", [])
            trace = ep.get("trace", "")
            if is_proc:
                ci = _correct_proc_sequence_from_solution(ref)
                di = _proc_demo_sequences_from_trace(trace)
            else:
                ci = _correct_item_from_solution(ref)
                di = _property_demo_items_from_trace(trace)
            in_demo = int(bool(ci and ci in di))
            bias_scores.append(in_demo / len(di) if (in_demo and di) else 0.0)

    if not successes:
        return None

    sr          = float(np.mean(successes))
    eff_on_succ = float(np.mean(effs)) if effs else 0.0
    norm_eff    = max((eff_on_succ - floor) / (1.0 - floor), 0.0) if eff_on_succ > 0 else 0.0
    return dict(
        sr            = sr,
        ecsr          = sr * norm_eff,
        bias          = float(np.mean(bias_scores)),
        variant_ecsrs = variant_ecsrs,
    )


def _bootstrap_net_gain(
    v0: list[float],
    vstar: list[float],
    n_boot: int = 1000,
    ci: float = 95,
) -> tuple[float, float, float]:
    """Bootstrap CI for mean(vstar) - mean(v0) (unpaired samples).
    Returns (mean_diff, ci_lo, ci_hi)."""
    if not v0 or not vstar:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(42)
    diffs = [
        float(np.mean(rng.choice(vstar, size=len(vstar), replace=True)) -
              np.mean(rng.choice(v0,    size=len(v0),    replace=True)))
        for _ in range(n_boot)
    ]
    mean_diff = float(np.mean(vstar)) - float(np.mean(v0))
    lo = float(np.percentile(diffs, (100 - ci) / 2))
    hi = float(np.percentile(diffs, 100 - (100 - ci) / 2))
    return mean_diff, lo, hi


def _load_coverage_ecsr(
    coverage_dir: Path,
    task: str,
    k: int,
    model_stem: str = "gpt-5.4-mini",
) -> tuple[float | None, float | None]:
    """Return (ecsr, sr) — thin wrapper kept for backward compatibility."""
    s = _load_coverage_stats(coverage_dir, task, k, model_stem)
    if s is None:
        return None, None
    return s["ecsr"], s["sr"]


def make_figure10(
    output_path: Path,
    coverage_entries: list[tuple[str, Path, str, str]] | None = None,
    main_results_dir: Path | None = None,
) -> None:
    """2×4 grid: ECSR vs normalised coverage k/k* for GPT-5.4-mini.

    coverage_entries: list of (display_label, coverage_dir, coverage_stem, main_fstem).
    The k=k* point is loaded from main_results_dir using main_fstem so it matches
    the main benchmark table exactly.
    """
    if coverage_entries is None:
        raise ValueError("coverage_entries required")

    ENTRY_COLORS = ["#4878CF", "#E84646", "#2CA02C", "#FF7F0E"]

    _apply_nature_style()

    fig, axes = plt.subplots(
        2, 4,
        figsize=(NATURE_FULL_W, NATURE_FULL_W * 0.65),
        sharey=False,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.18,
                        hspace=0.60, wspace=0.42)

    row_tasks  = [PROPERTY_TASKS, PROC_TASKS]
    row_labels = ["Attribute induction", "Procedural induction"]

    for row_idx, (task_row, row_label) in enumerate(zip(row_tasks, row_labels)):
        is_property = row_idx == 0
        for col_idx, task in enumerate(task_row):
            ax  = axes[row_idx][col_idx]
            ks  = _COVERAGE_GRIDS[task]
            src = _COVERAGE_SRC_SIZE[task]

            for entry_idx, (label, cov_dir, model_stem, main_fstem) in enumerate(coverage_entries):
                color = ENTRY_COLORS[entry_idx % len(ENTRY_COLORS)]
                valid_ks, ecsrs, biases, variant_ecsrs_list = [], [], [], []
                for k in ks:
                    if k == src and main_results_dir is not None:
                        s = _load_main_benchmark_stats(main_results_dir, task, main_fstem)
                    else:
                        s = _load_coverage_stats(Path(cov_dir), task, k, model_stem)
                    if s is not None:
                        valid_ks.append(k / src)
                        ecsrs.append(s["ecsr"])
                        biases.append(s["bias"])
                        variant_ecsrs_list.append(s["variant_ecsrs"])

                if valid_ks:
                    if any(b > 0 for b in biases):
                        ax.plot(valid_ks, biases, color="#999999", linewidth=0.9,
                                linestyle=":", marker="^", markersize=3, zorder=3)
                    ax.plot(valid_ks, ecsrs, color=color, linewidth=1.3,
                            marker="o", markersize=4, zorder=5)

            ax.axvline(1.0, color="#888888", linewidth=0.8, linestyle=":", zorder=2)
            norm_ticks = [0, 1/3, 2/3, 1, 2]
            ax.set_xticks(norm_ticks)
            ax.set_xticklabels(["0", "⅓", "⅔", "1", "2"])
            ax.set_xlim(-0.1, 2.2)
            ax.set_ylim(-0.02, 1.05)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
            ax.set_title(TASK_DISPLAY[task], pad=3)
            ax.set_xlabel(r"Coverage ($k\,/\,k^*$)", labelpad=2)
            if col_idx == 0:
                ax.set_ylabel("ECSR / Bias", labelpad=2)
            _despine(ax)

        axes[row_idx][0].annotate(
            row_label, xy=(-0.55, 0.50), xycoords="axes fraction",
            fontsize=7, fontweight="bold", va="center", ha="left", rotation=90,
        )

    legend_handles = [
        plt.Line2D([0], [0], color=ENTRY_COLORS[i % len(ENTRY_COLORS)], linewidth=1.3,
                   marker="o", markersize=4, label=f"ECSR ({label})")
        for i, (label, *_) in enumerate(coverage_entries)
    ] + [
        plt.Line2D([0], [0], color="#999999", linewidth=0.9, linestyle=":",
                   marker="^", markersize=3, label="Bias"),
        plt.Line2D([0], [0], color="#888888", linewidth=0.8, linestyle=":",
                   label=r"Identifiability threshold ($k^*$)"),
    ]
    fig.legend(handles=legend_handles,
               loc="lower center", bbox_to_anchor=(0.50, -0.02),
               ncol=len(legend_handles), frameon=False, handlelength=1.4,
               borderpad=0, columnspacing=1.2)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_figure10b(
    output_path: Path,
    coverage_entries: list[tuple[str, Path, str, str]] | None = None,
    main_results_dir: Path | None = None,
) -> None:
    """4×2 single-column figure: ECSR vs k/k* for GPT and GPT-OSS.
    Rows: [A-Add, A-Comp], [A-Cond, A-Over], [P-Add, P-Comp], [P-Cond, P-Over]."""
    if coverage_entries is None:
        raise ValueError("coverage_entries required")

    # Palette-consistent colours (match MODEL_PALETTE indices for gpt5, gptoss)
    ENTRY_COLORS = ["#4878CF", "#6BAF92"]   # blue=GPT, sage=GPT-OSS

    _apply_nature_style()

    task_grid = [
        [PROPERTY_TASKS[0], PROPERTY_TASKS[1]],
        [PROPERTY_TASKS[2], PROPERTY_TASKS[3]],
        [PROC_TASKS[0],     PROC_TASKS[1]],
        [PROC_TASKS[2],     PROC_TASKS[3]],
    ]

    fig, axes = plt.subplots(4, 2, figsize=(3.25, 6.5), sharey=False)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.96, bottom=0.10,
                        hspace=0.72, wspace=0.44)

    for row_idx, task_pair in enumerate(task_grid):
        for col_idx, task in enumerate(task_pair):
            ax  = axes[row_idx][col_idx]
            ks  = _COVERAGE_GRIDS[task]
            src = _COVERAGE_SRC_SIZE[task]

            for entry_idx, (label, cov_dir, model_stem, main_fstem) in enumerate(coverage_entries):
                color = ENTRY_COLORS[entry_idx % len(ENTRY_COLORS)]
                valid_ks, ecsrs, biases = [], [], []
                for k in ks:
                    if k == src and main_results_dir is not None:
                        s = _load_main_benchmark_stats(main_results_dir, task, main_fstem)
                    else:
                        s = _load_coverage_stats(Path(cov_dir), task, k, model_stem)
                    if s is not None:
                        valid_ks.append(k / src)
                        ecsrs.append(s["ecsr"])
                        biases.append(s["bias"])

                if valid_ks:
                    if any(b > 0 for b in biases):
                        ax.plot(valid_ks, biases, color="#999999", linewidth=0.9,
                                linestyle=":", marker="^", markersize=3, zorder=3)
                    ax.plot(valid_ks, ecsrs, color=color, linewidth=1.3,
                            marker="o", markersize=4, zorder=5)

            ax.axvline(1.0, color="#888888", linewidth=0.8, linestyle=":", zorder=2)
            ax.set_xticks([0, 1/3, 2/3, 1, 2])
            ax.set_xticklabels(["0", "⅓", "⅔", "1", "2"])
            ax.set_xlim(-0.1, 2.2)
            ax.set_ylim(-0.02, 1.05)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
            ax.set_title(TASK_DISPLAY[task], pad=3)
            if row_idx == 3:
                ax.set_xlabel(r"Coverage ($k\,/\,k^*$)", labelpad=2)
            if col_idx == 0:
                ax.set_ylabel("ECSR / Bias", labelpad=2)
            _despine(ax)

    # Legend below the figure
    legend_handles = [
        plt.Line2D([0], [0], color=ENTRY_COLORS[i % len(ENTRY_COLORS)], linewidth=1.3,
                   marker="o", markersize=4, label=label)
        for i, (label, *_) in enumerate(coverage_entries)
    ] + [
        plt.Line2D([0], [0], color="#999999", linewidth=0.9, linestyle=":",
                   marker="^", markersize=3, label="Bias"),
        plt.Line2D([0], [0], color="#888888", linewidth=0.8, linestyle=":",
                   label=r"$k^*$ threshold"),
    ]
    fig.legend(handles=legend_handles,
               loc="lower center", bbox_to_anchor=(0.57, 0.01),
               ncol=len(legend_handles), frameon=False,
               handlelength=1.2, columnspacing=1.0, borderpad=0, fontsize=6.5)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_figure10c(
    output_path: Path,
    coverage_entries: list[tuple[str, Path, str, str]] | None = None,
    main_results_dir: Path | None = None,
) -> None:
    """2×4 horizontal figure: ECSR vs k/k* for GPT and GPT-OSS.
    Row 0: A-Add, A-Comp, A-Cond, A-Over
    Row 1: P-Add, P-Comp, P-Cond, P-Over"""
    if coverage_entries is None:
        raise ValueError("coverage_entries required")

    ENTRY_COLORS = ["#4878CF", "#6BAF92"]   # blue=GPT, sage=GPT-OSS

    _apply_nature_style()

    task_grid = [
        [PROPERTY_TASKS[0], PROPERTY_TASKS[1], PROPERTY_TASKS[2], PROPERTY_TASKS[3]],
        [PROC_TASKS[0],     PROC_TASKS[1],     PROC_TASKS[2],     PROC_TASKS[3]],
    ]

    # sharey='row': panels in the same row share y limits — allows hiding
    # redundant interior y-tick labels and shrinking wspace substantially.
    fig, axes = plt.subplots(2, 4, figsize=(6.5, 2.3), sharey="row")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.97, bottom=0.18,
                        hspace=0.42, wspace=0.12)

    for row_idx, task_row in enumerate(task_grid):
        for col_idx, task in enumerate(task_row):
            ax  = axes[row_idx][col_idx]
            ks  = _COVERAGE_GRIDS[task]
            src = _COVERAGE_SRC_SIZE[task]

            for entry_idx, (label, cov_dir, model_stem, main_fstem) in enumerate(coverage_entries):
                color = ENTRY_COLORS[entry_idx % len(ENTRY_COLORS)]
                valid_ks, ecsrs, biases = [], [], []
                for k in ks:
                    if k == src and main_results_dir is not None:
                        s = _load_main_benchmark_stats(main_results_dir, task, main_fstem)
                    else:
                        s = _load_coverage_stats(Path(cov_dir), task, k, model_stem)
                    if s is not None:
                        valid_ks.append(k / src)
                        ecsrs.append(s["ecsr"])
                        biases.append(s["bias"])

                if valid_ks:
                    if any(b > 0 for b in biases):
                        ax.plot(valid_ks, biases, color="#999999", linewidth=0.9,
                                linestyle=":", marker="^", markersize=3, zorder=3)
                    ax.plot(valid_ks, ecsrs, color=color, linewidth=1.3,
                            marker="o", markersize=4, zorder=5)

            ax.axvline(1.0, color="#888888", linewidth=0.8, linestyle=":", zorder=2)
            ax.set_xticks([0, 1/3, 2/3, 1, 2])
            ax.set_xticklabels(["0", "⅓", "⅔", "1", "2"])
            ax.set_xlim(-0.1, 2.2)
            ax.set_ylim(-0.02, 1.05)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
            # Task label as inset annotation — no title line overhead
            ax.text(0.05, 0.95, TASK_DISPLAY[task],
                    transform=ax.transAxes, fontsize=6, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="#aaaaaa", linewidth=0.5, alpha=0.85),
                    zorder=6)
            if col_idx != 0:
                ax.tick_params(labelleft=False)
            _despine(ax)

    # y-label centered on the plot area (bottom=0.18, top=0.97 → midpoint=0.575)
    fig.supylabel("ECSR / Bias", fontsize=7, x=0.01, y=0.575)

    # Legend and x-label share the same bottom line:
    # x-label centered, legend right-aligned
    legend_handles = [
        plt.Line2D([0], [0], color=ENTRY_COLORS[i % len(ENTRY_COLORS)], linewidth=1.3,
                   marker="o", markersize=4, label=label)
        for i, (label, *_) in enumerate(coverage_entries)
    ] + [
        plt.Line2D([0], [0], color="#999999", linewidth=0.9, linestyle=":",
                   marker="^", markersize=3, label="Bias"),
        plt.Line2D([0], [0], color="#888888", linewidth=0.8, linestyle=":",
                   label=r"$k^*$ threshold"),
    ]
    fig.text(0.5, 0.01, r"Coverage ($k\,/\,k^*$)",
             ha="center", va="bottom", fontsize=7)
    fig.legend(handles=legend_handles,
               loc="lower right", bbox_to_anchor=(0.98, 0.0),
               ncol=len(legend_handles), frameon=False,
               handlelength=1.2, columnspacing=1.0, borderpad=0, fontsize=6.5)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_visualization(
    results_dir: str,
    models: list[str],
    tasks: list[str],
    include_humans: bool,
    output_dir: str | None,
) -> None:
    results_dir = Path(results_dir)
    output_dir  = Path(output_dir) if output_dir else Path("analysis/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # nonce variants are only used for fig9 — exclude from all other figures
    _nonce_only = {"gpt5_nonce", "qwen_nonce"}
    plot_models = [m for m in models if m not in _nonce_only]

    print(f"Loading data — models: {models}  tasks: {tasks}  humans: {include_humans}")
    data = load_all_data(results_dir, models, include_humans=include_humans)
    print(
        f"  Episodes: {len(data['episodes'])} rows  |  "
        f"QA: {len(data['qa'])} rows"
    )

    print("Generating figures …")
    make_figure1(data, plot_models, output_dir / "fig1_success_efficiency.pdf",
                 include_humans=include_humans)
    make_figure1b(data, plot_models, output_dir / "fig1b_sr_efficiency_split.pdf",
                  include_humans=include_humans)
    make_figure1b_normeff(data, plot_models, output_dir / "fig1b_normeff.pdf",
                          include_humans=include_humans)
    make_figure2(data, plot_models, output_dir / "fig2_mean_efficiency.pdf",
                 include_humans=include_humans)
    make_figure3(data, plot_models, tasks,  output_dir / "fig3_rule_score.pdf")
    make_figure3b(data, plot_models, tasks, output_dir / "fig3b_subscores.pdf")
    make_figure4(data, plot_models, tasks,  output_dir / "fig4_efficiency_vs_rule.pdf")
    make_figure4_combined(data, plot_models, tasks, output_dir / "fig4_combined_efficiency_vs_rule.pdf")
    make_figure5(data, plot_models, tasks,  output_dir / "fig5_instance_vs_rule.pdf")
    make_figure6(data, plot_models, tasks,  output_dir / "fig6_efficiency_vs_instance.pdf")
    make_figure4b(data, plot_models, tasks, output_dir / "fig4b_efficiency_vs_rule_per_model.pdf")
    make_figure5b(data, plot_models, tasks, output_dir / "fig5b_instance_vs_rule_per_model.pdf")
    make_figure6b(data, plot_models, tasks, output_dir / "fig6b_efficiency_vs_instance_per_model.pdf")
    make_figure7(data, plot_models,        output_dir / "fig7_format_comparison.pdf")
    make_figure7b(data, plot_models,       output_dir / "fig7b_format_gap.pdf")
    make_figure8_comparison(data, plot_models, output_dir / "fig8_comparison.pdf")
    nonce_pairs = []
    if "gpt5" in models and "gpt5_nonce" in models:
        nonce_pairs.append(("gpt5", "gpt5_nonce"))
    if "qwen27b" in models and "qwen_nonce" in models:
        nonce_pairs.append(("qwen27b", "qwen_nonce"))
    if nonce_pairs:
        make_figure9(data, output_dir / "fig9_nonce_comparison.pdf",
                     model_pairs=nonce_pairs)
    _fig8_models = [m for m in plot_models if m in {"qwen27b", "gpt5"}]
    if _fig8_models:
        make_figure8(data, _fig8_models, output_dir / "fig8_gap_ecsr.pdf")
    # Coverage sweep (fig10) — only include models present in plot_models
    _cov_entries = []
    for _m, (_lbl, _subdir, _stem, _main_fstem) in COVERAGE_MODEL_MAP.items():
        if _m in plot_models and (results_dir / _subdir).exists():
            _cov_entries.append((_lbl, results_dir / _subdir, _stem, _main_fstem))
    if _cov_entries:
        make_figure10(output_dir / "fig10_coverage_sweep.pdf",
                      coverage_entries=_cov_entries,
                      main_results_dir=results_dir)
    # fig10b: 4×2 single-column, GPT + GPT-OSS only
    _cov_entries_b = []
    for _m in ("gpt5", "gptoss"):
        if _m in COVERAGE_MODEL_MAP:
            _lbl, _subdir, _stem, _main_fstem = COVERAGE_MODEL_MAP[_m]
            if (results_dir / _subdir).exists():
                _cov_entries_b.append((_lbl, results_dir / _subdir, _stem, _main_fstem))
    if _cov_entries_b:
        make_figure10b(output_dir / "fig10b_coverage_sweep.pdf",
                       coverage_entries=_cov_entries_b,
                       main_results_dir=None)
        make_figure10c(output_dir / "fig10c_coverage_sweep.pdf",
                       coverage_entries=_cov_entries_b,
                       main_results_dir=None)
    # ECSR variants
    make_figure2_ecsr(data, plot_models,   output_dir / "fig2_ecsr.pdf",
                      include_humans=include_humans)
    make_figure4_ecsr(data, plot_models, tasks,  output_dir / "fig4_ecsr.pdf")
    make_figure4_combined_ecsr(data, plot_models, tasks,
                                     output_dir / "fig4_combined_ecsr.pdf")
    make_figure4b_ecsr(data, plot_models, tasks, output_dir / "fig4b_ecsr.pdf")
    make_figure6_ecsr(data, plot_models, tasks,  output_dir / "fig6_ecsr.pdf")
    make_figure6b_ecsr(data, plot_models, tasks, output_dir / "fig6b_ecsr.pdf")
    save_metrics_csv(data, plot_models, tasks, output_dir / "metrics.csv")
    print("ECSR–RV correlations by task family:")
    print_family_correlations(data, plot_models)
    print(f"Done. Figures saved to {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir",   default="results")
    parser.add_argument("--models",        default="qwen27b,gpt5,gptoss,llama,gemini",
                        help="Comma-separated model names")
    parser.add_argument("--tasks",         default=",".join(ALL_TASKS),
                        help="Comma-separated task names (default: property tasks)")
    parser.add_argument("--include_humans", action="store_true", default=False)
    parser.add_argument("--output_dir",    default=None)
    parser.add_argument("--only",          default=None,
                        help="Regenerate only this figure, e.g. fig9")
    args = parser.parse_args()

    models     = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks      = [t.strip() for t in args.tasks.split(",") if t.strip()]
    output_dir = Path(args.output_dir) if args.output_dir else Path("analysis/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(args.results_dir)

    if args.only == "fig9":
        data = load_all_data(results_dir, models, include_humans=False)
        nonce_pairs = []
        if "gpt5" in models and "gpt5_nonce" in models:
            nonce_pairs.append(("gpt5", "gpt5_nonce"))
        if "qwen27b" in models and "qwen_nonce" in models:
            nonce_pairs.append(("qwen27b", "qwen_nonce"))
        make_figure9(data, output_dir / "fig9_nonce_comparison.pdf",
                     model_pairs=nonce_pairs)
    elif args.only == "fig8":
        data = load_all_data(results_dir, models, include_humans=False)
        _fig8_models = [m for m in models if m in {"qwen27b", "gpt5"}]
        if _fig8_models:
            make_figure8(data, _fig8_models, output_dir / "fig8_gap_ecsr.pdf")
        else:
            print("Neither qwen27b nor gpt5 found in models list")
    elif args.only == "fig10":
        _cov_entries = []
        for _m, (_lbl, _subdir, _stem, _main_fstem) in COVERAGE_MODEL_MAP.items():
            if _m in models and (results_dir / _subdir).exists():
                _cov_entries.append((_lbl, results_dir / _subdir, _stem, _main_fstem))
        if _cov_entries:
            make_figure10(output_dir / "fig10_coverage_sweep.pdf",
                          coverage_entries=_cov_entries,
                          main_results_dir=results_dir)
        _cov_entries_b = []
        for _m in ("gpt5", "gptoss"):
            if _m in COVERAGE_MODEL_MAP:
                _lbl, _subdir, _stem, _main_fstem = COVERAGE_MODEL_MAP[_m]
                if (results_dir / _subdir).exists():
                    _cov_entries_b.append((_lbl, results_dir / _subdir, _stem, _main_fstem))
        if _cov_entries_b:
            make_figure10b(output_dir / "fig10b_coverage_sweep.pdf",
                           coverage_entries=_cov_entries_b,
                           main_results_dir=None)
            make_figure10c(output_dir / "fig10c_coverage_sweep.pdf",
                           coverage_entries=_cov_entries_b,
                           main_results_dir=None)
        else:
            print("No coverage dirs found under", results_dir)
    else:
        run_visualization(
            results_dir   = args.results_dir,
            models        = models,
            tasks         = tasks,
            include_humans= args.include_humans,
            output_dir    = str(output_dir),
        )


if __name__ == "__main__":
    main()
