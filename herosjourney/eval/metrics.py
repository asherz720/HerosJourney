"""
eval/metrics.py
ECSR (efficiency-calibrated success rate) and supporting helpers.

Formula (Hero's Journey §4.2.2):
    eff       = reference_length / num_runs   (per successful episode)
    floor     = 1 / n_tries
    norm_eff  = max((mean_eff - floor) / (1 - floor), 0)
    ECSR      = success_rate × norm_eff

n_tries is the brute-force enumeration ceiling for the task, stored as
TaskSpec.max_tries.  For the eight built-in tasks:
    additive: 5  compositional: 9  conditional: 6  override: 4
    proc_add: 3  proc_comp: 4     proc_cond: 3    proc_over: 4
"""

from __future__ import annotations

from typing import Optional, Sequence

from herosjourney.eval.result import EpisodeResult


def episode_efficiency(result: EpisodeResult) -> Optional[float]:
    """ref_length / num_runs for a successful episode; None for failures."""
    if not result.success or result.num_runs == 0 or result.reference_length == 0:
        return None
    return result.reference_length / result.num_runs


def success_rate(results: Sequence[EpisodeResult]) -> float:
    """Fraction of successful episodes."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.success) / len(results)


def compute_norm_eff(results: Sequence[EpisodeResult], n_tries: int) -> float:
    """Normalized efficiency (mean over successful episodes, floor-adjusted to [0, 1])."""
    if not results or n_tries <= 1:
        return 0.0
    floor = 1.0 / n_tries
    effs  = [e for r in results if (e := episode_efficiency(r)) is not None]
    eff_mean = sum(effs) / len(effs) if effs else 0.0
    return max((eff_mean - floor) / (1.0 - floor), 0.0)


def compute_ecsr(results: Sequence[EpisodeResult], n_tries: int) -> float:
    """Compute ECSR for a collection of episodes from the same task type.

    Args:
        results: Episode results (should all share the same task type).
        n_tries: Brute-force ceiling for this task (TaskSpec.max_tries).

    Returns:
        ECSR ∈ [0, 1].  Returns 0.0 on empty input.
    """
    if not results:
        return 0.0
    if n_tries <= 1:
        raise ValueError(f"n_tries must be > 1, got {n_tries}")
    return success_rate(results) * compute_norm_eff(results, n_tries)
