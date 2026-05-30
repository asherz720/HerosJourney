"""
eval/result.py
Typed episode result — the single type shared between the runner and eval packages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EpisodeResult:
    """Complete result for one evaluation episode.

    Produced directly by pipeline.adventure_episode.run_single_episode().
    Can also be reconstructed from a saved JSON dict via EpisodeResult.from_dict().
    """

    # --- Identity ---
    episode_idx: int
    task_type: str
    split: Optional[str]

    # --- Outcome ---
    success: bool = False
    terminated: bool = False
    num_runs: int = 0
    reference_length: int = 0
    reference_solution: List[str] = field(default_factory=list)
    # ref_length / num_runs for successful episodes; None for failures.
    efficiency: Optional[float] = None

    # --- Budget (needed for ECSR floor: 1/num_tries) ---
    num_tries: int = 2

    # --- Traces ---
    full_trace: str = ""
    action_history: List[str] = field(default_factory=list)
    action_obs_reasoning_history: List[Dict[str, Any]] = field(default_factory=list)
    completion_map: Dict[str, float] = field(default_factory=dict)

    # --- Token usage (cost tracking) ---
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # --- Teaching / hypothesis (strategies) ---
    teaching_message: str = ""
    hypothesis_history: List[str] = field(default_factory=list)

    @property
    def completion_rate(self) -> float:
        cm = self.completion_map
        return sum(cm.values()) / len(cm) if cm else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dict compatible with the results files."""
        return {
            "index":            self.episode_idx,
            "task_type":        self.task_type,
            "split":            self.split,
            "success":          self.success,
            "terminated":       self.terminated,
            "num_runs":         self.num_runs,
            "reference_length": self.reference_length,
            "reference_solution": self.reference_solution,
            "efficiency":       self.efficiency,
            "num_tries":        self.num_tries,
            "completion_rate":  self.completion_rate,
            "completion_map":   self.completion_map,
            "trace":            self.full_trace,
            "actions":          self.action_history,
            "action_obs_reasoning_history": self.action_obs_reasoning_history,
            "prompt_tokens":    self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "teaching_message": self.teaching_message,
            "hypothesis_history": self.hypothesis_history,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EpisodeResult":
        """Reconstruct from a dict produced by to_dict() or a legacy results JSON."""
        num_runs = d.get("num_runs", 0) or 0
        ref_len  = d.get("reference_length", 0) or 0
        success  = bool(d.get("success", False))
        eff = (ref_len / num_runs) if (success and num_runs > 0 and ref_len > 0) else None

        return cls(
            episode_idx=d.get("index", d.get("episode_idx", 0)),
            task_type=d.get("task_type", ""),
            split=d.get("split"),
            success=success,
            terminated=bool(d.get("terminated", d.get("terminate", False))),
            num_runs=num_runs,
            reference_length=ref_len,
            reference_solution=d.get("reference_solution", []),
            efficiency=eff,
            num_tries=d.get("num_tries", 2),
            full_trace=d.get("trace", d.get("full_trace", "")),
            action_history=d.get("actions", d.get("action_history", [])),
            action_obs_reasoning_history=d.get("action_obs_reasoning_history", []),
            completion_map=d.get("completion_map", {}),
            prompt_tokens=d.get("prompt_tokens", d.get("_prompt_tokens", 0)),
            completion_tokens=d.get("completion_tokens", d.get("_completion_tokens", 0)),
            teaching_message=d.get("teaching_message", ""),
            hypothesis_history=d.get("hypothesis_history", []),
        )
