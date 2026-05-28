"""
Evaluation metrics for adventure story episodes.

Extracted from eval_error_analysis.ipynb for use in teacher self-refinement
and elsewhere. Provides episode-level metrics and step-level analysis
(repetition, observation types).
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd


# --- Episode-level metrics ---

# Terms to detect in reasoning/thinking for generalization (ported from eval_error_analysis.ipynb; extend as needed).
GENERALIZATION_TERMS = ["demon_slayer_sword", "jein", "jein_dain"]

# Maps ref_generalization key → terms relevant to that task type.
# Used by metrics_from_episode_result to compute task-specific metrics (ever_mentioned, ever_attempted).
# Add new task types here as they are introduced.
TASK_TYPE_TERMS: Dict[str, List[str]] = {
    "sub_regular": ["demon_slayer_sword"],
    "sub_nonceword": ["jein", "jein_dain"],
}


def _count_generalization_mentions(
    action_obs_reasoning_history: List[Dict[str, Any]],
    terms: List[str] = GENERALIZATION_TERMS,
) -> Tuple[int, Dict[str, int], List[str]]:
    """
    Count generalization-term mentions in step-level reasoning/thinking.
    Returns (total_mentions, mentions_by_term, terms_mentioned_sorted).
    """
    if not action_obs_reasoning_history or not terms:
        return 0, {}, []

    counts: Dict[str, int] = {t: 0 for t in terms}
    for entry in action_obs_reasoning_history:
        reasoning = entry.get("reasoning", "") or ""
        thinking = entry.get("thinking_traces", "") or ""
        haystack = f"{reasoning}\n{thinking}".lower()
        for t in terms:
            tl = t.lower()
            if not tl:
                continue
            counts[t] += haystack.count(tl)

    total = sum(counts.values())
    mentioned = sorted([t for t, c in counts.items() if c > 0])
    counts = {t: c for t, c in counts.items() if c > 0}
    return total, counts, mentioned


def _compute_substitution_metrics(result: Dict, terms: List[str]) -> Dict:
    """
    Compute substitution-task-specific metrics.

    ever_mentioned: True if any term appears (case-insensitive) in reasoning or
        thinking_traces of any step in action_obs_reasoning_history.
    ever_attempted: True if any term appears (case-insensitive) in any action
        string in action_history.
    """
    if not terms:
        return {"ever_mentioned": False, "ever_attempted": False}
    terms_lower = [t.lower() for t in terms]
    history = result.get("action_obs_reasoning_history", []) or []
    ever_mentioned = False
    for entry in history:
        reasoning = (entry.get("reasoning", "") or "").lower()
        thinking = (entry.get("thinking_traces", "") or "").lower()
        if any(t in reasoning or t in thinking for t in terms_lower):
            ever_mentioned = True
            break
    actions = result.get("action_history", []) or []
    ever_attempted = any(
        any(t in (a or "").lower() for t in terms_lower) for a in actions
    )
    return {"ever_mentioned": ever_mentioned, "ever_attempted": ever_attempted}


def extract_episode_metrics(episodes: Dict) -> Tuple[List[int], List[bool], List[float]]:
    """Extract single-value metrics per episode. Sorted by episode index."""
    episode_indices = []
    success_list = []
    completion_rates = []

    for episode_key, episode_data in episodes.items():
        episode_indices.append(episode_data["index"])

        if "success" in episode_data:
            success_list.append(episode_data["success"])
        else:
            success_list.append(episode_data.get("completion_rate", 0) == 1.0)

        completion_rates.append(episode_data.get("completion_rate", 0.0))

    sorted_data = sorted(zip(episode_indices, success_list, completion_rates))
    episode_indices, success_list, completion_rates = zip(*sorted_data)

    return list(episode_indices), list(success_list), list(completion_rates)


def metrics_from_episode_result(result: Dict, task_type: Optional[str] = None) -> Dict:
    """
    Build a metrics dict from a single episode result (e.g. from run_single_episode).
    Used by teacher dynamic strategies (reflexion, observe_then_teach, ace) via metrics_fn in curate().
    """
    completion_map = result.get("completion_map") or {}
    completion_rate = (
        sum(completion_map.values()) / len(completion_map)
        if completion_map
        else (1.0 if result.get("success") else 0.0)
    )
    actions = result.get("action_history", []) or []
    abs_rep, _, _ = analyze_repetition(actions, repetition_threshold=2)
    repetition_fraction = (len(abs_rep) / len(actions)) if actions else 0.0

    mention_total, mentions_by_term, terms_mentioned = _count_generalization_mentions(
        result.get("action_obs_reasoning_history", []) or []
    )
    metrics = {
        "success": result.get("success", False),
        "completion_rate": completion_rate,
        "num_runs": result.get("num_runs", 0),
        "repetition_fraction": repetition_fraction,
        "generalization_mentions_total": mention_total,
        "generalization_mentions_by_term": mentions_by_term,
        "generalization_terms_mentioned": terms_mentioned,
        "terminate": result.get("terminate", False),
    }
    if task_type is not None and task_type in TASK_TYPE_TERMS:
        metrics.update(_compute_substitution_metrics(result, TASK_TYPE_TERMS[task_type]))
    return metrics


# --- Repetition analysis ---

KNOWN_VERBS = {"go", "buy", "get", "defeat", "rescue", "check"}
CHECK_SUB = {"inventory", "location"}


def analyze_repetition(
    actions: List[str], repetition_threshold: int = 2
) -> Tuple[List[int], List[float], List[str]]:
    """Returns (absolute_positions, normalized_positions, repeated_actions)."""
    absolute_p = []
    normalized_p = []
    repeated_actions = []
    epi_length = len(actions)

    if epi_length < 2:
        return [], [], []

    for i in range(1, epi_length):
        window_start = max(0, i - repetition_threshold)
        previous_in_window = actions[window_start:i]
        current_action = actions[i]

        if current_action in previous_in_window:
            absolute_p.append(i)
            normalized_p.append(i / epi_length)
            repeated_actions.append(current_action)

    return absolute_p, normalized_p, repeated_actions


def classify_repetition_action(action: str) -> str:
    """Parse action and return category: check inventory, check location, go, buy, get, defeat, rescue, or other."""
    s = action.strip().lower()
    if not s:
        return "other"
    parts = s.split(None, 1)
    verb = parts[0]
    if verb == "check" and len(parts) > 1:
        second = parts[1].split(None, 1)[0]
        if second in CHECK_SUB:
            return f"check {second}"
        return "other"
    if verb in KNOWN_VERBS:
        return verb
    return "other"


def extract_repetition_data(
    episodes: Dict,
    repetition_threshold: int = 2,
    include_reference: bool = True,
):
    """
    Extract repetition positions for all episodes, with action type.
    Returns DataFrame if pandas available, else list of dicts.
    """
    data = []
    for episode_key, episode_data in episodes.items():
        episode_idx = episode_data["index"]
        actions = episode_data.get("actions", [])
        reference_solution = (
            episode_data.get("reference_solution", []) if include_reference else []
        )

        abs_positions, norm_positions, repeated_actions = analyze_repetition(
            actions, repetition_threshold
        )
        for abs_pos, norm_pos, action in zip(
            abs_positions, norm_positions, repeated_actions
        ):
            data.append(
                {
                    "episode_index": episode_idx,
                    "absolute_position": abs_pos,
                    "normalized_position": norm_pos,
                    "episode_length": len(actions),
                    "action": action,
                    "action_type": classify_repetition_action(action),
                    "trajectory": "model",
                }
            )

        if reference_solution:
            abs_ref, norm_ref, rep_ref = analyze_repetition(
                reference_solution, repetition_threshold
            )
            for abs_pos, norm_pos, action in zip(abs_ref, norm_ref, rep_ref):
                data.append(
                    {
                        "episode_index": episode_idx,
                        "absolute_position": abs_pos,
                        "normalized_position": norm_pos,
                        "episode_length": len(reference_solution),
                        "action": action,
                        "action_type": classify_repetition_action(action),
                        "trajectory": "reference",
                    }
                )

    if pd is not None:
        return pd.DataFrame(data)
    return data


# --- Observation type classification ---

OBS_TYPES_TO_RECORD = {
    "invalid_action",
    "not_implemented",
    "hidden_location",
    "already_at_location",
    "get_failure",
    "buy_failure",
    "not_enough_currency",
    "requirements_not_met",
    "wrong_location",
    "wrong_kind",
}


def classify_observation_type(observation: str) -> str:
    """
    Classify step outcome from observation text only.
    Categories: invalid_action, not_implemented, hidden_location, already_at_location,
    get_failure, buy_failure, not_enough_currency, requirements_not_met, wrong_location, wrong_kind, other.
    """
    if not observation:
        return "other"
    obs = observation.strip()
    if obs.startswith("Invalid action "):
        return "invalid_action"
    if "Action handler not implemented" in obs:
        return "not_implemented"
    if obs.startswith("Cannot go to ") and "hidden location" in obs:
        return "hidden_location"
    if obs.startswith("Already at "):
        return "already_at_location"
    if obs.startswith("Not enough currency"):
        return "not_enough_currency"
    if "cannot be bought here" in obs:
        return "wrong_location"
    if "is not for sale" in obs:
        return "buy_failure"
    if obs.startswith("Cannot get ") and "It costs" in obs:
        return "get_failure"
    if obs.startswith("Cannot get ") and "but you're at" in obs:
        return "wrong_location"
    if " yet. Wrong " in obs and (
        "Cannot defeat " in obs or "Cannot rescue " in obs
    ):
        return "wrong_kind"
    if " yet. Requirements not met" in obs:
        return "requirements_not_met"
    if ("Cannot defeat " in obs or "Cannot rescue " in obs) and "Requirements not met" in obs:
        return "wrong_location"
    if obs.startswith("Error executing action") or "Episode already finished" in obs:
        return "other"
    return "other"


def analyze_obs(action_obs_reasoning_history: List[Dict]) -> List[Dict]:
    """
    Parse every step; if the observation falls under one of our obs classifications (not "other"),
    mark that step and record its position relative to full episode length.
    """
    if not action_obs_reasoning_history:
        return []
    n = len(action_obs_reasoning_history)
    result = []
    for entry in action_obs_reasoning_history:
        step = entry.get("step", 0)
        action = entry.get("action", "")
        observation = entry.get("observation", "")
        obs_type = classify_observation_type(observation)
        if obs_type not in OBS_TYPES_TO_RECORD:
            continue
        result.append(
            {
                "absolute_position": step,
                "normalized_position": step / n if n else 0,
                "action": action,
                "action_type": classify_repetition_action(action),
                "observation_type": obs_type,
            }
        )
    return result


def extract_obs_data(episodes: Dict):
    """
    For each episode, parse every step; if observation falls under one of our classifications,
    record that step and its position. Returns DataFrame if pandas available, else list of dicts.
    """
    data = []
    for episode_key, episode_data in episodes.items():
        history = episode_data.get("action_obs_reasoning_history", [])
        if not history:
            continue
        episode_idx = episode_data["index"]
        rows = analyze_obs(history)
        for r in rows:
            data.append(
                {
                    "episode_index": episode_idx,
                    "absolute_position": r["absolute_position"],
                    "normalized_position": r["normalized_position"],
                    "episode_length": len(history),
                    "action": r["action"],
                    "action_type": r["action_type"],
                    "observation_type": r["observation_type"],
                    "trajectory": "model",
                }
            )
    if pd is not None:
        return pd.DataFrame(data)
    return data
