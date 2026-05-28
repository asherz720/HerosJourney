"""
Teacher module for the adventure story inductive reasoning benchmark.

Four steering methods for Phase 2:
  react  (static)  — single prompt appended to every episode; no curation.
  ace    (dynamic) — offline playbook built from Phase 1 gen failure traces;
                     frozen playbook used as teaching message in Phase 2.
  hr     (dynamic) — per-episode hypothesis generation call in the episode runner;
                     curate() returns ("", []).
  idea   (dynamic) — I→D→A loop inside the episode runner;
                     curate() returns ("", []).

Phase 1.1 is only meaningful for ACE. HR and IDEA are handled entirely inside
adventure_episode.run_single_episode() when episode_mode is set appropriately.
"""

from typing import List, Tuple, Dict, Any, Optional
import json as _json
import random
import warnings
from dataclasses import dataclass, field

from pipeline.prompts import (
    REACT_STUDENT_PROMPT,
    ACE_REFLECTOR_PROMPT_TEMPLATE,
    ACE_DEDUP_PROMPT_TEMPLATE,
)
from pipeline.models import agent_response, teacher_json_repair_small, teacher_json_repair_gemini

# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STATIC_STRATEGIES  = {"react"}
DYNAMIC_STRATEGIES = {"ace", "hr", "idea"}
ALL_STRATEGIES     = STATIC_STRATEGIES | DYNAMIC_STRATEGIES

# Model path used for ACE reflector calls.
# None means fall back to the student model_path passed in the pipeline.
TEACHER_MODEL_PATH: Optional[str] = None

ACE_CONFIG: Dict[str, Any] = {
    "max_traces": 10,   # max Phase 1 failure traces fed to reflector
    "random_seed": 42,
}

# ---------------------------------------------------------------------------
# Playbook data structures (used by ACE)
# ---------------------------------------------------------------------------

@dataclass
class PlaybookEntry:
    id: str
    content: str
    helpful_count: int = 0
    harmful_count: int = 0


class Playbook:
    """Itemized collection of induction strategy bullets, updated incrementally."""

    def __init__(self):
        self.entries: List[PlaybookEntry] = []
        self._next_id: int = 1

    def _make_id(self) -> str:
        return f"b{self._next_id:03d}"

    def is_empty(self) -> bool:
        return len(self.entries) == 0

    def merge_delta(self, new_bullets: List[str]) -> List[str]:
        new_ids = []
        for content in new_bullets:
            if not content or not content.strip():
                continue
            eid = self._make_id()
            self._next_id += 1
            self.entries.append(PlaybookEntry(id=eid, content=content.strip()))
            new_ids.append(eid)
        return new_ids

    def update_feedback(self, helpful_ids: List[str], harmful_ids: List[str]) -> None:
        id_map = {e.id: e for e in self.entries}
        for bid in helpful_ids:
            if bid in id_map:
                id_map[bid].helpful_count += 1
        for bid in harmful_ids:
            if bid in id_map:
                id_map[bid].harmful_count += 1

    def remove_ids(self, ids_to_remove: List[str]) -> None:
        remove_set = set(ids_to_remove)
        self.entries = [e for e in self.entries if e.id not in remove_set]

    def to_teaching_message(self) -> str:
        if not self.entries:
            return ""
        lines = ["Here are strategies to keep in mind:"]
        for e in self.entries:
            lines.append(f"[{e.id}] {e.content}")
        return "\n".join(lines)

    def to_dict(self) -> List[Dict[str, Any]]:
        return [
            {"id": e.id, "content": e.content,
             "helpful": e.helpful_count, "harmful": e.harmful_count}
            for e in self.entries
        ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _teacher_call(
    teacher_model_path: str, prompt: str
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    response, reasoning, token_counts = agent_response(teacher_model_path, prompt, max_tokens=1024)
    text = response.strip() if (response and isinstance(response, str)) else response
    return text, reasoning, token_counts


_REFLECTOR_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "new_bullets": {"type": "array", "items": {"type": "string"}},
        "helpful_ids": {"type": "array", "items": {"type": "string"}},
        "harmful_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["new_bullets", "helpful_ids", "harmful_ids"],
}

_DEDUP_JSON_SCHEMA = {
    "type": "object",
    "properties": {"remove_ids": {"type": "array", "items": {"type": "string"}}},
    "required": ["remove_ids"],
}


def _repair_json(text: str, schema: dict, converter_model: str) -> str:
    if converter_model == "gemini":
        return teacher_json_repair_gemini(text, schema)
    return teacher_json_repair_small(text, schema)


def _parse_reflector_output(
    text: str, converter_model: str = "small"
) -> Tuple[List[str], List[str], List[str]]:
    def _extract(raw: str) -> Optional[Tuple[List[str], List[str], List[str]]]:
        try:
            start, end = raw.find('{'), raw.rfind('}') + 1
            if start >= 0 and end > start:
                data = _json.loads(raw[start:end])
                return (
                    [b for b in data.get("new_bullets", []) if b and isinstance(b, str)],
                    [i for i in data.get("helpful_ids", []) if i and isinstance(i, str)],
                    [i for i in data.get("harmful_ids", []) if i and isinstance(i, str)],
                )
        except Exception:
            pass
        return None

    result = _extract(text)
    if result is not None:
        return result
    print("Warning: reflector output not valid JSON; attempting repair.")
    try:
        repaired = _repair_json(text, _REFLECTOR_JSON_SCHEMA, converter_model)
        result = _extract(repaired)
        if result is not None:
            return result
    except Exception as e:
        print(f"Warning: JSON repair failed: {e}")
    return [], [], []


def _playbook_deduplicate(
    playbook: Playbook, teacher_model_path: str, converter_model: str = "small"
) -> None:
    if len(playbook.entries) < 2:
        return
    playbook_text = "\n".join(f"[{e.id}] {e.content}" for e in playbook.entries)
    prompt = ACE_DEDUP_PROMPT_TEMPLATE.format(playbook_text=playbook_text)
    response, _, _ = _teacher_call(teacher_model_path, prompt)
    if not response:
        return

    def _extract_remove_ids(raw: str) -> Optional[List[str]]:
        try:
            start, end = raw.find('{'), raw.rfind('}') + 1
            if start >= 0 and end > start:
                data = _json.loads(raw[start:end])
                return [i for i in data.get("remove_ids", []) if i and isinstance(i, str)]
        except Exception:
            pass
        return None

    remove_ids = _extract_remove_ids(response)
    if remove_ids is None:
        try:
            repaired = _repair_json(response, _DEDUP_JSON_SCHEMA, converter_model)
            remove_ids = _extract_remove_ids(repaired)
        except Exception as e:
            print(f"Warning: dedup JSON repair failed: {e}")
    if remove_ids:
        playbook.remove_ids(remove_ids)


# ---------------------------------------------------------------------------
# Static teaching messages
# ---------------------------------------------------------------------------

def build_static_teaching_messages(
    teaching_strategy: str,
    learning_tree_list: List[Tuple[Any, str]],
    **_ignored,
) -> List[str]:
    """Return one teaching message per episode for static strategies."""
    if not learning_tree_list:
        return []
    if teaching_strategy == "react":
        return [REACT_STUDENT_PROMPT] * len(learning_tree_list)
    return [""] * len(learning_tree_list)


# ---------------------------------------------------------------------------
# ACE curation: offline playbook from Phase 1 failures
# ---------------------------------------------------------------------------

def curate_teaching_message_ace(
    phase1_gen_results: Dict,
    teacher_model_path: str,
    max_traces: int = 10,
    random_seed: int = 42,
    converter_model: str = "small",
    verbose: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build a strategy playbook from Phase 1 gen failure traces (no new episodes, no solutions).
    One reflector call per failure trace → frozen playbook used as Phase 2 teaching message.
    """
    random.seed(random_seed)
    gen_episodes = phase1_gen_results.get("episodes", {})
    gen_ep_list = sorted(gen_episodes.values(), key=lambda e: e.get("index", 0))

    failed = [e for e in gen_ep_list if not e.get("success", True)]
    sampled = random.sample(failed, min(max_traces, len(failed))) if failed else []

    if not sampled:
        return "", []

    playbook = Playbook()
    curation_history: List[Dict[str, Any]] = []

    for i, ep in enumerate(sampled):
        trace_text = ep.get("full_trace", "") or ep.get("trace", "")
        if not trace_text:
            continue

        playbook_text = (
            "\n".join(f"[{e.id}] {e.content}" for e in playbook.entries)
            if not playbook.is_empty() else "(empty)"
        )
        prompt = ACE_REFLECTOR_PROMPT_TEMPLATE.format(
            playbook_text=playbook_text,
            trace_text=trace_text,
        )
        response, reasoning, token_counts = _teacher_call(teacher_model_path, prompt)
        new_bullets, helpful_ids, harmful_ids = _parse_reflector_output(
            response or "", converter_model
        )
        new_ids = playbook.merge_delta(new_bullets)
        playbook.update_feedback(helpful_ids, harmful_ids)

        if verbose:
            print(
                f"[ACE] Trace {i+1}/{len(sampled)}: "
                f"+{len(new_ids)} bullets, total={len(playbook.entries)}"
            )

        curation_history.append({
            "stage": "reflect",
            "trace_index": i,
            "new_bullet_ids": new_ids,
            "helpful_ids": helpful_ids,
            "harmful_ids": harmful_ids,
            "teacher_reasoning": reasoning,
            "teacher_token_counts": token_counts,
        })

    if not playbook.is_empty():
        _playbook_deduplicate(playbook, teacher_model_path, converter_model)

    if verbose:
        print(f"[ACE] Final playbook ({len(playbook.entries)} bullets):\n{playbook.to_teaching_message()}")

    return playbook.to_teaching_message(), curation_history


# ---------------------------------------------------------------------------
# Teacher class
# ---------------------------------------------------------------------------

class Teacher:
    """
    Wraps a steering method and its config.
    Static methods (react): get_static_teaching_messages() returns per-episode prompts.
    Dynamic methods:
      ace:  curate() builds a playbook from Phase 1 traces → returns teaching message.
      hr:   curate() returns ("", []) — handled in episode runner (episode_mode="hr").
      idea: curate() returns ("", []) — handled in episode runner (episode_mode="idea").
    """

    def __init__(
        self,
        method: str,
        model_path: Optional[str] = None,
        **strategy_config: Any,
    ):
        self.method = method
        self.model_path = model_path
        self.strategy_config = dict(strategy_config)

    @classmethod
    def from_strategy(
        cls,
        strategy: str,
        teacher_model_path: Optional[str] = None,
    ) -> "Teacher":
        if strategy not in ALL_STRATEGIES:
            raise ValueError(
                f"Unknown teaching strategy '{strategy}'. "
                f"Valid strategies: {sorted(ALL_STRATEGIES)}"
            )
        model_path = teacher_model_path or TEACHER_MODEL_PATH or None
        config: Dict[str, Any] = {}
        if strategy == "ace":
            config = {
                "max_traces": ACE_CONFIG.get("max_traces", 10),
                "random_seed": ACE_CONFIG.get("random_seed", 42),
            }
        return cls(method=strategy, model_path=model_path, **config)

    def get_static_teaching_messages(
        self, learning_tree_list: List[Tuple[Any, str]]
    ) -> List[str]:
        return build_static_teaching_messages(self.method, learning_tree_list)

    def curate(
        self,
        phase1_gen_results: Dict,
        verbose: bool = False,
        converter_model: str = "small",
        **_ignored,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Build teaching message from Phase 1 results.
        ACE: offline reflector pass over failure traces → playbook.
        HR / IDEA: no pre-episode message; episode runner handles these.
        """
        if self.method == "ace":
            return curate_teaching_message_ace(
                phase1_gen_results=phase1_gen_results,
                teacher_model_path=self.model_path or "",
                max_traces=self.strategy_config.get("max_traces", 10),
                random_seed=self.strategy_config.get("random_seed", 42),
                converter_model=converter_model,
                verbose=verbose,
            )
        return "", []


# ---------------------------------------------------------------------------
# RV analysis utility
# ---------------------------------------------------------------------------

def extract_stated_hypothesis(trace: str) -> Optional[str]:
    """
    Extract the student's stated rule hypothesis from a Phase 2 episode trace.
    Looks for lines beginning with 'Hypothesis:' (used by HR, IDEA, ReAct).
    Returns the last such statement found (most refined), or None.
    """
    hypothesis = None
    for line in trace.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("hypothesis:"):
            hypothesis = stripped.split(":", 1)[1].strip()
    return hypothesis
