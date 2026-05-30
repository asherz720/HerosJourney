"""Q&A evaluation: instance prediction and structure explanation with LLM-as-judge scoring."""

from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from herosjourney.core.generator import GeneratedTask
from herosjourney.runner.models import agent_response, json_converter_small, json_converter_gemini
from herosjourney.runner.prompts import (
    format_teaching_block,
    QA_INSTANCE_QUESTION,
    QA_INSTANCE_PROC_QUESTION,
    QA_STRUCTURE_EXP_QUESTION,
    QA_STRUCTURE_EXP_QUESTION_PROC,
    QA_JUDGE_PROMPT,
    QA_JUDGE_PROMPT_PROC,
    QA_TEACHER_PROMPT_TEMPLATE,
)


def _get_correct_rule(task_type: str) -> str:
    from herosjourney.core.registry import get_task
    return get_task(task_type).correct_rule


def _is_proc(task_type: str) -> bool:
    return task_type.startswith("proc_")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_correct_item(task: GeneratedTask) -> Optional[str]:
    """Return the hidden buy-item name for a property task."""
    skip_set = set(task.rules_to_skip)
    for node in task.tree.nodes.values():
        if node.meta.get("incoming_edge") == "buy" and node.argument in skip_set:
            return node.argument
    return None


def _get_correct_process(task: GeneratedTask) -> Dict[str, Any]:
    """Return hidden extra-step info (action, argument, ordering, count) for a procedural task."""
    tree = task.tree
    skip_set = set(task.rules_to_skip)

    extra_entries: List[Tuple[int, str, str]] = []  # (exec_order, action, argument)
    buy_eo: Optional[int] = None

    for node_id, node in tree.nodes.items():
        action = node.meta.get("incoming_edge", "")
        # execution_order is stored on the out-edge (child → parent edge)
        out_idx_list = tree.out_edges.get(node_id, [])
        eo = (tree.edges[out_idx_list[0]].meta.get("execution_order", 99)
              if out_idx_list else 99)

        if action == "buy":
            buy_eo = eo
        elif action in ("perform", "drink") and node.argument in skip_set:
            extra_entries.append((eo, action, node.argument))

    if not extra_entries:
        return {"action": "none", "argument": "none", "ordering": "none", "count": 0}

    first_eo, first_action, first_arg = extra_entries[0]
    if buy_eo is None or first_eo < buy_eo:
        ordering = "before_buy"
    else:
        ordering = "after_buy"

    return {
        "action":   first_action,
        "argument": first_arg,
        "ordering": ordering,
        "count":    len(extra_entries),
    }


def _entity_card(task: GeneratedTask) -> str:
    """Build RPG entity card string from root node properties."""
    root = task.tree.nodes[task.tree.root_id]
    p = root.properties
    parts = [f"{n}: {v}" for n, v in
             zip(p.get("attribute_names", []), p.get("attribute_values", [])) if n and v]
    entity = root.argument
    return f"[ {entity} ]  {'  |  '.join(parts)}" if parts else entity


def _attr_labels(tasks: List[GeneratedTask]) -> List[str]:
    """Return the list of attribute display names from the first task with properties."""
    for task in tasks:
        p = task.tree.nodes[task.tree.root_id].properties
        names = p.get("attribute_names", [])
        if names:
            return names
    return []


def _extract_fields_from_truncated(raw: str, schema: Optional[Dict]) -> Optional[Dict]:
    """
    Last-resort extractor for truncated JSON (e.g. reasoning cut off mid-string).
    Pulls each required field from the raw text using regex.
    Returns a dict of found fields, or None if no required fields could be extracted.
    """
    if not schema:
        return None
    required = schema.get("required", [])
    props    = schema.get("properties", {})
    result: Dict[str, Any] = {}

    for field in props:
        prop_type = props[field].get("type", "string")
        if prop_type == "string":
            m = re.search(rf'"{re.escape(field)}"\s*:\s*"([^"]*)"?', raw)
            if m:
                result[field] = m.group(1)
        elif prop_type == "integer":
            m = re.search(rf'"{re.escape(field)}"\s*:\s*(\d+)', raw)
            if m:
                result[field] = int(m.group(1))

    if all(f in result for f in required):
        return result
    return None


def _call_model(
    prompt: str,
    model: str,
    converter_model: str = "small",
    schema: Optional[Dict] = None,
    verbose: bool = False,
    label: str = "",
    max_tokens: int = 1024,
) -> Tuple[Optional[Dict], Optional[str]]:
    """Model call with JSON parse fallback chain: direct → converter → regex extraction."""
    if verbose:
        prefix = f"[QA:{label}] " if label else "[QA] "
        print(f"\n{prefix}{'=' * 60}")
        print(f"{prefix}PROMPT:")
        print(f"{prefix}{'=' * 60}")
        print(prompt)
        print(f"{prefix}{'=' * 60}")

    try:
        raw, _, _ = agent_response(model, prompt, max_tokens=max_tokens)
    except Exception as e:
        print(f"[QA] API error: {e}")
        return None, None

    if verbose:
        prefix = f"[QA:{label}] " if label else "[QA] "
        print(f"{prefix}RESPONSE:")
        print(raw)
        print(f"{prefix}{'=' * 60}\n")

    if not raw or not raw.strip():
        return None, raw

    # 1. Direct parse
    try:
        return json.loads(raw), raw
    except Exception:
        pass

    # 2. Converter fallback
    if schema:
        try:
            converted = (
                json_converter_gemini(raw, schema)
                if converter_model == "gemini"
                else json_converter_small(raw, schema)
            )
            return json.loads(converted), raw
        except Exception:
            pass

    # 3. Regex extraction (handles truncated responses)
    extracted = _extract_fields_from_truncated(raw, schema)
    if extracted is not None:
        print(f"[QA] Used regex extraction for truncated response ({label})", file=sys.stderr)
        return extracted, raw

    print(f"[QA] JSON parse failed (all methods) | response: {raw[:200]}")
    return None, raw


def _soft_match(predicted, correct) -> bool:
    """Case-insensitive, int/str-tolerant equality check."""
    if predicted is None:
        return False
    try:
        if isinstance(correct, int):
            return int(predicted) == correct
    except (ValueError, TypeError):
        pass
    return str(predicted).strip().lower() == str(correct).strip().lower()


def _normalize_ordering(s: str) -> str:
    """Normalize ordering string to 'before_buy', 'after_buy', or 'none'.
    Handles spaces, hyphens, and common model phrasings."""
    if not s:
        return "none"
    s = s.strip().lower().replace(" ", "_").replace("-", "_")
    if "before" in s:
        return "before_buy"
    if "after" in s:
        return "after_buy"
    if s in ("none", "n/a", "na", "0"):
        return "none"
    return s  # leave unrecognised strings as-is so _soft_match still catches exact matches


def _normalize_action(s: str) -> str:
    """Lowercase-strip action string; accept common aliases."""
    if not s:
        return "none"
    s = s.strip().lower()
    return s


# ---------------------------------------------------------------------------
# Mode 1 — Instance Q&A
# ---------------------------------------------------------------------------

_INSTANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "item":      {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["item"],
}

_INSTANCE_PROC_SCHEMA = {
    "type": "object",
    "properties": {
        "action":    {"type": "string"},
        "argument":  {"type": "string"},
        "ordering":  {"type": "string"},
        "count":     {"type": "integer"},
        "reasoning": {"type": "string"},
    },
    "required": ["action", "argument", "ordering", "count"],
}


def run_qa_instance(
    task: GeneratedTask,
    demo_context: str,
    model: str,
    base_prompt: str = "",
    converter_model: str = "small",
    teaching_message: str = "",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Instance-level Q&A for one gen task.

    Dispatches to the property or procedural variant based on task_type.
    Returns a result dict with entity info, correct answer, prediction, and a
    boolean ``correct`` field.
    """
    if _is_proc(task.task_type):
        return _run_qa_instance_proc(
            task, demo_context, model, base_prompt, converter_model,
            teaching_message, verbose,
        )
    return _run_qa_instance_property(
        task, demo_context, model, base_prompt, converter_model,
        teaching_message, verbose,
    )


def _run_qa_instance_property(
    task: GeneratedTask,
    demo_context: str,
    model: str,
    base_prompt: str = "",
    converter_model: str = "small",
    teaching_message: str = "",
    verbose: bool = False,
) -> Dict[str, Any]:
    root         = task.tree.nodes[task.tree.root_id]
    entity_name  = root.argument
    entity_card  = _entity_card(task)
    correct_item = _get_correct_item(task)

    question = QA_INSTANCE_QUESTION.format(
        entity_card=entity_card,
        entity_name=entity_name,
    )
    teaching = format_teaching_block(teaching_message)
    prompt = "\n\n".join(filter(None, [base_prompt, demo_context])) + teaching
    prompt = "\n\n".join(filter(None, [prompt, question]))

    parsed, raw = _call_model(prompt, model, converter_model, _INSTANCE_SCHEMA,
                              verbose=verbose, label=f"instance:{entity_name}")
    predicted = (parsed.get("item") or "").strip() if parsed else None
    correct   = bool(predicted and correct_item and
                     predicted.lower() == correct_item.lower())

    return {
        "mode":           "instance",
        "entity_name":    entity_name,
        "entity_card":    entity_card,
        "correct_item":   correct_item,
        "predicted_item": predicted,
        "correct":        correct,
        "task_type":      task.task_type,
        "split":          task.split,
        "root_id":        task.tree.root_id,
        "raw_response":   raw,
    }


def _run_qa_instance_proc(
    task: GeneratedTask,
    demo_context: str,
    model: str,
    base_prompt: str = "",
    converter_model: str = "small",
    teaching_message: str = "",
    verbose: bool = False,
) -> Dict[str, Any]:
    root        = task.tree.nodes[task.tree.root_id]
    entity_name = root.argument
    entity_card = _entity_card(task)
    correct     = _get_correct_process(task)

    question = QA_INSTANCE_PROC_QUESTION.format(
        entity_card=entity_card,
        entity_name=entity_name,
    )
    teaching = format_teaching_block(teaching_message)
    prompt = "\n\n".join(filter(None, [base_prompt, demo_context])) + teaching
    prompt = "\n\n".join(filter(None, [prompt, question]))

    parsed, raw = _call_model(prompt, model, converter_model, _INSTANCE_PROC_SCHEMA,
                              verbose=verbose, label=f"instance_proc:{entity_name}")

    pred_action   = _normalize_action  ((parsed.get("action")   or "") if parsed else "")
    pred_argument = (parsed.get("argument") or "").strip().lower() if parsed else ""
    pred_ordering = _normalize_ordering((parsed.get("ordering") or "") if parsed else "")
    pred_count    = parsed.get("count") if parsed else None
    # Normalise None → empty string so _soft_match comparisons are clean
    if not pred_action:   pred_action   = None
    if not pred_argument: pred_argument = None
    if not pred_ordering: pred_ordering = None

    action_ok   = _soft_match(pred_action,   correct["action"])
    argument_ok = _soft_match(pred_argument, correct["argument"])
    ordering_ok = _soft_match(pred_ordering, correct["ordering"])
    # count only checked for proc_add; for others it's always 1
    count_ok    = _soft_match(pred_count,    correct["count"])

    # Overall correct: action + argument + ordering must match;
    # count additionally required for proc_add
    is_proc_add = task.task_type == "proc_add"
    if is_proc_add:
        all_correct = action_ok and argument_ok and ordering_ok and count_ok
    else:
        all_correct = action_ok and argument_ok and ordering_ok

    return {
        "mode":             "instance",
        "entity_name":      entity_name,
        "entity_card":      entity_card,
        "correct_process":  correct,
        "predicted_action":   pred_action,
        "predicted_argument": pred_argument,
        "predicted_ordering": pred_ordering,
        "predicted_count":    pred_count,
        "action_ok":        action_ok,
        "argument_ok":      argument_ok,
        "ordering_ok":      ordering_ok,
        "count_ok":         count_ok,
        "correct":          all_correct,
        "task_type":        task.task_type,
        "split":            task.split,
        "root_id":          task.tree.root_id,
        "raw_response":     raw,
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Mode 2 — Structure Q&A: free-form explanation + LLM judge
# ---------------------------------------------------------------------------

_STRUCTURE_EXP_SCHEMA = {
    "type": "object",
    "properties": {"explanation": {"type": "string"}},
    "required": ["explanation"],
}

# Unified 4-dimension judge schema for both property and procedural tasks.
_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "input_score":          {"type": "integer"},
        "output_score":         {"type": "integer"},
        "rule_score":           {"type": "integer"},
        "generalization_score": {"type": "integer"},
        "reasoning":            {"type": "string"},
    },
    "required": ["input_score", "output_score", "rule_score", "generalization_score"],
}

# Keep as alias so existing import references don't break.
_JUDGE_PROC_SCHEMA = _JUDGE_SCHEMA


def judge_structure_explanation(
    explanation: str,
    task_type: str,
    model: str,
    converter_model: str = "small",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Score a rule explanation via LLM judge; returns overall (0–1) and per-dimension scores."""
    is_proc_task = _is_proc(task_type)
    correct_rule = _get_correct_rule(task_type)

    if is_proc_task:
        prompt = QA_JUDGE_PROMPT_PROC.format(
            correct_rule=correct_rule,
            model_explanation=explanation,
        )
        schema = _JUDGE_PROC_SCHEMA
    else:
        prompt = QA_JUDGE_PROMPT.format(
            correct_rule=correct_rule,
            model_explanation=explanation,
        )
        schema = _JUDGE_SCHEMA

    parsed, raw = _call_model(prompt, model, converter_model, schema,
                              verbose=verbose, label="judge")

    if parsed is None:
        return {
            "input_score": 0, "output_score": 0,
            "rule_score": 0, "generalization_score": 0,
            "overall": 0.0, "reasoning": "", "judge_failed": True,
            "raw_judge_response": raw,
        }

    def _clamp(v):
        try:
            return max(0, min(2, int(v)))
        except (TypeError, ValueError):
            return 0

    in_s   = _clamp(parsed.get("input_score",          0))
    out_s  = _clamp(parsed.get("output_score",         0))
    rule_s = _clamp(parsed.get("rule_score",           0))
    gen_s  = _clamp(parsed.get("generalization_score", 0))
    overall = (in_s + out_s + rule_s + gen_s) / 8.0
    return {
        "input_score":          in_s,
        "output_score":         out_s,
        "rule_score":           rule_s,
        "generalization_score": gen_s,
        "overall":              overall,
        "reasoning":            parsed.get("reasoning", ""),
        "judge_failed":         False,
        "raw_judge_response":   raw,
    }


def run_qa_structure_exp(
    source_tasks: List[GeneratedTask],
    gen_tasks: List[GeneratedTask],
    demo_context: str,
    task_type: str,
    model: str,
    base_prompt: str = "",
    converter_model: str = "small",
    teaching_message: str = "",
    verbose: bool = False,
    judge_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Ask model to explain the rule in free form; judge and return scored result dict."""
    question = (QA_STRUCTURE_EXP_QUESTION_PROC if _is_proc(task_type)
                else QA_STRUCTURE_EXP_QUESTION)
    teaching = format_teaching_block(teaching_message)
    prompt = "\n\n".join(filter(None, [base_prompt, demo_context])) + teaching
    prompt = "\n\n".join(filter(None, [prompt, question]))

    parsed, raw    = _call_model(prompt, model, converter_model, _STRUCTURE_EXP_SCHEMA,
                                 verbose=verbose, label="structure_exp")
    raw_exp     = parsed.get("explanation") if parsed else None
    if isinstance(raw_exp, str):
        explanation = raw_exp.strip()
    elif raw_exp:
        explanation = json.dumps(raw_exp)
    else:
        explanation = ""

    if explanation:
        judge_result = judge_structure_explanation(explanation, task_type,
                                                   judge_model or model,
                                                   converter_model, verbose=verbose)
    else:
        judge_result = {
            "input_score": 0, "output_score": 0,
            "rule_score": 0, "generalization_score": 0,
            "overall": 0.0, "reasoning": "No explanation provided.",
            "judge_failed": True,
        }

    return {
        "mode":                 "structure_exp",
        "task_type":            task_type,
        "correct_rule":         _get_correct_rule(task_type),
        "explanation":          explanation,
        "overall":              judge_result.get("overall", 0.0),
        "input_score":          judge_result.get("input_score",          0),
        "output_score":         judge_result.get("output_score",         0),
        "rule_score":           judge_result.get("rule_score",           0),
        "generalization_score": judge_result.get("generalization_score", 0),
        "judge_reasoning":      judge_result.get("reasoning", ""),
        "judge_failed":         judge_result.get("judge_failed", False),
        "raw_response":         raw,
        "raw_judge":            judge_result.get("raw_judge_response"),
    }


# ---------------------------------------------------------------------------
# QA-based phase 1.1: curate teaching message from QA failures
# ---------------------------------------------------------------------------

def _format_qa_failure_summary(qa_variant_result: Dict, task_type: str) -> str:
    """
    Summarise what the student got wrong in QA phase 1, without revealing the answer.
    Used as input for the QA teacher prompt.
    """
    lines: List[str] = []

    inst_data = qa_variant_result.get("instance")
    if inst_data:
        n   = inst_data.get("total", 0)
        n_c = inst_data.get("correct", 0)
        acc = n_c / n if n else 0.0
        lines.append(f"Instance prediction accuracy: {n_c}/{n} ({acc:.0%})")
        wrong = [r for r in inst_data.get("results", []) if not r.get("correct")]
        for r in wrong[:3]:
            lines.append(f"  Entity: {r.get('entity_card', '?')}")
            if _is_proc(task_type):
                cp = r.get("correct_process", {})
                lines.append(
                    f"  Student: action={r.get('predicted_action')!r}, "
                    f"ordering={r.get('predicted_ordering')!r}, "
                    f"count={r.get('predicted_count')!r}"
                )
                lines.append(
                    f"  Correct: action={cp.get('action')!r}, "
                    f"ordering={cp.get('ordering')!r}, "
                    f"count={cp.get('count')!r}"
                )
            else:
                lines.append(
                    f"  Student predicted: {r.get('predicted_item')!r}  |  "
                    f"Correct: {r.get('correct_item')!r}"
                )

    mc_data = qa_variant_result.get("structure_mc")
    if mc_data:
        outcome = "correct" if mc_data.get("correct") else "wrong"
        lines.append(
            f"\nRule-type MC: student chose '{mc_data.get('predicted_choice')}', "
            f"correct was '{mc_data.get('correct_choice')}' ({outcome})"
        )
        reasoning = (mc_data.get("reasoning") or "")[:300]
        if reasoning:
            lines.append(f"  Student reasoning: {reasoning}")

    exp_data = qa_variant_result.get("structure_exp")
    if exp_data:
        overall = exp_data.get("overall", 0.0)
        lines.append(f"\nRule explanation score: {overall:.3f}/1.0")
        lines.append(
            f"  Input: {exp_data.get('input_score', 0)}/2  "
            f"Output: {exp_data.get('output_score', 0)}/2  "
            f"Rule: {exp_data.get('rule_score', 0)}/2  "
            f"Generalization: {exp_data.get('generalization_score', 0)}/2"
        )
        explanation = (exp_data.get("explanation") or "")[:500]
        if explanation:
            lines.append(f"  Student explanation: \"{explanation}\"")
        judge_note = (exp_data.get("judge_reasoning") or "")[:300]
        if judge_note:
            lines.append(f"  Evaluator feedback: {judge_note}")

    return "\n".join(lines) if lines else "No Q&A results available."


def curate_from_qa_results(
    qa_variant_result: Dict,
    task_type: str,
    model: str,
    converter_model: str = "small",
    verbose: bool = False,
) -> str:
    """Generate a teaching message from QA phase 1 failure summary (QA-native phase 1.1)."""
    failure_summary = _format_qa_failure_summary(qa_variant_result, task_type)
    correct_rule    = _get_correct_rule(task_type)

    prompt = QA_TEACHER_PROMPT_TEMPLATE.format(
        qa_failure_summary=failure_summary,
        correct_rule=correct_rule,
    )

    parsed, raw = _call_model(
        prompt, model, converter_model,
        schema=None,        # free-form text, no JSON schema
        verbose=verbose,
        label="qa_teacher",
        max_tokens=512,
    )

    # The response is plain text, not JSON — use raw directly
    if raw and raw.strip():
        return raw.strip()
    return ""
