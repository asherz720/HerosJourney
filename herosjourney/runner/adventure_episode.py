"""Episode runners and result formatting for the generalization benchmark pipeline."""

import sys
import hashlib
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Tuple, Any

from pydantic import BaseModel

from herosjourney.env.env import AdventureEnv
from herosjourney.eval.result import EpisodeResult
from herosjourney.core.generator import GeneratedTask
from herosjourney.core.demo_generator import Demo


# ---------------------------------------------------------------------------
# Model interface
# ---------------------------------------------------------------------------

class ModelError(RuntimeError):
    """Raised by a model_fn to signal a terminal, non-retryable failure.
    The episode loop catches this and sets terminate=True."""
    pass


# model_fn signature: (prompt: str, max_tokens: int = 512)
#     -> (response: str | None, thinking: str | None, token_counts: dict | None)
# Return (None, None, None) for a transient / retryable error.
# Raise ModelError for a terminal failure (e.g., exceeded retry budget).


def _make_model_fn(model_path: str, max_api_errors: int = 2) -> Callable:
    """Build a model_fn from a model_path string.

    Wraps pipeline.models.agent_response with provider error handling so the
    core episode loop has no dependency on openai or other provider SDKs.
    Lazy-imports pipeline.models so the framework core stays provider-free.
    """
    from herosjourney.runner.models import agent_response  # noqa — experiment infra
    import openai as _openai                    # noqa — provider SDK

    error_count = [0]

    def model_fn(prompt: str, max_tokens: int = 512) -> tuple:
        try:
            return agent_response(model_path, prompt, max_tokens=max_tokens)
        except _openai.APITimeoutError:
            error_count[0] += 1
            if error_count[0] >= max_api_errors:
                raise ModelError(f"API timeout: reached {max_api_errors} errors")
            return (None, None, None)
        except _openai.APIError as exc:
            error_count[0] += 1
            if error_count[0] >= max_api_errors:
                raise ModelError(f"API error: reached {max_api_errors} errors — {exc}")
            return (None, None, None)

    return model_fn


def _make_converter_fn(converter_model: str) -> Optional[Callable]:
    """Build a JSON-repair callable from a converter_model name string.
    Returns None if converter_model is unrecognised (no repair attempted)."""
    if converter_model == "gemini":
        from herosjourney.runner.models import json_converter_gemini  # noqa
        return json_converter_gemini
    if converter_model == "small":
        from herosjourney.runner.models import json_converter_small   # noqa
        return json_converter_small
    return None


# ---------------------------------------------------------------------------
# Internal loop result  (private — callers receive EpisodeResult)
# ---------------------------------------------------------------------------

@dataclass
class _LoopResult:
    """Raw output from one adventure_run / _adventure_run_idea call."""
    success: bool
    terminated: bool
    full_trace: str
    action_history: List[str]
    num_runs: int
    completion_map: Dict[str, Any]
    base_user_prompt: str
    action_obs_reasoning_history: List[Dict[str, Any]]
    prompt_tokens: int
    completion_tokens: int
    hypothesis_history: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Action schema
# ---------------------------------------------------------------------------

class Action(BaseModel):
    action: str
    argument: str
    reasoning: Optional[str] = None


# ---------------------------------------------------------------------------
# Demo context builder
# ---------------------------------------------------------------------------

def construct_demo_context(demos: List[Demo], include_world: bool = True) -> str:
    """Format Demo objects into the in-context demo prompt section."""
    if not demos:
        return ""

    lines: List[str] = []

    if include_world:
        world = demos[0].metadata.get("world_listing", "")
        if world:
            lines.append(world)
            lines.append("")

    lines.append("[Start of Demonstration Episodes]")
    for demo in demos:
        lines.append(demo.format(show_world=False))
        lines.append("")
    lines.append("[End of Demonstration Episodes]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core episode loop  (model-agnostic)
# ---------------------------------------------------------------------------

def adventure_run(
    prompt: str,
    initial_observation: str,
    env: AdventureEnv,
    model_fn: Callable,
    converter_fn: Optional[Callable] = None,
    max_runs: int = 50,
    verbose: bool = False,
    truncate_window: Optional[int] = None,
    episode_prefix: str = "",
    teaching_message: str = "",
    no_reasoning: bool = False,
):
    """Iterative agent loop for one episode.

    Args:
        model_fn:     Callable (prompt, max_tokens=512) -> (response, thinking, token_counts).
                      Return (None, None, None) for a transient error; raise ModelError for
                      terminal failure.
        converter_fn: Optional callable (response_str, json_schema) -> str for repairing
                      malformed JSON responses.  If None, parse failures count against the
                      patience budget without repair.

    Returns a _LoopResult.
    """
    from herosjourney.runner.prompts import format_teaching_block, ACTION_JSON_REMINDER, ACTION_JSON_REMINDER_NO_REASONING

    teaching_message = teaching_message or ""
    anchor = (ACTION_JSON_REMINDER_NO_REASONING if no_reasoning else ACTION_JSON_REMINDER) if teaching_message else ""
    base_user_prompt = f"{prompt}\n\n{initial_observation}{format_teaching_block(teaching_message, anchor_text=anchor)}\n"

    action_obs_history: List[str] = []
    action_obs_reasoning_history: List[Dict] = []
    full_trace = base_user_prompt
    action_history: List[str] = []

    max_patience   = 10
    patience_counter = 0
    success   = False
    terminate = False
    num_runs  = 0
    done      = False
    initial_currency = env.currency
    total_prompt_tokens     = 0
    total_completion_tokens = 0
    truncation_applied = False
    prefix = f"[Ep {episode_prefix}] " if episode_prefix else ""
    completion_map: Dict = {}

    if verbose:
        print(f"{prefix}{'=' * 60}")
        print(f"{prefix}FULL PROMPT SENT TO MODEL:")
        print(f"{prefix}{'=' * 60}")
        print(base_user_prompt)
        print(f"{prefix}{'=' * 60}")
        sys.stdout.flush()

    for i in range(1, max_runs + 1):
        if truncate_window is not None and len(action_obs_history) > truncate_window:
            truncation_applied = True
            state_summary = (
                f"[TRUNCATED CONTEXT NOTICE] you are at {env.current_location}, "
                f"inventory: {dict(env.inventory) if env.inventory else 'empty'}.\n\n"
            )
            running_prompt = state_summary + "".join(action_obs_history[-truncate_window:])
            full_trace += state_summary
        else:
            running_prompt = "".join(action_obs_history)

        full_user_prompt = base_user_prompt + running_prompt
        thinking_traces = None
        token_counts    = None

        try:
            response, thinking_traces, token_counts = model_fn(full_user_prompt)
        except ModelError:
            terminate = True
            break

        if token_counts:
            total_prompt_tokens     += token_counts.get("prompt_tokens", 0)
            total_completion_tokens += token_counts.get("candidates_tokens", 0)

        if not response or not response.strip():
            patience_counter += 1
            if patience_counter >= max_patience:
                terminate = True
                break
            continue

        try:
            action_obj = Action.model_validate_json(response)
        except Exception:
            repaired = False
            if converter_fn is not None:
                try:
                    converted  = converter_fn(response, Action.model_json_schema())
                    action_obj = Action.model_validate_json(converted)
                    repaired   = True
                except Exception:
                    pass
            if not repaired:
                print(f"{prefix}JSON parse failed | response: {response[:200]}")
                patience_counter += 1
                if patience_counter >= max_patience:
                    terminate = True
                    break
                continue

        action    = action_obj.action
        argument  = action_obj.argument
        reasoning = action_obj.reasoning
        num_runs += 1

        full_action, obs_obj, completion_map = env.step(action, argument)
        observation = obs_obj.message
        done        = obs_obj.done
        action_history.append(full_action)

        if not obs_obj.success:
            patience_counter += 1
            if patience_counter >= max_patience:
                terminate = True
                break
        else:
            patience_counter = 0

        if verbose:
            print(f"{prefix}Act {i}: {full_action}")
            print(f"{prefix}Obs {i}: {observation}")
            sys.stdout.flush()

        pair = f"Action {i}: {full_action}\nObs {i}: {observation}\n>"
        full_trace += pair
        action_obs_history.append(pair)
        action_obs_reasoning_history.append({
            "step":            i,
            "action":          full_action,
            "observation":     observation,
            "reasoning":       reasoning,
            "thinking_traces": thinking_traces,
            "success":         obs_obj.success if hasattr(obs_obj, "success") else None,
        })

        if done:
            success = env.done and not terminate
            return _LoopResult(
                success=success,
                terminated=terminate,
                full_trace=full_trace,
                action_history=action_history,
                num_runs=num_runs,
                completion_map=completion_map,
                base_user_prompt=base_user_prompt,
                action_obs_reasoning_history=action_obs_reasoning_history,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
            )

    # Max steps reached or ModelError
    return _LoopResult(
        success=False,
        terminated=True,
        full_trace=full_trace,
        action_history=action_history,
        num_runs=num_runs,
        completion_map=completion_map,
        base_user_prompt=base_user_prompt,
        action_obs_reasoning_history=action_obs_reasoning_history,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
    )


# ---------------------------------------------------------------------------
# HR and IDEA helpers  (model-agnostic)
# ---------------------------------------------------------------------------

def _extract_idea_final_hypothesis(trace: str) -> str:
    """Return the last hypothesis reached during an IDEA episode."""
    final = ""
    for line in trace.splitlines():
        s = line.strip()
        if s.startswith("[Revised Hypothesis]:"):
            final = s[len("[Revised Hypothesis]:"):].strip()
        elif s.startswith("[Initial Hypothesis]:") and not final:
            final = s[len("[Initial Hypothesis]:"):].strip()
    return final


def _parse_idea_hypothesis_plan(text: str) -> Tuple[str, str]:
    """Extract hypothesis and plan from an IDEA abduction/induction response."""
    hypothesis = ""
    plan = ""
    if not text:
        return hypothesis, plan
    if "Hypothesis:" in text:
        hyp_part = text.split("Hypothesis:", 1)[1]
        if "Plan:" in hyp_part:
            hypothesis = hyp_part.split("Plan:")[0].strip()
            plan = hyp_part.split("Plan:", 1)[1].strip()
        else:
            hypothesis = hyp_part.strip()
    elif "Plan:" in text:
        plan = text.split("Plan:", 1)[1].strip()
    return hypothesis, plan


def _hr_hypothesis_call(
    demo_context: str,
    model_fn: Callable,
    num_hypotheses: int = 3,
) -> str:
    """Pre-episode call: generate hypothesis text from demos (plain-text, no JSON)."""
    from herosjourney.runner.strategies import HR_HYPOTHESIS_PROMPT
    from herosjourney.runner.prompts import REASONING_CONTEXT_PREAMBLE
    reasoning_base = (
        REASONING_CONTEXT_PREAMBLE + "\n\n" + demo_context
        if demo_context else REASONING_CONTEXT_PREAMBLE
    )
    prompt = reasoning_base + "\n\n" + HR_HYPOTHESIS_PROMPT.format(num_hypotheses=num_hypotheses)
    result, _, _ = model_fn(prompt, 2048)
    return (result or "").strip()


def _adventure_run_idea(
    prompt: str,
    initial_observation: str,
    env: "AdventureEnv",
    model_fn: Callable,
    converter_fn: Optional[Callable] = None,
    max_runs: int = 50,
    verbose: bool = False,
    truncate_window: Optional[int] = None,
    episode_prefix: str = "",
    reasoning_prompt: str = "",
) -> tuple:
    """IDEA loop: abduction at start, induction on defeat failure; returns 15-tuple."""
    from herosjourney.runner.strategies import IDEA_ABDUCTION_PROMPT, IDEA_INDUCTION_PROMPT

    base_user_prompt = f"{prompt}\n\n{initial_observation}\n"
    action_obs_history: List[str] = []
    action_obs_reasoning_history: List[Dict] = []
    full_trace = base_user_prompt
    action_history: List[str] = []
    hypothesis_history: List[str] = []

    _reasoning_base = reasoning_prompt if reasoning_prompt else base_user_prompt

    # --- Initial Abduction ---
    abduction_input = _reasoning_base + "\n" + IDEA_ABDUCTION_PROMPT
    hyp_text, _, _ = model_fn(abduction_input, 2048)
    hypothesis, plan = _parse_idea_hypothesis_plan(hyp_text or "")
    if hypothesis:
        full_trace += f"[Initial Hypothesis]: {hypothesis}\n[Plan]: {plan}\n"
        hypothesis_history.append(hypothesis)
    if verbose:
        pfx = f"[Ep {episode_prefix}] " if episode_prefix else ""
        print(f"{pfx}[IDEA] Initial hypothesis: {hypothesis}")

    max_patience   = 10
    patience_counter = 0
    success   = False
    terminate = False
    num_runs  = 0
    done      = False
    initial_currency = env.currency
    total_prompt_tokens     = 0
    total_completion_tokens = 0
    truncation_applied = False
    prefix = f"[Ep {episode_prefix}] " if episode_prefix else ""
    completion_map: Dict = {}

    for i in range(1, max_runs + 1):
        hypothesis_block = (
            f"\n[Current Hypothesis]: {hypothesis}\n[Current Plan]: {plan}\n\n"
            if hypothesis else ""
        )

        if truncate_window is not None and len(action_obs_history) > truncate_window:
            truncation_applied = True
            state_summary = (
                f"[TRUNCATED] at {env.current_location}, "
                f"inventory: {dict(env.inventory) if env.inventory else 'empty'}.\n\n"
            )
            running_prompt = state_summary + "".join(action_obs_history[-truncate_window:])
            full_trace += state_summary
        else:
            running_prompt = "".join(action_obs_history)

        full_user_prompt = base_user_prompt + hypothesis_block + running_prompt

        try:
            response, thinking_traces, token_counts = model_fn(full_user_prompt)
        except ModelError:
            terminate = True
            break

        if token_counts:
            total_prompt_tokens     += token_counts.get("prompt_tokens", 0)
            total_completion_tokens += token_counts.get("candidates_tokens", 0)

        if not response or not response.strip():
            patience_counter += 1
            if patience_counter >= max_patience:
                terminate = True
                break
            continue

        try:
            action_obj = Action.model_validate_json(response)
        except Exception:
            repaired = False
            if converter_fn is not None:
                try:
                    converted  = converter_fn(response, Action.model_json_schema())
                    action_obj = Action.model_validate_json(converted)
                    repaired   = True
                except Exception:
                    pass
            if not repaired:
                patience_counter += 1
                if patience_counter >= max_patience:
                    terminate = True
                    break
                continue

        action    = action_obj.action
        argument  = action_obj.argument
        reasoning = action_obj.reasoning
        num_runs += 1

        full_action, obs_obj, completion_map = env.step(action, argument)
        observation = obs_obj.message
        done        = obs_obj.done
        action_history.append(full_action)

        if not obs_obj.success:
            patience_counter += 1
            if patience_counter >= max_patience:
                terminate = True
                break
        else:
            patience_counter = 0

        if verbose:
            print(f"{prefix}Act {i}: {full_action}")
            print(f"{prefix}Obs {i}: {observation}")
            sys.stdout.flush()

        pair = f"Action {i}: {full_action}\nObs {i}: {observation}\n>"
        full_trace += pair
        action_obs_history.append(pair)
        action_obs_reasoning_history.append({
            "step":            i,
            "action":          full_action,
            "observation":     observation,
            "reasoning":       reasoning,
            "thinking_traces": thinking_traces,
            "success":         obs_obj.success if hasattr(obs_obj, "success") else None,
        })

        # --- Induction after defeat failure ---
        if action == "defeat" and not obs_obj.success and not done and num_runs < max_runs:
            recent_obs = "".join(action_obs_history[-5:])
            induction_input = (
                _reasoning_base + "\n"
                + IDEA_INDUCTION_PROMPT.format(
                    hypothesis=hypothesis,
                    plan=plan,
                    observations=recent_obs,
                )
            )
            hyp_text, _, ind_counts = model_fn(induction_input, 2048)
            if ind_counts:
                total_prompt_tokens     += ind_counts.get("prompt_tokens", 0)
                total_completion_tokens += ind_counts.get("candidates_tokens", 0)
            new_hyp, new_plan = _parse_idea_hypothesis_plan(hyp_text or "")
            if new_hyp:
                hypothesis, plan = new_hyp, new_plan
                hypothesis_history.append(hypothesis)
                full_trace += f"[Revised Hypothesis]: {hypothesis}\n[Revised Plan]: {plan}\n"
                if verbose:
                    print(f"{prefix}[IDEA] Revised hypothesis: {hypothesis}")

        if done:
            success = env.done and not terminate
            return _LoopResult(
                success=success,
                terminated=terminate,
                full_trace=full_trace,
                action_history=action_history,
                num_runs=num_runs,
                completion_map=completion_map,
                base_user_prompt=base_user_prompt,
                action_obs_reasoning_history=action_obs_reasoning_history,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                hypothesis_history=hypothesis_history,
            )

    return _LoopResult(
        success=False,
        terminated=True,
        full_trace=full_trace,
        action_history=action_history,
        num_runs=num_runs,
        completion_map=completion_map,
        base_user_prompt=base_user_prompt,
        action_obs_reasoning_history=action_obs_reasoning_history,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        hypothesis_history=hypothesis_history,
    )


# ---------------------------------------------------------------------------
# Single episode runner  (experiment-facing API — accepts model_path string)
# ---------------------------------------------------------------------------

def run_single_episode(
    episode_idx: int,
    task: GeneratedTask,
    demo_context: str,
    max_runs: Optional[int],
    verbose: bool,
    truncate_window: Optional[int],
    model_path: str,
    initial_currency: int,
    converter_model: str = "small",
    teaching_message: str = "",
    num_tries: int = 2,
    episode_label: Optional[str] = None,
    source_tasks: Optional[List] = None,
    episode_mode: str = "standard",
    no_reasoning: bool = False,
) -> Dict:
    """Set up env for one gen task, assemble prompt, run episode, return result dict.

    This is the experiment-facing entry point.  It builds a model_fn and converter_fn
    from the string parameters and delegates to the model-agnostic core loops.
    """
    from herosjourney.runner.prompts import (
        GENERALIZATION_BASE_PROMPT,
        GENERALIZATION_BASE_PROMPT_NO_REASONING,
        REASONING_CONTEXT_PREAMBLE,
    )

    model_fn     = _make_model_fn(model_path)
    converter_fn = _make_converter_fn(converter_model)

    all_trees = [(task.tree, task.tree.root_id)]
    if source_tasks:
        for st in source_tasks:
            all_trees.append((st.tree, st.tree.root_id))

    env = AdventureEnv(
        trees=all_trees,
        initial_currency=initial_currency,
        initial_location="GameStart",
    )

    rule_seed   = int(hashlib.md5(task.tree.root_id.encode()).hexdigest()[:8], 16) % (2**31)
    initial_obs = env.reset(
        tree_index=0,
        initial_currency=initial_currency,
        initial_location="GameStart",
        seed=rule_seed,
        rules_to_skip=task.rules_to_skip,
        task_label="Your task",
    )

    root_node   = task.tree.nodes[task.tree.root_id]
    goal_action = root_node.meta.get("incoming_edge", "")
    goal_target = root_node.argument

    reference_solution = task.tree.get_solution()

    base_prompt = GENERALIZATION_BASE_PROMPT_NO_REASONING if no_reasoning else GENERALIZATION_BASE_PROMPT
    prompt = base_prompt + ("\n\n" + demo_context if demo_context else "")

    reasoning_prompt = REASONING_CONTEXT_PREAMBLE + ("\n\n" + demo_context if demo_context else "")

    if max_runs is None:
        max_runs = len(reference_solution) * num_tries

    if episode_mode == "hr":
        hypothesis_text = _hr_hypothesis_call(
            demo_context=demo_context,
            model_fn=model_fn,
        )
        if hypothesis_text:
            teaching_message = (
                hypothesis_text + ("\n\n" + teaching_message if teaching_message else "")
            ).strip()

    ep_label = episode_label if episode_label is not None else str(episode_idx)

    if episode_mode == "idea":
        loop = _adventure_run_idea(
            prompt=prompt,
            initial_observation=initial_obs,
            env=env,
            model_fn=model_fn,
            converter_fn=converter_fn,
            max_runs=max_runs,
            verbose=verbose,
            truncate_window=truncate_window,
            episode_prefix=ep_label,
            reasoning_prompt=reasoning_prompt,
        )
        teaching_message = loop.hypothesis_history[-1] if loop.hypothesis_history else ""
    else:
        loop = adventure_run(
            prompt=prompt,
            initial_observation=initial_obs,
            env=env,
            model_fn=model_fn,
            converter_fn=converter_fn,
            max_runs=max_runs,
            verbose=verbose,
            truncate_window=truncate_window,
            episode_prefix=ep_label,
            teaching_message=teaching_message,
            no_reasoning=no_reasoning,
        )

    ref_len = len(reference_solution)
    efficiency = (
        ref_len / loop.num_runs
        if loop.success and loop.num_runs > 0 and ref_len > 0 else None
    )

    hyp_history = (
        loop.hypothesis_history if episode_mode == "idea"
        else ([teaching_message] if episode_mode == "hr" and teaching_message else [])
    )

    return EpisodeResult(
        episode_idx=episode_idx,
        task_type=task.task_type,
        split=task.split,
        success=loop.success,
        terminated=loop.terminated,
        num_runs=loop.num_runs,
        reference_length=ref_len,
        reference_solution=reference_solution,
        efficiency=efficiency,
        num_tries=num_tries,
        full_trace=loop.full_trace,
        action_history=loop.action_history,
        action_obs_reasoning_history=loop.action_obs_reasoning_history,
        completion_map=loop.completion_map,
        prompt_tokens=loop.prompt_tokens,
        completion_tokens=loop.completion_tokens,
        teaching_message=teaching_message,
        hypothesis_history=hyp_history,
    )


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def format_episode_result(result: EpisodeResult) -> Dict:
    """Serialise an EpisodeResult to the standard JSON-safe dict used by the pipeline."""
    return result.to_dict()
