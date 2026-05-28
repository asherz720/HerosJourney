"""
Single-episode execution for the generalization benchmark pipeline.

Contains:
  - Action schema
  - adventure_run          — core episode loop (one task, iterative actions)
  - construct_demo_context — format Demo objects into a prompt section
  - run_single_episode     — env setup + prompt assembly + episode run
  - format_episode_result  — standardised result dict
"""

import sys
import hashlib
from typing import Optional, List, Dict, Tuple, Callable

import openai
from pydantic import BaseModel

from env.env import AdventureEnv
from tree_management.generator import GeneratedTask
from tree_management.demo_generator import Demo
from pipeline.models import agent_response, json_converter_small, json_converter_gemini


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
    """
    Build the in-context demo section from a list of Demo objects.

    If ``include_world`` is True and the first demo has a world_listing in its
    metadata, the world listing is prepended once before all the episodes.

    Returns a formatted string ready to be inserted into the prompt.
    """
    if not demos:
        return ""

    lines: List[str] = []

    # World listing — prepend once if available
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
# Core episode loop
# ---------------------------------------------------------------------------

def adventure_run(
    prompt: str,
    initial_observation: str,
    env: AdventureEnv,
    converter_model: str = "small",
    max_runs: int = 50,
    verbose: bool = False,
    model: str = "gemini",
    truncate_window: Optional[int] = None,
    episode_prefix: str = "",
    teaching_message: str = "",
    no_reasoning: bool = False,
):
    """
    Run the agent on one episode.

    Returns
    -------
    (success, terminate, full_trace, action_history, num_runs, currency_used,
     completion_map, base_user_prompt, action_obs_reasoning_history,
     num_search_node_actions, prompt_tokens, completion_tokens,
     truncate_window_used, truncation_applied)
    """
    from pipeline.prompts import format_teaching_block, ACTION_JSON_REMINDER, ACTION_JSON_REMINDER_NO_REASONING
    teaching_message = teaching_message or ""
    anchor = (ACTION_JSON_REMINDER_NO_REASONING if no_reasoning else ACTION_JSON_REMINDER) if teaching_message else ""
    base_user_prompt = f"{prompt}\n\n{initial_observation}{format_teaching_block(teaching_message, anchor_text=anchor)}\n"

    action_obs_history: List[str] = []
    action_obs_reasoning_history: List[Dict] = []
    full_trace = base_user_prompt
    action_history: List[str] = []

    max_patience   = 10
    max_api_errors = 2
    patience_counter = 0
    api_error_counter = 0
    success   = False
    terminate = False
    num_runs  = 0
    done      = False
    initial_currency  = env.currency
    total_prompt_tokens      = 0
    total_completion_tokens  = 0
    truncation_applied = False
    prefix = f"[Ep {episode_prefix}] " if episode_prefix else ""

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
            response, thinking_traces, token_counts = agent_response(model, full_user_prompt)
        except openai.APITimeoutError:
            api_error_counter += 1
            print(f"{prefix}API timeout ({api_error_counter}/{max_api_errors})")
            if api_error_counter >= max_api_errors:
                terminate = True
                break
            continue
        except openai.APIError as e:
            api_error_counter += 1
            print(f"{prefix}API error ({api_error_counter}/{max_api_errors}): {e}")
            if api_error_counter >= max_api_errors:
                terminate = True
                break
            continue

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
            try:
                if converter_model == "gemini":
                    converted = json_converter_gemini(response, Action.model_json_schema())
                else:
                    converted = json_converter_small(response, Action.model_json_schema())
                action_obj = Action.model_validate_json(converted)
            except Exception as e2:
                print(f"{prefix}JSON parse failed: {e2} | response: {response[:200]}")
                patience_counter += 1
                if patience_counter >= max_patience:
                    terminate = True
                    break
                continue

        action   = action_obj.action
        argument = action_obj.argument
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
            "step":           i,
            "action":         full_action,
            "observation":    observation,
            "reasoning":      reasoning,
            "thinking_traces": thinking_traces,
            "success":        obs_obj.success if hasattr(obs_obj, "success") else None,
        })

        if done:
            success = env.done and not terminate
            currency_used = initial_currency - env.currency
            return (
                success, terminate, full_trace, action_history, num_runs,
                currency_used, completion_map, base_user_prompt,
                action_obs_reasoning_history, env.num_search_node_actions,
                total_prompt_tokens, total_completion_tokens,
                truncate_window, truncation_applied,
            )

    # Max steps reached
    success = False
    terminate = True
    currency_used = initial_currency - env.currency
    return (
        success, terminate, full_trace, action_history, num_runs,
        currency_used, completion_map if "completion_map" in dir() else {},
        base_user_prompt, action_obs_reasoning_history,
        env.num_search_node_actions,
        total_prompt_tokens, total_completion_tokens,
        truncate_window, truncation_applied,
    )


# ---------------------------------------------------------------------------
# HR and IDEA helpers
# ---------------------------------------------------------------------------

def _extract_idea_final_hypothesis(trace: str) -> str:
    """Return the last hypothesis reached during an IDEA episode.
    Scans for [Revised Hypothesis] lines first; falls back to [Initial Hypothesis]."""
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
    # Split on "Plan:" first to avoid it swallowing the hypothesis
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
    model_path: str,
    converter_model: str = "small",
    num_hypotheses: int = 3,
) -> str:
    """
    One pre-episode LLM call: generate and verify hypotheses from source demos.
    Returns the hypothesis text to prepend to the teaching message.
    Uses REASONING_CONTEXT_PREAMBLE (no JSON action instruction) so the model
    outputs plain-text hypothesis analysis rather than a JSON action object.
    """
    from pipeline.prompts import HR_HYPOTHESIS_PROMPT, REASONING_CONTEXT_PREAMBLE
    reasoning_base = REASONING_CONTEXT_PREAMBLE + "\n\n" + demo_context if demo_context else REASONING_CONTEXT_PREAMBLE
    prompt = reasoning_base + "\n\n" + HR_HYPOTHESIS_PROMPT.format(num_hypotheses=num_hypotheses)
    result, _, _ = agent_response(model_path, prompt, max_tokens=2048)
    return (result or "").strip()


def _adventure_run_idea(
    prompt: str,
    initial_observation: str,
    env: "AdventureEnv",
    model: str,
    converter_model: str = "small",
    max_runs: int = 50,
    verbose: bool = False,
    truncate_window: Optional[int] = None,
    episode_prefix: str = "",
    reasoning_prompt: str = "",
) -> tuple:
    """
    IDEA episode loop: Abduction at start → Deduction per step → Induction after defeat failure.

    Adaptation note: the original IDEA fires abduction on per-step feedback. In this
    environment intermediate steps (go, buy) are always successful and uninformative about
    the hidden rule. Induction therefore fires only after a defeat failure — the one
    genuinely informative signal — which maps onto the existing multi-attempt budget.
    """
    from pipeline.prompts import IDEA_ABDUCTION_PROMPT, IDEA_INDUCTION_PROMPT

    base_user_prompt = f"{prompt}\n\n{initial_observation}\n"
    action_obs_history: List[str] = []
    action_obs_reasoning_history: List[Dict] = []
    full_trace = base_user_prompt
    action_history: List[str] = []
    hypothesis_history: List[str] = []

    # reasoning_prompt uses REASONING_CONTEXT_PREAMBLE (no JSON action instruction)
    # so abduction/induction calls get plain-text responses instead of JSON actions.
    _reasoning_base = reasoning_prompt if reasoning_prompt else base_user_prompt

    # --- Initial Abduction ---
    abduction_input = _reasoning_base + "\n" + IDEA_ABDUCTION_PROMPT
    hyp_text, _, _ = agent_response(model, abduction_input, max_tokens=2048)
    hypothesis, plan = _parse_idea_hypothesis_plan(hyp_text or "")
    if hypothesis:
        full_trace += f"[Initial Hypothesis]: {hypothesis}\n[Plan]: {plan}\n"
        hypothesis_history.append(hypothesis)
    if verbose:
        prefix_str = f"[Ep {episode_prefix}] " if episode_prefix else ""
        print(f"{prefix_str}[IDEA] Initial hypothesis: {hypothesis}")

    max_patience   = 10
    max_api_errors = 2
    patience_counter = 0
    api_error_counter = 0
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
        # Inject current hypothesis into the running context
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
            response, thinking_traces, token_counts = agent_response(model, full_user_prompt)
        except openai.APITimeoutError:
            api_error_counter += 1
            if api_error_counter >= max_api_errors:
                terminate = True
                break
            continue
        except openai.APIError as e:
            api_error_counter += 1
            if api_error_counter >= max_api_errors:
                terminate = True
                break
            continue

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
            try:
                if converter_model == "gemini":
                    converted = json_converter_gemini(response, Action.model_json_schema())
                else:
                    converted = json_converter_small(response, Action.model_json_schema())
                action_obj = Action.model_validate_json(converted)
            except Exception:
                patience_counter += 1
                if patience_counter >= max_patience:
                    terminate = True
                    break
                continue

        action   = action_obj.action
        argument = action_obj.argument
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
            "step":           i,
            "action":         full_action,
            "observation":    observation,
            "reasoning":      reasoning,
            "thinking_traces": thinking_traces,
            "success":        obs_obj.success if hasattr(obs_obj, "success") else None,
        })

        # --- Induction after defeat failure ---
        is_defeat_failure = (action == "defeat" and not obs_obj.success)
        if is_defeat_failure and not done and num_runs < max_runs:
            recent_obs = "".join(action_obs_history[-5:])
            induction_input = (
                _reasoning_base + "\n"
                + IDEA_INDUCTION_PROMPT.format(
                    hypothesis=hypothesis,
                    plan=plan,
                    observations=recent_obs,
                )
            )
            hyp_text, _, ind_counts = agent_response(model, induction_input, max_tokens=2048)
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
            currency_used = initial_currency - env.currency
            return (
                success, terminate, full_trace, action_history, num_runs,
                currency_used, completion_map, base_user_prompt,
                action_obs_reasoning_history, env.num_search_node_actions,
                total_prompt_tokens, total_completion_tokens,
                truncate_window, truncation_applied,
                hypothesis_history,
            )

    success = False
    terminate = True
    currency_used = initial_currency - env.currency
    return (
        success, terminate, full_trace, action_history, num_runs,
        currency_used, completion_map, base_user_prompt,
        action_obs_reasoning_history, env.num_search_node_actions,
        total_prompt_tokens, total_completion_tokens,
        truncate_window, truncation_applied,
        hypothesis_history,
    )


# ---------------------------------------------------------------------------
# Single episode runner
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
    """
    Set up the environment for one gen task and run an episode.

    Parameters
    ----------
    episode_idx    : index for logging / result key
    task           : a gen-split GeneratedTask
    demo_context   : pre-formatted demo string (from construct_demo_context)
    teaching_message : optional teaching hint prepended in Phase 2
    source_tasks   : source-split GeneratedTask list whose trees are added to the
                     env so that all demo items/entities are valid action targets.
                     This ensures the agent can attempt to buy any item shown in
                     the world listing, not just the one in the gen task's tree.
    """
    from pipeline.prompts import GENERALIZATION_BASE_PROMPT

    # Build tree list: gen task tree first (index 0), then all source trees.
    # The env uses tree_index=0 as the active goal; source trees extend world knowledge
    # so that action validators find all items and entities from the demo context.
    all_trees = [(task.tree, task.tree.root_id)]
    if source_tasks:
        for st in source_tasks:
            all_trees.append((st.tree, st.tree.root_id))

    env = AdventureEnv(
        trees=all_trees,
        initial_currency=initial_currency,
        initial_location="GameStart",
    )

    rule_seed = int(hashlib.md5(task.tree.root_id.encode()).hexdigest()[:8], 16) % (2**31)
    # env.reset returns the complete formatted initial observation: header + entity cards
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

    # Build prompt: base system prompt + demo context
    from pipeline.prompts import REASONING_CONTEXT_PREAMBLE, GENERALIZATION_BASE_PROMPT_NO_REASONING
    base_prompt = GENERALIZATION_BASE_PROMPT_NO_REASONING if no_reasoning else GENERALIZATION_BASE_PROMPT
    prompt = base_prompt
    if demo_context:
        prompt = prompt + "\n\n" + demo_context

    # reasoning_prompt: same game context but without the JSON action output instruction,
    # used by HR and IDEA for their plain-text hypothesis/hypothesis-revision calls.
    reasoning_prompt = REASONING_CONTEXT_PREAMBLE
    if demo_context:
        reasoning_prompt = reasoning_prompt + "\n\n" + demo_context

    if max_runs is None:
        max_runs = len(reference_solution) * num_tries

    # HR: one pre-episode hypothesis generation call; then standard runner.
    if episode_mode == "hr":
        hypothesis_text = _hr_hypothesis_call(
            demo_context=demo_context,
            model_path=model_path,
            converter_model=converter_model,
        )
        if hypothesis_text:
            teaching_message = (
                hypothesis_text + ("\n\n" + teaching_message if teaching_message else "")
            ).strip()

    hypothesis_history: List[str] = []
    if episode_mode == "idea":
        (success, terminate, full_trace, action_history, num_runs,
         currency_used, completion_map, base_user_prompt,
         action_obs_reasoning_history, num_search_node_actions,
         prompt_tokens, completion_tokens,
         truncate_window_used, truncation_applied,
         hypothesis_history) = _adventure_run_idea(
            prompt=prompt,
            initial_observation=initial_obs,
            env=env,
            model=model_path,
            converter_model=converter_model,
            max_runs=max_runs,
            verbose=verbose,
            truncate_window=truncate_window,
            episode_prefix=episode_label if episode_label is not None else str(episode_idx),
            reasoning_prompt=reasoning_prompt,
        )
        # Final hypothesis = last entry in history; saved as teaching_message for
        # reuse in QA Phase 2
        teaching_message = hypothesis_history[-1] if hypothesis_history else ""
    else:
        (success, terminate, full_trace, action_history, num_runs,
         currency_used, completion_map, base_user_prompt,
         action_obs_reasoning_history, num_search_node_actions,
         prompt_tokens, completion_tokens,
         truncate_window_used, truncation_applied) = adventure_run(
            prompt=prompt,
            initial_observation=initial_obs,
            env=env,
            converter_model=converter_model,
            max_runs=max_runs,
            verbose=verbose,
            model=model_path,
            truncate_window=truncate_window,
            episode_prefix=episode_label if episode_label is not None else str(episode_idx),
            teaching_message=teaching_message,
            no_reasoning=no_reasoning,
        )

    efficiency = (
        len(action_history) / len(reference_solution)
        if success and reference_solution else None
    )

    result = {
        "episode_idx":                  episode_idx,
        "root_id":                      task.tree.root_id,
        "goal":                         (goal_action, goal_target),
        "goal_str":                     f"{goal_action} {goal_target}",
        "task_type":                    task.task_type,
        "split":                        task.split,
        "success":                      success,
        "terminate":                    terminate,
        "full_trace":                   full_trace,
        "action_history":               action_history,
        "num_runs":                     num_runs,
        "currency_used":                currency_used,
        "currency_remaining":           env.currency,
        "completion_map":               completion_map,
        "base_user_prompt":             base_user_prompt,
        "action_obs_reasoning_history": action_obs_reasoning_history,
        "num_search_node_actions":      num_search_node_actions,
        "reference_solution":           reference_solution,
        "reference_length":             len(reference_solution),
        "num_items":                    task.metadata.get("num_items", 1),
        "num_tries":                    num_tries,
        "max_runs_allowed":             max_runs,
        "efficiency":                   efficiency,
        "truncate_window":              truncate_window_used,
        "truncation_applied":           truncation_applied,
        "teaching_message":             teaching_message,
        "hypothesis_history":           (
            hypothesis_history if episode_mode == "idea"
            else ([teaching_message] if episode_mode == "hr" and teaching_message else [])
        ),
        "_prompt_tokens":               prompt_tokens,
        "_completion_tokens":           completion_tokens,
    }
    return result


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def format_episode_result(result: Dict, episode_idx: int) -> Dict:
    """Standardised episode dict for phase results (strips leading-underscore fields)."""
    cm = result.get("completion_map") or {}
    return {
        "index":            episode_idx,
        "root_id":          result.get("root_id", ""),
        "goal":             result.get("goal", []),
        "goal_str":         result.get("goal_str", ""),
        "task_type":        result.get("task_type", ""),
        "split":            result.get("split", ""),
        "completion_rate":  sum(cm.values()) / len(cm) if cm else 0.0,
        "reference_solution": result.get("reference_solution", []),
        "reference_length": result.get("reference_length", 0),
        "action_obs_reasoning_history": result.get("action_obs_reasoning_history", []),
        "trace":            result.get("full_trace", ""),
        "actions":          result.get("action_history", []),
        "completion_map":   cm,
        "success":          result.get("success", False),
        "terminated":       result.get("terminate", False),
        "num_runs":         result.get("num_runs", 0),
        "currency_used":    result.get("currency_used", 0),
        "currency_remaining": result.get("currency_remaining", 0),
        "efficiency":       result.get("efficiency"),
        "num_search_node_actions": result.get("num_search_node_actions", 0),
        "num_items":        result.get("num_items", 1),
        "num_tries":        result.get("num_tries", 2),
        "max_runs_allowed": result.get("max_runs_allowed"),
        "truncate_window":  result.get("truncate_window"),
        "truncation_applied": result.get("truncation_applied", False),
        "teaching_message": result.get("teaching_message", ""),
        "hypothesis_history": result.get("hypothesis_history", []),
    }
