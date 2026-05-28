# Steering Methods for Inductive Reasoning

Four methods that can be applied in Phase 2 to improve the student model's ability to induce the hidden rule from source demonstrations.

---

## Overview

| Method | Type | Extra LLM Calls | When Hypothesis Forms |
|--------|------|-----------------|----------------------|
| ReAct  | Static prompt | 0 | Per action step (in reasoning field) |
| HR     | Pre-episode call | 1 before episode | Before first action |
| IDEA   | Online loop | 1 at start + 1 per defeat failure | At start and after each defeat failure |
| ACE    | Offline curation | N (one per Phase 1 failure trace) | During Phase 1.1 preprocessing |

---

## ReAct

**Reference:** Yao et al. (2022), ReAct: Synergizing Reasoning and Acting in Language Models.

### What it does

Appends a single static prompt to every episode's teaching message that instructs the student to write an explicit hypothesis in its `reasoning` field before each action.

### Pipeline

```
Phase 1 (baseline)
  └─ Run gen episodes with no teaching message.

Phase 2 (ReAct)
  └─ For each gen episode:
       teaching_message = REACT_STUDENT_PROMPT
       └─ adventure_run() (standard loop, no extra calls)
            Each step: model writes
              Hypothesis: [current best guess at the rule]
              Evidence:   [which demos support it]
              Action rationale: [why this action follows]
```

### Key prompt (`REACT_STUDENT_PROMPT`)

```
Before each action, explicitly state your current hypothesis about the hidden rule
in your reasoning field.

Structure your reasoning as:
  Hypothesis: [your current best guess at the mapping rule]
  Evidence:   [which demonstration examples support this]
  Action rationale: [why this action follows from your hypothesis]

Revise your hypothesis whenever you receive new evidence.
```

### Implementation

- `teacher.py`: `build_static_teaching_messages("react", ...)` returns `[REACT_STUDENT_PROMPT] * n`
- `adventure_pipeline.py`: `p2_episode_mode = "standard"` (no special runner needed)
- `adventure_episode.py`: `run_single_episode` with `episode_mode="standard"`, teaching message prepended via `format_teaching_block()`

---

## HR — Hypothesis Refinement

**Reference:** Qiu et al. (2024), Hypothesis Refinement for Inductive Reasoning.

### What it does

Makes one extra LLM call at the start of each episode (before the first action). The model generates K candidate rules, verifies each against the source demos, and selects the best-supported one. The selected hypothesis is then prepended to the teaching message so the agent acts under an informed prior.

### Pipeline

```
Phase 1 (baseline)
  └─ Run gen episodes with no teaching message.

Phase 2 (HR)
  └─ For each gen episode:
       1. Pre-episode call (_hr_hypothesis_call):
            Input:  base_prompt + demo_context + HR_HYPOTHESIS_PROMPT
            Output: hypothesis text (selected rule + confidence + key evidence)
       2. teaching_message = hypothesis_text + original_teaching_message
       3. adventure_run() (standard loop, hypothesis visible from step 1)
```

### Key prompt (`HR_HYPOTHESIS_PROMPT`)

```
Before acting, reason systematically about the hidden rule using the source
demonstrations above.

Step 1 — Generate hypotheses: propose {num_hypotheses} distinct candidate rules
  explaining how entity attributes determine the required item.

Step 2 — Verify each: check whether each candidate correctly predicts the item
  for every demonstration entity. Mark each as consistent or inconsistent.

Step 3 — Select: choose the best-supported hypothesis.

Output your selected hypothesis, then proceed to act:
Selected hypothesis: [rule]
Confidence: [high/medium/low]
Key evidence: [which demos support it]
```

### Implementation

- `adventure_episode.py`: `_hr_hypothesis_call(base_prompt, model_path, num_hypotheses=3)` — one `agent_response()` call
- `adventure_episode.py`: `run_single_episode` with `episode_mode="hr"` — calls `_hr_hypothesis_call` then falls through to `adventure_run()`
- `adventure_pipeline.py`: `p2_episode_mode = "hr"`; teaching message is per-episode (hypothesis varies per entity)

---

## IDEA — Abduction, Deduction, Induction

**Reference:** He et al. (2025), IDEA: Interactive Deductive-Abductive Reasoning for LLMs.

### What it does

Replaces the standard episode loop with a three-phase cycle:

- **Abduction** (once, at episode start): form an initial hypothesis and action plan from source demos
- **Deduction** (every step): inject the current hypothesis/plan into context; select next action
- **Induction** (after each defeat failure): update the hypothesis using recent observations; revise plan

This is the only method that modifies the episode runner itself — it uses a separate `_adventure_run_idea()` function instead of `adventure_run()`.

### Adaptation note

The original IDEA fires abduction on per-step feedback. In this environment, intermediate steps (go, buy) are always successful and carry no information about the hidden rule. The only informative signal is a defeat failure (wrong item chosen). Induction therefore fires only after defeat failures, which maps onto the existing multi-attempt budget.

### Pipeline

```
Phase 1 (baseline)
  └─ Run gen episodes with no teaching message.

Phase 2 (IDEA)
  └─ For each gen episode  →  _adventure_run_idea():

       Abduction (LLM call 1):
         Input:  base_prompt + demo_context + IDEA_ABDUCTION_PROMPT
         Output: Hypothesis: [rule guess], Plan: [action sequence]

       Episode loop (max_runs steps):
         Each step (Deduction):
           Input:  base_prompt + [Current Hypothesis] + [Current Plan] + action_obs_history
           Output: {"action": ..., "argument": ..., "reasoning": ...}

         If action == "defeat" and obs.success == False  (Induction):
           LLM call:
             Input:  base_prompt + IDEA_INDUCTION_PROMPT(hypothesis, plan, recent_obs)
             Output: Hypothesis: [revised rule], Plan: [updated plan]
           → hypothesis and plan updated in place for next steps
```

### Key prompts

**`IDEA_ABDUCTION_PROMPT`**
```
A hidden rule determines which item to buy to defeat the target entity.
Before acting, study the source demonstrations above and form an initial hypothesis.

Abduction — form a hypothesis:
Hypothesis: [your best guess at the hidden rule]
Plan:       [your step-by-step action plan based on this hypothesis]
```

**`IDEA_INDUCTION_PROMPT`**
```
Your attempt to defeat the entity failed, which means your item choice was wrong.

Your previous hypothesis: {hypothesis}
Your previous plan:       {plan}

Recent observations:
{observations}

Revise your hypothesis using this new evidence, then update your plan.

Hypothesis: [revised rule hypothesis]
Plan:       [updated action plan]
```

### Implementation

- `adventure_episode.py`: `_parse_idea_hypothesis_plan(text)` — extracts `Hypothesis:` and `Plan:` fields
- `adventure_episode.py`: `_adventure_run_idea(prompt, initial_observation, env, model, ...)` — full I→D→A loop
- `adventure_episode.py`: `run_single_episode` with `episode_mode="idea"` routes to `_adventure_run_idea()`
- `adventure_pipeline.py`: `p2_episode_mode = "idea"`

---

## ACE — Agentic Context Engineering

**Reference:** ACE (Agentic Context Engineering), simplified variant.

### What it does

Builds a reusable strategy playbook from Phase 1 failure traces — offline, before Phase 2 runs. The playbook is a list of induction strategy bullets (e.g., "compare entities across both attribute dimensions systematically"). It is frozen before Phase 2 and used as a single shared teaching message for all gen episodes.

No gen solutions are fed to the reflector; it only sees what the student tried (the failure trace). This prevents answer leakage.

### Pipeline

```
Phase 1 (baseline)
  └─ Run gen episodes with no teaching message.
     Save results (full traces of failed episodes).

Phase 1.1 — ACE curation (teacher.curate()):
  For each Phase 1 failure trace (up to max_traces=10, sampled randomly):
    Reflector LLM call:
      Input:  current_playbook + failure_trace  (ACE_REFLECTOR_PROMPT_TEMPLATE)
      Output: {"new_bullets": [...], "helpful_ids": [...], "harmful_ids": [...]}
    → playbook.merge_delta(new_bullets)
    → playbook.update_feedback(helpful_ids, harmful_ids)

  After all traces:
    Dedup LLM call:
      Input:  full playbook  (ACE_DEDUP_PROMPT_TEMPLATE)
      Output: {"remove_ids": [...]}
    → playbook.remove_ids(ids)

  Result: frozen teaching message = playbook.to_teaching_message()

Phase 2 (ACE)
  └─ For each gen episode:
       teaching_message = frozen_playbook_message
       adventure_run()  (standard loop, no extra calls per episode)
```

### Key prompt (`ACE_REFLECTOR_PROMPT_TEMPLATE`)

```
You are analyzing a failed episode trace to improve a strategy playbook for
rule induction tasks.

The agent must infer a hidden rule from source demonstrations and apply it
to a novel entity. Study the failure and extract lessons about HOW to reason
about the rule — not what the specific answer is.

Current playbook (may be empty):
---
{playbook_text}
---

Failed episode trace:
---
{trace_text}
---

Extract induction strategies from this failure. Focus on reasoning patterns
(e.g. "compare entities across both attribute dimensions systematically"),
not task-specific answers.

Output ONLY valid JSON:
{"new_bullets": ["strategy 1", ...], "helpful_ids": ["b001"], "harmful_ids": ["b003"]}
```

### Playbook structure

Each bullet gets a stable ID (`b001`, `b002`, …) and tracks `helpful_count` / `harmful_count` feedback. The final teaching message is formatted as:

```
Here are strategies to keep in mind:
[b001] Compare entities across both attribute dimensions systematically.
[b002] Check whether changing one attribute changes only one output property.
...
```

### Implementation

- `teacher.py`: `Playbook` class — `merge_delta`, `update_feedback`, `remove_ids`, `to_teaching_message`
- `teacher.py`: `curate_teaching_message_ace(phase1_gen_results, teacher_model_path, max_traces, random_seed, ...)` — the full offline curation loop
- `teacher.py`: `Teacher.curate(phase1_gen_results=..., verbose=..., converter_model=...)` — entry point called by pipeline
- `adventure_pipeline.py`: after Phase 1, if `teaching_strategy == "ace"`, calls `teacher.curate()` to get `variant_teaching_message`; Phase 2 then uses `p2_episode_mode = "standard"`
