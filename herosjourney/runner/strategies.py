"""
pipeline/strategies.py
Prompts for induction-specific steering strategies and the ACE playbook system.

These are optional add-ons to the core episode loop — pass the result as a
teaching_message or inject into the reasoning context as appropriate.
"""

# =============================================================================
# ReAct
# =============================================================================

REACT_STUDENT_PROMPT = """Before each action, think about your current hypothesis about the hidden rule.

You can reason about the following aspects:
  Hypothesis: [your current best guess at the mapping rule]
  Evidence: [which demonstration examples support this]
  Action rationale: [why this action follows from your hypothesis]

Follow the output JSON format and do not include reasoning in your response."""


# =============================================================================
# Hypothesis Refinement (HR)
# =============================================================================

HR_HYPOTHESIS_PROMPT = """Before acting, reason systematically about the hidden rule using the source demonstrations above.

Step 1 — Generate hypotheses: propose {num_hypotheses} distinct candidate rules explaining how different goals are achieved. Consider different structural possibilities for how the attributes might combine. Note that not all goals show a uniform pattern.

Step 2 — Verify each: check whether each candidate correctly predicts the item for every demonstration entity. Mark each as consistent or inconsistent.

Step 3 — Select: choose the best-supported hypothesis.

Output your selected hypothesis, then proceed to act:
Selected hypothesis: [rule]
Confidence: [high/medium/low]
Key evidence: [which demos support it]"""


# =============================================================================
# IDEA (Induction–Deduction–Abduction)
# =============================================================================

IDEA_ABDUCTION_PROMPT = """A hidden rule determines how to achieve different goals. Before acting, study the source demonstrations above and form an initial hypothesis. Note that not all goals show a uniform pattern so try to find out those that do show any patterns.

Abduction — form a hypothesis:
Hypothesis: [your best guess at the hidden rule]
Plan: [your step-by-step action plan based on this hypothesis]"""


IDEA_INDUCTION_PROMPT = """Your attempt to defeat the entity failed, which means your action sequence was wrong.

Your previous hypothesis: {hypothesis}
Your previous plan: {plan}

Recent observations:
{observations}

Revise your hypothesis using this new evidence, then update your plan.

Hypothesis: [revised rule hypothesis]
Plan: [updated action plan]"""


# =============================================================================
# ACE (Abstract Causal Exploration) — reflector and deduplication
# =============================================================================

ACE_REFLECTOR_PROMPT_TEMPLATE = """You are analyzing a failed episode trace to improve a strategy playbook for rule induction tasks.

The agent must infer a hidden rule from source demonstrations and apply it to a novel entity. Study the failure and extract lessons about HOW to reason about the rule — not what the specific answer is.

Current playbook (may be empty):
---
{playbook_text}
---

Failed episode trace:
---
{trace_text}
---

Extract induction strategies from this failure. Focus on reasoning patterns (e.g. "compare entities across both attribute dimensions systematically"), not task-specific answers.

Output ONLY valid JSON:
{{
  "new_bullets": ["strategy 1", "strategy 2"],
  "helpful_ids": ["b001"],
  "harmful_ids": ["b003"]
}}
Empty lists are fine."""

ACE_DEDUP_PROMPT_TEMPLATE = """Below is a strategy playbook. Identify any bullets that are semantically redundant (same advice worded differently). For each redundant group, keep the most informative one and remove the rest.

Playbook:
---
{playbook_text}
---

Output ONLY valid JSON:
{{"remove_ids": ["b002", "b005"]}}
If nothing is redundant, output: {{"remove_ids": []}}"""
