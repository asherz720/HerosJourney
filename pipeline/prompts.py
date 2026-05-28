"""
Centralized prompts for the adventure story pipeline.

- Student: base game prompt, steering method prompts (ReAct, HR, IDEA).
- Teacher: ACE playbook reflector and deduplication.
- Q&A evaluation prompts.
"""


# =============================================================================
# Generalization benchmark prompt
# =============================================================================

GENERALIZATION_BASE_PROMPT = '''\
You are playing an adventure game. The world contains various entities — some of which \
you will encounter in demonstration episodes below, others in the task you must solve.

Available actions:
  go [location]       — travel to a location
  buy [object]        — purchase an item at the current location
  get [object]        — pick up a free item at the current location
  defeat [enemy]      — defeat an enemy (requires being at their location)
  rescue [npc]        — rescue an NPC (requires being at their location)
  perform [ritual]    — perform a ritual
  drink [potion]      — drink a potion
  check_inventory     — list your current items
  check_location      — show your current location

Rules:
- You can only perform ONE action at a time.
- Structure each action as {"action": "<name>", "argument": "<value>", "reasoning": "<brief reason>"}.
- The rules shown for your task may not be complete — some requirements must be inferred \
from the demonstration episodes.
- Study the demonstration episodes carefully.

Output your response as a JSON object with this exact structure:
{
  "action": "<action_name>",
  "argument": "<argument_value>",
  "reasoning": "<your_reasoning>"
}\n\nExample: To go to the market, output:
{
  "action": "go",
  "argument": "market",
  "reasoning": "I need to go to the market to buy a sword."
}Output ONLY the JSON object, nothing else.
'''

GENERALIZATION_BASE_PROMPT_NO_REASONING = '''\
You are playing an adventure game. The world contains various entities — some of which \
you will encounter in demonstration episodes below, others in the task you must solve.

Available actions:
  go [location]       — travel to a location
  buy [object]        — purchase an item at the current location
  get [object]        — pick up a free item at the current location
  defeat [enemy]      — defeat an enemy (requires being at their location)
  rescue [npc]        — rescue an NPC (requires being at their location)
  perform [ritual]    — perform a ritual
  drink [potion]      — drink a potion
  check_inventory     — list your current items
  check_location      — show your current location

Rules:
- You can only perform ONE action at a time.
- Structure each action as {"action": "<name>", "argument": "<value>"}.
- The rules shown for your task may not be complete — some requirements must be inferred \
from the demonstration episodes.
- Study the demonstration episodes carefully.

Output your response as a JSON object with this exact structure:
{
  "action": "<action_name>",
  "argument": "<argument_value>"
}\n\nExample: To go to the market, output:
{
  "action": "go",
  "argument": "market"
}Output ONLY the JSON object, nothing else.
'''

REASONING_CONTEXT_PREAMBLE = '''\
You are studying an adventure game. The world contains various entities — some of which \
you will encounter in demonstration episodes below, others in the task you must solve.

Available actions:
  go [location]       — travel to a location
  buy [object]        — purchase an item at the current location
  get [object]        — pick up a free item at the current location
  defeat [enemy]      — defeat an enemy (requires being at their location)
  rescue [npc]        — rescue an NPC (requires being at their location)
  perform [ritual]    — perform a ritual
  drink [potion]      — drink a potion
  check_inventory     — list your current items
  check_location      — show your current location

Study the demonstration episodes carefully. Respond in plain text.
'''


# =============================================================================
# Q&A evaluation prompt
# =============================================================================

ACTION_JSON_REMINDER = '''\
Output your response as a JSON object with this exact structure:
{
  "action": "<action_name>",
  "argument": "<argument_value>",
  "reasoning": "<your_reasoning>"
}

Example: To go to the market, output:
{
  "action": "go",
  "argument": "market",
  "reasoning": "I need to go to the market to buy a sword."
}Output ONLY the JSON object, nothing else.'''

ACTION_JSON_REMINDER_NO_REASONING = '''\
Output your response as a JSON object with this exact structure:
{
  "action": "<action_name>",
  "argument": "<argument_value>"
}

Example: To go to the market, output:
{
  "action": "go",
  "argument": "market"
}Output ONLY the JSON object, nothing else.'''


def format_teaching_block(teaching_message: str, anchor_text: str = "") -> str:
    """Wrap a teaching message in the standard delimiters used by both episode and Q&A pipelines.

    anchor_text: if non-empty, appended after the block to re-anchor the model to JSON output.
                 Pass ACTION_JSON_REMINDER or ACTION_JSON_REMINDER_NO_REASONING for episode calls.
    """
    if not teaching_message or not teaching_message.strip():
        return ""
    block = f"\n[START of Teaching Message]\n{teaching_message}\n[END of Teaching Message]"
    if anchor_text:
        block += f"\n\n{anchor_text}"
    return block


QA_BASE_PROMPT = """\
You are studying demonstration episodes from an adventure game. Your task is to \
analyze the patterns in the demonstrations and answer questions about the underlying rules.

Each demonstration shows an entity's attributes and the complete sequence of actions \
taken to achieve it. Study these carefully to identify what determines which item \
each entity needs.
"""


# =============================================================================
# Q&A prompts — property tasks (item-selection)
# =============================================================================

QA_INSTANCE_QUESTION = """\
Based on the demonstration episodes above, answer the following:

Entity: {entity_card}
What item must be bought to defeat {entity_name}?

Respond with ONLY the JSON object below — no other text, no markdown:
{{"item": "<item_name>", "reasoning": "<brief reasoning>"}}

The "item" field must appear first."""

# QA_STRUCTURE_MC_QUESTION = """\
# Based on the demonstration episodes above, which of the following best describes \
# the rule that determines which item each entity needs?
#
# A. Only one attribute matters; the other is completely irrelevant to the item choice.
# B. The two attributes independently control separate observable properties of the item — \
# varying one attribute changes one property while the other stays fixed, and vice versa.
# C. Both attributes shift the item along a single ordering — the contribution of each \
# attribute value is the same regardless of what the other attribute is.
# D. One attribute determines which dimension the other attribute controls — changing the \
# first attribute changes what effect the second attribute has on the item.
# E. One attribute provides a default item per entity; one specific value of the other \
# attribute always overrides this to a single fixed item regardless of the first attribute.
#
#
# Respond with ONLY the JSON object below — no other text, no markdown:
# {{"choice": "A"|"B"|"C"|"D"|"E", "reasoning": "<brief reasoning>"}}
#
# The "choice" field must appear first."""

QA_STRUCTURE_EXP_QUESTION = """\
Based only on the demonstration episodes above, describe the general rule that \
determines which item each entity needs.

Your explanation should address three things:

1. **Inputs (X)**: Which attributes of the entities determine the item, \
and what role each one plays.
2. **Output (Y)**: What the required item consists of — describe which observable \
properties of the item vary across entities.
3. **Rule structure (g)**: How the input attributes map to the output item — \
precise enough that someone who had not seen any demonstrations could use it \
to predict the correct item for a new entity.

Respond with ONLY the JSON object below — no other text, no markdown:
{{"explanation": "<your description of the rule, covering X, Y, and g>"}}"""

QA_JUDGE_PROMPT = """\
You are evaluating whether a model correctly understood a generalization rule \
from demonstration episodes — specifically, the rule that determines which item \
each entity requires.

Important context: the model only observed item names in the demos — it never saw \
internal numeric values, size indices, or modifier integers. A correct explanation \
should state a general rule that could predict the item for any new entity — \
not merely enumerate specific instances seen in the demonstrations.

The correct rule is:
---
{correct_rule}
---

The model's explanation:
---
{model_explanation}
---

Score the explanation on FOUR dimensions, each from 0 to 2:

**input_score** — Did the model correctly identify which attributes determine the item, \
and the role each plays?
  2: Correctly identifies all relevant attributes and what each one contributes
  1: Identifies at least one attribute's role correctly; misses or misstates the other
  0: Wrong — misidentifies which attributes matter or assigns them to the wrong roles

**output_score** — Did the model correctly describe the output item?
  (Relevant dimensions vary by task: item type, size, color, etc. Only dimensions \
  that vary across entities matter.)
  2: Correctly identifies all applicable output dimensions that vary
  1: Identifies some but not all applicable output dimensions
  0: Misidentifies or ignores the output dimensions entirely

**rule_score** — Did the model correctly describe the mapping structure connecting \
attributes to the item?
  2: Correctly captures the relational structure (e.g., consistent additive shift, \
     independent dimensions, conditional regime, or override exception), even if \
     using different words
  1: Gets the direction right but misses an important aspect (e.g., says "both matter" \
     but misdescribes the interaction)
  0: Wrong structural claim

**generalization_score** — Did the model state a general rule or just describe seen instances?
  2: States a clear general rule that would apply to unseen attribute combinations
  1: Mix — some general language but falls back on listing specific demo instances
  0: Only lists specific seen instances with no general rule

Do NOT penalize for different wording, for using item names instead of abstract terms,
or for not enumerating every value. DO penalize for wrong structure, missing output
dimensions, or instance-only reasoning.

Respond with ONLY the JSON object below — no other text, no markdown:
{{
  "input_score": 0|1|2,
  "output_score": 0|1|2,
  "rule_score": 0|1|2,
  "generalization_score": 0|1|2,
  "reasoning": "<two or three sentences covering all four dimensions>"
}}

The four score fields must appear before "reasoning"."""


# =============================================================================
# Q&A prompts — procedural tasks (process-selection)
# =============================================================================

QA_INSTANCE_PROC_QUESTION = """\
The base action sequence for this task type is:
  go [weapon shop] → buy [weapon] → go [entity location] → defeat [entity]

Some entities require one or more extra steps inserted into this sequence. \
The rules text does not show the extra step — you must infer it from the demonstrations.

Entity: {entity_card}
Based on the demonstrations, what extra step(s) must be performed for {entity_name}?

Respond with ONLY the JSON object below — no other text, no markdown:
{{"action": "perform|drink|none", "argument": "<ritual_or_potion_name_or_none>", \
"ordering": "before_buy|after_buy|none", "count": <integer, 0 if none>, \
"reasoning": "<brief reasoning>"}}

The "action" field must appear first."""

# QA_STRUCTURE_MC_QUESTION_PROC = """\
# Based on the demonstration episodes above, which of the following best describes \
# the rule that determines what extra step (if any) each entity must perform, and when?
#
# A. Only one attribute matters; the other is completely irrelevant to the extra step.
# B. The two attributes independently control separate dimensions of the extra step — \
# one determines the action type (perform vs. drink), and the other determines the \
# timing relative to buying the weapon (before vs. after). Neither affects the dimension \
# controlled by the other.
# C. Both attributes contribute to the same quantity — the total number of times the \
# action must be performed equals the sum of independent contributions from each attribute.
# D. One attribute selects a "regime" that determines what the other attribute controls — \
# changing the first attribute changes what effect the second attribute has on the process.
# E. One attribute determines a default process per entity; one specific value of the other \
# attribute always overrides this to a fixed process regardless of the first attribute.
#
#
# Respond with ONLY the JSON object below — no other text, no markdown:
# {{"choice": "A"|"B"|"C"|"D"|"E", "reasoning": "<brief reasoning>"}}
#
# The "choice" field must appear first."""

QA_STRUCTURE_EXP_QUESTION_PROC = """\
The base action sequence for this task type is:
  go [weapon shop] → buy [weapon] → go [entity location] → defeat [entity]

Some entities require one or more extra steps inserted into this sequence. \
The extra step and its details are NOT shown in the rules text — they must be inferred \
from the demonstrations.

Based only on the demonstration episodes above, describe the general rule that \
determines the extra step. Your explanation should identify three things:

1. **Inputs (X)**: Which attributes of the entities determine the extra step, \
and what role each one plays.
2. **Output dimensions (Y)**: What the extra step consists of — describe which \
observable properties of the step vary across entities.
3. **Rule structure (g)**: How the input attributes map to the output dimensions — \
precise and general enough that someone who had not seen any demonstrations could use it \
to predict the correct extra step for a new entity.

Respond with ONLY the JSON object below — no other text, no markdown:
{{"explanation": "<your description of the rule, covering X, Y, and g>"}}"""

QA_TEACHER_PROMPT_TEMPLATE = """\
You are a teacher reviewing a student's Q&A answers about an adventure-game rule.

The student studied demonstration episodes and then answered questions about the \
underlying rule. Here are their results:

{qa_failure_summary}

The correct rule is:
---
{correct_rule}
---

Write a concise teaching message that will help the student identify the correct rule \
when they try again. Do not reveal the exact answer. Focus specifically on the aspects \
the student got wrong — guide them to notice the pattern themselves.

Output only the teaching message (plain text), no preamble."""


QA_JUDGE_PROMPT_PROC = """\
You are evaluating whether a model correctly understood a generalization rule \
from demonstration episodes — specifically, the rule that determines what extra step \
(if any) each entity must perform, and when.

Important context: the model only observed action names and argument names in the \
demos — it did not see internal rule representations. A correct explanation should \
state a general rule that could predict the extra step for any new entity — \
not merely enumerate specific instances seen in the demonstrations.

The correct rule is:
---
{correct_rule}
---

The model's explanation:
---
{model_explanation}
---

Score the explanation on FOUR dimensions, each from 0 to 2:

**input_score** — Did the model correctly identify which attributes control which process dimensions?
  2: Correctly identifies all relevant attributes and what each one determines
  1: Identifies at least one attribute's role correctly; misses or misstates the other
  0: Wrong — misidentifies which attributes matter or assigns them to the wrong dimensions

**output_score** — Did the model identify all observable dimensions of the extra step?
  (Relevant dimensions vary by task: action type [perform vs. drink], ordering [before vs. after buy],
   count [how many times]. Only dimensions that vary across entities are relevant.)
  2: Correctly identifies all applicable output dimensions
  1: Identifies some but not all applicable dimensions
  0: Misidentifies or ignores the output dimensions entirely

**rule_score** — Did the model correctly describe the mapping structure connecting inputs to outputs?
  2: Correctly captures the relational structure (e.g., additive count, independent dimensions,
     conditional regime selection, or override exception), even if using different words
  1: Gets the direction right but misses an important aspect (e.g., says "both attributes matter"
     but misdescribes how they interact)
  0: Wrong structural claim

**generalization_score** — Did the model state a general rule or just describe seen instances?
  2: States a clear general rule that would apply to unseen attribute combinations
  1: Mix — some general language but falls back on listing specific demo instances
  0: Only lists specific seen instances with no general rule

Examples of good vs. weak answers (for an independent-dimensions task):

  Good (all scores = 2):
    "The class determines the type of extra step — warrior entities perform a ritual while
    mage entities drink a potion. The role independently determines the timing — scout role
    means the step happens before buying the weapon, guard role means after. The two
    attributes operate on separate dimensions and do not interact."

  Weak on rule_score (score 1):
    "Both the entity's class and role affect what must be done before defeating the entity,
    but I'm not sure exactly how they interact."

  Weak on output_score (score 1):
    "The class determines whether to perform a ritual or drink a potion, but I didn't
    notice a clear pattern for when to do it."

  Wrong (all scores = 0):
    "Each entity needs a specific weapon based on its class."

Do NOT penalize for different wording or for using concrete demo values as examples, \
as long as the general rule is also stated. DO penalize for wrong structure, missing \
output dimensions, or instance-only descriptions.

Respond with ONLY the JSON object below — no other text, no markdown:
{{
  "input_score": 0|1|2,
  "output_score": 0|1|2,
  "rule_score": 0|1|2,
  "generalization_score": 0|1|2,
  "reasoning": "<two or three sentences covering all four dimensions>"
}}

The four score fields must appear before "reasoning"."""


# =============================================================================
# Teacher prompts (Phase 1.1 curation and interactive teaching)
# =============================================================================

# --- Self-refinement: initial and refinement rounds ---
# --- ACE: reflector and deduplication ---

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


# =============================================================================
# Steering method prompts (Phase 2 student-side)
# =============================================================================

REACT_STUDENT_PROMPT = """Before each action, think about your current hypothesis about the hidden rule.

You can reason about the following aspeacts:
  Hypothesis: [your current best guess at the mapping rule]
  Evidence: [which demonstration examples support this]
  Action rationale: [why this action follows from your hypothesis]

Follow the output JSON format and do not include reasoning in your response."""


HR_HYPOTHESIS_PROMPT = """Before acting, reason systematically about the hidden rule using the source demonstrations above.

Step 1 — Generate hypotheses: propose {num_hypotheses} distinct candidate rules explaining how different goals are achieved. Consider different structural possibilities for how the attributes might combine. Note that not all goals show a uniform pattern.

Step 2 — Verify each: check whether each candidate correctly predicts the item for every demonstration entity. Mark each as consistent or inconsistent.

Step 3 — Select: choose the best-supported hypothesis.

Output your selected hypothesis, then proceed to act:
Selected hypothesis: [rule]
Confidence: [high/medium/low]
Key evidence: [which demos support it]"""


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
