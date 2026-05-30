"""
eval/judge.py
Prompts for the RV (rule verbalization) judge and QA evaluation.

RV judge usage:
    from herosjourney.eval.judge import QA_JUDGE_PROMPT, QA_JUDGE_PROMPT_PROC
    # Fill {correct_rule} and {model_explanation}, call your LLM judge,
    # parse the 4-dimension JSON score.
"""

# =============================================================================
# Q&A evaluation — property tasks (item-selection)
# =============================================================================

QA_BASE_PROMPT = """\
You are studying demonstration episodes from an adventure game. Your task is to \
analyze the patterns in the demonstrations and answer questions about the underlying rules.

Each demonstration shows an entity's attributes and the complete sequence of actions \
taken to achieve it. Study these carefully to identify what determines which item \
each entity needs.
"""

QA_INSTANCE_QUESTION = """\
Based on the demonstration episodes above, answer the following:

Entity: {entity_card}
What item must be bought to defeat {entity_name}?

Respond with ONLY the JSON object below — no other text, no markdown:
{{"item": "<item_name>", "reasoning": "<brief reasoning>"}}

The "item" field must appear first."""

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
# Q&A evaluation — procedural tasks (process-selection)
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
# Teacher prompt (used by the teaching pipeline to generate Phase 2 messages)
# =============================================================================

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
