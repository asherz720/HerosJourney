"""
pipeline/prompts.py
Core game-loop prompts used by the episode runner.

QA / judge prompts  →  eval.judge
Steering prompts    →  pipeline.strategies
"""

# =============================================================================
# Game loop prompts  (runner core)
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
    """Wrap a teaching message in the standard delimiters.

    anchor_text: if non-empty, appended after the block to re-anchor the model to JSON output.
                 Pass ACTION_JSON_REMINDER or ACTION_JSON_REMINDER_NO_REASONING for episode calls.
    """
    if not teaching_message or not teaching_message.strip():
        return ""
    block = f"\n[START of Teaching Message]\n{teaching_message}\n[END of Teaching Message]"
    if anchor_text:
        block += f"\n\n{anchor_text}"
    return block


# =============================================================================
# Backward-compat re-exports
# Code that imports QA / judge / strategy prompts from this module still works.
# New code should import from herosjourney.eval.judge or pipeline.strategies directly.
# =============================================================================

from herosjourney.eval.judge import (          # noqa: E402
    QA_BASE_PROMPT,
    QA_INSTANCE_QUESTION,
    QA_STRUCTURE_EXP_QUESTION,
    QA_JUDGE_PROMPT,
    QA_INSTANCE_PROC_QUESTION,
    QA_STRUCTURE_EXP_QUESTION_PROC,
    QA_JUDGE_PROMPT_PROC,
    QA_TEACHER_PROMPT_TEMPLATE,
)
from herosjourney.runner.strategies import (  # noqa: E402
    REACT_STUDENT_PROMPT,
    HR_HYPOTHESIS_PROMPT,
    IDEA_ABDUCTION_PROMPT,
    IDEA_INDUCTION_PROMPT,
    ACE_REFLECTOR_PROMPT_TEMPLATE,
    ACE_DEDUP_PROMPT_TEMPLATE,
)
