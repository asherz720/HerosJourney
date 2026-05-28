"""
tree_management/tasks/proc_over.py
P-Override: class determines the default extra step; one role value always
overrides to a fixed process regardless of class.

Rule file (proc_over.json) declares 3 classes and 4 roles (override_value=3).
The hidden rule: each class maps to a default extra step, but role=3 always
overrides to a fixed process (drink_after), independent of class.

  Base processes (class-determined):
    class=0 (warrior) → perform_before
    class=1 (mage)    → drink_before
    class=2 (rogue)   → perform_after

  Override (role=3 always wins):
    any class + role=3 (ranger) → drink_after

Demo structure (c4_override split, 3 classes × 4 roles):
  Source: full coverage for 2 of 3 classes (all 4 roles shown for each),
          plus 1 non-override role for the held-out class.
  Gen:    held-out class with override role + remaining non-override roles.

The key generalization test: the gen entity's class was never demonstrated
with the override role in source. The agent must recognize that role=3 always
produces drink_after, even for an unseen (class, override) combination.

Process templates (all reused from proc_comp — no new JSON files needed):
  0: p_proc_comp_perform_before.json
  1: p_proc_comp_drink_before.json
  2: p_proc_comp_perform_after.json
  3: p_proc_comp_drink_after.json   ← override process

Hidden node: extra_step
  Action type and ordering appear only in demos; the agent must induce
  both the base class→process rule and the override exception.
"""

# ---------------------------------------------------------------------------
# Attribute-to-process binding
# ---------------------------------------------------------------------------

# (class_idx, role_idx) → index into PROCESS_TEMPLATES
PROCESS_MAP: dict = {
    (0, 0): 0,   # warrior + scout   → perform_before  (class base)
    (0, 1): 0,   # warrior + guard   → perform_before  (class base)
    (0, 2): 0,   # warrior + herald  → perform_before  (class base)
    (0, 3): 3,   # warrior + ranger  → drink_after     (override)
    (1, 0): 1,   # mage    + scout   → drink_before    (class base)
    (1, 1): 1,   # mage    + guard   → drink_before    (class base)
    (1, 2): 1,   # mage    + herald  → drink_before    (class base)
    (1, 3): 3,   # mage    + ranger  → drink_after     (override)
    (2, 0): 2,   # rogue   + scout   → perform_after   (class base)
    (2, 1): 2,   # rogue   + guard   → perform_after   (class base)
    (2, 2): 2,   # rogue   + herald  → perform_after   (class base)
    (2, 3): 3,   # rogue   + ranger  → drink_after     (override)  ← partial gen
}

PROCESS_TEMPLATES: list = [
    "tree_management/processes/proc_comp/p_proc_comp_perform_before.json",  # 0
    "tree_management/processes/proc_comp/p_proc_comp_drink_before.json",    # 1
    "tree_management/processes/proc_comp/p_proc_comp_perform_after.json",   # 2
    "tree_management/processes/proc_comp/p_proc_comp_drink_after.json",     # 3  (override)
]

# ---------------------------------------------------------------------------
# Slot bindings, hidden nodes, property bindings
# ---------------------------------------------------------------------------

SLOT_BINDINGS: dict = {
    "defeat_0": "proc_over:input",
}

HIDDEN_NODES: list = ["extra_step"]

PROPERTY_BINDINGS: dict = {
    "buy_0": {"cost": 30},
}

EXTRA_STEP_POOLS: dict = {
    "activity.ritual": {"name": ["rituals"], "n_items": 1},
    "activity.potion": {"name": ["potions"], "n_items": 1},
}

CORRECT_RULE = (
    "Each entity's class determines a default extra step (perform or drink, and its ordering "
    "relative to buying the weapon). However, one specific role value always overrides this "
    "default: whenever that role appears, the extra step is always the same fixed process "
    "regardless of the entity's class. "
    "All other role values leave the class-determined process intact. "
    "A model that observes only non-override role demos will see a clean class→process mapping; "
    "the override is only visible when that one role value appears."
)

MC_ANSWER = "E"  # override → base rule + one role always overrides

DESCRIPTION = (
    "class → default extra step (perform/drink + before/after); "
    "one role value always overrides to a fixed process; "
    "PROCESS_MAP in task spec, rule file declares inputs + composition.override_value"
)

from tree_management.generator import register_procedural_task_type

register_procedural_task_type(
    name              = "proc_over",
    process_templates = PROCESS_TEMPLATES,
    process_map       = PROCESS_MAP,
    slot_bindings     = SLOT_BINDINGS,
    hidden_nodes      = HIDDEN_NODES,
    property_bindings = PROPERTY_BINDINGS,
    extra_step_pools  = EXTRA_STEP_POOLS,
    split             = {"fn": "c4_override", "seed": 0},
    rules             = "tree_management/rules/proc_over.json",
    correct_rule      = CORRECT_RULE,
    mc_answer         = MC_ANSWER,
    description       = DESCRIPTION,
    validate_fn       = None,
    max_tries         = 4,   # 4 distinct process templates (perform_before/after, drink_before/after)
)
