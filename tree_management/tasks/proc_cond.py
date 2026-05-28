"""
tree_management/tasks/proc_cond.py
P-Conditional: process selection driven by a two-regime conditional mapping.

Rule file (proc_cond.json) declares 4 classes and 2 roles.
Classes are partitioned into two regimes (2 per regime).
The regime determines which dimension of the extra step role controls:

  Regime 0 (warrior, mage — class 0, 1):
    role controls the ACTION TYPE; ordering is fixed (before buy).
      role=0 (scout) → perform_before
      role=1 (guard) → drink_before

  Regime 1 (rogue, paladin — class 2, 3):
    role controls the ORDERING; action type is fixed (drink).
      role=0 (scout) → drink_before
      role=1 (guard) → drink_after   ← gen: never demonstrated (leaf-one-out)

The agent cannot identify the rule from either attribute alone:
  - Class only → determines regime membership, not the full process.
  - Role only → ambiguous (role=0 maps to perform_before in regime 0 and drink_before in regime 1).
  The conditional structure must be induced from demonstration patterns.

Process templates reused from proc_comp (no new JSON files needed):
  0: p_proc_comp_perform_before.json
  1: p_proc_comp_drink_before.json
  2: p_proc_comp_drink_after.json

Split: independent_leave_one_out on the 4×2 grid.
  (conditional_two_offset degenerates to full coverage for n_roles=2, so leave-one-out
  is the correct minimal split here.)
  With seed=0, the held-out pair is in regime 1, ensuring the gen entity tests
  both regime identification and within-regime mapping.

Hidden node: extra_step
  Appears only in demo traces; the agent must induce action type, ordering,
  and argument from source demonstrations.
"""

# ---------------------------------------------------------------------------
# Attribute-to-process binding
# ---------------------------------------------------------------------------

# (class_idx, role_idx) → index into PROCESS_TEMPLATES
PROCESS_MAP: dict = {
    (0, 0): 0,   # warrior + scout → perform_before  (regime 0: role controls action type)
    (0, 1): 1,   # warrior + guard → drink_before
    (1, 0): 0,   # mage    + scout → perform_before
    (1, 1): 1,   # mage    + guard → drink_before
    (2, 0): 1,   # rogue   + scout → drink_before    (regime 1: role controls ordering)
    (2, 1): 2,   # rogue   + guard → drink_after
    (3, 0): 1,   # paladin + scout → drink_before
    (3, 1): 2,   # paladin + guard → drink_after     ← gen: never demonstrated
}

PROCESS_TEMPLATES: list = [
    "tree_management/processes/proc_comp/p_proc_comp_perform_before.json",  # 0
    "tree_management/processes/proc_comp/p_proc_comp_drink_before.json",    # 1
    "tree_management/processes/proc_comp/p_proc_comp_drink_after.json",     # 2
]

# ---------------------------------------------------------------------------
# Slot bindings, hidden nodes, property bindings
# ---------------------------------------------------------------------------

SLOT_BINDINGS: dict = {
    "defeat_0": "proc_cond:input",
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
    "The entity's class determines which dimension of the required extra step the role "
    "controls. For classes in regime 0, role determines the action type (whether to perform "
    "a ritual or drink a potion) while the ordering relative to buying the weapon is fixed. "
    "For classes in regime 1, role determines the ordering (before or after buying the weapon) "
    "while the action type is fixed. "
    "Neither attribute can be understood in isolation — the class must first be identified as "
    "belonging to one of two regimes to know what role governs."
)

MC_ANSWER = "D"  # conditional/regime → first attribute selects what second controls

DESCRIPTION = (
    "class selects a regime (action-type regime or ordering regime); "
    "regime determines which process dimension role controls; "
    "PROCESS_MAP in task spec, rule file declares inputs only"
)

from tree_management.generator import register_procedural_task_type
from tree_management.function_specs.splits import validate_independent_split

register_procedural_task_type(
    name              = "proc_cond",
    process_templates = PROCESS_TEMPLATES,
    process_map       = PROCESS_MAP,
    slot_bindings     = SLOT_BINDINGS,
    hidden_nodes      = HIDDEN_NODES,
    property_bindings = PROPERTY_BINDINGS,
    extra_step_pools  = EXTRA_STEP_POOLS,
    split             = {"fn": "independent_leave_one_out", "seed": 0},
    rules             = "tree_management/rules/proc_cond.json",
    correct_rule      = CORRECT_RULE,
    mc_answer         = MC_ANSWER,
    description       = DESCRIPTION,
    validate_fn       = validate_independent_split,
    max_tries         = 3,   # 3 distinct process templates (perform_before, drink_before, drink_after)
)
