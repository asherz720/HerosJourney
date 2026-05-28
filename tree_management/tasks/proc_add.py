"""
tree_management/tasks/proc_add.py
P-Additive: total extra steps = base(class) + modifier(role).

Rule file (proc_add.json) declares 2 classes and 2 roles.
Both attributes contribute independently to the same structural knob:
how many times the ritual must be performed before buying the weapon.

  class=0 (warrior) → base = 1 step
  class=1 (mage)    → base = 2 steps

  role=0 (scout)    → modifier = 0
  role=1 (guard)    → modifier = 1

  total = base + modifier:
    (0, 0) → 1   (warrior + scout)
    (0, 1) → 2   (warrior + guard)
    (1, 0) → 2   (mage    + scout)
    (1, 1) → 3   (mage    + guard)  ← gen: never demonstrated

The agent sees 3 source demos and must decompose the additive contributions:
  (0,0)→1 step  (0,1)→2 steps  (1,0)→2 steps
From these it must infer base=[1,2] and modifier=[0,1], then produce 3 steps
for the gen entity.

All extra steps perform the same ritual (sampled once per variant). The agent
learns the COUNT, not the specific ritual identity.

The env enforces the count at defeat time via action_counts comparison against
the number of perform nodes in the tree (see env.py _count_required_actions).

Process templates:
  0: p_proc_add_1.json  (1 perform before buy)
  1: p_proc_add_2.json  (2 performs before buy)
  2: p_proc_add_3.json  (3 performs before buy)

Split: independent_leave_one_out on the 2×2 grid (same reasoning as proc_comp:
two_offset degenerates for n_a1=n_a2=2).
"""

# ---------------------------------------------------------------------------
# Attribute-to-process binding
# ---------------------------------------------------------------------------

# (class_idx, role_idx) → index into PROCESS_TEMPLATES
PROCESS_MAP: dict = {
    (0, 0): 0,   # warrior + scout → 1 extra step
    (0, 1): 1,   # warrior + guard → 2 extra steps
    (1, 0): 1,   # mage    + scout → 2 extra steps
    (1, 1): 2,   # mage    + guard → 3 extra steps  ← gen: never demonstrated
}

PROCESS_TEMPLATES: list = [
    "tree_management/processes/proc_add/p_proc_add_1.json",  # 0: 1 step
    "tree_management/processes/proc_add/p_proc_add_2.json",  # 1: 2 steps
    "tree_management/processes/proc_add/p_proc_add_3.json",  # 2: 3 steps
]

# ---------------------------------------------------------------------------
# Slot bindings, hidden nodes, property bindings
# ---------------------------------------------------------------------------

SLOT_BINDINGS: dict = {
    "defeat_0": "proc_add:input",
}

# All extra_step_N IDs across all templates — build_tree_from_process only adds
# to rules_to_skip for IDs that actually appear in the selected template.
HIDDEN_NODES: list = ["extra_step_0", "extra_step_1", "extra_step_2"]

PROPERTY_BINDINGS: dict = {
    "buy_0": {"cost": 30},
}

EXTRA_STEP_POOLS: dict = {
    "activity.ritual": {"name": ["rituals"], "n_items": 1},
}

CORRECT_RULE = (
    "Both the entity's class and role independently contribute a fixed number of "
    "ritual performances to a shared total count. The class determines the base "
    "number of performances, and the role adds a modifier. The total number of "
    "times the ritual must be performed before buying the weapon equals "
    "base(class) + modifier(role). "
    "Changing either attribute shifts the count by its independent contribution, "
    "regardless of the other attribute's value."
)

MC_ANSWER = "C"  # additive count → both shift a single quantity

DESCRIPTION = (
    "n_extra_steps = base(class) + modifier(role); "
    "both attributes operate on the same structural knob (step count); "
    "count enforced at defeat time via action_counts in env"
)

from tree_management.generator import register_procedural_task_type
from tree_management.function_specs.splits import validate_independent_split

register_procedural_task_type(
    name              = "proc_add",
    process_templates = PROCESS_TEMPLATES,
    process_map       = PROCESS_MAP,
    slot_bindings     = SLOT_BINDINGS,
    hidden_nodes      = HIDDEN_NODES,
    property_bindings = PROPERTY_BINDINGS,
    extra_step_pools  = EXTRA_STEP_POOLS,
    split             = {"fn": "independent_leave_one_out", "seed": 0},
    rules             = "tree_management/rules/proc_add.json",
    correct_rule      = CORRECT_RULE,
    mc_answer         = MC_ANSWER,
    description       = DESCRIPTION,
    validate_fn       = validate_independent_split,
    max_tries         = 3,   # 3 distinct step-count templates (1, 2, 3 extra steps)
)
