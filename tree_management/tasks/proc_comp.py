"""
tree_management/tasks/proc_comp.py
P-Compositional: procedural independent mapping.

Rule file (proc_comp.json) is a pure property specification:
  - Declares independent input dimensions (class, role)
  - No property-to-property f; all attributes are independent
  - output_category signals the output domain is a process, not an item

Task specification (this file) owns the attribute-to-process binding —
the hidden rule the agent must induce from demonstrations:

  class=0 (warrior), role=0 (scout)  →  perform_before
  class=0 (warrior), role=1 (guard)  →  perform_after
  class=1 (mage),    role=0 (scout)  →  drink_before
  class=1 (mage),    role=1 (guard)  →  drink_after   ← gen: never demonstrated

Hidden nodes: extra_step
  The agent sees only the base sequence (buy weapon → defeat entity) in the
  rules text. The extra step (perform/drink) appears only in demo traces; the
  agent must induce its action type and ordering and apply both to the gen entity.
  Neither perform nor drink has a location dependency — both are done directly.

Extra step argument consistency:
  Each action type resolves to exactly one named argument per task variant:
    perform → one ritual name  (activity.ritual pool, n_items=1)
    drink   → one potion name  (activity.potion pool,  n_items=1)
  The agent can learn the fixed argument from source demos and reproduce it
  for the gen entity.

Anti-copying guarantee (independent_leave_one_out split):
  Source demos cover all individual (class, role) values but no source demo
  shares the exact (class, role) combination of the gen entity. The gen process
  template is never demonstrated; correct behaviour requires independently
  applying the class→action-type binding and the role→ordering binding.
"""

# ---------------------------------------------------------------------------
# Attribute-to-process binding  (the hidden rule; lives in task spec, not rule file)
# ---------------------------------------------------------------------------

# Maps (class_idx, role_idx) → index into PROCESS_TEMPLATES.
# The agent must induce this mapping from source demonstrations.
PROCESS_MAP: dict = {
    (0, 0): 0,   # warrior + scout  →  perform_before
    (0, 1): 1,   # warrior + guard  →  perform_after
    (1, 0): 2,   # mage    + scout  →  drink_before
    (1, 1): 3,   # mage    + guard  →  drink_after
}

PROCESS_TEMPLATES: list = [
    "tree_management/processes/proc_comp/p_proc_comp_perform_before.json",
    "tree_management/processes/proc_comp/p_proc_comp_perform_after.json",
    "tree_management/processes/proc_comp/p_proc_comp_drink_before.json",
    "tree_management/processes/proc_comp/p_proc_comp_drink_after.json",
]

# ---------------------------------------------------------------------------
# Slot bindings, hidden nodes, property bindings
# ---------------------------------------------------------------------------

SLOT_BINDINGS: dict = {
    "defeat_0": "proc_comp:input",
    # No rule_output: the process template is the output.
    # buy_0 is visible; its argument is sampled from object.weapon (not hidden).
}

HIDDEN_NODES: list = ["extra_step"]

PROPERTY_BINDINGS: dict = {
    "buy_0": {"cost": 30},
}

# Extra step argument pools: one named item per pool per task variant so the
# argument is consistent across all entities sharing the same action type.
EXTRA_STEP_POOLS: dict = {
    "activity.ritual": {"name": ["rituals"], "n_items": 1},
    "activity.potion": {"name": ["potions"], "n_items": 1},
}

CORRECT_RULE = (
    "Each attribute independently controls a separate dimension of the required "
    "process. The entity's class determines the action type of the extra step "
    "(perform a ritual or drink a potion). The entity's role determines when "
    "that step occurs relative to buying the weapon (before or after). "
    "Neither attribute influences the dimension controlled by the other."
)

MC_ANSWER = "B"  # independent dimensions → each attribute controls one axis

DESCRIPTION = (
    "class → step_type (perform|get), role → ordering (before|after buy); "
    "attribute-to-process binding in task spec, rule file declares inputs only"
)

from tree_management.generator import register_procedural_task_type
from tree_management.function_specs.splits import validate_independent_split

register_procedural_task_type(
    name              = "proc_comp",
    process_templates = PROCESS_TEMPLATES,
    process_map       = PROCESS_MAP,
    slot_bindings     = SLOT_BINDINGS,
    hidden_nodes      = HIDDEN_NODES,
    property_bindings = PROPERTY_BINDINGS,
    extra_step_pools  = EXTRA_STEP_POOLS,
    split             = {"fn": "independent_leave_one_out", "seed": 0},
    rules             = "tree_management/rules/proc_comp.json",
    correct_rule      = CORRECT_RULE,
    mc_answer         = MC_ANSWER,
    description       = DESCRIPTION,
    validate_fn       = validate_independent_split,
    max_tries         = 4,   # 4 distinct process templates (perform/drink × before/after)
)
