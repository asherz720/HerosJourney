"""
tree_management/tasks/compositional.py
Property 2: independent/compositional mapping.
class → size dimension, role → color dimension (fully independent).
"""
from tree_management.generator import register_task_type, load_process
from tree_management.function_specs.splits import validate_independent_split

register_task_type(
    name          = "compositional",
    process       = load_process("property_flat"),
    slot_bindings = {
        "defeat_0": "compositional:input",
        "buy_0":    "compositional:output",
    },
    hidden_nodes      = ["buy_0"],
    property_bindings = {"buy_0": {"cost": 30}},
    split             = {"fn": "two_offset", "seed": 0},
    rules         = "tree_management/rules/compositional.json",
    correct_rule  = (
        "Each attribute independently controls a completely separate observable property of "
        "the item. Varying attribute-1 (class) changes one property of the item while the "
        "other property stays the same. Varying attribute-2 (role) changes the other property "
        "while the first stays the same. "
        "The required item is the one that simultaneously matches the property determined by "
        "class and the property determined by role. "
        "Neither attribute has any influence over the dimension controlled by the other."
    ),
    mc_answer           = "B",
    distractor_rules    = "tree_management/distractors/canonical_property_distractor.json",
    description         = "class → size, role → color (independent dimensions)",
    validate_fn         = validate_independent_split,
    max_tries           = 9,   # 3 sizes × 3 colors — 9 distinct items
)
