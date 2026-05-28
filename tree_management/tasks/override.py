"""
tree_management/tasks/override.py
Property 4: override/exception mapping.
class determines the default item; one role value always overrides to a fixed item.
"""
from tree_management.generator import (
    register_task_type,
    load_process,
)

register_task_type(
    name          = "override",
    process       = load_process("property_flat"),
    slot_bindings = {
        "defeat_0": "override:input",
        "buy_0":    "override:output",
    },
    hidden_nodes      = ["buy_0"],
    property_bindings = {"buy_0": {"cost": 30}},
    split             = {"fn": "c4_override", "seed": 0},
    rules         = "tree_management/rules/override.json",
    correct_rule  = (
        "Attribute-1 (class) determines a distinct default item for each entity. "
        "However, one specific value of attribute-2 (role) breaks this pattern entirely: "
        "whenever that role value appears, the item is always the same single item for every "
        "entity regardless of class. "
        "All other role values leave the class-determined item intact. "
        "A model that only looks at non-override role demos will see a clean class→item mapping; "
        "the override is only visible when that one role value appears."
    ),
    mc_answer           = "E",
    distractor_rules    = "tree_management/distractors/canonical_property_distractor.json",
    description         = "class → item by default; one role value always overrides to a fixed item",
    validate_fn         = None,
    max_tries           = 4,   # size ∈ {0,1,2} base + 1 override — 4 distinct items
)
