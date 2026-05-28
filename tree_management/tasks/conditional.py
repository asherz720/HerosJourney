"""
tree_management/tasks/conditional.py
Property 3: conditional mapping.
class selects a regime; the regime determines which dimension role controls.
"""
from tree_management.generator import register_task_type, load_process
from tree_management.function_specs.splits import validate_conditional_split

register_task_type(
    name          = "conditional",
    process       = load_process("property_flat"),
    slot_bindings = {
        "defeat_0": "conditional:input",
        "buy_0":    "conditional:output",
    },
    hidden_nodes      = ["buy_0"],
    property_bindings = {"buy_0": {"cost": 30}},
    split             = {"fn": "conditional_two_offset", "seed": 0},
    rules         = "tree_management/rules/conditional.json",
    correct_rule  = (
        "The effect of attribute-2 (role) on the item depends entirely on which attribute-1 "
        "(class) value is present. For some class values, changing role causes one observable "
        "property of the item to vary while another property stays fixed. For other class "
        "values, changing role causes the opposite: the previously fixed property varies and "
        "the previously varying property becomes fixed. "
        "In other words, class determines which dimension of the item role controls."
    ),
    mc_answer           = "D",
    distractor_rules    = "tree_management/distractors/canonical_property_distractor.json",
    description         = "class selects a regime; regime determines which dimension role controls",
    validate_fn         = validate_conditional_split,
    max_tries           = 6,   # regime-0: 3 items (size∈{0,1,2}, color fixed); regime-1: 3 items (size fixed, color∈{0,1,2})
)
