"""
tree_management/tasks/additive.py
Property 1: additive mapping.
item_size = base(class) + modifier(role)
"""
from tree_management.generator import register_task_type, load_process
from tree_management.function_specs.splits import validate_additive_split

register_task_type(
    name          = "additive",
    process       = load_process("property_flat"),
    slot_bindings = {
        "defeat_0": "additive:input",   # entity whose attributes the rule reads
        "buy_0":    "additive:output",  # item determined by the rule
    },
    hidden_nodes      = ["buy_0"],
    property_bindings = {"buy_0": {"cost": 30}},
    split             = {"fn": "two_offset", "seed": 0},
    rules         = "tree_management/rules/additive.json",
    correct_rule  = (
        "Both attributes affect the item, and their effects are consistent and separable. "
        "Changing attribute-1 (class) while keeping attribute-2 (role) fixed always shifts "
        "the item by the same amount on an ordering of items, regardless of which role value "
        "is held fixed. Likewise, changing role while keeping class fixed always produces the "
        "same shift regardless of class. "
        "The item can be read off as: the item you would get for that class with a 'neutral' "
        "role, shifted up or down according to the role's consistent contribution. "
        "In short: both attributes contribute independently, and the contribution of each "
        "attribute value is the same no matter what the other attribute is."
    ),
    mc_answer           = "C",
    distractor_rules    = "tree_management/distractors/canonical_property_distractor.json",
    description         = "item_size = base(class) + modifier(role)",
    validate_fn         = validate_additive_split,
    max_tries           = 5,   # size ∈ {0,1,2,3,4} — 5 distinct items
)
