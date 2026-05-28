"""
tree_management/generator.py
Core task-generation pipeline: process spec + filled elements → GeneratedTask list.

Three-layer architecture
------------------------
Layer 1 – Process JSON (pure structure):
    Describes which actions happen, their dependencies, execution order, and
    ordering constraints.  Contains no rule semantics — no rule_ref anywhere.

Layer 2 – Task Spec (binding layer, lives in tasks/my_task.py):
    slot_bindings maps process step IDs to rule roles:
      "rule_input"  – this step's resolved record provides rule input attributes.
                      The entity passed to build_tree_from_process is the rule input
                      by definition; this label documents that binding explicitly.
      "rule_output" – this step's argument is the rule output, looked up via
                      entity["item_id"] set during fill_elements().
      (absent)      – pool sampling: a random item is drawn from the output item pool.
                      Used for distractor tasks where no rule governs item choice.

Layer 3 – Rule File + Lexicon:
    fill_elements() computes entity["item_id"] for every entity according to the
    composition function and stores surface names from the lexicon.
    Distractors (no "functions" key) assign item_id randomly — same entity dict
    format, no special-casing in the generator.

Usage
-----
    from tree_management.elements import fill_elements, load_lexicons
    from tree_management.registry import get_task

    sem, nonce = load_lexicons()
    with open("tree_management/rules/additive.json") as f:
        rule = json.load(f)
    elements = fill_elements(rule, sem, nonce, seed=0)
    tasks = get_task("additive").gen_fn(elements, split="gen")

Adding a new task type
----------------------
Create tree_management/tasks/my_task.py:

    from tree_management.generator import register_task_type, load_process
    from tree_management.function_specs.splits import validate_additive_split

    register_task_type(
        name          = "my_task",
        process       = load_process("property_flat"),
        slot_bindings = {
            "defeat_0": "rule_input",   # entity whose attributes the rule reads
            "buy_0":    "rule_output",  # item determined by the rule
        },
        rules         = "tree_management/rules/my_task.json",
        correct_rule  = "The rule is ...",
        mc_answer     = "B",
        distractor_rules = "tree_management/distractors/canonical_property_distractor.json",
        validate_fn   = validate_additive_split,
    )

Then add one import to tree_management/tasks/__init__.py.

Distractor tasks
----------------
Distractors use the same pipeline with split=None (flat entity list) and an empty
slot_bindings (all object nodes default to pool sampling):

    from tree_management.elements import fill_elements, load_lexicons
    import json

    with open("tree_management/distractors/canonical_property_distractor.json") as f:
        spec = json.load(f)
    sem, nonce = load_lexicons(spec.get("lexicon", "default"))
    elements = fill_elements(spec, sem, nonce, seed=0)
    process  = load_process(spec["process"])
    tasks    = generate_tasks(elements, process, slot_bindings={"defeat_0": "rule_input"}, split=None)
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict
from itertools import groupby
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tree_management.goal_tree import GoalTree
from tree_management.function_specs.item_mappings import item_sem_name, item_nonce_name

_PROCESSES_DIR = os.path.join(os.path.dirname(__file__), "processes")


# ---------------------------------------------------------------------------
# TemplateSpec  (process structure + slot bindings)
# ---------------------------------------------------------------------------

@dataclass
class TemplateSpec:
    """
    Binds a v2 process dict to slot_bindings that map step IDs to rule roles.

    Fields
    ------
    name          : identifier used in TEMPLATE_REGISTRY
    process       : v2 process dict (must have "steps" key)
    slot_bindings : {step_id: role}
                      "<rule>:input"  – step's entity provides rule input attributes
                      "<rule>:output" – step's argument is the rule output (item_id lookup)
                    Steps absent from slot_bindings: type:object nodes use pool sampling,
                    type:entity nodes resolve normally.
    hidden_nodes  : list of step IDs whose arguments are omitted from the agent's rules text.
                    Populates rules_to_skip on each GeneratedTask.
    """
    name: str
    process: Dict
    slot_bindings: Dict[str, str] = field(default_factory=dict)
    hidden_nodes: List[str] = field(default_factory=list)
    property_bindings: Dict[str, Dict] = field(default_factory=dict)


TEMPLATE_REGISTRY: Dict[str, TemplateSpec] = {}


def register_template(spec: TemplateSpec) -> None:
    if spec.name in TEMPLATE_REGISTRY:
        raise ValueError(f"Template '{spec.name}' is already registered.")
    TEMPLATE_REGISTRY[spec.name] = spec


def get_template(name: str) -> TemplateSpec:
    if name not in TEMPLATE_REGISTRY:
        raise ValueError(
            f"Unknown template '{name}'. "
            f"Registered: {sorted(TEMPLATE_REGISTRY)}"
        )
    return TEMPLATE_REGISTRY[name]


# ---------------------------------------------------------------------------
# GeneratedTask
# ---------------------------------------------------------------------------

@dataclass
class GeneratedTask:
    """
    One fully-instantiated task: a GoalTree for a single entity + metadata.

    Fields
    ------
    tree              : GoalTree ready to pass to AdventureEnv
    task_type         : e.g. "additive", "distractor"
    split             : "source" | "gen" | None (None = flat list / distractor)
    rules_to_skip     : argument strings of hidden nodes (passed to env.reset)
    demo_entity_names : for gen tasks — names of source entities used as demos
    metadata          : variant_id, entity_instance, attributes, use_nonce, num_items, …
    """
    tree:              GoalTree
    task_type:         str
    split:             Optional[str]
    rules_to_skip:     List[str]      = field(default_factory=list)
    demo_entity_names: List[str]      = field(default_factory=list)
    metadata:          Dict[str, Any] = field(default_factory=dict)

    @property
    def root_id(self) -> Optional[str]:
        return self.tree.root_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type":         self.task_type,
            "split":             self.split,
            "rules_to_skip":     self.rules_to_skip,
            "demo_entity_names": self.demo_entity_names,
            "metadata":          self.metadata,
            "tree":              self.tree.to_dict(),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GeneratedTask":
        return GeneratedTask(
            tree=GoalTree.from_dict(d["tree"]),
            task_type=d["task_type"],
            split=d.get("split"),
            rules_to_skip=d.get("rules_to_skip", []),
            demo_entity_names=d.get("demo_entity_names", []),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Process loading
# ---------------------------------------------------------------------------

def load_process(name_or_path: str) -> Dict:
    """
    Load a v2 process spec from a JSON file.

    Parameters
    ----------
    name_or_path : bare name looked up in tree_management/processes/,
                   or an absolute/relative file path.
    """
    if os.path.isabs(name_or_path) or os.sep in name_or_path or "/" in name_or_path:
        path = name_or_path
    else:
        path = os.path.join(_PROCESSES_DIR, f"{name_or_path}.json")
    with open(path) as f:
        d = json.load(f)
    if "steps" not in d:
        raise ValueError(
            f"Process file '{path}' is missing a 'steps' key. "
            "Only v2 process format is supported."
        )
    validate_process(d)
    return d


def validate_process(process: Dict) -> None:
    """
    Validate semantic consistency of a v2 process spec.

    Cost is now a task-level concern (declared in property_bindings on the
    TemplateSpec / register_task_type call), not stored in the process JSON.
    This function validates structural consistency only.
    """
    for step in process["steps"]:
        action    = step["action"]
        if False:  # placeholder for future structural checks
            raise ValueError(
                f"Process step '{step['id']}': placeholder."
            )


# ---------------------------------------------------------------------------
# v2 resolver helpers
# ---------------------------------------------------------------------------

def _resolve_arg_record(
    step_id: str,
    step: Dict,
    entity: Dict,
    elements: Dict,
    use_nonce: bool,
    resolved: Dict,
    slot_bindings: Dict[str, str],
    rng: random.Random,
    used: Dict[str, set],
) -> Any:
    """
    Resolve one v2 argument slot → record dict or primitive.

    Resolution is driven by slot_bindings first, then the slot spec:

      {"from": "step_id.field"}
          Copy a field from an already-resolved step — no binding check needed.

      slot_bindings[step_id] == "rule_input"
          Use the current task entity.  All entity fields pass through;
          "name", "attribute_names", "attribute_values" are the use_nonce-aware
          variants pre-computed by fill_elements.  The slot's "pool" / "type"
          fields are ignored (overridden).

      slot_bindings[step_id] == "rule_output"
          Index elements[pool_path] with entity["item_id"] (rule-determined).
          Requires slot to declare "pool".

      no binding
          Sample randomly from the pool identified by "pool".
          Requires slot to declare "pool".

    pool values in process files are lexicon paths (e.g. "object.weapon"), matching
    the convention in rule files.  fill_elements() stores a _pool_map in elements
    that translates each lexicon path to its elements key.  The resolver falls back
    to treating pool as a direct dotted elements key if _pool_map has no entry for it.

    Item cost and other output properties are declared as process-level property slots
    (pool or value) and resolved by _resolve_prop_val, not by this function.

    For pool entries all fields pass through; "name" is normalised from
    semantic_name / nonce_name so downstream "from" references work uniformly.
    """
    slot = step["argument"]

    # Mode 1: from-reference — copy a field from a previously resolved step.
    if "from" in slot:
        parts = slot["from"].split(".")
        val = resolved[parts[0]]
        for part in parts[1:]:
            if not isinstance(val, dict):
                raise ValueError(
                    f"Cannot access .{part} on non-dict {val!r} (from: {slot['from']!r})"
                )
            val = val[part]
        return val

    raw_binding = slot_bindings.get(step_id)
    # Binding format: "<rule_name>:input" | "<rule_name>:output"
    # The rule_name prefix is informational (groups nodes by rule); only the role
    # suffix drives resolution behaviour.
    binding_role = raw_binding.split(":")[-1] if raw_binding else None

    # Mode 2: input — always the current entity; pool spec is overridden.
    if binding_role == "input":
        return {
            **{k: v for k, v in entity.items()
               if k not in ("instance", "nonce_instance",
                            "attribute_values",       "nonce_attribute_values",
                            "attribute_names",         "nonce_attribute_names",
                            "attribute_values_list",   "nonce_attribute_values_list")},
            "name":             entity["nonce_instance"              if use_nonce else "instance"],
            "attribute_names":  entity["nonce_attribute_names"       if use_nonce else "attribute_names"],
            "attribute_values": entity["nonce_attribute_values_list" if use_nonce else "attribute_values_list"],
        }

    # Modes 3 & 4: pool-based — resolve from elements or compute on-the-fly.
    pool_path = slot.get("pool")
    if not pool_path:
        raise ValueError(
            f"Step '{step_id}': argument slot has no 'from' and no 'pool', "
            f"and slot_bindings does not mark it as '<rule>:input'. "
            f"Declare a pool path (lexicon path, e.g. 'object.weapon')."
        )

    # --- Rule item pool (structured fill path) ---
    # If fill_elements ran in structured mode, item surface names are computed on
    # demand from _dim_* metadata rather than read from a pre-built list.
    # Pool-sampled nodes using the same output pool always exclude the rule item
    # (identified by entity["item_id"]) to prevent collision.
    output_category = elements.get("_output_category")
    if pool_path == output_category and "_dim_sem" in elements:
        dim_sem     = elements["_dim_sem"]
        dim_nonce   = elements["_dim_nonce"]
        dim_strides = elements["_dim_strides"]
        dim_order   = elements["_dim_order"]
        items_fn    = elements["_items_fn"]
        n_items     = elements["_n_items"]

        if binding_role == "output":
            item_id = entity.get("item_id")
            if item_id is None:
                raise ValueError(
                    f"Step '{step_id}' has slot binding 'rule_output' but the entity "
                    "has no 'item_id'. Ensure fill_elements computed item assignments."
                )
        else:
            # Exclude any item_id already in `used` (rule item and any earlier
            # pool-sampled items in this tree).
            excluded_ids = used.get("_item_ids", set())
            candidates   = [i for i in range(n_items) if i not in excluded_ids]
            if not candidates:
                candidates = list(range(n_items))
            item_id = rng.choice(candidates)
            used.setdefault("_item_ids", set()).add(item_id)

        # Decode item_id → per-dimension value indices via row-major strides.
        dv        = {}
        remaining = item_id
        for d in dim_order:
            dv[d]     = remaining // dim_strides[d]
            remaining %= dim_strides[d]

        if use_nonce:
            name = item_nonce_name(items_fn, dv, dim_order, dim_nonce,
                                   elements.get("_output_noun_nonce", ""))
        else:
            name = item_sem_name(items_fn, dv, dim_order, dim_sem, elements["_output_noun"])

        return {"name": name, **elements.get("_output_item_props", {})}

    # --- Entity pool and distractor item pool ---
    # _pool_map translates lexicon paths (e.g. "entity.npc") to elements keys
    # (e.g. "entities.all").  Distractor fill stores "items" and maps the output
    # category to it.  Falls back to treating pool_path as a direct elements key.
    pool_map     = elements.get("_pool_map", {})
    elements_key = pool_map.get(pool_path, pool_path)
    pool         = elements
    for key in elements_key.split("."):
        pool = pool[key]
    if not isinstance(pool, list) or not pool:
        raise ValueError(
            f"Step '{step_id}': pool path '{pool_path}' in elements must be a "
            f"non-empty list, got {type(pool).__name__!r}."
        )

    # Detect pool type by entry structure.
    # Entity entries have "instance" / "nonce_instance"; item entries have "semantic_name".
    is_entity_pool = pool and "instance" in pool[0]

    if is_entity_pool:
        # Entity pool: exclude any entity whose name appears in `used` (this includes
        # the rule entity and any entity already picked earlier in this tree).
        name_key      = "nonce_instance" if use_nonce else "instance"
        exclude_key   = "_entity_nonce_names" if use_nonce else "_entity_sem_names"
        excluded      = used.get(exclude_key, set())
        candidates    = [e for e in pool if e.get(name_key) not in excluded]
        if not candidates:
            candidates = pool
        entry = rng.choice(candidates)
        # Register newly picked entity so subsequent pool nodes in this tree avoid it.
        used.setdefault("_entity_sem_names",   set()).add(entry.get("instance",       ""))
        used.setdefault("_entity_nonce_names", set()).add(entry.get("nonce_instance", ""))
        return {
            **{k: v for k, v in entry.items()
               if k not in ("instance", "nonce_instance",
                            "attribute_values",        "nonce_attribute_values",
                            "attribute_names",          "nonce_attribute_names",
                            "attribute_values_list",    "nonce_attribute_values_list")},
            "name":             entry[name_key],
            "attribute_names":  entry["nonce_attribute_names"       if use_nonce else "attribute_names"],
            "attribute_values": entry["nonce_attribute_values_list" if use_nonce else "attribute_values_list"],
        }

    # Item pool (distractor path): entries have "semantic_name" / "nonce_name".
    if binding_role == "output":
        item_id = entity.get("item_id")
        if item_id is None:
            raise ValueError(
                f"Step '{step_id}' has slot binding 'rule_output' but the entity "
                "has no 'item_id'. Ensure fill_elements computed item assignments."
            )
        entry = pool[item_id]
    else:
        # Exclude item ids already in `used` (rule item + earlier pool-sampled items).
        excluded_ids = used.get("_item_ids", set())
        candidates   = [e for e in pool if e.get("id", 0) not in excluded_ids]
        if not candidates:
            candidates = pool
        entry   = rng.choice(candidates)
        item_id = entry.get("id", 0)
        used.setdefault("_item_ids", set()).add(item_id)

    return {
        "name": entry.get("nonce_name" if use_nonce else "semantic_name", entry.get("name", str(entry))),
        **elements.get("_output_item_props", {}),
    }


def _arg_str(val: Any) -> str:
    return val["name"] if isinstance(val, dict) else str(val)


# ---------------------------------------------------------------------------
# Used-values registry
# ---------------------------------------------------------------------------

def _build_used(entity: Dict, elements: Dict) -> Dict[str, set]:
    """
    Build the per-tree 'used' registry that prevents pool-sampled nodes from
    reusing values already taken by fill_elements or by earlier nodes in the
    same tree.

    Structure
    ---------
    "_entity_sem_names"   : set of semantic entity names already in this tree.
    "_entity_nonce_names" : set of nonce entity names already in this tree.
    "_item_ids"           : set of item_ids already in this tree.
    <full_pool_path>      : set of values already drawn from that lexicon pool.

    Seeded from two sources:

    1. elements["_reserved_variant_pools"] — values pre-sampled by fill_elements
       at variant level (e.g. the shop location).  These are the same for every
       tree in the variant.

    2. entity["_consumed_pool_values"] — values the *rule* entity occupies in
       each lexicon pool (e.g. its location pool path → "citadel").  This makes
       the system self-documenting: fill_elements declares what it consumed on
       the entity dict itself, so no hard-coded exclusion is needed in generator.py.

    _resolve_arg_record reads `used` for exclusion and writes back newly
    sampled values so subsequent nodes in the same tree stay unique.
    """
    used: Dict[str, set] = {}

    # 1. Variant-level reservations (output property pools: shop location, etc.)
    for path, vals in elements.get("_reserved_variant_pools", {}).items():
        used[path] = set(vals)

    # 2. Rule entity identity
    if "instance" in entity:
        used["_entity_sem_names"]   = {entity["instance"]}
    if "nonce_instance" in entity:
        used["_entity_nonce_names"] = {entity["nonce_instance"]}
    if entity.get("item_id") is not None:
        used["_item_ids"] = {entity["item_id"]}

    # 3. Rule entity's per-property-pool consumed values (location, etc.)
    for pool_path, val in entity.get("_consumed_pool_values", {}).items():
        used.setdefault(pool_path, set()).add(val)

    return used


# ---------------------------------------------------------------------------
# Tree builder (v2)
# ---------------------------------------------------------------------------

def build_tree_from_process(
    process: Dict,
    entity: Dict,
    elements: Dict,
    use_nonce: bool = False,
    slot_bindings: Optional[Dict[str, str]] = None,
    hidden_nodes: Optional[List[str]] = None,
    property_bindings: Optional[Dict[str, Dict]] = None,
    seed: int = 0,
) -> Tuple[GoalTree, List[str]]:
    """
    Build a GoalTree from a v2 process dict for one entity.

    Steps are resolved root-first (reverse list order) so forward "from"
    references always point to already-resolved steps.

    Ordering constraints are derived automatically from execution_order differences
    among sibling steps (same parent, different execution_order values).  Siblings
    with the same execution_order are treated as parallel (no constraint).  Manual
    ordering_constraints in the process JSON are unioned with the auto-derived set.

    Parameters
    ----------
    process       : v2 process dict
    entity        : the entity this task is about (provides rule input attributes
                    and item_id for rule_output nodes)
    elements      : filled elements dict (items table, attribute_labels, etc.)
    use_nonce     : use nonce surface names
    slot_bindings : {step_id: "<rule_name>:input" | "<rule_name>:output"}
                    absent step IDs → type:object nodes use pool sampling.
                    The rule_name prefix groups nodes that participate in the same
                    rule, enabling future tasks with multiple rules per process.
    hidden_nodes  : list of step IDs whose arguments are omitted from the agent's
                    rules text (rules_to_skip).  Declared in task spec, not process.
    seed          : seed for pool sampling rng (only used for pool-sampled nodes)
    """
    if slot_bindings is None:
        slot_bindings = {}
    if hidden_nodes is None:
        hidden_nodes = []
    if property_bindings is None:
        property_bindings = {}

    # "type" is metadata only — no behavioural effect on generation.

    steps      = process["steps"]
    root_sid   = process["root"]
    step_by_id = {s["id"]: s for s in steps}
    step_index = {s["id"]: i for i, s in enumerate(steps)}

    rng  = random.Random(seed)
    used = _build_used(entity, elements)

    # Fields on arg records that are internal implementation details and should not
    # be auto-promoted to node.properties.
    _SKIP_PROP_KEYS = frozenset({"item_id", "attributes"})

    # Resolve root-first (reverse order) so "from" references always look backward.
    # For each step: resolve argument, then auto-populate node properties from it.
    # arg record fields (excluding name and internal keys) become node properties
    # automatically — entity nodes expose location/attribute_names/attribute_values,
    # item nodes expose location (from _output_item_props).  Task-level fixed values
    # (e.g. cost) are overlaid from property_bindings.
    # Merged into `resolved` so later steps can reference earlier steps' properties
    # via "from": "step_id.prop_key".
    arg_records: Dict[str, Any]   = {}
    prop_dicts:  Dict[str, Dict]  = {}
    resolved:    Dict[str, Any]   = {}

    for step in reversed(steps):
        sid              = step["id"]
        arg_records[sid] = _resolve_arg_record(
            sid, step, entity, elements,
            use_nonce, resolved, slot_bindings, rng, used,
        )
        # Auto-populate node properties from arg record fields, then overlay
        # task-level fixed-value bindings (e.g. cost on buy steps).
        if isinstance(arg_records[sid], dict):
            auto_props = {
                k: v for k, v in arg_records[sid].items()
                if k != "name" and not k.startswith("_") and k not in _SKIP_PROP_KEYS
            }
        else:
            auto_props = {}
        prop_dicts[sid] = {**auto_props, **property_bindings.get(sid, {})}
        # Merge so cross-step "from" references can access both arg fields and properties.
        # "from"-reference steps (e.g. go) may return a primitive — store as-is.
        if isinstance(arg_records[sid], dict):
            resolved[sid] = {**arg_records[sid], **prop_dicts[sid]}
        else:
            resolved[sid] = arg_records[sid]

    tree          = GoalTree()
    rules_to_skip: List[str] = []
    node_ids:      Dict[str, str] = {}

    for step in steps:
        sid        = step["id"]
        arg_string = _arg_str(arg_records[sid])
        node_type  = "root" if sid == root_sid else "leaf"
        nid        = tree._add_node(
            type=node_type, argument=arg_string,
            properties=prop_dicts[sid], meta={"incoming_edge": step["action"]},
        )
        node_ids[sid] = nid
        if sid in hidden_nodes:
            rules_to_skip.append(arg_string)

    for step in steps:
        sid = step["id"]
        if sid == root_sid:
            continue
        parent_sid = step["parent"]
        eo = step.get("execution_order", step_index[sid])
        tree._add_edge(node_ids[sid], node_ids[parent_sid],
                       action=step_by_id[parent_sid]["action"],
                       meta={"execution_order": eo})

    tree.root_id = node_ids[root_sid]

    # Auto-derive ordering constraints from execution_order differences among siblings.
    # For each parent, sort children by execution_order; consecutive groups with
    # different values generate "every step in earlier group before every step in
    # later group" constraints.  Same execution_order = parallel, no constraint.
    parent_children: Dict[str, List] = defaultdict(list)
    for step in steps:
        sid = step["id"]
        if sid == root_sid:
            continue
        eo = step.get("execution_order", step_index[sid])
        parent_children[step["parent"]].append((eo, sid))

    derived: set = set()
    for children in parent_children.values():
        children.sort(key=lambda x: x[0])
        groups = [list(g) for _, g in groupby(children, key=lambda x: x[0])]
        for i in range(len(groups) - 1):
            for _, b_sid in groups[i]:
                for _, a_sid in groups[i + 1]:
                    c = (
                        (step_by_id[b_sid]["action"], _arg_str(resolved[b_sid])),
                        (step_by_id[a_sid]["action"], _arg_str(resolved[a_sid])),
                    )
                    derived.add(c)

    # Manual overrides from process JSON (union with auto-derived; duplicates ignored)
    for constraint in process.get("ordering_constraints", []):
        b_sid, a_sid = constraint["before"], constraint["after"]
        c = (
            (step_by_id[b_sid]["action"], _arg_str(resolved[b_sid])),
            (step_by_id[a_sid]["action"], _arg_str(resolved[a_sid])),
        )
        derived.add(c)

    tree.ordering_constraints.extend(derived)

    return tree, rules_to_skip


def _build_tree(
    template: TemplateSpec,
    entity: Dict,
    elements: Dict,
    use_nonce: bool = False,
    seed: int = 0,
) -> Tuple[GoalTree, List[str]]:
    """Build a GoalTree from a TemplateSpec."""
    return build_tree_from_process(
        template.process, entity, elements, use_nonce,
        slot_bindings=template.slot_bindings,
        hidden_nodes=template.hidden_nodes,
        property_bindings=template.property_bindings,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Core generate_tasks function
# ---------------------------------------------------------------------------

def _load_elements(elements_path_or_dict):
    if isinstance(elements_path_or_dict, str):
        with open(elements_path_or_dict) as f:
            elements = json.load(f)
        variant_id = os.path.splitext(os.path.basename(elements_path_or_dict))[0]
    else:
        elements = elements_path_or_dict
        vi = elements.get("_variant_info", {})
        seed = vi.get("seed")
        variant_id = f"v{seed}" if seed is not None else "inline"
    return elements, variant_id


def _get_entity_list(elements: Dict, split: Optional[str]) -> List[Dict]:
    """
    Return the list of entities to process.

    elements["entities"] is always a dict with an "all" key (flat list of every
    entity) and optionally "source" / "gen" keys for rule-based tasks.

    split=None   → elements["entities"]["all"]  (distractor / pool mode)
    split="source" / "gen" → structured split
    """
    entities_field = elements["entities"]
    if not isinstance(entities_field, dict):
        raise ValueError(
            "elements['entities'] must be a dict with an 'all' key "
            "(and optionally 'source'/'gen'). Got a plain list — "
            "re-generate elements with the current fill_elements()."
        )
    if split is None:
        return entities_field["all"]
    return entities_field[split]


def generate_tasks(
    elements_path_or_dict,
    process: Optional[Dict],
    split: Optional[str] = "source",
    use_nonce: bool = False,
    validate_fn: Optional[Callable] = None,
    task_type: str = "task",
    template_name: Optional[str] = None,
    slot_bindings: Optional[Dict[str, str]] = None,
    seed: int = 0,
) -> List[GeneratedTask]:
    """
    Build a list of GeneratedTask from a filled elements dict and a process spec.

    Parameters
    ----------
    elements_path_or_dict : path to a filled elements JSON or an already-loaded dict
    process               : v2 process dict (from load_process()); ignored when
                            template_name is given
    split                 : "source" | "gen" | None
                            None = flat entity list (distractor/pool mode)
    use_nonce             : use nonce surface names
    validate_fn           : optional callable(elements) → None for split validation
    task_type             : string stored on each GeneratedTask
    template_name         : if given, look up a registered TemplateSpec (its
                            slot_bindings take precedence over the slot_bindings arg)
    slot_bindings         : {step_id: "rule_input" | "rule_output"}
                            absent step IDs → type:object nodes use pool sampling
    seed                  : base seed for pool sampling rng; each entity gets seed+i
    """
    elements, variant_id = _load_elements(elements_path_or_dict)

    if validate_fn is not None:
        validate_fn(elements)

    if template_name is not None:
        template = get_template(template_name)
    else:
        template = TemplateSpec(
            name=task_type,
            process=process,
            slot_bindings=slot_bindings or {},
        )

    key        = "nonce_instance" if use_nonce else "instance"
    entities   = _get_entity_list(elements, split)

    # For gen split: collect source entity names to attach as demo_entity_names
    demo_names: List[str] = []
    if split == "gen":
        demo_names = [e[key] for e in elements["entities"].get("source", [])]

    tasks: List[GeneratedTask] = []
    for i, entity in enumerate(entities):
        tree, rules_to_skip = _build_tree(template, entity, elements, use_nonce, seed=seed + i)
        tasks.append(GeneratedTask(
            tree=tree,
            task_type=task_type,
            split=split,
            rules_to_skip=rules_to_skip,
            demo_entity_names=demo_names,
            metadata={
                "variant_id":      variant_id,
                "entity_instance": entity["nonce_instance" if use_nonce else "instance"],
                "entity_location": entity["location"],
                "attributes":      dict(entity.get("attributes", {})),
                "use_nonce":       use_nonce,
                "num_items":       len(elements.get("items", [])),
            },
        ))
    return tasks


# ---------------------------------------------------------------------------
# make_property_task_generator — factory for the registry gen_fn pattern
# ---------------------------------------------------------------------------

def make_property_task_generator(
    template_name: str,
    task_type: str,
    validate_fn: Optional[Callable] = None,
) -> Callable:
    """
    Return a generate_tasks function bound to a specific template + task_type.

    Signature of returned function:
        fn(elements_path_or_dict, split="source", use_nonce=False, validate=True)
            -> List[GeneratedTask]
    """
    def _gen(elements_path_or_dict, split="source", use_nonce=False, validate=True):
        return generate_tasks(
            elements_path_or_dict,
            process=None,
            split=split,
            use_nonce=use_nonce,
            validate_fn=validate_fn if validate else None,
            task_type=task_type,
            template_name=template_name,
        )
    _gen.__name__     = f"generate_{task_type}_tasks"
    _gen.__qualname__ = f"generate_{task_type}_tasks"
    return _gen


# ---------------------------------------------------------------------------
# make_procedural_task_generator — factory for procedural task gen_fn
# ---------------------------------------------------------------------------

def make_procedural_task_generator(
    task_type: str,
    process_templates: Dict,
    slot_bindings: Dict[str, str],
    hidden_nodes: List[str],
    property_bindings: Dict[str, Dict],
    extra_step_pools: Dict[str, Dict],
    validate_fn: Optional[Callable] = None,
) -> Callable:
    """
    Return a gen_fn for procedural tasks where entity attributes determine
    which process template is used.

    At call time the returned function:
      1. Samples one item per extra_step_pool from the lexicon (using the variant
         seed), stores them in elements._extra_items, and adds _pool_map entries
         so _resolve_arg_record picks them up as single-item pools.
      2. For each entity derives (attr_combo) and selects the matching process
         template, then calls build_tree_from_process.

    Parameters
    ----------
    task_type         : string stored on each GeneratedTask
    process_templates : {attr_combo -> loaded process dict}
                        attr_combo is a tuple of int attribute indices in _attr_order.
    slot_bindings     : step_id -> "<rule>:input" | "<rule>:output"
    hidden_nodes      : step IDs whose arguments are omitted from rules text
    property_bindings : {step_id: {key: value}} — fixed properties (e.g. cost)
    extra_step_pools  : {lexicon_pool_path -> {"name": [pool_key, ...], "n_items": int}}
                        One item is sampled per pool per variant and injected into
                        elements so the same argument is used across all entities
                        sharing the same action type.
    validate_fn       : optional callable(elements) -> None
    """
    def _gen(elements_path_or_dict, split="source", use_nonce=False, validate=True):
        from tree_management.elements import load_lexicons
        elements, variant_id = _load_elements(elements_path_or_dict)

        if validate and validate_fn is not None:
            validate_fn(elements)

        sem_lex, nonce_lex = load_lexicons()

        # Shallow-copy elements; deep-copy only the mutable sub-dicts we modify.
        elements = {
            **elements,
            "_pool_map":    dict(elements.get("_pool_map", {})),
            "_extra_items": {},
        }
        variant_seed = elements.get("_variant_info", {}).get("seed", 0)
        rng = random.Random(variant_seed * 31337 + 7)

        # --- Inject extra-step pools (single consistent item per pool per variant) ---
        for pool_path, pool_spec in extra_step_pools.items():
            node_sem   = sem_lex
            node_nonce = nonce_lex
            for part in pool_path.split("."):
                node_sem   = node_sem[part]
                node_nonce = node_nonce.get(part, {}) if isinstance(node_nonce, dict) else {}

            items: List[Dict] = []
            for name_key in pool_spec["name"]:
                sem_pool   = node_sem[name_key]
                nonce_pool = node_nonce.get(name_key, sem_pool)
                for _ in range(pool_spec.get("n_items", 1)):
                    items.append({
                        "semantic_name": rng.choice(sem_pool),
                        "nonce_name":    rng.choice(nonce_pool),
                    })

            safe_key = pool_path.replace(".", "_")
            elements["_extra_items"][safe_key] = items
            elements["_pool_map"][pool_path]   = f"_extra_items.{safe_key}"

        # --- Auto-build object item pools for any pool path referenced by process
        # templates that is neither the entity pool nor an extra-step pool.
        # This covers "buy"-style steps (e.g. object.weapon) which need an item list
        # plus a shop location for subsequent "go" steps that reference buy_0.location.
        entity_pool   = elements.get("_input_category", "")
        covered_paths = set(elements["_pool_map"])
        obj_pool_paths: set = set()
        for process in process_templates.values():
            for step in process["steps"]:
                p = step.get("argument", {}).get("pool")
                if p and p != entity_pool and p not in covered_paths:
                    obj_pool_paths.add(p)

        out_item_props: Dict[str, Any] = {}
        for pool_path in sorted(obj_pool_paths):
            if pool_path in elements["_pool_map"]:
                continue
            parts = pool_path.split(".")
            node_sem   = sem_lex
            node_nonce = nonce_lex
            for part in parts:
                node_sem   = node_sem[part]
                node_nonce = node_nonce.get(part, {}) if isinstance(node_nonce, dict) else {}

            # Dimension pools: list values that are not "locations" or "nouns"
            dim_sem   = {k: v for k, v in node_sem.items()
                         if isinstance(v, list) and k not in ("locations", "nouns")}
            noun_sem   = node_sem.get("nouns", [])
            noun_nonce = (node_nonce.get("nouns", noun_sem)
                          if isinstance(node_nonce, dict) else noun_sem)

            dim_names = sorted(dim_sem)
            n_items   = max((len(v) for v in dim_sem.values()), default=6)
            built_items: List[Dict] = []
            for k in range(n_items):
                s_parts = [dim_sem[d][k % len(dim_sem[d])] for d in dim_names]
                n_dim   = {d: (node_nonce.get(d, dim_sem[d])
                               if isinstance(node_nonce, dict) else dim_sem[d])
                           for d in dim_names}
                n_parts = [n_dim[d][k % len(n_dim[d])] for d in dim_names]
                if noun_sem:
                    s_parts.append(rng.choice(noun_sem))
                    n_parts.append(rng.choice(noun_nonce) if noun_nonce else s_parts[-1])
                built_items.append({
                    "id":            k,
                    "semantic_name": "_".join(s_parts),
                    "nonce_name":    "_".join(n_parts),
                })

            safe_key = pool_path.replace(".", "_")
            elements[f"_obj_{safe_key}"]     = built_items
            elements["_pool_map"][pool_path] = f"_obj_{safe_key}"

            loc_pool = node_sem.get("locations", ["shop"])
            out_item_props["location"] = rng.choice(loc_pool)

        if out_item_props:
            elements["_output_item_props"] = {
                **elements.get("_output_item_props", {}),
                **out_item_props,
            }

        key_field  = "nonce_instance" if use_nonce else "instance"
        entities   = _get_entity_list(elements, split)
        attr_order = elements.get("_attr_order", [])

        demo_names: List[str] = []
        if split == "gen":
            demo_names = [e[key_field] for e in elements["entities"].get("source", [])]

        tasks: List[GeneratedTask] = []
        for i, entity in enumerate(entities):
            combo   = tuple(entity["attributes"][n] for n in attr_order)
            process = process_templates[combo]
            tree, rules_to_skip = build_tree_from_process(
                process, entity, elements, use_nonce,
                slot_bindings=slot_bindings,
                hidden_nodes=hidden_nodes,
                property_bindings=property_bindings,
                seed=i,
            )
            tasks.append(GeneratedTask(
                tree=tree,
                task_type=task_type,
                split=split,
                rules_to_skip=rules_to_skip,
                demo_entity_names=demo_names,
                metadata={
                    "variant_id":      variant_id,
                    "entity_instance": entity[key_field],
                    "entity_location": entity["location"],
                    "attributes":      dict(entity.get("attributes", {})),
                    "use_nonce":       use_nonce,
                    "process_combo":   combo,
                },
            ))
        return tasks

    _gen.__name__     = f"generate_{task_type}_tasks"
    _gen.__qualname__ = f"generate_{task_type}_tasks"
    return _gen


# ---------------------------------------------------------------------------
# register_procedural_task_type — one-call convenience for procedural tasks
# ---------------------------------------------------------------------------

_CANONICAL_PROPERTY_DISTRACTOR = "tree_management/distractors/canonical_property_distractor.json"


def register_procedural_task_type(
    *,
    name: str,
    process_templates: List[str],
    process_map: Dict,
    slot_bindings: Optional[Dict[str, str]] = None,
    hidden_nodes: Optional[List[str]] = None,
    property_bindings: Optional[Dict[str, Dict]] = None,
    extra_step_pools: Optional[Dict[str, Dict]] = None,
    split: Optional[Dict] = None,
    rules: str,
    correct_rule: str = "",
    mc_answer: str = "",
    distractor_rules: Optional[str] = _CANONICAL_PROPERTY_DISTRACTOR,
    distractor_template: str = "additive",
    description: str = "",
    validate_fn: Optional[Callable] = None,
    max_tries: Optional[int] = None,
) -> None:
    """
    Register a procedural task type in one call (parallel to register_task_type).

    Each entry in process_map maps an attr_combo tuple to an index into
    process_templates.  The gen_fn selects the template per entity at generation
    time based on attribute values, rather than using a single fixed process.

    Parameters
    ----------
    name              : identifier (template name, task registry key, label)
    process_templates : ordered list of process file paths (or bare names);
                        indexed by the values in process_map.
    process_map       : {attr_combo_tuple -> int index into process_templates}
    slot_bindings     : {step_id: "<rule>:input" | "<rule>:output"}
    hidden_nodes      : step IDs whose arguments are omitted from rules text
    property_bindings : {step_id: {key: value}} — fixed properties (e.g. cost)
    extra_step_pools  : {lexicon_pool_path -> {"name": [...], "n_items": int}}
    split             : split spec dict e.g. {"fn": "independent_leave_one_out", "seed": 0}
    rules                : repo-relative path to rule JSON file
    validate_fn          : callable(elements) -> None for split validation
    max_tries            : upper bound for --num_tries max
    distractor_rules     : distractor rule file path; defaults to the canonical property
                           distractor so that flat-process entities are interleaved with
                           proc demos. Pass None to disable distractors.
    distractor_template  : template used to build distractor tasks; defaults to "additive"
                           (flat go→buy→go→defeat, matching the distractor process).
    """
    from tree_management.registry import TaskSpec, PropertySpec, ProcessSpec, register_task

    _slot   = slot_bindings or {}
    _hidden = hidden_nodes or []
    _props  = property_bindings or {}
    _pools  = extra_step_pools or {}

    loaded = {
        combo: load_process(process_templates[idx])
        for combo, idx in process_map.items()
    }

    gen_fn = make_procedural_task_generator(
        task_type         = name,
        process_templates = loaded,
        slot_bindings     = _slot,
        hidden_nodes      = _hidden,
        property_bindings = _props,
        extra_step_pools  = _pools,
        validate_fn       = validate_fn,
    )

    mapping_nodes = [sid for sid, role in _slot.items()
                     if role.split(":")[-1] == "output"]

    register_task(TaskSpec(
        property_spec    = PropertySpec(
            task_type    = name,
            rules        = rules,
            validate_fn  = validate_fn,
            correct_rule = correct_rule,
            mc_answer    = mc_answer,
            description  = description,
            max_tries    = max_tries,
            split        = split,
        ),
        processes        = [ProcessSpec(name=f"{name}_{i}", mapping_nodes=mapping_nodes)
                            for i in range(len(process_templates))],
        process_selector = None,
        gen_fn              = gen_fn,
        template_name       = None,
        distractor_rules    = distractor_rules,
        distractor_template = distractor_template,
    ))


# ---------------------------------------------------------------------------
# register_task_type — one-call convenience for tasks/ files
# ---------------------------------------------------------------------------

def register_task_type(
    *,
    name: str,
    process=None,
    slot_bindings: Optional[Dict[str, str]] = None,
    hidden_nodes: Optional[List[str]] = None,
    property_bindings: Optional[Dict[str, Dict]] = None,
    split: Optional[Dict] = None,
    rules: str,
    correct_rule: str = "",
    mc_answer: str = "",
    distractor_rules: Optional[str] = None,
    distractor_template: Optional[str] = None,
    description: str = "",
    validate_fn: Optional[Callable] = None,
    max_tries: Optional[int] = None,
) -> None:
    """
    Register a TemplateSpec and TaskSpec in one call.

    Parameters
    ----------
    name          : single identifier used as template name, task registry key,
                    and human-readable label (e.g. "additive", "conditional")
    process       : load with load_process("property_flat")
    slot_bindings : {step_id: "<rule>:input" | "<rule>:output"}
                    Steps absent from here use pool sampling for type:object nodes.
                    mapping_nodes for QA evaluation is derived as all "<rule>:output" steps.
    hidden_nodes  : list of step IDs whose arguments are omitted from the agent's rules text.
                    Stored on TemplateSpec and populates rules_to_skip on each GeneratedTask.
    property_bindings : {step_id: {key: value}} — fixed-value properties bound at task level.
                    Overlaid on top of auto-populated properties from the arg record.
                    Typical use: {"buy_0": {"cost": 30}} for purchase cost.
    split         : split spec dict, e.g. {"fn": "two_offset", "seed": 0}.
                    Passed as split_spec to fill_elements(). Takes precedence over any
                    "split" field in the rule JSON file.
    rules         : repo-relative path to the rule JSON file
    validate_fn   : callable(elements) → None for split validation
    max_tries     : upper bound on tries used when --num_tries max is passed.
                    Should equal the number of distinct items for the task type
                    (i.e. the worst-case number of attempts needed to exhaust all choices).
    """
    from tree_management.registry import TaskSpec, PropertySpec, ProcessSpec, register_task

    _slot_bindings     = slot_bindings or {}
    _hidden_nodes      = hidden_nodes or []
    _property_bindings = property_bindings or {}

    # Derive mapping_nodes (used by QA evaluation) from "<rule>:output" bindings
    mapping_nodes = [sid for sid, role in _slot_bindings.items() if role.split(":")[-1] == "output"]

    process_tmpl = TemplateSpec(
        name=name,
        process=process,
        slot_bindings=_slot_bindings,
        hidden_nodes=_hidden_nodes,
        property_bindings=_property_bindings,
    )
    register_template(process_tmpl)

    property_spec = PropertySpec(
        task_type    = name,
        rules        = rules,
        validate_fn  = validate_fn,
        correct_rule = correct_rule,
        mc_answer    = mc_answer,
        description  = description,
        max_tries    = max_tries,
        split        = split,
    )

    gen_fn = make_property_task_generator(name, name, validate_fn)

    def _constant_selector(entity, seed):
        return process_tmpl

    register_task(TaskSpec(
        property_spec       = property_spec,
        processes           = [ProcessSpec(name=name, mapping_nodes=mapping_nodes)],
        process_selector    = _constant_selector,
        gen_fn              = gen_fn,
        template_name       = name,
        distractor_rules    = distractor_rules,
        distractor_template = distractor_template,
    ))


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_tasks(tasks: List[GeneratedTask], path: str) -> None:
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w") as f:
        json.dump([t.to_dict() for t in tasks], f, indent=2)
