"""
tree_management/generator.py
Core task-generation pipeline: process spec + filled elements → GeneratedTask list.

Rule file output.type controls generation path:
  "item"    — property task: attributes → item to buy (hidden)
  "process" — procedural task: attributes → extension knobs (action, position, count)
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from itertools import groupby
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from herosjourney.core.goal_tree import GoalTree
from herosjourney.core.function_specs.item_mappings import item_sem_name, item_nonce_name

_TREE_MGMT_DIR  = Path(__file__).parent
_PROCESSES_DIR  = str(_TREE_MGMT_DIR / "processes")


# ---------------------------------------------------------------------------
# TemplateSpec  (process structure + slot bindings)
# ---------------------------------------------------------------------------

@dataclass
class TemplateSpec:
    """Process dict + slot_bindings. slot_bindings maps step IDs to rule roles
    ("<rule>:input" | "<rule>:output"); absent IDs use pool sampling."""
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
    """One instantiated task: GoalTree + metadata for a single entity."""
    tree:              GoalTree
    task_type:         str
    split:             Optional[str]
    rules_to_skip:     List[str]      = field(default_factory=list)
    demo_entity_names: List[str]      = field(default_factory=list)
    metadata:          Dict[str, Any] = field(default_factory=dict)

    @property
    def root_id(self) -> Optional[str]:
        return self.tree.root_id


# ---------------------------------------------------------------------------
# Process loading
# ---------------------------------------------------------------------------

def load_process(name_or_path: str) -> Dict:
    """Load a process JSON. Bare name resolves from herosjourney/core/processes/."""
    if os.path.isabs(name_or_path) or os.sep in name_or_path or "/" in name_or_path:
        path = name_or_path
    else:
        path = str(Path(_PROCESSES_DIR) / f"{name_or_path}.json")
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
    """Validate a v2 process dict; raise ValueError on the first problem found.

    Checks:
      - top-level "steps" is a non-empty list
      - every step has a unique string "id" and an "action" in ACTION_REGISTRY
      - every "parent" reference points to an existing step id
      - the graph (child -> parent edges) is acyclic
      - the declared "root" (if any) exists and has no parent
      - "execution_order" (if present) is an int
      - argument "from" references point to an existing step id
      - ordering_constraints reference existing step ids
    """
    from herosjourney.world_info.actions import ACTION_REGISTRY

    steps = process.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Process must have a non-empty 'steps' list.")

    ids: set = set()
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"Step {i} is not an object.")
        sid = step.get("id")
        if not isinstance(sid, str) or not sid:
            raise ValueError(f"Step {i} is missing a non-empty string 'id'.")
        if sid in ids:
            raise ValueError(f"Duplicate step id '{sid}'.")
        ids.add(sid)

    for step in steps:
        sid = step["id"]
        action = step.get("action")
        if action not in ACTION_REGISTRY:
            raise ValueError(
                f"Step '{sid}': unknown action '{action}'. "
                f"Valid actions: {sorted(ACTION_REGISTRY)}."
            )
        if "execution_order" in step and not isinstance(step["execution_order"], int):
            raise ValueError(
                f"Step '{sid}': execution_order must be an int, "
                f"got {step['execution_order']!r}."
            )
        parent = step.get("parent")
        if parent is not None and parent not in ids:
            raise ValueError(
                f"Step '{sid}': parent '{parent}' is not a defined step id."
            )
        # Argument "from" references must point to a known step.
        arg = step.get("argument", {})
        if isinstance(arg, dict) and "from" in arg:
            ref_head = str(arg["from"]).split(".")[0]
            if ref_head not in ids:
                raise ValueError(
                    f"Step '{sid}': argument from-reference '{arg['from']}' "
                    f"points to unknown step '{ref_head}'."
                )

    # Root, if declared, must exist and be parentless.
    root = process.get("root")
    if root is not None:
        if root not in ids:
            raise ValueError(f"root '{root}' is not a defined step id.")
        root_step = next(s for s in steps if s["id"] == root)
        if root_step.get("parent") is not None:
            raise ValueError(f"root step '{root}' must not have a parent.")

    # Cycle check over the child -> parent edges.
    parent_of = {s["id"]: s.get("parent") for s in steps}
    for start in ids:
        seen: set = set()
        node = start
        while node is not None:
            if node in seen:
                raise ValueError(f"Cycle detected in process graph at step '{node}'.")
            seen.add(node)
            node = parent_of.get(node)

    # ordering_constraints reference existing ids.
    for c in process.get("ordering_constraints", []) or []:
        for key in ("before", "after"):
            ref = c.get(key)
            if ref is not None and ref not in ids:
                raise ValueError(
                    f"ordering_constraint {c}: '{key}' references unknown step '{ref}'."
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
    Resolve one step's argument → record dict or primitive.
    Modes: from-reference, rule_input (current entity), rule_output (item_id lookup),
    or pool sampling. Fixed literal properties live in the step's "properties" list.
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
    """Build per-tree exclusion sets so pool-sampled nodes don't reuse values
    already taken by fill_elements (variant-level) or the rule entity."""
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
    """Build a GoalTree for one entity. Returns (tree, rules_to_skip).
    Steps resolved root-first; ordering constraints auto-derived from execution_order."""
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
    _SKIP_PROP_KEYS = frozenset({"item_id", "attributes", "proc_dims"})

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
        # process-level literal properties ({"key": k, "value": v} entries on the step),
        # then task-level escape-hatch bindings.
        if isinstance(arg_records[sid], dict):
            auto_props = {
                k: v for k, v in arg_records[sid].items()
                if k != "name" and not k.startswith("_") and k not in _SKIP_PROP_KEYS
            }
        else:
            auto_props = {}
        step_literal_props = {
            p["key"]: p["value"]
            for p in step.get("properties", [])
            if "value" in p
        }
        prop_dicts[sid] = {**auto_props, **step_literal_props, **property_bindings.get(sid, {})}
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
        if sid in hidden_nodes or step.get("is_hidden"):
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


def _resolve_extension_knobs(
    extension_spec: Dict,
    proc_dims: Dict[str, int],
) -> Tuple[int, str, str, Dict]:
    """Resolve (count, action, pool, position) from extension spec + entity's proc_dims.
    Each knob is a fixed value or {"from_dim": dim, "values": [...]} selector."""
    def _sel(val):
        if isinstance(val, dict) and "from_dim" in val:
            raw = proc_dims[val["from_dim"]]
            return val["values"][raw] if "values" in val else raw
        return val

    count    = int(_sel(extension_spec.get("count", 1)))
    action   = _sel(extension_spec.get("action", "perform"))
    position = _sel(extension_spec.get("position", {"before": "buy_0"}))

    pool_spec = extension_spec.get("pool")
    pool      = pool_spec["by_action"][action] if (
        isinstance(pool_spec, dict) and "by_action" in pool_spec
    ) else pool_spec

    return count, action, pool, position


def _insert_extension_steps(
    base_process: Dict,
    count: int,
    action: str,
    pool: str,
    position: Dict,
    is_hidden: bool = True,
) -> Dict:
    """Return a new process dict with `count` steps inserted at position.
    position: {"before": step_id} or {"after": step_id}."""
    import copy
    process = copy.deepcopy(base_process)
    steps   = process["steps"]

    if "before" in position:
        insert_before_id = position["before"]
    else:
        ref_id     = position["after"]
        ref_step   = next(s for s in steps if s["id"] == ref_id)
        ref_parent = ref_step.get("parent")
        ref_eo     = ref_step.get("execution_order", 0)
        next_sib   = min(
            (s for s in steps
             if s.get("parent") == ref_parent and s.get("execution_order", 0) > ref_eo),
            key=lambda s: s.get("execution_order", 0),
            default=None,
        )
        if next_sib is None:
            raise ValueError(
                f"Cannot insert after '{ref_id}': no next sibling found under the same parent."
            )
        insert_before_id = next_sib["id"]

    target_step   = next(s for s in steps if s["id"] == insert_before_id)
    target_parent = target_step.get("parent")
    target_eo     = target_step.get("execution_order", 0)

    for s in steps:
        if s.get("parent") == target_parent and s.get("execution_order", 0) >= target_eo:
            s["execution_order"] += 1

    new_steps = [
        {
            "id":              f"extension_step_{k}",
            "action":          action,
            "execution_order": target_eo,
            "parent":          target_parent,
            "is_hidden":       is_hidden,
            "argument":        {"pool": pool},
        }
        for k in range(count)
    ]

    insert_pos = next(i for i, s in enumerate(steps) if s["id"] == insert_before_id)
    process["steps"] = steps[:insert_pos] + new_steps + steps[insert_pos:]
    return process


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
    """Build GeneratedTask list from filled elements + process spec."""
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
# make_procedural_task_generator
# ---------------------------------------------------------------------------

def make_procedural_task_generator(
    task_type: str,
    base_process: Dict,
    slot_bindings: Optional[Dict[str, str]] = None,
    hidden_nodes: Optional[List[str]] = None,
    property_bindings: Optional[Dict[str, Dict]] = None,
    validate_fn: Optional[Callable] = None,
) -> Callable:
    _slot   = slot_bindings or {}
    _hidden = hidden_nodes or []
    _props  = property_bindings or {}

    def _gen(elements_path_or_dict, split="source", use_nonce=False, validate=True):
        from herosjourney.core.elements import load_lexicons
        elements, variant_id = _load_elements(elements_path_or_dict)

        if validate and validate_fn is not None:
            validate_fn(elements)

        sem_lex, nonce_lex = load_lexicons()

        elements = {
            **elements,
            "_pool_map": dict(elements.get("_pool_map", {})),
        }

        # Auto-build object item pools for pool paths referenced in base_process
        # that are not already covered (e.g. object.weapon for buy steps).
        entity_pool   = elements.get("_input_category", "")
        covered_paths = set(elements["_pool_map"])

        obj_pool_paths: set = set()
        for step in base_process["steps"]:
            p = step.get("argument", {}).get("pool")
            if p and p != entity_pool and p not in covered_paths:
                obj_pool_paths.add(p)

        rng = random.Random(elements.get("_variant_info", {}).get("seed", 0) * 31337 + 7)
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

        extension_spec = elements.get("_extension_spec")
        key_field      = "nonce_instance" if use_nonce else "instance"
        entities       = _get_entity_list(elements, split)

        demo_names: List[str] = []
        if split == "gen":
            demo_names = [e[key_field] for e in elements["entities"].get("source", [])]

        tasks: List[GeneratedTask] = []
        for i, entity in enumerate(entities):
            proc_dims = entity.get("proc_dims", {})
            count, action, pool, position = _resolve_extension_knobs(extension_spec, proc_dims)
            process = _insert_extension_steps(
                base_process, count, action, pool, position,
                is_hidden=extension_spec.get("is_hidden", True),
            )

            tree, rules_to_skip = build_tree_from_process(
                process, entity, elements, use_nonce,
                slot_bindings=_slot,
                hidden_nodes=_hidden,
                property_bindings=_props,
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
                },
            ))
        return tasks

    _gen.__name__     = f"generate_{task_type}_tasks"
    _gen.__qualname__ = f"generate_{task_type}_tasks"
    return _gen


_CANONICAL_PROPERTY_DISTRACTOR = str(
    _TREE_MGMT_DIR / "distractors" / "canonical_property_distractor.json"
)


# ---------------------------------------------------------------------------
# register_task — single entry point for all task types
# ---------------------------------------------------------------------------

def register_task(
    *,
    name: str,
    rules: str,
    process,                           # str | List[str]
    lexicon: str = "default",
    split: Optional[Dict] = None,
    eval: Optional[Dict] = None,       # {correct_rule, description}
    distractors: Optional[str] = _CANONICAL_PROPERTY_DISTRACTOR,
    property_bindings: Optional[Dict[str, Dict]] = None,
    validate_fn: Optional[Callable] = None,
    max_tries: Optional[int] = None,
    slot_bindings: Optional[Dict[str, str]] = None,
    hidden_nodes: Optional[List[str]] = None,
) -> None:
    """Register a task. output.type in the rule file selects "item" or "process" path.
    slot_bindings and hidden_nodes are auto-inferred if not provided."""
    from herosjourney.core.registry import (
        TaskSpec, PropertySpec, ProcessSpec,
        register_task as _registry_register_task,
    )

    eval_info = eval or {}

    with open(rules) as f:
        rule = json.load(f)

    output_block    = rule.get("output", {})
    output_type     = output_block.get("type", "item")
    input_category  = rule["input_category"]
    output_category = output_block.get("category") if output_type == "item" else None

    # Normalise process to list; use index 0 as the working template.
    process_names    = [process] if isinstance(process, str) else list(process)
    loaded_processes = [load_process(p) for p in process_names]
    working_process  = loaded_processes[0]

    if slot_bindings is None:
        slot_bindings = {}
        for step in working_process["steps"]:
            pool = step.get("argument", {}).get("pool")
            if pool == input_category:
                slot_bindings[step["id"]] = f"{name}:input"
            elif output_category and pool == output_category:
                slot_bindings[step["id"]] = f"{name}:output"

    if hidden_nodes is None:
        _hidden = [sid for sid, role in slot_bindings.items()
                   if role.split(":")[-1] == "output"]
    else:
        _hidden = hidden_nodes   # explicit escape hatch
    _props  = property_bindings or {}

    property_spec = PropertySpec(
        task_type    = name,
        rules        = rules,
        validate_fn  = validate_fn,
        correct_rule = eval_info.get("correct_rule", ""),
        description  = eval_info.get("description", ""),
        max_tries    = max_tries,
        split        = split,
    )

    if output_type == "item":
        template_spec = TemplateSpec(
            name=name,
            process=working_process,
            slot_bindings=slot_bindings,
            hidden_nodes=_hidden,
            property_bindings=_props,
        )
        register_template(template_spec)

        mapping_nodes = [sid for sid, role in slot_bindings.items()
                         if role.split(":")[-1] == "output"]
        gen_fn = make_property_task_generator(name, name, validate_fn)

        def _constant_selector(entity, seed):
            return template_spec

        _registry_register_task(TaskSpec(
            property_spec       = property_spec,
            processes           = [ProcessSpec(name=name, mapping_nodes=mapping_nodes)],
            process_selector    = _constant_selector,
            gen_fn              = gen_fn,
            template_name       = name,
            distractor_rules    = distractors,
            distractor_template = "additive",
        ))

    else:
        gen_fn = make_procedural_task_generator(
            task_type         = name,
            base_process      = working_process,
            slot_bindings     = slot_bindings,
            hidden_nodes      = _hidden,
            property_bindings = _props,
            validate_fn       = validate_fn,
        )

        _registry_register_task(TaskSpec(
            property_spec       = property_spec,
            processes           = [ProcessSpec(name=name, mapping_nodes=[])],
            process_selector    = None,
            gen_fn              = gen_fn,
            template_name       = None,
            distractor_rules    = distractors,
            distractor_template = "additive",
        ))

