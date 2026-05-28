"""
world_info/function_specs/compositions.py
Implementations of built-in composition functions.

Each eval_* function maps a set of attribute values to a dict of dimension values.

Signature
---------
    eval_*(funcs, func_by_name, comp, attr_vals) -> Dict[str, int]

    funcs        : List[Dict]       — function specs from the rule file
    func_by_name : Dict[str, Dict]  — funcs indexed by name
    comp         : str | Dict       — the composition spec
    attr_vals    : Dict[str, int]   — {attr_name: attr_value_index} for this entity

Returns {dim_name: dim_value_index}.

To add a new composition type
------------------------------
1. Write an eval_* function below.
2. Register it in COMPOSITION_REGISTRY at the bottom of this file.
3. Add documentation to world_info/function_specs/compositions.json.
"""

from typing import Dict, List


# ---------------------------------------------------------------------------
# n_values derivation
# ---------------------------------------------------------------------------

def derive_dim_nvals(dim_name: str, funcs: List[Dict], comp, comp_type: str) -> int:
    """
    Derive the number of distinct values for a dimension from the function maps.

    Called when n_values is absent from the dimension spec in the rule file.
    """
    if comp_type == "additive":
        total = sum(max(f["map"]) for f in funcs if f["output"] == dim_name)
        return total + 1

    max_val = max(
        (max(f["map"]) for f in funcs if f["output"] == dim_name),
        default=0,
    )
    if isinstance(comp, dict):
        for branch in comp.get("branches", []):
            if dim_name in branch.get("fixed", {}):
                max_val = max(max_val, branch["fixed"][dim_name])
        if comp.get("override_dim") == dim_name:
            max_val = max(max_val, comp.get("override_dim_value", 0))
    return max_val + 1


# ---------------------------------------------------------------------------
# Composition evaluators
# ---------------------------------------------------------------------------

def eval_additive(
    funcs: List[Dict],
    _func_by_name: Dict[str, Dict],  # unused: additive only needs the map values
    _comp,                            # unused: no branching logic
    attr_vals: Dict[str, int],
) -> Dict[str, int]:
    """
    All functions contribute to the same output dimension; item index = their sum.

    Works with any number of functions / attributes — each f["input"] is looked up
    in attr_vals by name, so the number of attributes is unrestricted.
    """
    result: Dict[str, int] = {}
    for f in funcs:
        val = f["map"][attr_vals[f["input"]]]
        result[f["output"]] = result.get(f["output"], 0) + val
    return result


def eval_independent(
    funcs: List[Dict],
    _func_by_name: Dict[str, Dict],  # unused: each function acts independently
    _comp,                            # unused: no branching logic
    attr_vals: Dict[str, int],
) -> Dict[str, int]:
    """
    Each function controls a separate output dimension independently.

    Works with any number of functions / attributes.
    """
    return {f["output"]: f["map"][attr_vals[f["input"]]] for f in funcs}


def eval_conditional(
    funcs: List[Dict],
    func_by_name: Dict[str, Dict],
    comp: Dict,
    attr_vals: Dict[str, int],
) -> Dict[str, int]:
    """
    A selector maps one attribute to a regime; the active regime determines
    which function controls which dimension.  All other dimensions take
    fixed values from the branch spec.

    ``comp`` must be a dict with keys: selector, branches.
    """
    sel    = comp["selector"]
    regime = sel["map"][attr_vals[sel["input"]]]
    branch = next(b for b in comp["branches"] if b["regime"] == regime)
    active = func_by_name[branch["active_fn"]]
    result = {active["output"]: active["map"][attr_vals[active["input"]]]}
    result.update(branch["fixed"])
    return result


def eval_override(
    funcs: List[Dict],
    func_by_name: Dict[str, Dict],
    comp: Dict,
    attr_vals: Dict[str, int],
) -> Dict[str, int]:
    """
    A base function determines the item by default.  When one specific attribute
    value is present the item is always a fixed dimension value, regardless of
    the base function's output.

    ``comp`` must be a dict with keys: base_fn, override_attr, override_value,
    override_dim, override_dim_value.
    """
    if attr_vals[comp["override_attr"]] == comp["override_value"]:
        return {comp["override_dim"]: comp["override_dim_value"]}
    base_fn = func_by_name[comp["base_fn"]]
    return {base_fn["output"]: base_fn["map"][attr_vals[base_fn["input"]]]}


# ---------------------------------------------------------------------------
# Registry and dispatcher
# ---------------------------------------------------------------------------

COMPOSITION_REGISTRY: Dict[str, callable] = {
    "additive":    eval_additive,
    "independent": eval_independent,
    "conditional": eval_conditional,
    "override":    eval_override,
}


def eval_composition(
    comp_type: str,
    funcs: List[Dict],
    func_by_name: Dict[str, Dict],
    comp,
    attr_vals: Dict[str, int],
) -> Dict[str, int]:
    """
    Dispatch to the correct composition evaluator.

    Parameters
    ----------
    comp_type    : "additive" | "independent" | "conditional" | "override"
                   (or any key registered in COMPOSITION_REGISTRY)
    funcs        : all function specs from the rule file
    func_by_name : funcs indexed by name
    comp         : the full composition value (str or dict) from the rule file
    attr_vals    : {attr_name: attr_value_index} for the current entity

    Returns
    -------
    {dim_name: dim_value_index}
    """
    fn = COMPOSITION_REGISTRY.get(comp_type)
    if fn is None:
        raise ValueError(
            f"Unknown composition type: {comp_type!r}. "
            f"Registered: {sorted(COMPOSITION_REGISTRY)}"
        )
    return fn(funcs, func_by_name, comp, attr_vals)
