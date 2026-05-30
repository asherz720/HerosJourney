"""Composition evaluators: map attribute value indices to dimension value indices."""

from typing import Dict, List


# ---------------------------------------------------------------------------
# n_values derivation
# ---------------------------------------------------------------------------

def derive_dim_nvals(dim_name: str, funcs: List[Dict], comp, comp_type: str) -> int:
    """Derive n_values for a dimension when absent from the rule file."""
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
        # multi-dim override
        if dim_name in comp.get("override_dims", {}):
            max_val = max(max_val, comp["override_dims"][dim_name])
        # legacy single-dim override
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
    """Sum all function outputs into the same dimension."""
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
    """Each function maps independently to its own output dimension."""
    return {f["output"]: f["map"][attr_vals[f["input"]]] for f in funcs}


def eval_conditional(
    funcs: List[Dict],
    func_by_name: Dict[str, Dict],
    comp: Dict,
    attr_vals: Dict[str, int],
) -> Dict[str, int]:
    """Selector picks a regime; active branch function determines output; fixed dims fill rest."""
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
    """Base function by default; one specific attribute value always overrides to fixed dims."""
    if attr_vals[comp["override_attr"]] == comp["override_value"]:
        if "override_dims" in comp:
            return dict(comp["override_dims"])
        return {comp["override_dim"]: comp["override_dim_value"]}
    base_fns = comp.get("base_fns") or [comp["base_fn"]]
    result: Dict[str, int] = {}
    for fn_name in base_fns:
        fn = func_by_name[fn_name]
        result[fn["output"]] = fn["map"][attr_vals[fn["input"]]]
    return result


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
    """Dispatch to the evaluator for comp_type; raises ValueError for unknown types."""
    fn = COMPOSITION_REGISTRY.get(comp_type)
    if fn is None:
        raise ValueError(
            f"Unknown composition type: {comp_type!r}. "
            f"Registered: {sorted(COMPOSITION_REGISTRY)}"
        )
    return fn(funcs, func_by_name, comp, attr_vals)
