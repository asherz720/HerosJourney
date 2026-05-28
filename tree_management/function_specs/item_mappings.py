"""
world_info/function_specs/item_mappings.py
Implementations of built-in item mapping functions.

An item mapping function converts a dict of dimension values into an item
surface name (semantic or nonce) and a dense integer item_id.

Helpers
-------
compute_strides  : row-major strides for item_id linearization over dimensions
item_sem_name    : build a semantic item name from dimension values
item_nonce_name  : build a nonce item name from dimension values

To add a new items fn
----------------------
1. Add a branch in item_sem_name and item_nonce_name below.
2. Document it in world_info/function_specs/item_mappings.json.
"""

from typing import Dict, List


def compute_strides(dims: List[Dict], dim_nvals: Dict[str, int]) -> Dict[str, int]:
    """
    Compute row-major strides for linearizing a multi-dimensional index.

    The last dimension has stride 1; earlier dimensions multiply up.
    item_id = sum(dim_val[d] * strides[d] for d in dims)

    Parameters
    ----------
    dims      : dimension specs in order (determines stride order)
    dim_nvals : {dim_name: n_values}
    """
    strides: Dict[str, int] = {}
    s = 1
    for d in reversed(dims):
        strides[d["name"]] = s
        s *= dim_nvals[d["name"]]
    return strides


def item_sem_name(
    items_fn: str,
    dim_vals: Dict[str, int],
    dim_order: List[str],
    dim_sem: Dict[str, List[str]],
    object_sem: str,
) -> str:
    """
    Build the semantic surface name for an item.

    Parameters
    ----------
    items_fn   : "join" | "index"
    dim_vals   : {dim_name: dim_value_index} for this item
    dim_order  : dimension names in order (determines name prefix order for "join")
    dim_sem    : {dim_name: [surface_name_0, surface_name_1, ...]}
    object_sem : the sampled object noun (e.g. "sword", "lance")

    "join"  → "{dim0_val}_{dim1_val}_..._{object}"  (all dimensions in order)
    "index" → "{dim0_val}_{object}"                 (single dimension only)
    """
    if items_fn == "join":
        parts = [dim_sem[d][dim_vals[d]] for d in dim_order]
        return "_".join(parts) + "_" + object_sem
    if items_fn == "index":
        d = dim_order[0]
        return f"{dim_sem[d][dim_vals[d]]}_{object_sem}"
    if items_fn == "numeric":
        # Exposes the numeric index directly: "{dim_name}_{index}_{noun}".
        # Makes the additive structure transparent — items are clearly ordered numbers.
        d = dim_order[0]
        return f"{d}_{dim_vals[d]}_{object_sem}"
    raise ValueError(
        f"Unknown items fn: {items_fn!r}. "
        "Add a branch here and document it in item_mappings.json."
    )


def item_nonce_name(
    items_fn: str,
    dim_vals: Dict[str, int],
    dim_order: List[str],
    dim_nonce: Dict[str, List[str]],
    object_nonce: str = "",
) -> str:
    """
    Build the nonce surface name for an item.

    Parameters
    ----------
    items_fn     : "join" | "index" | "numeric"
    dim_vals     : {dim_name: dim_value_index} for this item
    dim_order    : dimension names in order
    dim_nonce    : {dim_name: [nonce_syllable_0, nonce_syllable_1, ...]}
    object_nonce : nonce noun suffix (e.g. "plex"); appended when non-empty,
                   making nonce items structurally parallel to semantic ones.

    "join"    → "{dim0_syl}_{dim1_syl}_{noun}"
    "index"   → "{dim0_syl}_{noun}"
    "numeric" → "{dim_label}_{index}_{noun}"  (dim_label = first syllable of pool)
    """
    suffix = f"_{object_nonce}" if object_nonce else ""
    if items_fn == "join":
        parts = [dim_nonce[d][dim_vals[d]] for d in dim_order]
        return "_".join(parts) + suffix
    if items_fn == "index":
        d = dim_order[0]
        return dim_nonce[d][dim_vals[d]] + suffix
    if items_fn == "numeric":
        # Nonce: use the first syllable of the dim's nonce pool as the category label,
        # append the index. E.g., "grel_0_plex", "grel_1_plex", ...
        d = dim_order[0]
        label = dim_nonce[d][0]
        return f"{label}_{dim_vals[d]}{suffix}"
    raise ValueError(
        f"Unknown items fn: {items_fn!r}. "
        "Add a branch here and document it in item_mappings.json."
    )
