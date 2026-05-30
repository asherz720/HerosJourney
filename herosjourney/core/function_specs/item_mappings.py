"""Build semantic and nonce item surface names from dimension value indices."""

from typing import Dict, List


def compute_strides(dims: List[Dict], dim_nvals: Dict[str, int]) -> Dict[str, int]:
    """Row-major strides for linearizing dimensions into a single item_id integer."""
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
    """Build semantic item name: join → all dims joined; index → first dim only."""
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
    """Build nonce item name using syllable pools; numeric mode exposes the index directly."""
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
