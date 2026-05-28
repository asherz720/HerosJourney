"""
tree_management/function_specs/splits.py
Split functions and split validators — the full split contract.

Split functions (construction)
-------------------------------
Partition the full N-dimensional attribute grid into source (demo) entities
and gen (eval) entities from a rule spec.

Signature (for registry functions):
    fn(attr_nvals, spec_dict, comp_dict) -> (source_pairs, gen_pairs)

    attr_nvals : List[int]  — n_values for each attribute, in attribute order
    spec_dict  : Dict       — the full split spec from the rule file (fn, seed, ...)
    comp_dict  : Dict       — the composition spec (needed by c4_override)

Each pair is a Tuple[int, ...] of attribute value indices, one per attribute.

Split validators (post-fill verification)
------------------------------------------
Check that a fully-filled elements dict satisfies the identifiability
conditions for a given task type.  Called after fill_elements() populates
elements["entities"]["source"] and elements["entities"]["gen"].

To add a new split function
-----------------------------
1. Write a function with the signature above.
2. Register it in SPLIT_REGISTRY at the bottom of this file.
3. Document it in tree_management/function_specs/split_functions.json.
"""

import random
from collections import defaultdict
from typing import Dict, List, Tuple

Pair = Tuple[int, ...]


# ---------------------------------------------------------------------------
# Split functions
# ---------------------------------------------------------------------------

def two_offset(
    attr_nvals: List[int],
    spec_dict: Dict,
    comp_dict: Dict,
) -> Tuple[List[Pair], List[Pair]]:
    """
    Bipartite two-offset split for exactly 2 attributes.

    For each attr1 value a1_i the source gets two pairs:
        (a1_i, a2_{i % n_a2})  and  (a1_i, a2_{(i+1) % n_a2})

    This guarantees bipartite connectivity (every attr1 shares at least one
    attr2 partner with every other attr1), which is sufficient for concept
    identifiability in additive, independent, and conditional tasks.

    Gen = all N×M pairs not in source.
    """
    if len(attr_nvals) != 2:
        raise ValueError(
            f"two_offset requires exactly 2 attributes, got {len(attr_nvals)}. "
            "For 3+ attributes use a different split function."
        )
    n_a1, n_a2 = attr_nvals
    seed = spec_dict.get("seed", 0)
    rng  = random.Random(seed)
    a1   = list(range(n_a1)); rng.shuffle(a1)
    a2   = list(range(n_a2)); rng.shuffle(a2)
    source: set = set()
    for i in range(n_a1):
        source.add((a1[i], a2[i % n_a2]))
        source.add((a1[i], a2[(i + 1) % n_a2]))
    all_pairs = [(v1, v2) for v1 in a1 for v2 in a2]
    gen = [p for p in all_pairs if p not in source]
    return list(source), gen


def conditional_two_offset(
    attr_nvals: List[int],
    spec_dict: Dict,
    comp_dict: Dict,
) -> Tuple[List[Pair], List[Pair]]:
    """
    Regime-aware two-offset split for conditional tasks with exactly 2 attributes.

    Applies a within-regime two-offset cover to each group of attr1 values that share
    a regime (as determined by comp_dict["selector"]["map"]).  This guarantees:

      1. Each attr1 value appears with ≥ 2 distinct attr2 partners
         (enabling regime identification).
      2. Every attr2 value appears within each regime at least once
         (enabling identification of the within-regime mapping).

    Precondition: n_regime_classes >= n_a2 - 1 for each regime, where
    n_regime_classes is the number of attr1 values in that regime.
    (For conditional: 2 classes per regime, 3 roles → 2 >= 3-1 = 2 ✓)

    Gen = all N×M pairs not in source.
    """
    if len(attr_nvals) != 2:
        raise ValueError(
            f"conditional_two_offset requires exactly 2 attributes, got {len(attr_nvals)}."
        )
    n_a1, n_a2 = attr_nvals
    seed = spec_dict.get("seed", 0)
    rng  = random.Random(seed)

    # Group attr1 values by regime using the selector map.
    selector   = comp_dict.get("selector", {})
    regime_map = selector.get("map", list(range(n_a1)))
    regimes: Dict[int, List[int]] = {}
    for class_val, regime in enumerate(regime_map):
        regimes.setdefault(regime, []).append(class_val)

    # Shared role ordering — same shuffle across all regimes for consistency.
    a2 = list(range(n_a2)); rng.shuffle(a2)

    source: set = set()
    for regime_classes in regimes.values():
        rng.shuffle(regime_classes)
        n_rc = len(regime_classes)
        # Verify within-regime coverage is achievable.
        if n_rc < n_a2 - 1:
            raise ValueError(
                f"conditional_two_offset: regime has {n_rc} class value(s) but "
                f"{n_a2} role values — need at least {n_a2 - 1} classes per regime "
                "for two-offset to cover all role values within the regime. "
                "Use a different split function or increase n_values."
            )
        for i, c in enumerate(regime_classes):
            source.add((c, a2[i % n_a2]))
            source.add((c, a2[(i + 1) % n_a2]))

    all_pairs = [(v1, v2) for v1 in range(n_a1) for v2 in range(n_a2)]
    gen = [p for p in all_pairs if p not in source]
    return list(source), gen


def c4_override(
    attr_nvals: List[int],
    spec_dict: Dict,
    comp_dict: Dict,
) -> Tuple[List[Pair], List[Pair]]:
    """
    Override-task split for exactly 2 attributes.

    Provides full (class × role) coverage for all but one held-out class,
    and partial coverage for the held-out class: only one non-override role
    value is shown in source.  Gen tests the held-out class with both the
    override role value and all remaining non-override role values.

    This is the minimal split that makes the override rule identifiable:
    the agent can distinguish "this role always overrides" from "this class
    happens to need the same item" only when it has seen the held-out class
    with at least one non-override role.

    seed (optional): shuffles the class ordering before applying the split
    structure, permuting which class is held out and which non-override role
    is revealed in source.  Different seeds produce structurally equivalent
    but surface-distinct splits.
    """
    if len(attr_nvals) != 2:
        raise ValueError(
            f"c4_override requires exactly 2 attributes, got {len(attr_nvals)}."
        )
    n_a1, n_a2   = attr_nvals
    override_col = comp_dict.get("override_value", n_a2 - 1)
    non_ov       = [j for j in range(n_a2) if j != override_col]

    seed = spec_dict.get("seed")
    if seed is not None:
        rng = random.Random(seed)
        classes = list(range(n_a1)); rng.shuffle(classes)
        rng.shuffle(non_ov)
    else:
        classes = list(range(n_a1))

    source: List[Pair] = []
    for c in classes[:-1]:
        for j in range(n_a2):
            source.append((c, j))
    held_out = classes[-1]
    source.append((held_out, non_ov[0]))
    gen: List[Pair] = [(held_out, override_col)] + [(held_out, j) for j in non_ov[1:]]
    return source, gen


def independent_leave_one_out(
    attr_nvals: List[int],
    spec_dict: Dict,
    comp_dict: Dict,
) -> Tuple[List[Pair], List[Pair]]:
    """
    Leave-one-out split for independent tasks with a 2×2 attribute grid.

    Source = all N×M combinations except one (the gen pair).
    Gen = [gen_pair], chosen via seed so that both attribute values of the
    gen entity appear in source with at least one cross-partner each.

    Use when two_offset would degenerate to an empty gen set (i.e., n_a1 == n_a2 == 2).
    For any grid with n_a1>=2 and n_a2>=2, removing one pair always leaves every
    attribute value covered in source.
    """
    if len(attr_nvals) != 2:
        raise ValueError(
            f"independent_leave_one_out requires exactly 2 attributes, "
            f"got {len(attr_nvals)}."
        )
    n_a1, n_a2 = attr_nvals
    seed = spec_dict.get("seed", 0)
    rng  = random.Random(seed)

    all_pairs = [(v1, v2) for v1 in range(n_a1) for v2 in range(n_a2)]
    rng.shuffle(all_pairs)
    gen_pair = all_pairs[0]
    source   = [p for p in all_pairs if p != gen_pair]
    return source, [gen_pair]


# ---------------------------------------------------------------------------
# Registry and dispatcher
# ---------------------------------------------------------------------------

SPLIT_REGISTRY: Dict[str, callable] = {
    "two_offset":                two_offset,
    "conditional_two_offset":    conditional_two_offset,
    "c4_override":               c4_override,
    "independent_leave_one_out": independent_leave_one_out,
}


def apply_split(
    split_spec: Dict,
    attr_nvals: List[int],
    comp,
) -> Tuple[List[Pair], List[Pair]]:
    """
    Dispatch to the correct split function.

    Parameters
    ----------
    split_spec : the split dict from the rule file, e.g. {"fn": "two_offset", "seed": 0}
    attr_nvals : [n_values_attr0, n_values_attr1, ...]
    comp       : the composition value (str or dict) from the rule file

    Returns
    -------
    (source_pairs, gen_pairs) — lists of index tuples, one int per attribute
    """
    fn = split_spec["fn"]

    # "explicit" is handled inline — no registry entry needed
    if fn == "explicit":
        return (
            [tuple(p) for p in split_spec["source"]],
            [tuple(p) for p in split_spec["gen"]],
        )

    if fn not in SPLIT_REGISTRY:
        raise ValueError(
            f"Unknown split fn: {fn!r}. "
            f"Registered: {sorted(SPLIT_REGISTRY)}. "
            "Add it to SPLIT_REGISTRY and document it in split_functions.json."
        )

    comp_dict = comp if isinstance(comp, dict) else {}
    return SPLIT_REGISTRY[fn](attr_nvals, split_spec, comp_dict)


# ---------------------------------------------------------------------------
# Split validators (post-fill identifiability checks)
# ---------------------------------------------------------------------------

def _check_bipartite_connected(
    attr1_to_attr2: Dict[int, set],
    attr2_to_attr1: Dict[int, set],
) -> None:
    """Raise ValueError if the bipartite coverage graph is disconnected."""
    adj: Dict[str, set] = defaultdict(set)
    for a1, partners in attr1_to_attr2.items():
        for a2 in partners:
            adj[f"1:{a1}"].add(f"2:{a2}")
            adj[f"2:{a2}"].add(f"1:{a1}")
    all_nodes = set(adj)
    if not all_nodes:
        return
    visited = {next(iter(all_nodes))}
    queue = list(visited)
    while queue:
        for nb in adj[queue.pop()]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    unreachable = all_nodes - visited
    if unreachable:
        raise ValueError(
            f"Coverage graph is disconnected. Unreachable: "
            f"{[n.split(':',1)[1] for n in unreachable]}. "
            f"Add source entities to bridge disconnected components."
        )


def validate_additive_split(elements: Dict) -> None:
    """
    Validate the source/gen split for additive (prop_1) tasks.

    Identifiability condition: the bipartite coverage graph (attr1 values ↔ attr2
    values, edges = observed source pairs) must be connected, and each value must
    appear with ≥ 2 partners on the other side.  This is the standard two-way ANOVA
    identifiability condition: base(class) and modifier(role) can only be separated
    if each is observed in enough contexts to cancel the other's effect.

    Checks:
      1. Every attr1 value appears in source with ≥ 2 distinct attr2 partners.
      2. Every attr2 value appears in source with ≥ 2 distinct attr1 partners.
      3. The source bipartite graph is connected.
      4. No gen pair duplicates a source pair.
    """
    attr_names = list(elements["attribute_labels"].keys())
    source = [tuple(int(e["attributes"][a]) for a in attr_names)
              for e in elements["entities"]["source"]]
    gen    = [tuple(int(e["attributes"][a]) for a in attr_names)
              for e in elements["entities"]["gen"]]

    a1_to_a2: Dict[int, set] = defaultdict(set)
    a2_to_a1: Dict[int, set] = defaultdict(set)
    for i, j in source:
        a1_to_a2[i].add(j)
        a2_to_a1[j].add(i)

    for val, partners in a1_to_a2.items():
        if len(partners) < 2:
            raise ValueError(
                f"attr1={val} appears in source with only {len(partners)} attr2 partner(s). Need ≥ 2."
            )
    for val, partners in a2_to_a1.items():
        if len(partners) < 2:
            raise ValueError(
                f"attr2={val} appears in source with only {len(partners)} attr1 partner(s). Need ≥ 2."
            )
    _check_bipartite_connected(a1_to_a2, a2_to_a1)

    source_set = set(source)
    for pair in gen:
        if pair in source_set:
            raise ValueError(f"Gen pair {pair} already in source.")


# Keep old name as an alias so existing call sites don't break.
validate_property_split = validate_additive_split


def validate_independent_split(elements: Dict) -> None:
    """
    Validate the source/gen split for independent (prop_2) tasks.

    Identifiability condition: since class → size and role → color are completely
    independent mappings, each can be identified from source entities that exhibit
    it alone.  Every attr1 value must appear at least once (to identify its size),
    and every attr2 value must appear at least once (to identify its color).
    No cross-attribute bipartite connectivity is required.

    Checks:
      1. Every attr1 value appears in source ≥ 1 time.
      2. Every attr2 value appears in source ≥ 1 time.
      3. No gen pair duplicates a source pair.
    """
    attr_names = list(elements["attribute_labels"].keys())
    n_a1 = max(int(e["attributes"][attr_names[0]]) for e in elements["entities"]["all"]) + 1
    n_a2 = max(int(e["attributes"][attr_names[1]]) for e in elements["entities"]["all"]) + 1
    source = [tuple(int(e["attributes"][a]) for a in attr_names)
              for e in elements["entities"]["source"]]
    gen    = [tuple(int(e["attributes"][a]) for a in attr_names)
              for e in elements["entities"]["gen"]]

    seen_a1 = {p[0] for p in source}
    seen_a2 = {p[1] for p in source}
    for val in range(n_a1):
        if val not in seen_a1:
            raise ValueError(f"attr1={val} never appears in source. Independent mapping not identifiable.")
    for val in range(n_a2):
        if val not in seen_a2:
            raise ValueError(f"attr2={val} never appears in source. Independent mapping not identifiable.")

    source_set = set(source)
    for pair in gen:
        if pair in source_set:
            raise ValueError(f"Gen pair {pair} already in source.")


def validate_conditional_split(elements: Dict) -> None:
    """
    Validate the source/gen split for conditional (prop_3) tasks.

    Identifiability conditions:
      1. Each attr1 value appears with ≥ 2 distinct attr2 partners in source —
         needed to identify which regime a class belongs to (you need to see the
         same class in enough contexts to distinguish its behaviour).
      2. Every attr2 value appears within each regime at least once in source —
         needed to identify the within-regime mapping (e.g. f_size and f_color
         independently, per regime).
      3. No gen pair duplicates a source pair.

    Reads comp["selector"]["map"] from elements["_composition"] to determine
    which attr1 values belong to which regime.
    """
    comp = elements.get("_composition")
    if comp is None:
        raise ValueError(
            "validate_conditional_split requires '_composition' in elements. "
            "Ensure fill_elements stores it (structured path only)."
        )
    selector   = comp.get("selector", {})
    regime_map = selector.get("map")
    if regime_map is None:
        raise ValueError("Composition has no selector.map — cannot determine regime structure.")

    attr_names = list(elements["attribute_labels"].keys())
    source = [tuple(int(e["attributes"][a]) for a in attr_names)
              for e in elements["entities"]["source"]]
    gen    = [tuple(int(e["attributes"][a]) for a in attr_names)
              for e in elements["entities"]["gen"]]

    a1_to_a2: Dict[int, set] = defaultdict(set)
    for a1, a2 in source:
        a1_to_a2[a1].add(a2)
    for val, partners in a1_to_a2.items():
        if len(partners) < 2:
            raise ValueError(
                f"attr1={val} appears in source with only {len(partners)} attr2 partner(s). "
                "Need ≥ 2 for regime identifiability."
            )

    n_a2 = max(int(e["attributes"][attr_names[1]]) for e in elements["entities"]["all"]) + 1
    regimes: Dict[int, List[int]] = {}
    for class_val, regime in enumerate(regime_map):
        regimes.setdefault(regime, []).append(class_val)

    for regime_id, regime_classes in regimes.items():
        regime_source = [(a1, a2) for a1, a2 in source if a1 in regime_classes]
        seen_a2 = {a2 for _, a2 in regime_source}
        for val in range(n_a2):
            if val not in seen_a2:
                raise ValueError(
                    f"attr2={val} never appears in source within regime {regime_id} "
                    f"(classes {regime_classes}). Within-regime mapping not identifiable."
                )

    source_set = set(source)
    for pair in gen:
        if pair in source_set:
            raise ValueError(f"Gen pair {pair} already in source.")
