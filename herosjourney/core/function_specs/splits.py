"""Split functions and validators: partition the attribute grid into source/gen pairs."""

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
    """Bipartite two-offset split for 2 attributes; each attr1 value paired with 2 attr2 values."""
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
    """Regime-aware two-offset split: within each regime every attr2 value appears at least once."""
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
    """Full coverage for all classes except one held-out; gen tests held-out with override and non-override roles."""
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
    """Source = all pairs minus one; gen = that one pair. Use for 2×2 grids where two_offset degenerates."""
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
    """Dispatch to the split function named in split_spec["fn"]; return (source_pairs, gen_pairs)."""
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
    """Check ≥2 partners per value, bipartite connectivity, and no gen/source overlap."""
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
    """Check every attribute value appears in source ≥1 time and no gen/source overlap."""
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
    """Check ≥2 attr2 partners per attr1 value, full attr2 coverage within each regime, no gen/source overlap."""
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
