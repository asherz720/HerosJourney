"""
tree_management/elements.py
Fill abstract rule files with surface names from a lexicon.

Rule files store only structural information (attribute counts, function maps,
composition type, split algorithm) plus optional input/output pool specs.
This module generates fully-named elements dicts from that abstract spec by
sampling names from a lexicon — a different seed produces a different surface
variant.

What fill_elements does
-----------------------
fill_elements is the surface-realization layer: rule file + lexicon + seed →
elements dict with concrete names.  What gets filled is driven entirely by
which fields the rule file declares:

  input_category + input_meta (always required)
      Entity names, attribute values, entity properties (e.g. location).

  output_category + output_meta (optional)
      Item names and item properties (e.g. shop location).
      Absent for procedural tasks — only entity names are filled.

  functions (optional)
      Deterministic item assignment via composition f.
      Absent for distractors (random assignment) and procedural tasks (no items).

  split / split_spec (optional)
      Source/gen split of entities.
      Absent for distractor specs → flat entity list.

Primary entry point
-------------------
    from tree_management.elements import fill_elements, load_lexicons

    sem_lex, nonce_lex = load_lexicons()
    with open("tree_management/rules/additive.json") as f:
        rule = json.load(f)

    filled = fill_elements(rule, sem_lex, nonce_lex, seed=3)

    # Run 10 variants:
    variants = [fill_elements(rule, sem_lex, nonce_lex, seed=i) for i in range(10)]

CLI usage (print split for an attribute grid)
---------------------------------------------
    python -m tree_management.elements split \\
        --attr1_values warrior mage rogue \\
        --attr2_values scout guard herald \\
        --seed 0
"""

import copy
import json
import random
import argparse
import os
from itertools import product as _iproduct
from typing import Dict, List, Tuple, Optional, Any

from tree_management.function_specs.compositions import derive_dim_nvals, eval_composition
from tree_management.function_specs.item_mappings import compute_strides, item_sem_name, item_nonce_name
from tree_management.function_specs.splits import apply_split

LEXICON_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "world_info", "lexicons")

# ---------------------------------------------------------------------------
# Default pool specs  (used when rule file omits "input" / "output")
# ---------------------------------------------------------------------------
# Uses type-based dispatch into the lexicon's category hierarchy:
#   input  type "entity" → sem_lex["entity"], nonce_lex["entity"]
#   output type "object", category "weapon" → sem_lex["object"]["weapon"]
#   dimensions → sem_lex["dimensions"][dim_name], nonce_lex["dimensions"][dim_name]


# ---------------------------------------------------------------------------
# Lexicon loading
# ---------------------------------------------------------------------------

def _load_lexicon(name: str) -> Dict:
    path = os.path.join(LEXICON_DIR, f"{name}_lexicon.json")
    with open(path) as f:
        return json.load(f)


def load_lexicons(lexicon_ref: str = "default") -> Tuple[Dict, Dict]:
    """
    Load semantic and nonce lexicons.

    ``lexicon_ref`` is either "default" or a path like
    "world_info/lexicons/distractor" (basename is used as prefix):
      - "default"     → semantic_lexicon.json + nonce_lexicon.json
      - "distractor"  → distractor_semantic_lexicon.json + distractor_nonce_lexicon.json

    Returns (sem_lex, nonce_lex).
    """
    ref = os.path.basename(lexicon_ref.rstrip("/\\"))
    if ref in ("default", ""):
        return _load_lexicon("semantic"), _load_lexicon("nonce")
    return _load_lexicon(f"{ref}_semantic"), _load_lexicon(f"{ref}_nonce")


# ---------------------------------------------------------------------------
# Pool resolution
# ---------------------------------------------------------------------------

def _resolve_pool(lex: Dict, path: str):
    """
    Resolve a dotted path into a lexicon dict, supporting integer list indices.

    Examples:
        _resolve_pool(sem_lex,   "entity.attributes.class")  → ["warrior", "mage", ...]
        _resolve_pool(nonce_lex, "entity.attribute_slots.0") → ["vrel_A", "vrel_B", ...]
        _resolve_pool(nonce_lex, "entity.attribute_labels.0")→ "vrel"
        _resolve_pool(sem_lex,   "dimensions.size")          → ["tiny", "small", ...]
    """
    val = lex
    for key in path.split("."):
        if isinstance(val, list):
            val = val[int(key)]
        else:
            val = val[key]
    return val


# ---------------------------------------------------------------------------
# Lexicon validation
# ---------------------------------------------------------------------------

def _validate_lexicon(
    inputs: List[Dict],
    outputs: List[Dict],
    nonce_lex: Dict,
    sem_lex: Dict,
    input_category: str,
    output_category: str,
    input_meta: Dict,
    output_meta: Dict,
) -> None:
    """
    Validate that all category fields and pool paths resolve in the lexicon.

    Checks:
      1. input_category and output_category resolve in sem_lex.
      2. input_meta.surface paths resolve under input_category in both sem_lex and nonce_lex.
      3. input_meta.properties paths resolve under input_category in sem_lex.
      4. output_meta.surface paths resolve under output_category in sem_lex.
      5. output_meta.properties paths resolve under output_category in sem_lex.
      6. Each input attribute pool resolves under input_category in sem_lex.
      7. Each output dimension pool resolves under output_category in sem_lex.
      8. Each input nonce_pool resolves in nonce_lex with enough values.
      9. Each input nonce_label resolves to a string in nonce_lex.
    """
    for category, kind in [(input_category, "input"), (output_category, "output")]:
        try:
            _resolve_pool(sem_lex, category)
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"Top-level {kind}_category '{category}' does not resolve in "
                f"semantic lexicon: {e}"
            )

    # --- input_meta name pool validation ---
    # name: used for both semantic and nonce names → validate in both lexicons.
    # repeat_name: semantic-only (nonce repeats use an offset into the primary nonce pool).
    for rel_path in input_meta.get("name", []):
        for lex, lex_name in [(sem_lex, "semantic"), (nonce_lex, "nonce")]:
            try:
                _resolve_pool(lex, f"{input_category}.{rel_path}")
            except (KeyError, TypeError) as e:
                raise ValueError(
                    f"input_meta name path '{rel_path}' does not resolve "
                    f"under input_category '{input_category}' in {lex_name} lexicon: {e}"
                )
    for rel_path in input_meta.get("repeat_name", []):
        try:
            _resolve_pool(sem_lex, f"{input_category}.{rel_path}")
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"input_meta repeat_name path '{rel_path}' does not resolve "
                f"under input_category '{input_category}' in semantic lexicon: {e}"
            )
    for key, rel_path in input_meta.get("properties", {}).items():
        try:
            _resolve_pool(sem_lex, f"{input_category}.{rel_path}")
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"input_meta.properties['{key}']: path '{rel_path}' does not resolve "
                f"under input_category '{input_category}' in semantic lexicon: {e}"
            )

    # --- output_meta name pool validation ---
    for rel_path in output_meta.get("name", []):
        try:
            _resolve_pool(sem_lex, f"{output_category}.{rel_path}")
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"output_meta name path '{rel_path}' does not resolve "
                f"under output_category '{output_category}' in semantic lexicon: {e}"
            )
    for key, rel_path in output_meta.get("properties", {}).items():
        try:
            _resolve_pool(sem_lex, f"{output_category}.{rel_path}")
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"output_meta.properties['{key}']: path '{rel_path}' does not resolve "
                f"under output_category '{output_category}' in semantic lexicon: {e}"
            )

    # --- attribute and dimension pool validation ---
    for v in inputs:
        try:
            _resolve_pool(sem_lex, f"{input_category}.{v['pool']}")
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"Input '{v['name']}': pool '{v['pool']}' does not resolve under "
                f"input_category '{input_category}': {e}"
            )

    for v in outputs:
        try:
            _resolve_pool(sem_lex, f"{output_category}.{v['pool']}")
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"Output '{v['name']}': pool '{v['pool']}' does not resolve under "
                f"output_category '{output_category}': {e}"
            )

    for v in inputs:
        n_vals = v["n_values"]
        full_nonce_pool = f"{input_category}.{v['nonce_pool']}"
        try:
            pool = _resolve_pool(nonce_lex, full_nonce_pool)
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(
                f"Input '{v['name']}': cannot resolve nonce_pool '{v['nonce_pool']}' "
                f"under input_category '{input_category}' in nonce lexicon: {e}"
            )
        if not isinstance(pool, list) or len(pool) < n_vals:
            raise ValueError(
                f"Input '{v['name']}': nonce_pool '{v['nonce_pool']}' resolved to "
                f"{len(pool) if isinstance(pool, list) else type(pool).__name__!r} "
                f"but needs at least {n_vals} values."
            )
        full_nonce_label = f"{input_category}.{v['nonce_label']}"
        try:
            label = _resolve_pool(nonce_lex, full_nonce_label)
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(
                f"Input '{v['name']}': cannot resolve nonce_label '{v['nonce_label']}' "
                f"under input_category '{input_category}' in nonce lexicon: {e}"
            )
        if not isinstance(label, str):
            raise ValueError(
                f"Input '{v['name']}': nonce_label '{v['nonce_label']}' should resolve "
                f"to a string, got {type(label).__name__!r}."
            )


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _sample_n(pool: List, n: int, rng: random.Random) -> List:
    if n > len(pool):
        raise ValueError(f"Pool too small: need {n}, have {len(pool)} ({pool[:3]}...)")
    return rng.sample(pool, n)


def _pick(pool: List, rng: random.Random):
    return rng.choice(pool)


def _make_names(part_pools: List[List[str]], n: int, rng: random.Random) -> List[str]:
    """
    Build n instance names by sampling one element from each pool and joining with '_'.
    Single-pool → bare word; two pools → "First_Last"; N pools → N-part joined name.
    e.g. part_pools=[first, last], n=3 → ["Gareth_Stonehall", "Mira_Duskwood", ...]
    """
    sampled = [_sample_n(pool, n, rng) for pool in part_pools]
    return ["_".join(vals) for vals in zip(*sampled)]


def _make_nonce_names(part_pools: List[List[str]], n: int, offset: int = 0) -> List[str]:
    """
    Build n nonce instance names by cycling through each pool and joining with '_'.
    """
    return [
        "_".join(pool[(i + offset) % len(pool)] for pool in part_pools)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# fill_elements — primary public API
# ---------------------------------------------------------------------------

def fill_elements(
    abstract: Dict,
    sem_lex: Dict,
    nonce_lex: Dict,
    seed: int = 0,
    split_spec: Optional[Dict] = None,
) -> Dict:
    """
    Convert an abstract rule dict (indices only, no surface names) into a
    fully-named dict compatible with build_tree_from_process.

    Rule files with "functions" key use a rule-determined item assignment and
    produce a source/gen split:
        entities: {"source": [...], "gen": [...]}

    Rule files without "functions" (distractor/pool mode) assign item_ids
    randomly and produce a flat entity list:
        entities: [...]

    Both formats produce entity dicts with the same fields:
        instance, nonce_instance, location,
        attributes, attribute_values, nonce_attribute_values, item_id

    Lexicon pools are driven by the top-level 'input_category', 'output_category',
    'input_meta', and 'output_meta' fields in the rule file.  Both rule files and
    distractor files must declare all four fields explicitly.
    """
    rng          = random.Random(seed)
    inputs       = abstract["inputs"]
    outputs_spec = abstract.get("outputs", abstract.get("dimensions", []))

    input_category  = abstract["input_category"]
    output_category = abstract.get("output_category")
    input_meta      = abstract["input_meta"]
    output_meta     = abstract.get("output_meta", {})

    if output_category is not None:
        _validate_lexicon(inputs, outputs_spec, nonce_lex, sem_lex,
                          input_category=input_category,
                          output_category=output_category,
                          input_meta=input_meta,
                          output_meta=output_meta)

    # --- Input name pools (from input_meta.name) ---
    name_sem_pools   = [_resolve_pool(sem_lex,   f"{input_category}.{p}") for p in input_meta["name"]]
    name_nonce_pools = [_resolve_pool(nonce_lex, f"{input_category}.{p}") for p in input_meta["name"]]

    # --- Output name (from output_meta.name, shared across all items in this variant) ---
    if output_category is not None:
        output_noun = "_".join(
            _pick(_resolve_pool(sem_lex, f"{output_category}.{p}"), rng)
            for p in output_meta["name"]
        )
        # Nonce noun: sampled from nonce_lex if a "nouns" pool exists there; otherwise
        # falls back to empty string (preserving prior behaviour for lexicons without it).
        _nonce_noun_parts = []
        for p in output_meta["name"]:
            pool_path = f"{output_category}.{p}"
            try:
                _nonce_noun_parts.append(_pick(_resolve_pool(nonce_lex, pool_path), rng))
            except (KeyError, IndexError):
                pass
        output_noun_nonce = "_".join(_nonce_noun_parts) if _nonce_noun_parts else ""

        def _build_output_prop_samples(entity_prop_samples: Dict[str, List]) -> Dict[str, Any]:
            """
            Pre-sample one value per output property (shared by all items in this variant).
            Returns a dict keyed by FULL lexicon pool path so _resolve_prop_val can look
            up values by the "pool" path declared in the process.

            For each output property, values already used as entity properties with the
            same key are excluded, preventing location collisions between item shops and
            entity positions.
            """
            # Collect the set of values used by entities for each property key.
            entity_used: Dict[str, set] = {
                key: set(vals) for key, vals in entity_prop_samples.items()
            }
            samples: Dict[str, Any] = {}
            for key, rel_path in output_meta["properties"].items():
                full_path = f"{output_category}.{rel_path}"
                pool      = _resolve_pool(sem_lex, full_path)
                excluded  = entity_used.get(key, set())
                candidates = [v for v in pool if v not in excluded]
                samples[full_path] = rng.choice(candidates if candidates else pool)
            return samples
    else:
        output_noun = ""
        output_noun_nonce = ""

        def _build_output_prop_samples(_entity_prop_samples: Dict[str, List]) -> Dict[str, Any]:
            return {}

    # --- Shared attribute surface names (inputs) ---
    # Semantic: sampled from the declared pool.
    # Nonce values: deterministic slice from nonce_pool (same mapping across seeds).
    # Nonce labels: single string from nonce_label path (display name for the attribute).
    attr_sem: Dict[str, List[str]] = {
        v["name"]: _sample_n(
            _resolve_pool(sem_lex, f"{input_category}.{v['pool']}"), v["n_values"], rng
        )
        for v in inputs
    }
    attr_nonce: Dict[str, List[str]] = {
        v["name"]: _resolve_pool(nonce_lex, f"{input_category}.{v['nonce_pool']}")[:v["n_values"]]
        for v in inputs
    }
    attr_nonce_labels: Dict[str, str] = {
        v["name"]: _resolve_pool(nonce_lex, f"{input_category}.{v['nonce_label']}")
        for v in inputs
    }

    # --- Helpers ---
    _attr_order = [v["name"] for v in inputs]

    def _entity(sem_name: str, nonce_name: str, props: Dict, attrs_dict: Dict) -> Dict:
        """Build one entity dict.  props is a dict of sampled entity properties
        (e.g. {"location": "citadel"}); all keys spread into the entity at top level.

        _consumed_pool_values maps each full lexicon pool path to the value this
        entity occupies in that pool.  build_tree_from_process seeds the per-tree
        `used` registry from this field so that pool-sampled nodes automatically
        exclude values already taken by the rule entity, regardless of which
        properties fill_elements declares.
        """
        attr_sem_v   = {n: attr_sem[n][v]   for n, v in attrs_dict.items()}
        attr_nonce_v = {n: attr_nonce[n][v] for n, v in attrs_dict.items()}
        consumed: Dict[str, Any] = {
            f"{input_category}.{rel_path}": props[key]
            for key, rel_path in input_meta["properties"].items()
            if key in props
        }
        return {
            "instance":                    sem_name,
            "nonce_instance":              nonce_name,
            **props,
            "attributes":                  attrs_dict,
            "attribute_values":            attr_sem_v,
            "nonce_attribute_values":      attr_nonce_v,
            # Pre-computed ordered lists — resolver picks the right variant via use_nonce.
            "attribute_names":             _attr_order,
            "nonce_attribute_names":       [attr_nonce_labels[n] for n in _attr_order],
            "attribute_values_list":       [attr_sem_v[n]   for n in _attr_order],
            "nonce_attribute_values_list": [attr_nonce_v[n] for n in _attr_order],
            # Consumed pool values — used to seed the `used` registry in tree building.
            "_consumed_pool_values":       consumed,
        }

    def _sample_entity_props(n: int) -> Dict[str, List]:
        """For each property in input_meta.properties sample n values (one per entity)."""
        result: Dict[str, List] = {}
        for key, path in input_meta["properties"].items():
            pool    = _resolve_pool(sem_lex, f"{input_category}.{path}")
            samples = _sample_n(pool, min(n, len(pool)), rng)
            while len(samples) < n:
                samples = samples + samples
            result[key] = samples[:n]
        return result

    attribute_labels = {
        v["name"]: {"semantic_name": v["name"], "nonce_name": attr_nonce_labels[v["name"]]}
        for v in inputs
    }

    # =========================================================
    # Procedural path: no output items; process template drives
    # the output. Detected by absence of output_category, functions,
    # and n_items in the rule file.
    # =========================================================
    if output_category is None and "functions" not in abstract and "n_items" not in abstract:
        _split_spec = split_spec or abstract.get("split")
        if _split_spec is None:
            raise ValueError(
                "Procedural tasks require a split specification. "
                "Pass split_spec= to fill_elements() or add 'split' to the rule file."
            )
        comp_for_split = abstract.get("composition", {})
        source_pairs, gen_pairs = apply_split(
            {**_split_spec, "seed": seed}, [v["n_values"] for v in inputs], comp_for_split
        )
        n_total            = len(source_pairs) + len(gen_pairs)
        sem_names          = _make_names(name_sem_pools, n_total, rng)
        nonce_names        = _make_nonce_names(name_nonce_pools, n_total)
        input_prop_samples = _sample_entity_props(n_total)

        def _make_proc_entity(pair, idx):
            props = {key: vals[idx] for key, vals in input_prop_samples.items()}
            return _entity(sem_names[idx], nonce_names[idx], props,
                           {inputs[i]["name"]: int(pair[i]) for i in range(len(inputs))})

        source = [_make_proc_entity(p, i) for i, p in enumerate(source_pairs)]
        gen    = [_make_proc_entity(p, len(source_pairs) + i)
                  for i, p in enumerate(gen_pairs)]
        return {
            "attribute_labels": attribute_labels,
            "entities":         {"source": source, "gen": gen, "all": source + gen},
            "_pool_map":        {input_category: "entities.all"},
            "_input_category":  input_category,
            "_input_meta":      input_meta,
            "_attr_order":      _attr_order,
            "_variant_info":    {"seed": seed, "variant_type": "procedural_filled"},
        }

    # =========================================================
    # Structured path: rule-determined items + source/gen split
    # =========================================================
    if "functions" in abstract:
        outputs    = outputs_spec
        funcs      = abstract["functions"]
        comp       = abstract["composition"]
        items_fn   = abstract["items"]
        # split_spec: prefer the caller-supplied value (from task spec); fall back to
        # the rule file's own "split" key for backward compatibility.
        split_spec = split_spec or abstract.get("split")
        if split_spec is None:
            raise ValueError(
                "No split specification found. "
                "Pass split_spec= to fill_elements() or add 'split' to the rule file."
            )
        comp_type    = comp if isinstance(comp, str) else comp["fn"]
        func_by_name = {f["name"]: f for f in funcs}

        dim_nvals: Dict[str, int] = {
            d["name"]: (d.get("n_values") or derive_dim_nvals(d["name"], funcs, comp, comp_type))
            for d in outputs
        }
        dim_order   = [d["name"] for d in outputs]
        dim_strides = compute_strides(outputs, dim_nvals)

        # Output surface names: both semantic and nonce are sampled (vary per seed).
        dim_sem: Dict[str, List[str]] = {
            d["name"]: _sample_n(
                _resolve_pool(sem_lex,   f"{output_category}.{d['pool']}"),       dim_nvals[d["name"]], rng
            )
            for d in outputs
        }
        dim_nonce: Dict[str, List[str]] = {
            d["name"]: _sample_n(
                _resolve_pool(nonce_lex, f"{output_category}.{d['nonce_pool']}"), dim_nvals[d["name"]], rng
            )
            for d in outputs
        }

        n_items = 1
        for d in dim_order:
            n_items *= dim_nvals[d]

        input_nvals = [v["n_values"] for v in inputs]
        combo_to_item: Dict[Tuple, int] = {
            combo: sum(
                eval_composition(comp_type, funcs, func_by_name, comp,
                                 {inputs[i]["name"]: combo[i] for i in range(len(inputs))})[d]
                * dim_strides[d] for d in dim_order
            )
            for combo in _iproduct(*[range(n) for n in input_nvals])
        }

        # Pass the fill seed to the split so structural variation (e.g. which
        # class is held out in c4_override, or which pairs are in source for
        # two_offset) also varies across fill variants.  The rule file's split
        # seed is overridden — it serves only as documentation of the seed=0
        # canonical structure.
        source_pairs, gen_pairs = apply_split({**split_spec, "seed": seed}, input_nvals, comp)
        n_total             = len(source_pairs) + len(gen_pairs)
        sem_names           = _make_names(name_sem_pools, n_total, rng)
        nonce_names         = _make_nonce_names(name_nonce_pools, n_total)
        input_prop_samples  = _sample_entity_props(n_total)
        output_prop_samples = _build_output_prop_samples(input_prop_samples)

        def _make_split_entity(pair, idx):
            props = {key: vals[idx] for key, vals in input_prop_samples.items()}
            e = _entity(sem_names[idx], nonce_names[idx], props,
                        {inputs[i]["name"]: int(pair[i]) for i in range(len(inputs))})
            e["item_id"] = combo_to_item[tuple(pair)]
            return e

        source = [_make_split_entity(p, i) for i, p in enumerate(source_pairs)]
        gen    = [_make_split_entity(p, len(source_pairs) + i) for i, p in enumerate(gen_pairs)]
        return {
            "n_items":          n_items,
            "attribute_labels": attribute_labels,
            "entities":         {"source": source, "gen": gen, "all": source + gen},
            # Item metadata — _resolve_arg_record uses these to compute item
            # surface names on demand instead of reading from a pre-built list.
            # Cost is NOT stored here: it is read from the process step spec.
            "_output_category": output_category,
            "_dim_sem":         dim_sem,
            "_dim_nonce":       dim_nonce,
            "_dim_strides":     dim_strides,
            "_dim_order":       dim_order,
            "_items_fn":              items_fn,
            "_output_noun":           output_noun,
            "_output_noun_nonce":     output_noun_nonce,
            "_output_prop_samples":   output_prop_samples,
            # Semantic-key → value map for auto-populating item node properties.
            # Maps property key (e.g. "location") → sampled value (e.g. "forge").
            "_output_item_props": {
                key: output_prop_samples[f"{output_category}.{rel_path}"]
                for key, rel_path in output_meta.get("properties", {}).items()
            },
            # Variant-level reserved pool values: what fill_elements pre-sampled globally.
            # _build_used() in generator.py seeds the per-tree `used` registry from this.
            # Keys are full lexicon pool paths; values are sets of already-taken values.
            "_reserved_variant_pools": {
                full_path: {val} for full_path, val in output_prop_samples.items()
            },
            "_n_items":         n_items,
            # Entity pool only — item pool is resolved dynamically via _dim_* above.
            "_pool_map":        {input_category: "entities.all"},
            "_composition":     comp,
            "_input_category":  input_category,
            "_input_meta":      input_meta,   # for generate_entity_repeat
            "_variant_info":    {"seed": seed, "variant_type": "filled"},
        }

    # =========================================================
    # Unstructured path: random item assignment, flat entity list
    # =========================================================
    n_items    = abstract["n_items"]
    n_entities = abstract["n_entities"]

    # Nonce dimension pools for distractor item names live under output_category in nonce_lex.
    try:
        obj_nonce_lex = _resolve_pool(nonce_lex, output_category)
    except (KeyError, TypeError):
        obj_nonce_lex = {}
    nonce_size_pool  = obj_nonce_lex.get("size", [])
    nonce_color_pool = obj_nonce_lex.get("color", nonce_size_pool)

    # Optional shape_labels in output_meta.properties; fallback to numbered labels.
    shape_path = output_meta["properties"].get("shape_labels")
    shape_pool = (
        _resolve_pool(sem_lex, f"{output_category}.{shape_path}")
        if shape_path else [f"shape_{j}" for j in range(n_items)]
    )

    nonce_noun_pool = obj_nonce_lex.get("nouns", [])
    items: List[Dict] = []
    for k in range(n_items):
        if nonce_size_pool:
            nonce_name = (
                f"{nonce_size_pool[(k // max(len(nonce_color_pool), 1)) % len(nonce_size_pool)]}"
                f"_{nonce_color_pool[k % len(nonce_color_pool)]}"
            )
        else:
            nonce_name = f"item_{k}"
        if nonce_noun_pool:
            nonce_noun = _pick(nonce_noun_pool, rng)
            nonce_name = f"{nonce_name}_{nonce_noun}"
        items.append({
            "id":            k,
            "semantic_name": f"{shape_pool[k % len(shape_pool)]}_{output_noun}",
            "nonce_name":    nonce_name,
        })

    item_ids = list(range(n_items))
    rng.shuffle(item_ids)

    sem_names           = _make_names(name_sem_pools, n_entities, rng)
    nonce_names         = _make_nonce_names(name_nonce_pools, n_entities)
    input_prop_samples  = _sample_entity_props(n_entities)
    output_prop_samples = _build_output_prop_samples(input_prop_samples)

    entities: List[Dict] = []
    for k in range(n_entities):
        props = {key: vals[k] for key, vals in input_prop_samples.items()}
        e = _entity(sem_names[k], nonce_names[k], props,
                    {v["name"]: k % v["n_values"] for v in inputs})
        e["item_id"] = item_ids[k % n_items]
        entities.append(e)

    return {
        "n_items":               n_items,
        "items":                 items,
        "attribute_labels":      attribute_labels,
        "entities":              {"all": entities},
        "_pool_map":             {output_category: "items", input_category: "entities.all"},
        "_input_category":       input_category,
        "_input_meta":           input_meta,
        "_output_prop_samples":  output_prop_samples,
        "_output_item_props": {
            key: output_prop_samples[f"{output_category}.{rel_path}"]
            for key, rel_path in output_meta.get("properties", {}).items()
        },
        "_reserved_variant_pools": {
            full_path: {val} for full_path, val in output_prop_samples.items()
        },
        "_variant_info":         {"seed": seed, "variant_type": "distractor_filled"},
    }


# ---------------------------------------------------------------------------
# Entity repeat — new entity names/locations, same attribute structure
# ---------------------------------------------------------------------------

def generate_entity_repeat(
    elements: Dict,
    seed: int,
    sem_lex: Optional[Dict] = None,
    nonce_lex: Optional[Dict] = None,
) -> Dict:
    """
    Return a copy of elements with new instance names and locations for
    source entities only.  Attribute values, items, and gen entities are unchanged.
    """
    if sem_lex is None:
        sem_lex = _load_lexicon("semantic")
    if nonce_lex is None:
        nonce_lex = _load_lexicon("nonce")

    rng = random.Random(seed)
    out = copy.deepcopy(elements)
    src = out["entities"]["source"]
    n   = len(src)

    input_category = elements["_input_category"]
    input_meta     = elements["_input_meta"]

    # Semantic: prefer repeat_name pools if declared (disjoint name set); fall back to name.
    # Nonce: always use name pools — no separate repeat pool; uniqueness comes from offset.
    repeat_sem_pools = input_meta.get("repeat_name", input_meta["name"])
    name_sem_pools   = [_resolve_pool(sem_lex,   f"{input_category}.{p}") for p in repeat_sem_pools]
    name_nonce_pools = [_resolve_pool(nonce_lex, f"{input_category}.{p}") for p in input_meta["name"]]

    new_names = _make_names(name_sem_pools, n, rng)

    loc_pool        = _resolve_pool(sem_lex, f"{input_category}.{input_meta['properties']['location']}")
    unique_old_locs = list({e["location"] for e in src})
    new_locs        = _sample_n(loc_pool, len(unique_old_locs), rng)
    loc_map         = dict(zip(unique_old_locs, new_locs))

    offset = (seed * 7) % len(name_nonce_pools[0])

    for i, entity in enumerate(src):
        entity["instance"]       = new_names[i]
        entity["nonce_instance"] = "_".join(
            pool[(i + offset) % len(pool)] for pool in name_nonce_pools
        )
        entity["location"] = loc_map.get(entity["location"], entity["location"])

    out.setdefault("_variant_info", {})["entity_repeat_seed"] = seed
    return out


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

class _SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return sorted(obj)
        return super().default(obj)


def save_variant(elements: Dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(elements, f, indent=2, cls=_SetEncoder)
    print(f"Saved: {out_path}")

