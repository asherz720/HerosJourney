# Rule File Format

A **rule file** is the only thing you need to write to add a new property-induction task.
It is a JSON file at `tree_management/rules/<name>.json` that specifies the hidden mapping
$f: V_1 \times V_2 \to \mathcal{P}$ in abstract terms (integer indices, no surface names).

Surface names are supplied by a lexicon at fill time; different seeds produce different variants.

---

## Top-level fields

```jsonc
{
  "input_category":  "entity.npc",    // Required. Lexicon root for all input entities.
  "input_meta":      { ... },         // Required. Surface name and property pools for entities.
  "output_category": "object.weapon", // Required. Lexicon root for all output items.
  "output_meta":     { ... },         // Required. Surface name and property pools for items.
  "inputs":     [...],   // Required. Input attribute spaces.
  "outputs":    [...],   // Required. Output property dimensions.
  "functions":  [...],   // Required. Component mapping functions.
  "composition": ...,    // Required. How functions combine.
  "items": ...,          // Required. How combined output maps to item identity.
  "split": { ... }       // Required. How to partition into source / gen entities.
}
```

`input_category` and `output_category` are **lexicon root paths** that drive all
surface-name resolution for entities and items respectively.  All pool paths in
`inputs`, `outputs`, `input_meta`, and `output_meta` are **relative** to their
respective category — the full lexicon path is `{category}.{rel_path}`.

---

## `input_meta` and `output_meta`

These two objects fully describe the surface-name and property pools for the
entity and item "node types" that the task uses.  Nothing in `fill_elements` is
hardcoded — all subkeys are declared here.

```jsonc
"input_meta": {
  "surface": {
    // Pools for constructing entity instance names (joined with '_').
    // Both sem_lex and nonce_lex are queried with the same relative paths.
    "name_first":        "name_pools.first",
    "name_last":         "name_pools.last",
    // Optional: disjoint repeat pools for entity-repeat variants.
    "repeat_name_first": "name_pools.repeat_first",
    "repeat_name_last":  "name_pools.repeat_last"
  },
  "properties": {
    // Each key becomes a top-level field on every entity dict.
    // Value is a relative pool path resolved against input_category in sem_lex.
    "location": "locations"
    // Add more to attach additional per-entity fields, e.g. "faction": "factions"
  }
},

"output_meta": {
  "surface": {
    // "noun" is the item-type word appended to dimension values in item names.
    "noun": "nouns"
  },
  "properties": {
    // Each key becomes a top-level field on every item record.
    // One value is sampled per variant (shared across all items).
    "location": "locations"
    // Add more for additional item properties, e.g. "rarity": "rarities"
  }
}
```

To add a new entity or item type, add the category to the lexicon and update
`input_category` / `output_category` and the `*_meta` pools accordingly.

---

## `inputs`

List of attribute specs. Each attribute is an abstract input dimension.

- `pool`, `nonce_pool`, `nonce_label` are **relative paths** under `input_category`.
  e.g. with `input_category: "entity.npc"`, `"pool": "attributes.class"` resolves to
  `entity.npc.attributes.class` in the lexicon.

```jsonc
"inputs": [
  {
    "name":        "class",
    "n_values":    3,
    "pool":        "attributes.class",
    "nonce_pool":  "attribute_slots.0",
    "nonce_label": "attribute_labels.0"
  },
  {
    "name":        "role",
    "n_values":    3,
    "pool":        "attributes.role",
    "nonce_pool":  "attribute_slots.1",
    "nonce_label": "attribute_labels.1"
  }
]
```

`name` is the key used to reference this attribute in `functions` and `composition`.

---

## `outputs`

List of output property dimensions. `n_values` is optional and derived from the
function maps if absent.

- `pool` and `nonce_pool` are **relative paths** under `output_category`.
  e.g. with `output_category: "object.weapon"`, `"pool": "size"` resolves to
  `object.weapon.size` in the lexicon.
- Item cost and other output properties (e.g. location) are declared in the **process
  file** as property slots on the relevant step, not in the rule file.  See
  `processes/property_flat.json` for an example.

To use a different item type (e.g. tools), add the corresponding category to the
lexicon and set `output_category` accordingly at the top level.

```jsonc
"outputs": [
  {"name": "size",  "n_values": 3, "pool": "size",  "nonce_pool": "size"},
  {"name": "color", "n_values": 3, "pool": "color", "nonce_pool": "color"}
]
```

For `items: "join"`, the order of dimensions here determines:
  - the item_id linearization (last dimension has stride 1)
  - the semantic name order (`{dim0}_{dim1}_..._{object}`)

---

## `functions`

List of component mapping functions. Each maps one input to one output dimension.

```jsonc
"functions": [
  {"name": "f_size",  "input": "class", "output": "size",  "map": [2, 1, 0]},
  {"name": "f_color", "input": "role",  "output": "color", "map": [0, 1, 2]}
]
```

`map[i]` is the output dimension index for attribute value at index `i`.
Values are always integer indices into the sampled dimension surface name list.

---

## `composition`

How component functions combine to produce an item. A bare string for simple types:

| Value | Meaning |
|---|---|
| `"additive"` | All functions map to the same dimension; item = sum of outputs |
| `"independent"` | Each function controls a separate dimension independently |

Structured object for complex types:

```jsonc
// conditional: one attribute selects a regime; regime determines which dimension role controls
"composition": {
  "fn": "conditional",
  "selector": {"input": "class", "map": [0, 0, 1, 1]},
  "branches": [
    {"regime": 0, "active_fn": "f_size",  "fixed": {"color": 2}},
    {"regime": 1, "active_fn": "f_color", "fixed": {"size":  3}}
  ]
}

// override: base rule applies except when one role value triggers a fixed item
"composition": {
  "fn": "override",
  "base_fn": "f_base",
  "override_attr": "role",
  "override_value": 2,
  "override_dim": "size",
  "override_dim_value": 3
}
```

See `tree_management/function_specs/compositions.json` for full parameter documentation.

---

## `items`

How the composition output (a dict of dimension value indices) maps to an item.

| Value | Meaning |
|---|---|
| `"join"` | item_id = row-major index over dimensions; name = `{dim0}_{dim1}_.._{object}` |
| `"index"` | single dimension; item_id = dimension value directly; name = `{dim_val}_{object}` |

The full cross-product of dimension values is pre-generated so item_ids form a dense list.

See `tree_management/function_specs/item_mappings.json` for details.

---

## `split`

Algorithm to partition the full attribute grid into source (demos) and gen (eval) entities.

```jsonc
{"fn": "two_offset", "seed": 0}    // for additive, independent, conditional
{"fn": "c4_override"}              // for override — guarantees override identifiability
{"fn": "explicit",                 // manual specification
 "source": [[0,0],[1,1],...],
 "gen":    [[0,2],[1,0],...]}
```

See `tree_management/function_specs/split_functions.json` for full documentation.

---

## Lexicon contract

The semantic lexicon must have pools at the paths declared in each input's `pool` and
each output's `pool`. The nonce lexicon must have pools at the paths declared in each
input's `nonce_pool` and `nonce_label`. Pool size must be ≥ `n_values`.

Paths are resolved relative to the top-level `input_category` / `output_category`:

| Field | Resolved as |
|---|---|
| Entity instance names | `{input_category}.name_pools.first` / `.last` |
| Entity locations | `{input_category}.locations` |
| Item nouns | `{output_category}.nouns` |
| Item craft locations | `{output_category}.locations` |
| Item dimension values | `{output_category}.{dim_name}` (e.g. `.size`, `.color`) |

---

## Registering a new task type

1. Write `tree_management/rules/my_task.json`
2. Create `tree_management/tasks/my_task.py` and call `register_task_type()`
3. Add one import to `tree_management/tasks/__init__.py`

```python
# tree_management/tasks/my_task.py
from tree_management.generator import register_task_type, load_process
from tree_management.function_specs.splits import validate_additive_split

register_task_type(
    name          = "my_task",
    task_type     = "my_task",
    process       = load_process("property_flat"),
    slot_bindings = {
        "defeat_0": "rule_input",   # entity whose attributes the rule reads
        "buy_0":    "rule_output",  # item determined by the rule (hidden from agent)
    },
    rules         = "tree_management/rules/my_task.json",
    display_name  = "My Task",
    correct_rule  = "The rule is ...",
    mc_answer     = "A",
    description   = "One-line description",
    validate_fn   = validate_additive_split,
)
```
