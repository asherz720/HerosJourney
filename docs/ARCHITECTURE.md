# Adventure Story Generalization Benchmark

## What this project is

A benchmark for testing whether language models can **induce hidden rules from demonstrations** in a text-based adventure game environment.

The agent plays an RPG-style game. It sees a set of rules for the current task but some rules are deliberately hidden. It must infer the missing requirements by studying **demonstration episodes** (in-context examples from the source split) and then apply the inferred pattern to a novel entity (the gen split).

The key research question: can models generalize a learned property to unseen combinations of attributes — and how does this vary across property types and surface representations (semantic vs nonce names)?

---

## Task taxonomy

### Property induction tasks — the main focus right now

All property tasks share the same flat process structure:
`go item_location → buy item → go entity_location → defeat entity`

The hidden requirement is **which item to buy**, determined by the entity's attributes.

| Task | Property | Description |
|------|----------|-------------|
| additive | Additive | `item_size = base(class) + modifier(role)` |
| compositional | Compositional | `class → size, role → color` (independent dimensions) |
| conditional | Conditional | class selects a regime; regime determines which dimension role controls |
| override | Override | base rule `class → item`; one role value always overrides to a fixed item |

### Procedural tasks (P-class) — scaffolded, not the current focus

P_SEQ_1, P_SEQ_2, P_NEST_2 — multi-step tasks with ordering constraints and hidden procedural steps. Elements files exist but evaluation pipeline is not yet wired up.

---

## Repository structure

The framework is a single installable package, `herosjourney/`. Paper-specific
experiment code lives under `experiments/` (not installed). Install with `pip install -e .`
(add `[runner]` for model providers, `[analysis]` for figures, `[yaml]` for YAML task files).

```
herosjourney/                # The installable package (single namespace)
  __init__.py                # Public API: get_task, register_task, AdventureEnv,
                             #   EpisodeResult, compute_ecsr, load_task_file
                             #   (builtins are eager-loaded into TASK_REGISTRY at import)
  world_info/
    lexicons/
      semantic_lexicon.json            # Surface name pools (attribute values, entities, locations)
      nonce_lexicon.json               # Nonce syllable pools
      distractor_semantic_lexicon.json
      distractor_nonce_lexicon.json
    actions.py                         # RPG action definitions (ActionDef + ACTION_REGISTRY)
  core/                      # (was tree_management/) task generation
    rules/                   # Abstract rule files (functions + composition, no surface names)
      additive.json, compositional.json, conditional.json, override.json,
      proc_add.json, proc_comp.json, proc_cond.json, proc_over.json
      RULE_FORMAT.md         # Rule file format reference
    processes/
      property_flat.json     # Flat go→buy→go→defeat process (used by all current tasks)
    distractors/
      canonical_property_distractor.json
    function_specs/          # Built-in mapping functions (compositions, item_mappings, splits)
    tasks/                   # One <name>.json + 2-line <name>.py wrapper per task type
                             #   .json = declarative spec; .py = load_task_file(...)
      __init__.py            # Imports all task modules to trigger registration
    elements.py              # fill_elements(rule, lexicon, seed, split_spec) → elements dict
    goal_tree.py             # GoalTree data structure
    generator.py             # build tasks; register_task(), load_process(), validate_process()
    registry.py              # TASK_REGISTRY + PropertySpec/ProcessSpec/TaskSpec dataclasses
    task_file.py             # load_task_file(): declarative JSON/YAML task registration
    demo_generator.py        # Demo dataclass + generate_demos / generate_mixed_demos
  env/
    env.py                   # AdventureEnv: stateful episode environment
  eval/
    result.py                # EpisodeResult dataclass (to_dict / from_dict)
    metrics.py               # compute_ecsr, compute_norm_eff, success_rate (ECSR)
    judge.py                 # QA / RV-judge prompts
  runner/                    # (was pipeline/) model-agnostic episode loop
    prompts.py               # Game-loop prompts
    strategies.py            # ReAct / HR / IDEA / ACE steering prompts
    models.py                # Provider adapters shim → experiments/models.py
    adventure_episode.py     # run_single_episode() → EpisodeResult; adventure_run() core loop
    adventure_pipeline.py    # run_two_phase_pipeline, run_qa_pipeline, run_episodes_batch
    run_adventure_story.py   # CLI entry point (console script: `adventure-story`)
    teacher.py, qa_episode.py

experiments/                 # NOT installed — paper-specific (gitignored outputs)
  models.py                  # Real provider adapters (vLLM / Gemini / Claude / GPT)
  analysis/                  # viz_results.py, visualization.py (ECSR/RV figures)
  human_study/               # Human experiment server + annotation tools
  scripts/                   # experiments.json, run_experiments.sh, ...

tools/
  process_editor.html        # Visual web-based process editor (open in any browser)
```

---

## Task generation pipeline

Task generation happens in three steps, each corresponding to a distinct abstraction:

### Step 1 — Fill surface names (`elements.py`)

A rule file contains only structural information (attribute counts, function maps, composition type). `fill_elements()` samples concrete names from a lexicon and applies the source/gen split:

```python
import json
from herosjourney import get_task
from herosjourney.core.elements import fill_elements, load_lexicons

spec = get_task("additive")
sem_lex, nonce_lex = load_lexicons()
with open(spec.rules) as f:
    rule = json.load(f)
elements = fill_elements(rule, sem_lex, nonce_lex, seed=0, split_spec=spec.split)
# Different seed → different surface names, same structure.
# split_spec is required (comes from the task spec); fill_elements does not
# read it from the rule file.
```

### Step 2 — Build tasks from process + elements (`generator.py`)

`build_tree_from_process()` resolves each step in the process JSON against the filled elements dict to produce a `GoalTree`. One elements dict produces one tree per entity in the relevant split.

```python
tasks = spec.gen_fn(elements, split="source", use_nonce=False)
# tasks: List[GeneratedTask]
```

### Step 3 — Demo generation (`demo_generator.py`)

`generate_demos()` runs the environment against each task to produce a demo trace, then `construct_demo_context()` formats it as an in-context example. `build_world_listing()` generates the `=== World ===` block shown at the start of each context.

---

## Process JSON schema (v2)

Process files live in `tree_management/processes/`. The schema is fully declarative — no Python build logic needed.

```json
{
  "_schema": "process_v2",
  "_note": "Human-readable description",
  "type": "sequence",
  "root": "defeat_0",
  "ordering_constraints": [],
  "steps": [
    {
      "id": "go_item",
      "action": "go",
      "execution_order": 0,
      "parent": "buy_0",
      "argument": {"from": "buy_0.location"}
    },
    {
      "id": "buy_0",
      "action": "buy",
      "execution_order": 1,
      "parent": "defeat_0",
      "is_hidden": true,
      "argument": {"pool": "object.weapon"},
      "properties": [
        {"key": "location", "from": "argument.location"},
        {"key": "cost",     "from": "argument.cost"}
      ]
    },
    {
      "id": "go_entity",
      "action": "go",
      "execution_order": 2,
      "parent": "defeat_0",
      "argument": {"from": "defeat_0.location"}
    },
    {
      "id": "defeat_0",
      "action": "defeat",
      "argument": {"pool": "entity"},
      "properties": [
        {"key": "location",         "from": "argument.location"},
        {"key": "attribute_names",  "from": "argument.attribute_names"},
        {"key": "attribute_values", "from": "argument.attribute_values"}
      ]
    }
  ]
}
```

### Key step fields

| Field | Description |
|-------|-------------|
| `id` | Unique step identifier, used by other steps as a `"from"` reference target |
| `action` | Must be a key in `ACTION_REGISTRY` (go, get, buy, perform, defeat, rescue) |
| `parent` | ID of the step this feeds into (absent on root) |
| `execution_order` | Integer; lower = earlier in demo trace. Steps with the same value are parallel. Defaults to list position if omitted |
| `is_hidden` | If true, this step's argument is omitted from the agent's rules text |
| `argument` | Argument resolution spec — see below |
| `properties` | List of `{key, from}` pairs storing extra data (location, cost, etc.) on the tree node |

### Argument resolution (`argument` field)

Two modes, distinguished by which key is present:

| Mode | Example | Resolver behaviour |
|------|---------|-------------------|
| **from-reference** | `{"from": "buy_0.location"}` | Copies a field from a previously-resolved step |
| **pool** | `{"pool": "object.weapon"}` | Resolves via `slot_bindings` first, then samples from the pool |

`pool` values are **lexicon paths** (e.g. `"object.weapon"`, `"entity"`) — the same convention as `pool` in rule files. `fill_elements()` adds a `_pool_map` to the elements dict that maps each lexicon path to its elements key (`"object.weapon"` → `"items"`, `"entity"` → `"entities.all"`).

### Rule binding — `slot_bindings` (in task spec, not process JSON)

The process JSON describes pure structure. Which steps carry rule semantics is declared in the task registration via `slot_bindings`:

- `"rule_input"` — always resolves to the current task entity; `pool` is ignored
- `"rule_output"` — resolves to the item at `entity["item_id"]` in the pool (rule-determined)
- absent — random sampling from the pool (distractor path)

| Field | Controls |
|-------|---------|
| `pool` | Lexicon path identifying which pool to sample from; overridden by `slot_bindings` for `rule_input` |
| `is_hidden` | Whether the argument appears in the agent's rules text |

### `ordering_constraints` — temporal ordering

**Ordering constraints are auto-derived from `execution_order` differences among sibling steps** (same parent, different `execution_order` values). Steps with the same `execution_order` are treated as parallel — no constraint generated.

The `ordering_constraints` field in the process JSON is an escape hatch for cases that can't be expressed via `execution_order` alone (e.g., ordering across different parents). In practice you rarely need it.

```json
"ordering_constraints": [
  {"before": "perform_ritual", "after": "buy_0"}
]
```

Constraints are stored on the `GoalTree` and validated by the environment **at terminal goal time** (defeat/rescue), not eagerly. This means an agent can attempt steps in the wrong order and only be rejected at the end — the right failure signal for generalization testing.

Step IDs are resolved to `(action, argument)` pairs at build time. The env tracks a timestamp (`_step_counter`) for each completed action and checks all constraints when the terminal goal is reached.

---

## Registry design

### Three-level structure

```
TaskSpec
├── property_spec: PropertySpec   ← what is being tested (rule file + metadata)
└── processes: List[ProcessSpec]  ← how the task is structured (process + mapping nodes)
    process_selector: Callable    ← picks which ProcessSpec to use per entity/seed
```

**`PropertySpec`** — the rule being tested:
- `task_type`, `rules` (path to rule file), `validate_fn`, `correct_rule`, `description`

**`ProcessSpec`** — one process structure:
- `name` (matches a registered `TemplateSpec`), `mapping_nodes` (step IDs that carry the hidden property)

**`TaskSpec`** — bundles everything; convenience properties delegate to `property_spec` (`task_type`, `correct_rule`, `rules`, `description`).

Supporting multiple `ProcessSpec`s on one `TaskSpec` allows future procedural tasks where the process structure itself varies per episode, not just the node values.

---

## Adding a new task type

### Recommended — declarative file (no Python needed)

Write two files and call `register_task` with the spec path:

1. A **rule file** `my_task_rule.json` (see `RULE_FORMAT.md`) — the abstract
   attribute → item/process mapping.

2. A **task spec** `my_task.json`:

```json
{
  "name": "my_task",
  "rules": "my_task_rule.json",
  "process": "property_flat",
  "split": {"fn": "two_offset", "seed": 0},
  "max_tries": 5,
  "validate_fn": "validate_additive_split",
  "eval": {"correct_rule": "The rule is ...", "description": "One-line description"}
}
```

3. Register it:

```python
import herosjourney
herosjourney.register_task("path/to/my_task.json")   # .yaml also works (needs [yaml] extra)
```

`rules` may be absolute or relative to the task file. `validate_fn` accepts the
built-in validator names (`validate_additive_split`, `validate_independent_split`,
`validate_conditional_split`) or `null`. The spec is schema-checked at load time.

To ship a task as a **built-in**, drop `my_task.json` in `herosjourney/core/tasks/`,
add a 2-line `my_task.py` next to it:

```python
from pathlib import Path
from herosjourney.core.task_file import load_task_file
load_task_file(Path(__file__).with_suffix(".json"))
```

and add `from herosjourney.core.tasks import my_task  # noqa: F401` to
`herosjourney/core/tasks/__init__.py`.

### Programmatic — keyword arguments (custom validators / slot_bindings)

```python
from herosjourney import register_task
from herosjourney.core.function_specs.splits import validate_additive_split

register_task(
    name        = "my_task",
    rules       = "/abs/path/to/my_task_rule.json",
    process     = "property_flat",
    split       = {"fn": "two_offset", "seed": 0},
    eval        = {"correct_rule": "The rule is ...", "description": "One-line"},
    validate_fn = validate_additive_split,
    max_tries   = 5,
)
```

For a non-flat structure, create a new process JSON in `herosjourney/core/processes/`
(validated by `validate_process()` on load). Use `tools/process_editor.html` to build it.

---

## Process editor (`tools/process_editor.html`)

A self-contained single-file web app for visually creating and editing process JSON files. Open directly in any browser — no build step, no server.

**Features:**
- Draggable node canvas; scroll to zoom, drag background to pan
- Add steps via topbar buttons (+ go, + get, + buy, + perform, + drink, + defeat, + rescue)
- Dependency arrows (solid, colored by action) flow from child → parent
- Ordering constraint arrows (dashed orange) show temporal constraints
- Left sidebar: process config (note, type, root, ordering constraints) + per-node editor
  (id, action, execution order, parent, is_hidden, argument, properties). Argument is either
  a **from**-reference (e.g. `buy_0.location`) or a **pool** lexicon path (e.g. `object.weapon`,
  `entity.npc`). Properties are either a `from`-reference or a literal `value` (e.g. `cost = 30`).
- Autocomplete suggestions for `from` references (populated from step IDs + their available fields)
- Import JSON / Export JSON / Copy / Download
- Load Example loads the `property_flat.json` process as a starting point
- Auto Layout button arranges nodes in tree order (root at bottom, dependencies above)

---

## Actions (`world_info/actions.py`)

All available actions are defined in `ACTION_REGISTRY` as `ActionDef` entries.

| Action | `generates_rule` | `is_acquisition` | Notes |
|--------|-----------------|-----------------|-------|
| `go` | False | False | Navigation; shown as "must be at {loc}" in parent rule |
| `get` | False | True | Acquire item for free |
| `buy` | False | True | Acquire item for cost |
| `perform` | False | False | Ritual / process step |
| `defeat` | True | False | Terminal goal; triggers ordering constraint check |
| `rescue` | True | False | Terminal goal; triggers ordering constraint check |

`is_acquisition=True` means `demo_generator.build_world_listing()` includes these items in the world listing block.

To add a new action: add an `ActionDef` entry to `ACTION_REGISTRY` and add a `_handle_<name>()` method to `AdventureEnv`.

---

## Key design decisions

### Source / gen split
- **Source split**: entities whose rules are fully shown in demos → used as in-context demonstrations
- **Gen split**: novel entities with hidden item requirement → what we actually evaluate
- Each gen task carries `demo_entity_names` pointing to its source entities
- `variant_id` = elements filename stem, for future multi-variant scaling

### Distractor episodes
Distractors are entities with arbitrary item assignments and **no property pattern**. They use the same attribute labels (`class`/`role`) as property entities but with out-of-vocabulary values (`goblin`, `vampire`, etc.), making them structurally indistinguishable from property demos. This forces the agent to identify the relevant pattern rather than just using all demos.

Distractors currently only work for property tasks (flat process structure). P-task distractors would need a procedural template — the `template` parameter on `generate_distractor_tasks` is the extension point.

### RPG world listing
Every demo context starts with a `=== World ===` block listing all entities (property + distractor), their attributes, and locations. This gives spatial knowledge upfront without revealing causal rules. Items acquired by `is_acquisition=True` actions (buy, get) are also listed here.

### Nonce names
Setting `use_nonce=True` replaces all semantic names (entity names, attribute values, item names) with nonsense syllables (e.g., `vrel_A`, `korv_Z`). Tests whether models rely on world knowledge vs pure in-context pattern induction.

### Episode budget
`max_runs = ref_length × num_tries` (default `num_tries=2`).

This gives every task type the same budget in units of "number of full attempts", making success rates comparable across all four property tasks. `num_items` is stored in task metadata for post-hoc analysis but is not used for the budget calculation. `efficiency = num_runs / ref_length` tells you how many tries the model needed.

---

## Two-phase evaluation pipeline

**Phase 1 — demo only**: gen tasks are run with source demos as in-context examples, no teaching message. Baseline: can the model induce the property from demonstrations alone?

**Phase 2 — demo + teaching**: same gen tasks, same demos, plus an explicit teaching message. Measures how much a hint about the hidden pattern helps.

The `teacher.py` module (inherited from prior work) is retained for Phase 2 teaching message curation. Dynamic strategies (reflexion, ACE, observe-then-teach) generate teaching messages from Phase 1 failure traces.

### Running an experiment

```bash
# Start vLLM server first (example for Qwen 27B on GPUs 2,3):
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
    --model /path/to/qwen27b --tensor-parallel-size 2 --port 8001

# Then run (from repo root):
./experiments/scripts/run_experiments.sh all_qwen27b           # additive only (as configured)
./experiments/scripts/run_experiments.sh all_nonce_qwen27b     # sweep all four tasks, nonce
./experiments/scripts/run_experiments.sh all_nonce_qwen27b --only conditional

# Or directly (console script, or python -m):
adventure-story \
    --task_type additive \
    --elements herosjourney/core/rules/additive.json \
    --model /path/to/qwen27b \
    --num_tries 2 \
    --num_workers 4
# equivalently: python -m herosjourney.runner.run_adventure_story ...
```

Results are saved to `results/phase_1_<save_name>.json`.

---

## Concepts to keep in mind

- **Rule file is abstract**: surface names (attribute values, dimension values, item names) are never stored in rule files. `fill_elements(rule, sem_lex, nonce_lex, seed=N)` produces a fully-named variant. Different seeds = different surface names, same structure.
- **Coverage graph connectivity**: for additive/compositional, the bipartite graph (attr1 values ↔ attr2 values, edges = observed source pairs) must be connected for property identifiability. ≥2 per value is necessary; connectivity is the true sufficient condition (checked by `_check_bipartite_connected` in `generator.py`).
- **`rules_to_skip`**: the list of argument strings hidden from the agent's rules text. Built from all steps with `is_hidden: true`. Stored on each `GeneratedTask`; passed to `env.reset()`.
- **`slot_bindings` vs `is_hidden`**: these are independent. `slot_bindings` (in the task spec, not the process file) says *what rule role* a step plays — `rule_input` or `rule_output` — which controls how the argument is resolved. `is_hidden` (on the process step) says *whether* to show the argument to the agent in the rules text. Both can be set independently.
- **Ordering constraints are lazy**: violations are only detected at terminal goal time (defeat/rescue), not when individual prerequisite actions are taken. This is intentional — the agent can attempt steps in the wrong order and only learn the constraint failed at the end.
- **`execution_order`**: controls demo trace ordering (lower = earlier). Same value = parallel (get_solution() treats ties as concurrent). Does not enforce runtime ordering — use `ordering_constraints` for that.
- **`num_items` in metadata**: stored for analysis (e.g., does larger item pool correlate with lower success?), not used in the budget calculation.
