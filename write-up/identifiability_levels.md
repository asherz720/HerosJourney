# Identifiability Coverage Levels — Experiment Design

Each task has a fixed gen entity (what the model is tested on). We vary the
**source demo set** across four levels while keeping everything else identical
(world listing, rules text, distractor episodes, gen entity). The key
manipulation is how much of the attribute-combination space is *covered* by
the demos — specifically whether the coverage is sufficient to uniquely
determine the rule for the gen entity.

**Level definitions:**

| Level | Coverage | Identifiable? | Description |
|---|---|---|---|
| **A** | Partial, disconnected | ✗ No | Demos exist but cannot uniquely pin down the rule — multiple rules are consistent with all demos |
| **B** | Minimal spanning | ✓ Yes | Smallest demo set from which the rule is uniquely determinable |
| **C** | Current setup | ✓ Yes | Current source split (at or one step above B) |
| **D** | Near-complete | ✓ Yes | All attribute combinations covered except gen entities |

**Context-size control:** total number of demo *episodes* is held constant across
levels by filling any remaining slots with repeated combinations (same
attribute values, different entity name, same item outcome). This isolates
*coverage structure* from *context length*.

---

## Task 1 — Additive

**Rule:** `item_size = base(class) + modifier(role)`
- `base`: C0→2, C1→1, C2→0
- `modifier`: R0→2, R1→1, R2→0
- Items are indexed by size (0 = smallest, 4 = largest)

**Full combination space (3 × 3 = 9):**

| | R0 | R1 | R2 |
|---|---|---|---|
| **C0** | size 4 | size 3 | size 2 |
| **C1** | size 3 | size 2 | size 1 |
| **C2** | size 2 | size 1 | size 0 |

**Gen entities (current split, seed=0):** `(C0,R0)→4`, `(C1,R1)→2`, `(C2,R2)→0`

**Identifiability condition:** the bipartite graph (classes ↔ roles, edge per
observed demo pair) must be *connected*. A spanning tree on 3+3 nodes requires
**≥ 5 edges**. With 4 or fewer demos the graph is always disconnected and
the rule is not uniquely determinable.

---

### Level A — 4 demos, non-identifiable

Demo set: `(C0,R1)`, `(C0,R2)`, `(C1,R0)`, `(C2,R0)`

| Demo | Class | Role | Item (size) |
|---|---|---|---|
| 1 | C0 | R1 | 3 |
| 2 | C0 | R2 | 2 |
| 3 | C1 | R0 | 3 |
| 4 | C2 | R0 | 2 |

**Bipartite graph:**
```
C0 —— R1
C0 —— R2
C1 —— R0
C2 —— R0
```
Two components: `{C0, R1, R2}` and `{C1, C2, R0}`. Disconnected.

**What's determined:** `base(C0)` and `mod(R1)`, `mod(R2)` are jointly
constrained (but not individually anchored). `base(C1)−base(C2)` is known.
**What's undetermined:** absolute values of `base` and `mod` are not pinned —
multiple additive decompositions fit the observed sizes.

**Can model determine gen entities?**
- `(C0,R0)`: needs `base(C0)+mod(R0)` — `mod(R0)` unknown → ✗
- `(C1,R1)`: needs `base(C1)+mod(R1)` — cross-component → ✗
- `(C2,R2)`: needs `base(C2)+mod(R2)` — `mod(R2)` known relative but `base(C2)` not anchored → ✗

---

### Level B — 5 demos, minimally identifiable

Demo set: `(C0,R1)`, `(C0,R2)`, `(C1,R0)`, `(C1,R2)`, `(C2,R0)`

| Demo | Class | Role | Item (size) |
|---|---|---|---|
| 1 | C0 | R1 | 3 |
| 2 | C0 | R2 | 2 |
| 3 | C1 | R0 | 3 |
| 4 | C1 | R2 | 1 |
| 5 | C2 | R0 | 2 |

**Bipartite graph:**
```
C0 —— R1
C0 —— R2
C1 —— R0
C1 —— R2   ← bridge connecting the two components
C2 —— R0
```
Connected spanning tree (5 edges on 6 nodes). ✓

**What's determined:** all `base` and `mod` values uniquely solvable:
`mod(R2)=0, mod(R1)=1, mod(R0)=2; base(C0)=2, base(C1)=1, base(C2)=0`

**Can model determine gen entities?** ✓ All three: 4, 2, 0.

---

### Level C — 6 demos (current split)

Demo set: `(C0,R1)`, `(C0,R2)`, `(C1,R0)`, `(C1,R2)`, `(C2,R0)`, `(C2,R1)`

Adds `(C2,R1)→1` to Level B. C2 now appears twice; confirms `base(C2)=0` from
two independent equations.

| Demo | Class | Role | Item (size) |
|---|---|---|---|
| 1 | C0 | R1 | 3 |
| 2 | C0 | R2 | 2 |
| 3 | C1 | R0 | 3 |
| 4 | C1 | R2 | 1 |
| 5 | C2 | R0 | 2 |
| 6 | C2 | R1 | 1 |

**Gen entities:** ✓ all determinable. Each `base` and `mod` now has two
constraints — overdetermined, no ambiguity.

---

### Level D — 8 demos, near-complete

All 9 combinations except gen entities `{(C0,R0), (C1,R1), (C2,R2)}`.

| Demo | Class | Role | Item (size) |
|---|---|---|---|
| 1 | C0 | R1 | 3 |
| 2 | C0 | R2 | 2 |
| 3 | C1 | R0 | 3 |
| 4 | C1 | R2 | 1 |
| 5 | C2 | R0 | 2 |
| 6 | C2 | R1 | 1 |
| 7 | C0 | R2 | 2 *(repeat)* |
| 8 | C1 | R0 | 3 *(repeat)* |

> Note: only 6 distinct combos available without hitting gen entities, so
> slots 7–8 are repeats (context-size padding). Alternatively, run with
> n=6 demos for Level D and treat it as "distinct-maximised."

---

## Task 2 — Compositional

**Rule:** `class → size` AND `role → color` (independently)
- `size`: C0→2, C1→1, C2→0
- `color`: R0→color_0, R1→color_1, R2→color_2
- Items are (size, color) pairs

**Full combination space (3 × 3 = 9):**

| | R0 | R1 | R2 |
|---|---|---|---|
| **C0** | (2, c0) | (2, c1) | (2, c2) |
| **C1** | (1, c0) | (1, c1) | (1, c2) |
| **C2** | (0, c0) | (0, c1) | (0, c2) |

**Gen entities (current split, seed=0):** `(C0,R0)→(2,c0)`, `(C1,R1)→(1,c1)`, `(C2,R2)→(0,c2)`

**Identifiability condition:** because the two dimensions are independent,
identifiability is simpler than additive — you just need each *class* to
appear in at least one demo (to fix its size) **and** each *role* to appear
in at least one demo (to fix its color). For gen entity `(Ci, Rj)`: need at
least one demo with class Ci AND at least one demo with role Rj.

The threshold is much lower here: **3 demos** suffice (one per class, one per
role, achievable simultaneously).

---

### Level A — 3 demos, non-identifiable

Demo set covers only 2 of 3 classes and 2 of 3 roles — gen entity's class
or role is absent.

| Demo | Class | Role | Item |
|---|---|---|---|
| 1 | C1 | R1 | (1, c1) |
| 2 | C1 | R2 | (1, c2) |
| 3 | C2 | R1 | (0, c1) |

**What's determined:** `size(C1)=1`, `size(C2)=0`; `color(R1)=c1`, `color(R2)=c2`

**What's missing:** C0 never seen → `size(C0)=?`; R0 never seen → `color(R0)=?`

**Gen entities:**
- `(C0,R0)`: both unknown → ✗
- `(C1,R1)`: both known → ✓ *(this gen entity happens to be identifiable even at Level A)*
- `(C2,R2)`: size(C2) known, color(R2) known → ✓

> Key observation: Level A is non-identifiable for only the gen entity whose
> class *and* role are both absent. The other two gen entities may already be
> identifiable. This is a property of compositional tasks — identifiability
> is per-attribute, not global.

---

### Level B — 3 demos, minimally identifiable (all gen entities)

All 3 classes and all 3 roles covered.

| Demo | Class | Role | Item |
|---|---|---|---|
| 1 | C0 | R1 | (2, c1) |
| 2 | C1 | R2 | (1, c2) |
| 3 | C2 | R0 | (0, c0) |

**What's determined:** `size(C0)=2`, `size(C1)=1`, `size(C2)=0`; `color(R0)=c0`, `color(R1)=c1`, `color(R2)=c2`

**Gen entities:** ✓ all three determinable.

---

### Level C — 5 demos (one step above)

| Demo | Class | Role | Item |
|---|---|---|---|
| 1 | C0 | R1 | (2, c1) |
| 2 | C0 | R2 | (2, c2) |
| 3 | C1 | R0 | (1, c0) |
| 4 | C1 | R2 | (1, c2) |
| 5 | C2 | R0 | (0, c0) |

Each class and role appears 1–2 times. Rule well-supported.

---

### Level D — 6 demos, near-complete

All 9 combos minus the 3 gen entities = 6 demos. Every class appears twice,
every role appears twice. Rule structure fully saturated.

---

## Task 3 — Conditional

**Rule:** class selects a regime; the regime determines which dimension role
controls.
- **Regime 0** (classes C0, C1): role → size (color fixed = c2)
  - R0→size 0, R1→size 1, R2→size 2
- **Regime 1** (classes C2, C3): role → color (size fixed = 3)
  - R0→color 2, R1→color 1, R2→color 0

**Full combination space (4 × 3 = 12):**

| | R0 | R1 | R2 |
|---|---|---|---|
| **C0** (regime 0) | (0, c2) | (1, c2) | (2, c2) |
| **C1** (regime 0) | (0, c2) | (1, c2) | (2, c2) |
| **C2** (regime 1) | (3, c2) | (3, c1) | (3, c0) |
| **C3** (regime 1) | (3, c2) | (3, c1) | (3, c0) |

**Gen entities (current split, seed=0):** `(C0,R0)`, `(C1,R1)`, `(C2,R1)`, `(C3,R0)`

**Identifiability conditions:**
1. At least one class from **each regime** must appear in demos (to reveal two-regime structure)
2. Within each regime, **≥ 2 distinct role values** must appear (to reveal which dimension varies)
3. The gen entity's role must appear in its regime's demos (to fix the specific mapping)

Minimum identifiable: **4 demos** (2 per regime, different roles each).

---

### Level A — 4 demos, non-identifiable (one regime only)

| Demo | Class | Role | Item |
|---|---|---|---|
| 1 | C0 | R1 | (1, c2) |
| 2 | C0 | R2 | (2, c2) |
| 3 | C1 | R0 | (0, c2) |
| 4 | C1 | R2 | (2, c2) |

Only regime 0 demos. Model sees: role controls size, color is fixed at c2.

**What's missing:** no regime 1 demo → model cannot discover that regime 1
exists or that a different mapping applies. For gen entities in regime 1
(`(C2,R1)`, `(C3,R0)`): model would likely predict regime 0 behavior → ✗

---

### Level A' — 4 demos, non-identifiable (both regimes, insufficient role variation)

| Demo | Class | Role | Item |
|---|---|---|---|
| 1 | C0 | R1 | (1, c2) |
| 2 | C0 | R2 | (2, c2) |
| 3 | C2 | R0 | (3, c2) |
| 4 | C3 | R2 | (3, c0) |

Both regimes present, but regime 1 shows only 2 role values. Crucially:
- `(C2,R0)→(3,c2)` and `(C3,R2)→(3,c0)`: color varies, size fixed — regime 1 structure visible
- But gen `(C2,R1)→(3,c1)` and `(C3,R0)→(3,c2)` — role R1 not seen in regime 1 for any class

For `(C3,R0)`: R0 appears in regime 1 via C2 → `color(R0)=c2` determinable. ✓
For `(C2,R1)`: R1 not seen in regime 1 → `color(R1)=?` → ✗

> Level A' distinguishes: having both regimes visible is not enough —
> you need role coverage within each regime too.

---

### Level B — 4 demos, minimally identifiable

| Demo | Class | Role | Item |
|---|---|---|---|
| 1 | C0 | R1 | (1, c2) |
| 2 | C0 | R2 | (2, c2) |
| 3 | C2 | R0 | (3, c2) |
| 4 | C2 | R1 | (3, c1) |

**What this reveals:**
- Regime 0 (from C0): role changes size, color is constant c2 → role controls size
- Regime 1 (from C2): role changes color, size is constant 3 → role controls color
- Gen `(C2,R1)→(3,c1)`: wait — `(C2,R1)` is in the demo! That's a source entity, not gen.

Let me reclarify: gen entities are `(C0,R0)`, `(C1,R1)`, `(C2,R1)`, `(C3,R0)`.

Revised Level B (ensuring no gen entity appears in demos):

| Demo | Class | Role | Item |
|---|---|---|---|
| 1 | C0 | R1 | (1, c2) |
| 2 | C0 | R2 | (2, c2) |
| 3 | C2 | R0 | (3, c2) |
| 4 | C2 | R2 | (3, c0) |

- Regime 0: R1→size1, R2→size2, color fixed=c2. Gen (C0,R0): need R0 in regime 0.
  - size(R0) not seen in regime 0 → (C0,R0) not fully determined ✗
  - But gen (C1,R1): C1 is regime 0, R1 seen → size(R1)=1, color=c2 → (1,c2) ✓

Minimum to determine ALL gen entities requires all role values in each regime to appear. With 4 classes and gen entities spread across both regimes and roles, the minimum rises to 5–6 demos in practice.

---

### Level C — 8 demos (current split)

Source from `conditional_two_offset`: `(C0,R1)`, `(C0,R2)`, `(C1,R0)`, `(C1,R2)`, `(C2,R0)`, `(C2,R2)`, `(C3,R1)`, `(C3,R2)`

| Demo | Class | Regime | Role | Item |
|---|---|---|---|---|
| 1 | C0 | 0 | R1 | (1, c2) |
| 2 | C0 | 0 | R2 | (2, c2) |
| 3 | C1 | 0 | R0 | (0, c2) |
| 4 | C1 | 0 | R2 | (2, c2) |
| 5 | C2 | 1 | R0 | (3, c2) |
| 6 | C2 | 1 | R2 | (3, c0) |
| 7 | C3 | 1 | R1 | (3, c1) |
| 8 | C3 | 1 | R2 | (3, c0) |

All roles (R0, R1, R2) covered within each regime. Gen entities fully determined.

---

### Level D — 10 demos, near-complete

All 12 combinations minus 4 gen entities = 8 distinct combos + 2 repeats for
context padding.

---

## Task 4 — Override

**Rule:** base rule `class → size`; role R2 *always* overrides to size 3
regardless of class.
- `base`: C0→0, C1→1, C2→2
- Override: any entity with R2 → size 3

**Full combination space (3 × 3 = 9):**

| | R0 | R1 | R2 (override) |
|---|---|---|---|
| **C0** | 0 | 0 | **3** |
| **C1** | 1 | 1 | **3** |
| **C2** | 2 | 2 | **3** |

**Gen entities (c4\_override, seed=0):** `(C1,R0)→1` (base rule for held-out
class), `(C1,R2)→3` (override for held-out class)

**Identifiability conditions:**
- For gen `(C1,R0)→1`: need `base(C1)=1`. Visible from any `(C1,Rx)` where `Rx≠R2`.
  Already shown via `(C1,R1)→1` in source. ✓ trivially identifiable.
- For gen `(C1,R2)→3` (override): need to see that R2 *always* maps to 3
  regardless of class. This requires at least **2 override demos with different
  classes** so the model can distinguish "C0 happens to need size 3 when role=R2"
  from "R2 always gives size 3".

Source (c4\_override, seed=0): `(C0,R0)`, `(C0,R1)`, `(C0,R2)`, `(C1,R1)`, `(C2,R0)`, `(C2,R1)`, `(C2,R2)`

---

### Level A — 5 demos, non-identifiable (override role R2 absent)

| Demo | Class | Role | Item (size) |
|---|---|---|---|
| 1 | C0 | R0 | 0 |
| 2 | C0 | R1 | 0 |
| 3 | C1 | R1 | 1 |
| 4 | C2 | R0 | 2 |
| 5 | C2 | R1 | 2 |

No R2 demos. Model learns base rule: C0→0, C1→1, C2→2.

**Gen `(C1,R2)`:** model predicts size 1 (applies base rule for C1). Correct
answer is 3 (override). **Model is wrong.** ✗

---

### Level B — 6 demos, partially informative (override seen once)

Add `(C0,R2)→3` to Level A:

| Demo | Class | Role | Item (size) |
|---|---|---|---|
| 1 | C0 | R0 | 0 |
| 2 | C0 | R1 | 0 |
| 3 | C0 | R2 | **3** |
| 4 | C1 | R1 | 1 |
| 5 | C2 | R0 | 2 |
| 6 | C2 | R1 | 2 |

Model sees: C0+R2→3 while C0+R0→0 and C0+R1→0. R2 is anomalous for C0.

**Ambiguity:** is this "C0 specifically needs size 3 with R2" or "R2 always
gives size 3"? With only one override demo, the agent **cannot distinguish**
these two hypotheses from the demonstrations alone. ✗ (partially informative
but not uniquely identifiable)

---

### Level C — 7 demos, identifiable (override seen for 2 classes)

Add `(C2,R2)→3` to Level B:

| Demo | Class | Role | Item (size) |
|---|---|---|---|
| 1 | C0 | R0 | 0 |
| 2 | C0 | R1 | 0 |
| 3 | C0 | R2 | **3** |
| 4 | C1 | R1 | 1 |
| 5 | C2 | R0 | 2 |
| 6 | C2 | R1 | 2 |
| 7 | C2 | R2 | **3** |

Now C0+R2→3 AND C2+R2→3 with C0 and C2 having very different base rules
(0 vs 2). The only consistent explanation: **R2 always gives size 3**.

**Gen `(C1,R2)`:** override → 3. ✓ Gen `(C1,R0)`: base(C1)=1 ✓

---

### Level D — Current split (7 demos, same as Level C)

The `c4_override` split already provides 7 source demos including two override
demos `(C0,R2)` and `(C2,R2)`. **Level C and D coincide for override** because
the design already provides the minimum override evidence without redundancy.

Level D can be extended to 8 demos by adding `(C0,R0)` repeat or a third
non-gen override class... but since only C0 and C2 are non-held-out classes,
and both already appear with R2, there is no new information to add.

---

## Summary: Levels per task

| Task | Grid | Gen entity/ies | Level A | Level B | Level C | Level D |
|---|---|---|---|---|---|---|
| **Additive** | 3×3 | (C0,R0), (C1,R1), (C2,R2) | 4 demos, disconnected | 5 demos, spanning tree | 6 demos (current) | 8 demos (repeat padding) |
| **Compositional** | 3×3 | (C0,R0), (C1,R1), (C2,R2) | 3 demos, missing gen's class+role | 3 demos, all classes+roles covered | 5 demos | 6 demos (all non-gen) |
| **Conditional** | 4×3 | (C0,R0), (C1,R1), (C2,R1), (C3,R0) | 4 demos, one regime | 4 demos, both regimes + role variation | 8 demos (current) | 10 demos |
| **Override** | 3×3 | (C1,R0), (C1,R2) | 5 demos, no override demos | 6 demos, override once (ambiguous) | 7 demos (current), override twice | — (same as C) |

---

## Procedural tasks (P-Add, P-Comp, P-Cond, P-Over)

These share the same attribute→process structure. Grid sizes are **2×2**
(P-Add, P-Cond) or **2×4** (P-Comp, P-Over with action×ordering), giving
only 3 source combinations after leave-one-out. The identifiability sweep
collapses to:

- **Level A**: 1–2 demos (one attribute value missing)
- **Level B**: 3 demos (current, minimally identifiable)
- **Level C/D**: not achievable without repeats

The coverage range is too narrow for a clean multi-level curve. **Recommend
keeping P-tasks at their current (Level B) setup** and running the sweep only
on the 4 property tasks.

---

## Implementation notes

1. **New split parameter:** add `coverage_level ∈ {A, B, C, D}` to the task
   registration / run config. The level selects which subset of source entities
   to include as demos.

2. **Context-size control:** if running Level B (fewer distinct combos) with
   the same episode count as Level C, pad by repeating existing source combos.
   The padded episodes are identical to the original — same entity name,
   same item, same trace — so they add no new information.

3. **Distractor demos:** kept fixed across all levels. Only the
   *property-bearing* source demos change.

4. **Gen entity is fixed:** the same gen entity is evaluated at all levels.
   Only what the model *sees as demos* changes.

5. **Models to run:** GPT-5 (semantic + nonce) + Qwen 27B (semantic + nonce)
   — gives a 2 (model) × 2 (surface) × 4 (level) design per task.
