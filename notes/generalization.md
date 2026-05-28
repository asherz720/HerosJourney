# Generalization: Stimuli Design, Theory, and Scenario Taxonomy

**Last updated:** 2026-03-26
**Purpose:** Living reference for generalization design decisions — what types to test, why the sequential setting is necessary, how we differ from existing benchmarks, and what scenarios to build.

---

## 1. Research Framing

**Setup (revised):** No learn phase. The model receives source demonstrations (N episodes of solved tasks) and a teaching message, then is tested on a generalization task. The question: does the teaching message steer better in-context generalization than demonstrations alone?

**What we are testing:** Whether the model can extract an unstated rule from demonstrations and apply it to a novel instance with matching structure but different surface form.

---

## 2. Two Generalization Axes

### Axis 1: Entity-item mapping (concept induction)

**What generalizes:** Observable shared property across entities → correct required item.
The property is not stated in the task rules; the item requirement is absent from the gen entity's rule. The model must have extracted the property→item mapping from demonstrations.

**Sequential necessity:** The shared property is only visible when the model navigates to the entity's location — not in the initial rules. The model must explore to discover the cue. This is intrinsically sequential: a QA setup would hand the model the property; here the model must generate the observation by acting.

**Validity criterion:** The mapping must not be inferable from the entity name alone. Satisfied by: (a) designed observable properties in env observations (not name-derived), and (b) nonce-word variants that eliminate all pretraining signal.

---

### Axis 2: Implicit ordering rule (structural induction)

**What generalizes:** A hidden causal dependency between subgoals → correct ordering of actions.
The ordering rule is not stated anywhere in the task description. The model must extract it from demonstrations. Wrong order produces a hard failure (not just suboptimality).

**Sequential necessity:** Causal ordering constraints only matter in sequential execution — a static QA task cannot test whether a model plans to do B before C when C only becomes possible after B changes the world state.

**Validity criterion (critical):** The rule must NOT be derivable from the task description by a competent planner. This rules out:
- Resource arithmetic ("I have 100g, sword costs 80g, quest gives 50g → do quest first"): pure planning, no demo needed
- Stated prerequisites ("defeat demon to unlock passage"): instruction following, no induction needed
- LLM world knowledge ("rescue missions typically require clearing guards"): pretraining, not in-context

The rule must be **world-specific and opaque from the task description**, only visible in demo traces. Nonce-word variants eliminate pretraining confounds for ordering rules too.

**Hard failure enforcement:** Attempting the wrong order must fail mechanically, not just be suboptimal. This means the enabling condition must literally not exist before the required precondition action is taken.

---

## 3. Validity Criteria Summary

For any generalization scenario to be a clean test:

| Criterion | Why required |
|-----------|-------------|
| Rule not stated in task description | Otherwise instruction following |
| Rule not derivable by planner from task description | Otherwise just planning, not generalization |
| Wrong behavior → hard failure (not suboptimality) | Otherwise accidental success possible |
| Rule extractable from demo traces | Must have an inductive signal |
| Surface form changes between demos and test | Otherwise sequence copying, not rule induction |
| Nonce-word variant available | Blocks pretraining confounds |

**Planner baseline:** Any scenario should be tested with a model receiving the task description only (no demos, no teaching message). If this model succeeds, the scenario is not testing in-context generalization — it is testing planning, which is not the target.

---

## 4. Scenario Taxonomy

Each scenario type is defined by: (1) which axis it tests, (2) what the hidden rule is, (3) what makes wrong behavior fail hard.

---

### Type 1A — Feature-to-item mapping *(Axis 1)*

**Hidden rule:** Entities sharing observable property F require item I.
F is visible in env observation when model navigates to entity. Not in rules. Not inferable from entity name.

**Demo signal:** Multiple episodes showing entities with property F, all solved with item I.

**Test:** New entity has property F; item rule is absent. Model must navigate to entity, observe F, infer item, acquire it.

**Hard failure:** Item absent from entity's rule; wrong item → defeat fails.

**Nonce variant:** F = "zorbex-radiant" (made-up descriptor); I = "grundle_blade" (made-up item). No pretraining association possible.

**Why sequential:** F only visible after navigating to entity's location. Model must act (navigate) to obtain the cue.

---

### Type 1B — Attribute-relational mapping *(Axis 1, harder)*

**Hidden rule:** Entity attributes (e.g., social class + occupation) → item attributes (e.g., size + color of required tool).

**Demo signal:** Multiple NPC episodes where the (class, occupation) → (tool size, tool color) pattern is consistent.

**Test:** New NPC with unseen (class, occupation) combination; item rule absent. Model must infer correct tool from attribute relation.

**Hard failure:** Wrong tool attributes → task fails.

**Nonce variant:** Attribute names and values are nonce words; relation is purely in-context.

**Why sequential:** NPC attributes may be revealed progressively (class visible on approach, occupation visible on interaction). Model must interact to gather full attribute profile.

---

### Type 2A — Enabling dependency *(Axis 2)*

**Hidden rule:** Completing subgoal B changes world state in a way that is not stated in any rule; this state change is a necessary precondition for subgoal C to succeed.

**Demo signal:** Multiple episodes where: B completed → C succeeds; episodes where C attempted before B → C fails with a specific failure message.

**Test:** New task with same structural pattern (B enables C) but different surface form: different entities, different locations, different nonce item names. Model must extract "B before C" from the structural pattern in demos, not from stated rules.

**Hard failure:** Attempting C before B → hard failure message. Model stuck.

**Nonce variant:** The causal relationship uses entirely invented mechanics with nonce names so no pretraining association between B-type actions and C-type actions exists.

**Why sequential:** The enabling relationship only manifests in sequential execution. State must actually change. A QA model asked "what order should you do these in?" might guess correctly from surface semantics — here the model must have extracted the actual causal dependency.

**Example surface:** Performing ritual X (nonce) on altar Y (nonce) → artifact Z (nonce) becomes interactive → can use Z to complete rescue. No natural-language association between ritual, altar, and artifact.

---

### Type 2B — Destructive constraint *(Axis 2)*

**Hidden rule:** Completing subgoal C first irreversibly destroys the precondition for subgoal B.

**Demo signal:** Episodes showing B then C succeeds; episodes where C first → B permanently fails.

**Test:** New task, same structural pattern, nonce surface. Model must infer "do B first" from demos.

**Hard failure:** Doing C first → B's precondition destroyed → B fails → overall task fails.

**Nonce variant:** C's effect on B's precondition is an invented world mechanic, not inferable from names.

---

### Type 3 — Combined: concept + ordering *(Axis 1 + Axis 2)*

**Hidden rules:**
- (Axis 1) Entity-item mapping: defeating demon-type entity requires specific item (learned from feature cue)
- (Axis 2) Enabling dependency: defeating that entity is a precondition for a second subgoal (rescue or collection), non-obviously so

**Structure:** Episode has two goals. Goal 1 (defeat) requires: identify entity category from observable feature → acquire correct item. Completing Goal 1 enables Goal 2 (rescue). The enabling relationship is not stated.

**Why both axes are independently necessary:**
- Without concept rule: does not know which item for the defeat → stuck on Goal 1
- Without ordering rule: attempts Goal 2 (rescue) before Goal 1 (defeat) → hard failure even if item is known
- Both needed: identify item (concept) AND infer enabling structure (ordering)

**Test:** New episode: new entity with same category feature, new rescue target, same causal structure. Item rule absent; ordering rule absent.

**Nonce variant:** Category feature name, item name, and causal mechanic name are all nonce words.

---

### Type 4 — Multi-instance reliability *(Axis 1 diagnostic)*

**What this tests:** Not a new generalization type — a diagnostic on top of Types 1A/1B. Same rule must be applied to N entity instances within one episode, including correct non-application to out-of-class entities.

**Why useful:** Single correct application could be accidental. N correct applications + K correct rejections rules out guessing. Tests reliability and rule scope simultaneously.

**Structure:** Gen episode contains 2–3 category members (all with absent item rules) and 2 out-of-class entities (complete rules, different items). Model must: apply rule correctly to all category members, not apply it to out-of-class entities.

---

## 5. Information Disclosure Design

What is given to the model at episode start vs. discovered in-episode vs. never given:

| Information | Given at episode start | Discovered in-episode | Never given |
|-------------|----------------------|----------------------|-------------|
| Goal and distractor rules (complete, for non-target entities) | Yes | — | — |
| Target entity's item requirement | No (absent from rule) | — | — |
| Observable feature of target entity (Type 1) | No | Yes, on navigation to entity | — |
| World connectivity (how to reach goal location) | Yes | — | — |
| Items available and where to buy them | Yes | — | — |
| Causal enabling relationship (Type 2) | No | Visible as failure/success in demos | — |
| Entity class/category label | No | No | — |
| Ordering rule | No | Visible in demo trace structure | — |

**Source demonstrations:** N solved episodes shown as context. These encode:
- For Axis 1: action-observation sequences where the entity's feature appears in an observation and the correct item is used
- For Axis 2: action sequences where subgoal ordering is visible, and ideally failure traces where wrong order produces a specific failure message

**Teaching message:** Generated from source demo traces. Its job is to make the hidden rule more explicit — pointing to patterns in demos, not asserting class rules the model cannot verify.

---

## 6. How This Differs from Existing Benchmarks

| | SCAN/COGS | ALFRED/ALFWorld | gSCAN | This work |
|--|--|--|--|--|
| Rule stated explicitly? | No | Yes (instructions given) | Yes (command given) | No — must be induced |
| Generalization target | Novel linguistic compositions | Novel object-action combinations | Novel composed navigation commands | Hidden entity-item rules + hidden causal ordering |
| Sequential necessity | No (static output) | Execution overhead | Execution overhead | Intrinsic: cue only on navigation; ordering only testable in execution |
| Teacher / teaching message | None | None | None | Central variable — what steers generalization |
| Nonce control | Partial | No | No | Systematic variant for all types |
| Planner can solve from description? | N/A | With instructions, yes | With command, yes | No — rule absent from description |

**Single-sentence framing:** Existing benchmarks test novel combinations of given rules (instruction-following) or novel compositions of given primitives (compositional generalization); this work tests whether a teaching message helps a model induce and apply rules that are never stated, only demonstrated.

---

## 7. Theoretical Grounding

Each scenario type operationalizes a defined generalization type:

| Scenario | Theoretical type | Key references |
|----------|-----------------|----------------|
| 1A (feature → item) | Category induction from perceptual observation | Ashby et al. (1998) COVIS implicit system; Shepard (1987) generalization gradient |
| 1B (attribute relation) | Relational concept learning | Gentner (1983) structure mapping; Holyoak & Thagard (1989) ACME |
| 2A (enabling dependency) | Causal model induction from sequential observation | Gershman & Niv (2010) latent structure induction in RL; Laskin et al. (2022) Algorithm Distillation |
| 2B (destructive constraint) | Constraint learning from negative outcomes | Tenenbaum & Griffiths (2001) Bayesian concept induction |
| 3 (combined) | Compositional policy generalization | Lake et al. (2019) meta seq2seq; Xie et al. (2022) ICL as Bayesian inference |
| 4 (multi-instance) | Systematic generalization | Fodor & Pylyshyn (1988) systematicity; Hupkes et al. (2020) substitutivity |

**Formal criterion for "rule learned" (not exemplar copied):**
- **Substitutivity** (Hupkes et al. 2020): rule applies uniformly to all novel instances of the category, regardless of surface similarity to training exemplars
- **Size principle compliance** (Tenenbaum & Griffiths 2001): sharper generalization boundary with fewer, tighter examples
- **Label-flip diagnostic** (Min et al. 2022): replacing demo item with wrong item (same outcome) should degrade performance if model is doing genuine rule induction, not format recognition

---

## 8. References

| Paper | Citation | Relevance |
|-------|----------|-----------|
| Fodor & Pylyshyn (1988) | *Cognition* 28, 3–71 | Systematicity criterion — rule-based generalization is symmetric across all class members |
| Shepard (1987) | *Science* 237, 1317–1323 | Universal generalization law — exponential gradient; Bayesian derivation |
| Tenenbaum & Griffiths (2001) | *BBS* 24(4), 629–640 | Size principle — narrow concept from tight examples; Bayesian concept induction |
| Nosofsky et al. (1994) RULEX | *Psych. Review* 101(1), 53–79 | Rule + exception model — when rules vs. exemplars govern generalization |
| Ashby et al. (1998) COVIS | *Psych. Review* 105(3), 442–481 | Dual-system categorization: explicit (verbal rule) vs. implicit (reinforcement from observation) |
| Gentner (1983) | *Cognitive Science* 7(2), 155–170 | Structure mapping — relational transfer, not surface similarity |
| Holyoak & Thagard (1989) | *Cognitive Science* 13(3), 295–355 | Multiconstraint analogy — structural constraint rejects out-of-class mappings |
| Lake et al. (2015) | *Science* 350(6266), 1332–1338 | One-shot concept learning via program induction |
| Lake & Baroni (2018) SCAN | ICML 2018 | Operationalized systematicity benchmark |
| Ruis et al. (2020) gSCAN | NeurIPS 2020 | Grounded compositional instruction following — closest existing benchmark |
| Hupkes et al. (2020) | *JAIR* 67, 757–795 | Five compositionality types — substitutivity and systematicity most relevant |
| Xie et al. (2022) | ICLR 2022, arXiv:2111.02080 | ICL as implicit Bayesian concept inference |
| Min et al. (2022) | EMNLP 2022 | Label-flip diagnostic — format recognition vs. genuine rule induction |
| Wei et al. (2023) | arXiv:2303.03846 | Emergent rule override at scale |
| Laskin et al. (2022) | ICLR 2023, arXiv:2210.14215 | Algorithm Distillation — in-context RL; sequential policy generalization |
| Gershman & Niv (2010) | *Curr. Opin. Neurobiol.* 20(2), 251–256 | Latent structure induction in RL — transfer via task-type recognition |
| Anderson (1982) | *Psych. Review* 89(4), 369–406 | Procedural vs. declarative knowledge; skill acquisition as rule compilation |
| Lake et al. (2019) | ICML 2019 | Meta seq2seq for compositional generalization from demonstrations |
