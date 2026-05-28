# Paper Framing: How Teaching Messages Steer In-Context Generalization

**Date:** 2026-03-24
**Reference paper structure:** arXiv:2510.14318 (Abdulhai et al. — deceptive dialogue + multi-turn RL)

---

## The Reference Paper's Structure (for imitation)

The paper we're modeling follows a tight arc:

1. **Phenomenon definition** — formally define the thing being studied (deception; here: in-context generalization failure)
2. **Evaluation framework** — a benchmark + metrics that isolate the phenomenon
3. **Condition space** — a systematic set of interventions (baselines → proposed method), not just one method
4. **Results organized by question** — each experiment answers one research question; tables have a clear claim
5. **Ablation layer** — what drives the effect? (reward model quality, single vs. multi-turn RL, model size)
6. **Analysis layer** — qualitative + failure-mode analysis explaining the mechanism, not just the number

The paper reads as an *evaluation study first, method paper second*. The RL training is the proposed fix, but the evaluation framework is the primary contribution.

---

## Reframing Our Project

**Big eval question:**
> How do different teaching messages steer a student LLM to pick up in-context generalization?

This reframes the project from "we built a teacher-student pipeline" → "we systematically evaluate what message properties/generation strategies enable a student to generalize."

**Why this framing is stronger:**
- The benchmark (controlled generalization axes, nonce-word probes, learn/gen split) becomes a first-class contribution, not scaffolding
- The teaching strategies become a condition space for an eval, not just ablations of one method
- Negative results (strategies that don't help) become findings, not failures

---

## Paper Structure Under This Framing

### 1. Introduction
- LLMs fail at compositional generalization even in tasks they can do individually
- In-context learning can help but only with the right context — what makes context effective?
- We study the *teacher-student steering* setting: given a student's failures, can a teacher craft a message that enables generalization?
- Contributions: (a) benchmark with controlled generalization axes, (b) taxonomy of teaching strategies, (c) systematic evaluation, (d) findings on what works and why

### 2. Problem Formulation (formal)
- Define: task tree, learn/gen split, teaching message M, three-phase evaluation protocol
- Define: generalization gap = student perf on gen tasks (Phase 1) vs. with teaching (Phase 3)
- Define: what "steering" means — message M steers student if gen gap closes by δ

### 3. Benchmark / Task Suite
- Controlled generalization axes (substitution, search, macro_reuse, hierarchical)
- Nonce-word probes as anti-memorization controls
- Learn/gen split design — what structural property must the student learn to bridge the gap
- Statistics: number of tasks, tree depth distribution, etc.

### 4. Teaching Strategies (the condition space)
- **Static strategies** — organized by information richness:
  - `none` (Phase 1 reuse) → `thinker` (hints) → `truthteller` (reference solution) → `egen` (explicit generalization instruction)
- **Dynamic strategies** — organized by curation sophistication:
  - `observe_then_teach` (one-shot offline) → `reflexion` (iterative rewrite) → `ace` (structured accumulation)
- Frame as: information content axis × generation process axis

### 5. Experiments
- **Main result:** generalization gap (Phase 1 vs Phase 3) for each condition
- **Sub-questions:**
  - RQ1: Does any static message help? (information richness)
  - RQ2: Does dynamic curation improve over static? (process sophistication)
  - RQ3: Which generalization axis is hardest? (task-type breakdown)
  - RQ4: Does iteration help dynamic strategies? (convergence curves from Phase 1.1)
  - RQ5: Does student model size modulate steerability? (7B vs 27B)
  - RQ6: Does teacher model quality matter? (teacher model ablation)

### 6. Analysis
- What semantic content in the best messages predicts gen performance?
- When does teaching fail? (per-axis failure modes)
- Qualitative examples: what did the best message say? what did the student do differently?
- `ever_mentioned` / `ever_attempted` as process metrics (does the student engage with the right concept?)

### 7. Related Work
- In-context learning and what context helps (survey)
- LLM steering / alignment feedback (RLHF, constitutional AI, process reward)
- TextWorld / adventure game LLM benchmarks (BALROG, ALFWorld, LLF-Bench, Steer-Bench)
- Teacher-student / knowledge distillation via language

### 8. Conclusion

---

## Elements We Already Have

| Element | Status |
|---------|--------|
| Benchmark with controlled gen axes | ✓ Have (sub_regular, sub_nonceword, search variants, macro_reuse, hierarchical) |
| Learn/gen split design | ✓ Have |
| Nonce-word probes (sub_nonceword) | ✓ Have |
| Phase 1 baseline (generalization gap exists) | ✓ Have (running) |
| Full static strategy set (none, truthteller, thinker, egen) | ✓ Have |
| Full dynamic strategy set (reflexion, observe_then_teach, ace) | ✓ Have |
| Per-iteration performance curves (Phase 1.1 curation_history) | ✓ Logged in result JSONs |
| Process metrics (ever_mentioned, ever_attempted) | ✓ Have (for sub tasks) |
| Student size ablation scaffold | ✓ Partial (7B + 27B being run) |

---

## Missing Elements (gaps to fill for a complete paper)

### Missing experiments

| Gap | What's needed | Priority |
|-----|--------------|----------|
| **Results across all gen axes** | Run full pipeline (or at least Phase 1 + static) on all 7 ref_gen types, not just sub_regular | HIGH — this is the core "systematic eval" claim |
| **Convergence curves** | Extract per-iteration gen performance from `curation_history` across dynamic strategies; plot as figure | HIGH — needed to argue dynamic > static |
| **Teacher model ablation** | Same strategy (e.g., reflexion) with different teacher models (gemini, claude4.5, smaller model) | MEDIUM |
| **Student size ablation** | 7B vs 27B on identical conditions | MEDIUM (partially running) |
| **Upper-bound condition** | Student with oracle message (e.g., truthteller + full ref_gen) as ceiling | HIGH — calibrates effect size |
| **Multiple seeds / reliability** | Error bars on gen performance; current runs may be single-seed | MEDIUM |

### Missing analysis

| Gap | What's needed |
|-----|--------------|
| **Message content analysis** | Annotate or categorize what the best messages contain vs. worst; correlate with gen performance |
| **Per-axis difficulty ordering** | Which gen axis is hardest? Does teaching help uniformly or only on some? |
| **Failure mode analysis** | When does Phase 3 still fail despite teaching? What did the student do wrong? |
| **Process metric extension** | `ever_mentioned`/`ever_attempted` currently only for sub tasks — extend or analogize for other axes |

### Missing structural elements

| Gap | What's needed |
|-----|--------------|
| **Formal problem statement** | Written-out formal definition of the steering problem (task tree, message M, gen gap δ) |
| **Benchmark statistics table** | Task counts, tree depth distribution, learn/gen ratio per axis |
| **Human baseline** | Even small-scale; reviewers will ask "what does a human do?" |
| **Related work positioning** | Systematic comparison to BALROG, LLF-Bench, Steer-Bench; what does this benchmark uniquely measure? |

---

## Recommended Priority Order

1. **Run full pipeline across all gen axes with static strategies** — this is the minimum for "systematic eval." Even Phase 1 + Phase 3 with `truthteller` (static upper bound) across all 7 axes gives the core table.
2. **Extract + plot convergence curves** from existing Phase 1.1 logs — already have the data.
3. **Write the formal problem statement** — unlocks the framing for everything else.
4. **Student size ablation** — already partially running; just needs identical conditions for both sizes.
5. **Message content analysis** — qualitative pass over best/worst Phase 1.1 messages; can be a small analysis section.
6. **Human baseline** — low cost, high reviewer value; even 3-5 humans on a subset.

---

## Key Framing Insight

The reference paper's strength is that it separates:
- **Measurement** (does deception exist? how much?) from
- **Intervention** (can we reduce it?)

We should do the same:
- **Measurement:** how large is the generalization gap across axes? what does it look like? (Phase 1)
- **Intervention:** which teaching strategies close the gap and by how much? (Phase 2/3)

This makes the benchmark + Phase 1 results a standalone contribution, and the teaching strategies an evaluation over interventions — not just "our method vs. nothing."
