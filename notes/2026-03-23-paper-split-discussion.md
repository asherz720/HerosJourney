# Paper Split Discussion — 2026-03-23

## Question
Should the project be split into two papers:
1. **Dataset paper** — the environment and stimuli (task suite, generalization splits)
2. **Method paper** — the teacher-student LLM steering pipeline

---

## What Dataset / Benchmark Papers Look Like

**Primary venues:** NeurIPS Datasets & Benchmarks Track, EMNLP, ACL (Resource track)

**Typical structure:**
1. **Motivation** — what gap in existing benchmarks does this fill? What capability does it probe that nothing else does?
2. **Dataset construction** — design principles, generation process, quality control
3. **Task taxonomy** — types of tasks, difficulty axes, splits (train/test/OOD)
4. **Baseline experiments** — current SOTA fails in interesting ways; shows the benchmark is non-trivial
5. **Dataset analysis** — statistics, inter-task relationships, what the benchmark reveals about model behavior
6. **Release** — code, data, leaderboard

Key reviewer question: **Would someone outside this group use this benchmark for a different purpose?**

---

## What Method Papers Look Like

**Primary venues:** NeurIPS main, ICLR, ICML, EMNLP

**Typical structure:**
1. **Problem formulation** — formal setup
2. **Method** — the core algorithm
3. **Experiments** — main results, ablations, baselines
4. **Analysis** — what drives performance, failure modes
5. **Related work**

---

## Honest Take on the Split

The two-paper route makes sense only if the dataset is independently compelling. The key question:

> *Would a researcher studying something other than teacher-student steering want to use this task suite?*

If the tasks probe a distinct capability missing from existing benchmarks (ScienceWorld, TWC, ALFWorld, BALROG), dataset paper stands alone. If the task design is primarily motivated by the method, they're entangled and splitting creates two weaker papers.

**Hybrid option:** Submit to **NeurIPS D&B track** as a single paper combining the benchmark + baseline results + the method. That track explicitly wants "benchmarks + analysis of existing systems on novel datasets."

**Hold off on committing to the split until knowing:** how many distinct task types exist and how diverse they are. If 4 generalization axes are well-covered with clean controlled splits + nonce-word probes, the dataset paper is defensible. If gen tasks are few or narrow, one combined paper is stronger.

---

## Specific Suggestions

### Dataset Paper

**Motivation angle:** Existing LLM adventure game benchmarks (TextWorld, ComplexWorld) test overall task completion but don't isolate *specific generalization axes*. This benchmark uniquely controls for:
- **Substitution generalization** — same structure, different entities; nonce-word variant probes memorization vs. structural learning
- **Compositional search** — multi-attribute NPC → tool mapping
- **Macro reuse** — repeating subtask patterns
- **Hierarchical planning** — bottom-up prerequisite satisfaction

The nonce-word variant is a particularly strong methodological contribution.

**Stimuli suggestions:**
- Make the learn/gen split structure the centerpiece — show statistically that gen tasks require structural generalization, not surface-level pattern matching
- Include a "leakage analysis" showing nonce-word performance cannot be explained by pretraining data
- Add a human baseline (even small-scale) — reviewers will ask

**Experiment suggestions:**
- Baseline LLMs of varying sizes (Qwen 7B, 27B, GPT-4o, Claude) with zero-shot, few-shot, CoT
- Show models succeed on learn tasks but fail on gen tasks — the generalization gap is the finding
- Difficulty calibration: completion rate vs. tree depth, number of substitution levels, etc.

### Method Paper

**Motivation angle:** LLMs fail to generalize compositionally even when they understand individual steps. Key insight: *failure traces contain information needed to synthesize a teaching message* — but this requires a teacher with meta-cognitive ability to reason about why the student failed.

**Experiment suggestions:**
- Core result: Phase 1 vs. Phase 3 generalization gap (with teaching)
- Ablations: static vs. dynamic strategies; teacher model quality; with/without ref_generalization
- Validation set curation loop convergence curve (does reflexion actually improve over iterations?)
- Qualitative analysis: what do the best teaching messages actually say?

---

## Strongest Combined Story

> "We introduce a benchmark specifically designed to test compositional generalization axes, show current LLMs fail on it, and show that teacher-student LLM steering closes most of the gap."

---

## Related Work / Benchmark Comparisons
- TextWorld, ComplexWorld (IJCAI 2023), ScienceWorld, ALFWorld
- BALROG (ICLR 2025) — agentic LLM benchmark including TextWorld
- LLF-Bench — evaluates AI agents learning from natural language feedback
- Steer-Bench (2025) — evaluates steerability of LLMs
