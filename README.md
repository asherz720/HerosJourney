# Hero's Journey

A benchmark for testing whether language models can **induce hidden rules from
demonstrations** and act on them in a goal-directed, text-based adventure game.

An agent plays an RPG-style game. It sees a set of rules for the current task,
but some rules are deliberately **hidden**. It must infer the missing
requirements by studying **demonstration episodes** (in-context examples) and
then apply the inferred pattern to a novel entity — and *execute* a multi-step
plan, not just state the answer.

See the paper for the full design and experiments. This package is the reusable
framework; our paper experiments live separately.

## Install

```bash
pip install herosjourney                 # core: task generation + env + eval
pip install "herosjourney[runner]"       # + a generic OpenAI-compatible model adapter
pip install "herosjourney[yaml]"         # + YAML task-definition files
pip install "herosjourney[analysis]"     # + pandas/numpy/matplotlib for metrics & figures
```

Python 3.10+.

## The four concepts

| Concept   | What it is                                            | Where it lives |
|-----------|-------------------------------------------------------|----------------|
| **Rule**  | The abstract attribute→item/process mapping (no surface names) | `*.json` rule file (`herosjourney/core/rules/`) |
| **Task**  | A rule + a process structure + a source/gen split     | a task spec `*.json` registered via `register_task` |
| **Agent** | Your model, wrapped as a `model_fn(prompt) -> text`   | any callable you pass to `run_single_episode` |
| **Method**| An induction strategy layered on the agent (ReAct/HR/IDEA/ACE) | `episode_mode=` + `herosjourney/runner/strategies.py` |

## Quick start

```python
import json
from herosjourney import get_task, compute_ecsr
from herosjourney.core.elements import fill_elements, load_lexicons
from herosjourney.core.demo_generator import generate_mixed_demos
from herosjourney.runner.adventure_episode import run_single_episode, construct_demo_context

# 1. RULE + TASK — pick a built-in task (additive | compositional | conditional |
#    override | proc_add | proc_comp | proc_cond | proc_over)
spec = get_task("additive")

# 2. Surface-realize the rule into a concrete variant (seed controls names)
sem_lex, nonce_lex = load_lexicons()
with open(spec.rules) as f:
    rule = json.load(f)
elements = fill_elements(rule, sem_lex, nonce_lex, seed=0, split_spec=spec.split)

# Source entities (shown in demos) and gen entities (what we evaluate)
source_tasks = spec.gen_fn(elements, split="source", use_nonce=False)
gen_tasks    = spec.gen_fn(elements, split="gen",    use_nonce=False)

# 3. Build the in-context demonstrations
demos        = generate_mixed_demos(source_tasks, distractor_tasks=[])
demo_context = construct_demo_context(demos)

# 4. AGENT — wrap your model as model_fn(prompt, max_tokens) -> (text, thinking, tokens)
def my_model_fn(prompt, max_tokens=512):
    text = my_llm(prompt)            # call your model however you like
    return text, None, None

# 5. Run one episode on a gen task and score it
result = run_single_episode(
    episode_idx=0,
    task=gen_tasks[0],
    demo_context=demo_context,
    max_runs=None,                   # defaults to reference_length * num_tries
    verbose=False,
    truncate_window=None,
    model_fn=my_model_fn,
    source_tasks=source_tasks,
)
print(result.success, result.efficiency)

# ECSR (efficiency-calibrated success rate) over a set of results
ecsr = compute_ecsr([result], n_tries=spec.max_tries)
```

### Using a hosted/local model without writing a `model_fn`

Install the `runner` extra and point the generic OpenAI-compatible adapter at any
endpoint (OpenAI, vLLM, LM Studio, Ollama, …):

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"   # your server
export OPENAI_API_KEY="EMPTY"                        # or your real key
```

```python
result = run_single_episode(..., model_path="my-model-name")
```

Or from the command line:

```bash
adventure-story --task_type additive \
    --elements herosjourney/core/rules/additive.json \
    --model my-model-name --num_tries 2 --num_workers 4
```

## Adding your own task (no Python required)

Write a **rule file** (`my_rule.json`, see
`herosjourney/core/rules/RULE_FORMAT.md`) and a **task spec**:

```json
{
  "name": "my_task",
  "rules": "my_rule.json",
  "process": "property_flat",
  "split": {"fn": "two_offset", "seed": 0},
  "max_tries": 5,
  "validate_fn": "validate_additive_split",
  "eval": {"correct_rule": "…", "description": "…"}
}
```

```python
import herosjourney
herosjourney.register_task("path/to/my_task.json")   # .yaml also works with [yaml]
spec = herosjourney.get_task("my_task")
```

For custom validators or per-episode process variation, `register_task(...)`
also accepts keyword arguments. See `CLAUDE.md` for the full architecture.

## Applying an induction method

```python
# episode_mode selects a steering strategy applied on top of your agent
result = run_single_episode(..., model_fn=my_model_fn, episode_mode="idea")
# "standard" (default), "react", "hr", "idea"
```

## Evaluation

- **ECSR** (efficiency-calibrated success rate) — `herosjourney.compute_ecsr`:
  `success_rate × normalized_efficiency`, where efficiency = `reference_length / num_runs`
  and the floor is `1 / n_tries`.
- **RV** (rule verbalization) — an LLM judge scores a model's free-text rule
  description; prompts are in `herosjourney.eval.judge`.

## License

MIT — see [LICENSE](LICENSE).
