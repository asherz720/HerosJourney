# Examples

Runnable examples for the `herosjourney` package, in increasing order of scope.

| File | What it shows | Needs a model? |
|------|---------------|----------------|
| `01_run_builtin_task.py` | RULE → TASK → demos → episode → ECSR, with a mock oracle agent | No (offline) |
| `02_custom_task.py` | Register a custom task from a declarative JSON spec | No (offline) |
| `03_custom_agent.py` | Plug in your own model as a `model_fn` (the "agent") — three patterns | No (template) |
| `04_methods.py` | Apply an induction method (ReAct / HR / IDEA) on top of an agent | Template |
| `05_run_benchmark.py` | **Run a model on all 8 tasks and report ECSR + RV** (how we run the benchmark) | Yes |

## Running the benchmark (example 05)

`05_run_benchmark.py` reproduces the paper's evaluation: for each of the eight
tasks it averages **ECSR** (efficiency-calibrated success rate, from episodes)
and **RV** (rule-verbalization score, from an LLM judge) over several variants.

Serve your model behind any OpenAI-compatible endpoint and point the adapter at it:

```bash
# Example: a local vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-27B-Instruct --tensor-parallel-size 2 --port 8000

export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY

# All 8 tasks, 5 variants each (paper uses 20); ECSR + RV table
python examples/05_run_benchmark.py --model Qwen/Qwen2.5-27B-Instruct --num-variants 5

# A subset, nonce names, separate judge model:
python examples/05_run_benchmark.py --model my-model --tasks additive conditional \
    --nonce --judge-model my-judge-model
```

Output is a per-task table of ECSR and RV plus the mean across tasks.

> The offline examples (01, 02) use an oracle/mock agent and need no API calls —
> good for verifying an install. Examples 03–05 require a live model endpoint.
