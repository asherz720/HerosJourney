"""
Example 03 — Plug in your own agent (model_fn).

The "agent" is any callable:
    model_fn(prompt: str, max_tokens: int = 512)
        -> (response_text: str | None, thinking: str | None, token_counts: dict | None)

You can wrap any LLM client. Three patterns are shown below; only one needs to
be active. Run:  python examples/03_custom_agent.py
"""
import json

from herosjourney import get_task
from herosjourney.core.elements import fill_elements, load_lexicons
from herosjourney.core.demo_generator import generate_mixed_demos
from herosjourney.runner.adventure_episode import (
    run_single_episode, construct_demo_context,
)


# --- Pattern A: the built-in generic OpenAI-compatible adapter (no model_fn) ---
# Set OPENAI_BASE_URL / OPENAI_API_KEY, then pass model_path="...".
#   result = run_single_episode(..., model_path="gpt-4o-mini")

# --- Pattern B: wrap the OpenAI SDK yourself ---
def openai_model_fn(prompt, max_tokens=512):
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY / OPENAI_BASE_URL from env
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    u = resp.usage
    return (
        resp.choices[0].message.content,
        None,
        {"prompt_tokens": u.prompt_tokens,
         "candidates_tokens": u.completion_tokens,
         "total_tokens": u.total_tokens},
    )


# --- Pattern C: any other provider (Anthropic, local HF pipeline, etc.) ---
def my_model_fn(prompt, max_tokens=512):
    # text = my_client.generate(prompt, max_tokens=max_tokens)
    # return text, None, None
    raise NotImplementedError("Wire up your model here.")


def main():
    spec = get_task("additive")
    sem_lex, nonce_lex = load_lexicons()
    with open(spec.rules) as f:
        rule = json.load(f)
    elements = fill_elements(rule, sem_lex, nonce_lex, seed=0, split_spec=spec.split)
    source_tasks = spec.gen_fn(elements, split="source", use_nonce=False)
    gen_tasks    = spec.gen_fn(elements, split="gen",    use_nonce=False)
    demo_context = construct_demo_context(generate_mixed_demos(source_tasks, distractor_tasks=[]))

    # Swap in openai_model_fn / my_model_fn, or use model_path= for Pattern A.
    print("Set up a real model_fn (see patterns above) to run this against a model.")
    print("Example call:")
    print("    run_single_episode(episode_idx=0, task=gen_tasks[0],")
    print("        demo_context=demo_context, max_runs=None, verbose=True,")
    print("        truncate_window=None, model_fn=openai_model_fn, source_tasks=source_tasks)")


if __name__ == "__main__":
    main()
