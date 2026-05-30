"""
herosjourney/runner/models.py
Generic OpenAI-compatible model adapter.

The episode loop is model-agnostic: it only needs a callable
    model_fn(prompt, max_tokens) -> (response_text, thinking, token_counts)
This module provides a default such callable backed by any OpenAI-compatible
chat-completions endpoint (OpenAI, vLLM, LM Studio, Ollama, TGI, ...).

Configuration (environment variables):
    HEROSJOURNEY_BASE_URL  or  OPENAI_BASE_URL   (default: https://api.openai.com/v1)
    HEROSJOURNEY_API_KEY   or  OPENAI_API_KEY    (default: "EMPTY", for local servers)

For provider-specific backends (Azure, Gemini, Bedrock) or bespoke decoding
parameters, write your own model_fn and pass it to run_single_episode(model_fn=...)
— you do not need to edit this file. The paper's exact provider adapters live in
experiments/models.py (not shipped with the package).

Requires the `openai` package:  pip install "herosjourney[runner]"
"""

from __future__ import annotations

import os
from typing import Optional, Tuple


def _client():
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "The generic model adapter needs the 'openai' package. "
            'Install it with:  pip install "herosjourney[runner]"\n'
            "Or supply your own model_fn to run_single_episode(model_fn=...)."
        ) from e
    base_url = os.environ.get("HEROSJOURNEY_BASE_URL") or os.environ.get(
        "OPENAI_BASE_URL", "https://api.openai.com/v1"
    )
    api_key = os.environ.get("HEROSJOURNEY_API_KEY") or os.environ.get(
        "OPENAI_API_KEY", "EMPTY"
    )
    return OpenAI(base_url=base_url, api_key=api_key, timeout=180)


def agent_response(
    model: str,
    prompt: str,
    max_tokens: int = 512,
) -> Tuple[Optional[str], Optional[str], Optional[dict]]:
    """Call an OpenAI-compatible chat endpoint.

    Returns (response_text, thinking, token_counts). `thinking` is always None
    for this generic adapter; token_counts uses the key names the episode loop
    expects ('prompt_tokens', 'candidates_tokens', 'total_tokens').
    """
    client = _client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=float(os.environ.get("HEROSJOURNEY_TEMPERATURE", "1.0")),
    )
    content = resp.choices[0].message.content
    usage = getattr(resp, "usage", None)
    token_counts = None
    if usage is not None:
        token_counts = {
            "prompt_tokens":     getattr(usage, "prompt_tokens", 0),
            "candidates_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens":      getattr(usage, "total_tokens", 0),
        }
    return (content.strip() if content else ""), None, token_counts


_JSON_CONVERTER_INSTRUCTION = (
    "You extract the single action a game agent wants to take from its text and "
    "return it as JSON matching this schema: {json_schema}. "
    "Action/argument formats: go [location], get [object], buy [object], "
    "perform [ritual], drink [potion], defeat [enemy], rescue [npc], "
    "check_inventory, check_location. Output ONLY the JSON object."
)


def json_converter(response_content: str, json_schema: dict, model: Optional[str] = None) -> str:
    """Repair/extract a JSON action from free-form model text via the same endpoint.

    `model` defaults to env HEROSJOURNEY_CONVERTER_MODEL, else HEROSJOURNEY_MODEL.
    """
    client = _client()
    conv_model = (
        model
        or os.environ.get("HEROSJOURNEY_CONVERTER_MODEL")
        or os.environ.get("HEROSJOURNEY_MODEL", "gpt-4o-mini")
    )
    instruction = _JSON_CONVERTER_INSTRUCTION.format(json_schema=json_schema)
    resp = client.chat.completions.create(
        model=conv_model,
        messages=[{
            "role": "user",
            "content": f"{instruction}\n\nModel generation:\n{response_content}",
        }],
        temperature=0,
        max_tokens=512,
    )
    return (resp.choices[0].message.content or "").strip()


# --- Backward-compatible aliases used by qa_episode.py / teacher.py ---
# In this generic adapter there is a single configured endpoint, so the
# "small" / "gemini" variants all route to the generic converter.

def json_converter_small(response_content: str, json_schema: dict) -> str:
    return json_converter(response_content, json_schema)


def json_converter_gemini(response_content: str, json_schema: dict) -> str:
    return json_converter(response_content, json_schema)


def teacher_json_repair_small(text: str, json_schema: dict) -> str:
    return json_converter(text, json_schema)


def teacher_json_repair_gemini(text: str, json_schema: dict) -> str:
    return json_converter(text, json_schema)
