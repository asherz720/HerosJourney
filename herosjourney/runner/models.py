"""
pipeline/models.py — backward-compat shim.

Model provider adapters (vLLM, OpenAI/Azure, Gemini, Bedrock) live in
experiments/models.py.  This shim re-exports everything from there so
existing imports of the form

    from herosjourney.runner.models import agent_response

continue to work without changes.  New code should import from
experiments.models directly.
"""
from experiments.models import *  # noqa: F401, F403
from experiments.models import (  # noqa: F401  (explicit for IDEs / type checkers)
    agent_response,
    json_converter,
    json_converter_small,
    json_converter_gemini,
    teacher_json_repair_small,
    teacher_json_repair_gemini,
)
