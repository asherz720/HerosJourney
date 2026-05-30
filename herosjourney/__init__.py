"""
Hero's Journey — benchmark for rule induction in goal-directed episodic tasks.

Quick start
-----------
    from herosjourney import get_task, AdventureEnv, EpisodeResult, compute_ecsr
    from herosjourney.core.elements import fill_elements, load_lexicons
    from herosjourney.core.demo_generator import generate_mixed_demos

    # 1. Pick a built-in task
    spec = get_task("additive")   # additive | compositional | conditional | override
                                  # proc_add | proc_comp    | proc_cond   | proc_over

    # 2. Generate surface-realized elements (seed controls lexicon sampling)
    import json
    sem_lex, nonce_lex = load_lexicons()
    with open(spec.rules) as f:
        rule = json.load(f)
    elements = fill_elements(rule, sem_lex, nonce_lex, seed=0, split_spec=spec.split)

    # 3. Build tasks and demos
    tasks = spec.gen_fn(elements, split="source", use_nonce=False)
    demos = generate_mixed_demos(tasks, distractor_tasks=[])

    # 4. Run episodes (supply your own model_fn: (prompt, max_tokens) -> (response, thinking, tokens))
    env = AdventureEnv(trees=[(t.tree, t.tree.root_id) for t in tasks])

    # 5. Compute ECSR
    ecsr = compute_ecsr(results, n_tries=spec.max_tries)

Adding a new task — file-based (recommended)
---------------------------------------------
    # my_task.json:
    # {
    #   "name": "my_task",
    #   "rules": "rules/my_task.json",
    #   "process": "property_flat",
    #   "split": {"fn": "two_offset", "seed": 0},
    #   "max_tries": 5,
    #   "validate_fn": "validate_additive_split",
    #   "eval": {"correct_rule": "...", "description": "..."}
    # }

    register_task("path/to/my_task.json")   # or .yaml with pyyaml installed

Adding a new task — keyword-based (programmatic)
-------------------------------------------------
    register_task(
        name        = "my_task",
        rules       = "/path/to/my_task_rule.json",
        process     = "property_flat",
        split       = {"fn": "two_offset", "seed": 0},
        eval        = {"correct_rule": "...", "description": "..."},
        validate_fn = None,
        max_tries   = 5,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__version__ = "0.1.0"

from herosjourney.core.registry  import get_task, TASK_REGISTRY, _ensure_builtins
from herosjourney.core.task_file import load_task_file
from herosjourney.env.env        import AdventureEnv
from herosjourney.eval.result    import EpisodeResult
from herosjourney.eval.metrics   import compute_ecsr, compute_norm_eff, success_rate

# Populate TASK_REGISTRY with the 8 built-in tasks at import time, so
# `herosjourney.TASK_REGISTRY` is non-empty without first calling get_task().
_ensure_builtins()


def register_task(path_or_name: Any = None, /, **kwargs) -> None:
    """Register a task from a JSON/YAML file path or keyword arguments.

    File-based (recommended — no Python required for new tasks):
        register_task("path/to/my_task.json")
        register_task("path/to/my_task.yaml")   # requires pyyaml

    Keyword-based (full control):
        register_task(name="my_task", rules="...", process="property_flat",
                      split={"fn": "two_offset", "seed": 0}, max_tries=5, ...)
    """
    if path_or_name is not None and not kwargs:
        load_task_file(path_or_name)
    else:
        from herosjourney.core.generator import register_task as _rt
        name = path_or_name if path_or_name is not None else kwargs.pop("name", None)
        _rt(name=name, **kwargs)


__all__ = [
    "__version__",
    # Task registry
    "get_task",
    "TASK_REGISTRY",
    "register_task",
    "load_task_file",
    # Environment
    "AdventureEnv",
    # Evaluation
    "EpisodeResult",
    "compute_ecsr",
    "compute_norm_eff",
    "success_rate",
]
