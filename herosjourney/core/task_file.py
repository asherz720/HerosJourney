"""
herosjourney/core/task_file.py
Declarative task registration from JSON (or YAML) files.

File format (JSON, .json):
    {
      "name": "my_task",
      "rules": "rules/my_task.json",
      "process": "property_flat",
      "split": {"fn": "two_offset", "seed": 0},
      "max_tries": 5,
      "validate_fn": "validate_additive_split",
      "eval": {
        "correct_rule": "...",
        "description": "..."
      }
    }

The rules path may be absolute or relative to the task file's directory.
YAML files (.yaml / .yml) are also accepted if pyyaml is installed.

Usage:
    from herosjourney import register_task
    register_task("path/to/my_task.json")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, field_validator

_VALID_SPLIT_FNS = frozenset({
    "two_offset",
    "conditional_two_offset",
    "c4_override",
    "independent_leave_one_out",
})

_VALID_VALIDATE_FNS = frozenset({
    "validate_additive_split",
    "validate_independent_split",
    "validate_conditional_split",
})


class TaskFileSpec(BaseModel):
    """Schema for a declarative task definition file."""
    name: str
    rules: str                              # path to rule JSON (relative to task file or absolute)
    process: str = "property_flat"          # registered process template name
    split: Optional[Dict[str, Any]] = None
    max_tries: Optional[int] = None
    validate_fn: Optional[str] = None       # name of a built-in validator, or null
    eval: Optional[Dict[str, str]] = None   # {correct_rule, description}
    distractors: Optional[str] = None       # path to distractor spec, or null for default
    lexicon: str = "default"

    @field_validator("validate_fn")
    @classmethod
    def _check_validate_fn(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in _VALID_VALIDATE_FNS:
            raise ValueError(
                f"Unknown validate_fn {v!r}. "
                f"Built-in options: {sorted(_VALID_VALIDATE_FNS)}. "
                "For a custom validator, use register_task() with keyword arguments instead."
            )
        return v

    @field_validator("split")
    @classmethod
    def _check_split(cls, v: Optional[Dict]) -> Optional[Dict]:
        if v is not None:
            fn = v.get("fn")
            if fn and fn not in _VALID_SPLIT_FNS:
                raise ValueError(
                    f"Unknown split.fn {fn!r}. Valid: {sorted(_VALID_SPLIT_FNS)}"
                )
        return v


def _resolve_validate_fn(name: Optional[str]):
    if name is None:
        return None
    from herosjourney.core.function_specs.splits import (
        validate_additive_split,
        validate_independent_split,
        validate_conditional_split,
    )
    return {
        "validate_additive_split":    validate_additive_split,
        "validate_independent_split": validate_independent_split,
        "validate_conditional_split": validate_conditional_split,
    }[name]


def load_task_file(path: str | Path) -> None:
    """Parse a task definition JSON/YAML file and register the task.

    Args:
        path: Path to a .json, .yaml, or .yml task definition file.
              The ``rules`` path inside the file may be relative to the
              task file's own directory.

    Raises:
        FileNotFoundError: If the task file or the rules file it references
            does not exist.
        ValueError: If the task file fails schema validation.
        ImportError: If a YAML file is given but pyyaml is not installed.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "pyyaml is required for YAML task files. "
                "Install it with:  pip install pyyaml"
            ) from None
        raw: Dict = yaml.safe_load(path.read_text())
    elif suffix == ".json":
        raw = json.loads(path.read_text())
    else:
        raise ValueError(
            f"Unsupported file extension {suffix!r}. Use .json, .yaml, or .yml."
        )

    spec = TaskFileSpec.model_validate(raw)

    # Resolve rules path relative to the task file's directory
    rules_path = Path(spec.rules)
    if not rules_path.is_absolute():
        rules_path = (path.parent / rules_path).resolve()
    if not rules_path.exists():
        raise FileNotFoundError(
            f"Rule file not found: {rules_path}  "
            f"(resolved from {spec.rules!r} relative to {path.parent})"
        )

    # Resolve distractors path if given
    distractors = spec.distractors
    if distractors:
        d_path = Path(distractors)
        if not d_path.is_absolute():
            d_path = (path.parent / d_path).resolve()
        distractors = str(d_path)

    from herosjourney.core.generator import register_task
    register_task(
        name        = spec.name,
        rules       = str(rules_path),
        process     = spec.process,
        split       = spec.split,
        eval        = spec.eval,
        validate_fn = _resolve_validate_fn(spec.validate_fn),
        max_tries   = spec.max_tries,
        distractors = distractors,
        lexicon     = spec.lexicon,
    )
