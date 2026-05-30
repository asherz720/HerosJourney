"""Maps task-type strings to TaskSpec metadata; use get_task() to look up registered tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from herosjourney.core.generator import GeneratedTask, TemplateSpec


# ---------------------------------------------------------------------------
# PropertySpec — what is being tested
# ---------------------------------------------------------------------------

@dataclass
class PropertySpec:
    task_type:   str
    rules:       str
    validate_fn: Optional[Callable]
    correct_rule: str
    description: str = ""
    max_tries:   Optional[int] = None
    split:       Optional[Dict] = None


# ---------------------------------------------------------------------------
# ProcessSpec — how the process is structured
# ---------------------------------------------------------------------------

@dataclass
class ProcessSpec:
    name:          str
    mapping_nodes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TaskSpec
# ---------------------------------------------------------------------------

@dataclass
class TaskSpec:
    property_spec:    PropertySpec
    processes:        List[ProcessSpec]
    process_selector: Callable
    gen_fn:           Callable
    template_name:    str
    distractor_rules: Optional[str] = None
    distractor_template: Optional[str] = None

    # Convenience pass-throughs to PropertySpec fields
    @property
    def task_type(self) -> str:
        return self.property_spec.task_type

    @property
    def correct_rule(self) -> str:
        return self.property_spec.correct_rule

    @property
    def rules(self) -> str:
        return self.property_spec.rules

    @property
    def description(self) -> str:
        return self.property_spec.description

    @property
    def max_tries(self) -> Optional[int]:
        return self.property_spec.max_tries

    @property
    def split(self) -> Optional[Dict]:
        return self.property_spec.split


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, TaskSpec] = {}


def register_task(spec: TaskSpec) -> None:
    """Register a TaskSpec. Raises if the task_type is already registered."""
    if spec.task_type in TASK_REGISTRY:
        raise ValueError(f"Task type '{spec.task_type}' is already registered.")
    TASK_REGISTRY[spec.task_type] = spec


def get_task(task_type: str) -> TaskSpec:
    """Return the TaskSpec for task_type, or raise a descriptive error."""
    _ensure_builtins()
    if task_type not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task_type '{task_type}'. "
            f"Registered types: {sorted(TASK_REGISTRY)}"
        )
    return TASK_REGISTRY[task_type]


# ---------------------------------------------------------------------------
# Built-in registration (lazy, triggered on first get_task call)
# ---------------------------------------------------------------------------

_builtins_loaded = False


def _ensure_builtins() -> None:
    global _builtins_loaded
    if not _builtins_loaded:
        _builtins_loaded = True
        import herosjourney.core.tasks  # noqa — triggers all task registrations
