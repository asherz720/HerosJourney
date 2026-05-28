"""
tree_management/registry.py
Task registry: maps task-type strings to all metadata needed to run them.

Usage
-----
    from tree_management.registry import get_task

    spec  = get_task("additive")
    tasks = spec.gen_fn(filled_elements, split="source", use_nonce=False)

Registering a new task type
----------------------------
    from tree_management.generator import register_task_type, load_process
    from tree_management.function_specs.splits import validate_additive_split

    register_task_type(
        name             = "my_task",
        process          = load_process("property_flat"),
        rules            = "tree_management/rules/my_task.json",
        correct_rule     = "The rule is ...",
        mc_answer        = "B",
        distractor_rules = "tree_management/distractors/canonical_property_distractor.json",
        validate_fn      = validate_additive_split,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tree_management.generator import GeneratedTask, TemplateSpec


# ---------------------------------------------------------------------------
# PropertySpec — what is being tested
# ---------------------------------------------------------------------------

@dataclass
class PropertySpec:
    """
    Describes the property (rule) being tested — independent of process structure.

    Fields
    ------
    task_type   : short identifier used in CLI flags and filenames
    rules       : repo-relative path to the canonical rule file
    validate_fn : optional split validator callable(elements) → None
    correct_rule: ground-truth rule description for the LLM judge (structure_exp)
    mc_answer   : correct multiple-choice letter ("A"–"E") for structure_mc
    description : one-line description shown in experiment listings
    """
    task_type:   str
    rules:       str
    validate_fn: Optional[Callable]
    correct_rule: str
    mc_answer:   str
    description: str = ""
    max_tries:   Optional[int] = None
    split:       Optional[Dict] = None


# ---------------------------------------------------------------------------
# ProcessSpec — how the process is structured
# ---------------------------------------------------------------------------

@dataclass
class ProcessSpec:
    """
    Describes one process structure (process JSON + mapping_nodes).

    A task may have multiple ProcessSpecs (e.g. for procedural tasks that
    vary in process structure).  The process_selector on TaskSpec picks which
    one to use per entity/seed.

    Fields
    ------
    name          : identifier (matches a registered TemplateSpec)
    mapping_nodes : step IDs in the process that carry the hidden property
    """
    name:          str
    mapping_nodes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TaskSpec
# ---------------------------------------------------------------------------

@dataclass
class TaskSpec:
    """
    All metadata needed to run one task type end-to-end.

    Fields
    ------
    property_spec    : PropertySpec describing what is being tested
    processes        : list of ProcessSpec (one for standard tasks, multiple
                       for procedural tasks that vary in process structure)
    process_selector : callable (entity, seed) -> ProcessSpec; selects which
                       process structure to use for a given entity/seed
    gen_fn           : callable (elements_dict, split, use_nonce) → List[GeneratedTask]
    template_name    : primary template name (matches processes[0].name for standard tasks)
    distractor_rules : repo-relative path to the default distractor rule file
    distractor_template : template name for building distractor tasks
    """
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
    def mc_answer(self) -> str:
        return self.property_spec.mc_answer

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
        import tree_management.tasks  # noqa — triggers all task registrations
