"""
tree_management/tasks/__init__.py
Import all built-in task modules to trigger their registrations.

To add a new task type:
  1. Create tree_management/tasks/my_task.py and call register_task_type()
  2. Add one import line here
  3. Create tree_management/rules/my_task.json
"""
from tree_management.tasks import additive      # noqa: F401
from tree_management.tasks import compositional  # noqa: F401
from tree_management.tasks import conditional    # noqa: F401
from tree_management.tasks import override       # noqa: F401
from tree_management.tasks import proc_comp      # noqa: F401
from tree_management.tasks import proc_cond      # noqa: F401
from tree_management.tasks import proc_over      # noqa: F401
from tree_management.tasks import proc_add       # noqa: F401
