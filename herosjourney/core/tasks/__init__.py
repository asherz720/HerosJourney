"""
herosjourney/core/tasks/__init__.py
Import all built-in task modules to trigger their registrations.

To add a new task type:
  1. Create herosjourney/core/tasks/my_task.py and call register_task()
  2. Add one import line here
  3. Create herosjourney/core/rules/my_task.json
"""
from herosjourney.core.tasks import additive      # noqa: F401
from herosjourney.core.tasks import compositional  # noqa: F401
from herosjourney.core.tasks import conditional    # noqa: F401
from herosjourney.core.tasks import override       # noqa: F401
from herosjourney.core.tasks import proc_comp      # noqa: F401
from herosjourney.core.tasks import proc_cond      # noqa: F401
from herosjourney.core.tasks import proc_over      # noqa: F401
from herosjourney.core.tasks import proc_add       # noqa: F401
